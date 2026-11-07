//! Joint planet / binary / noise evidence at one trial period.
//!
//! Sufficient statistics are folded over two periods (2 nb bins) so cycle
//! parity is retained: a template at phase j is paired with its odd
//! counterpart at j + nb. Each local model integrates a positive eclipse
//! depth with a half-normal prior and sums normalized discrete scenario
//! masses: secondary eclipses at quarter phases for the planet and binary
//! families, and alternating primaries at twice the period for the binary
//! family. Phase, duration and shape are marginalized with uniform phase and
//! duration masses and normalized shape masses. Scores are conditional
//! period evidence ratios.
//!
//! Layouts: `cum` holds, per field (v, w, 1), an exclusive prefix sum over
//! the 2 nb bins (2 nb + 1 positions, field-major); w serves as the baseline
//! projection and as the diagonal norm. `shapes` holds, per
//! (shape, layer), the fractional width, the increment and the squared-sum
//! coefficient. `parts` holds 15 values per (tile, selection).

use super::kernels::{CUM_BLOCK, FOLD_BLOCK, NF};
use super::math::{log_add, positive_amplitude_evidence, subspace_upper};
use super::scan::{DeviceReady, Gpu};
use crate::config::SearchConfig;
use crate::prep::curve::FamilyState;
use cuda_core::{DeviceBuffer, LaunchConfig};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF64};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

pub const EV_TILE: usize = 64;
pub const PART: usize = 15;
pub const NORB: usize = 3;
pub const NPARAM: usize = 30;
const NEG_INF: f64 = f64::NEG_INFINITY;
const KEY_MAX: u64 = u64::MAX;

#[inline(always)]
fn circular_sum(cum: &[f64], row: usize, start: usize, width: usize, nb2: usize) -> f64 {
    let j = start % nb2;
    let end = j + width;
    let left = cum[row + j];
    let right = cum[row + end.min(nb2)];
    let wrap = cum[row + end.saturating_sub(nb2)];
    right - left + wrap
}

/// Integral of a staircase (or its square) as nested interval sums.
#[inline(always)]
fn layered_sum(cum: &[f64], row: usize, start: usize, width: usize, shapes: &[f64], si: usize, nb2: usize, layers: usize, squared: bool) -> f64 {
    let mut value = 0.0;
    for layer in 0..layers {
        let fraction = shapes[(si * layers + layer) * 3];
        let inner = ((width as f64 * fraction + 0.5).floor() as usize).max(1);
        let coefficient = shapes[(si * layers + layer) * 3 + if squared { 2 } else { 1 }];
        value += coefficient * circular_sum(cum, row, start + (width - inner) / 2, inner, nb2);
    }
    value
}

#[cuda_module]
pub mod evidence_gpu {
    use super::*;

    /// Fold the four fields over two periods: bin = floor(t * scale) mod 2 nb.
    #[kernel]
    pub fn fold_two(time: &[f64], fields: &[f64], scale: f64, folded: &[DeviceAtomicF64], n: u32, nb2: u32) {
        let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let n = n as usize;
        if i >= n {
            return;
        }
        let cell = ((time[i] * scale).floor() as i64).rem_euclid(nb2 as i64) as usize;
        for k in 0..NF {
            folded[k * nb2 as usize + cell].fetch_add(fields[k * n + i], AtomicOrdering::Relaxed);
        }
    }

    /// One block per field: exclusive prefix sums over 2 nb bins.
    #[kernel]
    pub fn prefix_fields(folded: &[DeviceAtomicF64], mut cum: DisjointSlice<f64>, nb2: u32) {
        static mut SCAN: SharedArray<f64, CUM_BLOCK> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let field = thread::blockIdx_x() as usize;
        let nb2 = nb2 as usize;
        let row_out = field * (nb2 + 1);
        if lane == 0 {
            unsafe {
                *cum.get_unchecked_mut(row_out) = 0.0;
            }
        }
        let mut carry = 0.0;
        let segments = nb2.div_ceil(CUM_BLOCK);
        for seg in 0..segments {
            let b = seg * CUM_BLOCK + lane;
            let v = if b < nb2 { folded[field * nb2 + b].load(AtomicOrdering::Relaxed) } else { 0.0 };
            unsafe {
                SCAN[lane] = v;
            }
            thread::sync_threads();
            let mut stride = 1;
            while stride < CUM_BLOCK {
                let add = if lane >= stride { unsafe { SCAN[lane - stride] } } else { 0.0 };
                thread::sync_threads();
                if lane >= stride {
                    unsafe {
                        SCAN[lane] += add;
                    }
                }
                thread::sync_threads();
                stride *= 2;
            }
            if b < nb2 {
                unsafe {
                    *cum.get_unchecked_mut(row_out + b + 1) = SCAN[lane] + carry;
                }
            }
            carry += unsafe { SCAN[CUM_BLOCK - 1] };
            thread::sync_threads();
        }
    }

    /// Amplitude screen on the identical bank: per tile the maximum score, the
    /// planet and binary evidence lower bounds of the best-merit template and
    /// its key.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn comparison_screen(
        cum: &[f64],
        scale: f64,
        durations: &[f64],
        rho: &[f64],
        duration_log_mass: &[f64],
        shapes: &[f64],
        shape_priors: &[f64],
        parameters: &[f64],
        baseline: f64,
        n: u32,
        nb2: u32,
        nd: u32,
        ns: u32,
        layers: u32,
        mut screen: DisjointSlice<f64>,
    ) {
        static mut S_BEST: SharedArray<f64, EV_TILE> = SharedArray::UNINIT;
        static mut S_MERIT: SharedArray<f64, EV_TILE> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let tile = thread::blockIdx_x() as usize;
        let nb2 = nb2 as usize;
        let nb = nb2 / 2;
        let (nd, ns, layers) = (nd as usize, ns as usize, layers as usize);
        let j = tile * EV_TILE + lane;
        let jj = j.min(nb - 1);
        let n = n as f64;
        let c = baseline;
        let mut best = NEG_INF;
        let mut bound_merit = NEG_INF;
        let (mut best_a, mut best_q, mut best_null, mut best_ps, mut best_bs) = (0.0, 1.0, 0.0, 0.0, 0.0);
        let mut best_key = 0u64;
        for di in 0..nd {
            let raw_k = ((durations[di] * scale + 0.5).floor() as usize).max(1);
            let k = raw_k.min(nb2 / 8 - 1);
            let r = rho[di];
            let count = circular_sum(cum, 2 * (nb2 + 1), jj, k, nb2) + circular_sum(cum, 2 * (nb2 + 1), jj + nb, k, nb2);
            let usable = j < nb && raw_k < nb2 / 8 && count >= 3.0 && count <= n - 3.0;
            for si in 0..ns {
                let a = layered_sum(cum, 0, jj, k, shapes, si, nb2, layers, false) + layered_sum(cum, 0, jj + nb, k, shapes, si, nb2, layers, false);
                let pe = layered_sum(cum, nb2 + 1, jj, k, shapes, si, nb2, layers, false);
                let po = layered_sum(cum, nb2 + 1, jj + nb, k, shapes, si, nb2, layers, false);
                let ne = layered_sum(cum, nb2 + 1, jj, k, shapes, si, nb2, layers, true);
                let no = layered_sum(cum, nb2 + 1, jj + nb, k, shapes, si, nb2, layers, true);
                let (qe, qo, qeo) = (ne - pe * pe / c, no - po * po / c, -pe * po / c);
                let q = qe + qo + 2.0 * qeo;
                let score = a / (q * (r * r)).max(1e-300).sqrt();
                let score_valid = usable && a > 0.0 && q > 0.0;
                let merit = 0.5 * score * score + duration_log_mass[di] + shape_priors[si * 2];
                if score_valid && merit > bound_merit {
                    best_a = a / (r * r);
                    best_q = q / (r * r);
                    best_null = duration_log_mass[di];
                    best_ps = shape_priors[si * 2];
                    best_bs = shape_priors[si * 2 + 1];
                    best_key = ((si * nd + di) as u64) * nb as u64 + jj as u64;
                    bound_merit = merit;
                }
                if score_valid && score > best {
                    best = score;
                }
            }
        }
        unsafe {
            S_BEST[lane] = best;
            S_MERIT[lane] = bound_merit;
        }
        thread::sync_threads();
        if lane == 0 {
            let mut maximum = NEG_INF;
            let mut bound_maximum = NEG_INF;
            let mut chosen = EV_TILE;
            for l in 0..EV_TILE {
                let (b, m) = unsafe { (S_BEST[l], S_MERIT[l]) };
                if b > maximum {
                    maximum = b;
                }
                if m > bound_maximum {
                    bound_maximum = m;
                    chosen = l;
                }
            }
            unsafe {
                S_MERIT[0] = chosen as f64;
                S_BEST[0] = maximum;
                S_BEST[1] = bound_maximum;
            }
        }
        thread::sync_threads();
        let chosen = unsafe { S_MERIT[0] } as usize;
        if lane == chosen {
            let maximum = unsafe { S_BEST[0] };
            let bound_maximum = unsafe { S_BEST[1] };
            let lower_p = positive_amplitude_evidence(best_a, best_q, parameters[0]) + 0.75f64.ln() + best_null + best_ps;
            let lower_b = log_add(positive_amplitude_evidence(best_a, best_q, parameters[1]), positive_amplitude_evidence(best_a, best_q, parameters[2]))
                + 0.025f64.ln()
                + best_null
                + best_bs;
            let out = tile * 4;
            unsafe {
                *screen.get_unchecked_mut(out) = maximum;
                *screen.get_unchecked_mut(out + 1) = if bound_maximum > NEG_INF { lower_p } else { NEG_INF };
                *screen.get_unchecked_mut(out + 2) = if bound_maximum > NEG_INF { lower_b } else { NEG_INF };
                *screen.get_unchecked_mut(out + 3) = best_key as f64;
            }
        }
    }

    /// Evidence of every ephemeris in a tile; the block keeps the winners by
    /// joint evidence and by amplitude. In pilot mode every lane evaluates
    /// the single template `pilot_key` and the output provides the floors.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn joint_tiles(
        cum: &[f64],
        scale: f64,
        durations: &[f64],
        rho: &[f64],
        duration_log_mass: &[f64],
        shapes: &[f64],
        shape_priors: &[f64],
        parameters: &[f64],
        orbit: &[f64],
        floors: &[f64],
        baseline: f64,
        threshold: f64,
        n: u32,
        nb2: u32,
        nd: u32,
        ns: u32,
        layers: u32,
        pilot: u32,
        mut parts: DisjointSlice<f64>,
    ) {
        static mut S_VAL: SharedArray<f64, EV_TILE> = SharedArray::UNINIT;
        static mut S_KEY: SharedArray<u64, EV_TILE> = SharedArray::UNINIT;
        static mut S_SUM: SharedArray<f64, EV_TILE> = SharedArray::UNINIT;
        static mut S_OUT: SharedArray<f64, 16> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let tile = thread::blockIdx_x() as usize;
        let nb2 = nb2 as usize;
        let nb = nb2 / 2;
        let (nd, ns, layers) = (nd as usize, ns as usize, layers as usize);
        let pilot = pilot != 0;
        let pilot_key = floors[2] as u64;
        let j = if pilot { (pilot_key % nb as u64) as usize } else { tile * EV_TILE + lane };
        let jj = j.min(nb - 1);
        let c = baseline;
        let n = n as f64;
        let (tau_p, tau_b, tau_d) = (parameters[0], parameters[1], parameters[2]);
        let (floor_p, floor_b) = if pilot { (NEG_INF, NEG_INF) } else { (floors[0], floors[1]) };

        let mut best_joint = NEG_INF;
        let mut best_signal = NEG_INF;
        let mut joint_key = KEY_MAX;
        let mut signal_key = KEY_MAX;
        let mut j_out = [0.0f64; 9];
        let mut s_out = [0.0f64; 9];
        let mut sum_p = NEG_INF;
        let mut sum_b = NEG_INF;
        let mut omitted = 0u32;

        let (di_begin, di_end) = if pilot { let d = ((pilot_key / nb as u64) % nd as u64) as usize; (d, d + 1) } else { (0, nd) };
        for di in di_begin..di_end {
            let raw_k = ((durations[di] * scale + 0.5).floor() as usize).max(1);
            let k = raw_k.min(nb2 / 8 - 1);
            let rho2 = rho[di] * rho[di];
            let dlm = duration_log_mass[di];
            let count = circular_sum(cum, 2 * (nb2 + 1), jj, k, nb2) + circular_sum(cum, 2 * (nb2 + 1), jj + nb, k, nb2);
            let usable = j < nb && raw_k < nb2 / 8 && count >= 3.0 && count <= n - 3.0;
            let (si_begin, si_end) = if pilot { let s = (pilot_key / (nb as u64 * nd as u64)) as usize; (s, s + 1) } else { (0, ns) };
            for si in si_begin..si_end {
                let ae = layered_sum(cum, 0, jj, k, shapes, si, nb2, layers, false);
                let ao = layered_sum(cum, 0, jj + nb, k, shapes, si, nb2, layers, false);
                let pe = layered_sum(cum, nb2 + 1, jj, k, shapes, si, nb2, layers, false);
                let po = layered_sum(cum, nb2 + 1, jj + nb, k, shapes, si, nb2, layers, false);
                let ne = layered_sum(cum, nb2 + 1, jj, k, shapes, si, nb2, layers, true);
                let no = layered_sum(cum, nb2 + 1, jj + nb, k, shapes, si, nb2, layers, true);
                let (qe, qo, qeo) = (ne - pe * pe / c, no - po * po / c, -pe * po / c);
                let (ap, qp) = (ae + ao, qe + qo + 2.0 * qeo);
                let snr = ap / (qp * rho2).max(1e-300).sqrt();
                let depth = ap / qp.max(1e-300);
                let valid = usable && ap > 0.0 && qp > 0.0 && snr >= threshold;
                if raw_k < nb2 / 8 {
                    let mut lp = NEG_INF;
                    let mut lb = NEG_INF;
                    let mut component_best = NEG_INF;
                    let mut sec_ratio = 0.0;
                    let mut sec_phase = 0.5;
                    let mut odd_ratio = 1.0;
                    let mut binary_multiple = 1.0;
                    let mut omitted_upper = NEG_INF;
                    for phase_index in 0..NORB {
                        let orbital = (di * NORB + phase_index) * 4;
                        let phase = orbit[orbital];
                        let duration_ratio = orbit[orbital + 1];
                        let (logwp, logwb) = (orbit[orbital + 2], orbit[orbital + 3]);
                        let offset = (phase * nb as f64 + 0.5).floor() as i64;
                        let sk = (((k as f64 * duration_ratio + 0.5).floor() as i64).max(1)).min(nb as i64 - 1) as usize;
                        let s0 = (jj as i64 + offset + ((k as f64 - sk as f64) / 2.0).floor() as i64) as usize;
                        let s1 = s0 + nb;
                        let ass = layered_sum(cum, 0, s0, sk, shapes, si, nb2, layers, false) + layered_sum(cum, 0, s1, sk, shapes, si, nb2, layers, false);
                        let ps = layered_sum(cum, nb2 + 1, s0, sk, shapes, si, nb2, layers, false) + layered_sum(cum, nb2 + 1, s1, sk, shapes, si, nb2, layers, false);
                        let nsum = layered_sum(cum, nb2 + 1, s0, sk, shapes, si, nb2, layers, true) + layered_sum(cum, nb2 + 1, s1, sk, shapes, si, nb2, layers, true);
                        let (qs, qps) = (nsum - ps * ps / c, -(pe + po) * ps / c);
                        let (pa, pb, pq, pr, px) = (ap / rho2, ass / rho2, qp / rho2, qs / rho2, qps / rho2);
                        let cu = 1e-6;
                        let pu = subspace_upper(pa, pb, pq, pr, px, 1.0) + logwp + cu;
                        let bu = subspace_upper(pa, pb, pq, pr, px, 1.0) + logwb + cu;
                        let mut upper = (pu - floor_p).max(bu - floor_b) + dlm;
                        if !(logwp > NEG_INF || logwb > NEG_INF) {
                            upper = NEG_INF;
                        }
                        let raw_upper = if logwb > NEG_INF { bu } else { NEG_INF };
                        unsafe {
                            S_SUM[lane] = if j < nb { upper } else { NEG_INF };
                        }
                        thread::sync_threads();
                        let mut tile_max = NEG_INF;
                        for l in 0..EV_TILE {
                            let v = unsafe { S_SUM[l] };
                            if v > tile_max {
                                tile_max = v;
                            }
                        }
                        thread::sync_threads();
                        if tile_max >= 0.0 {
                            for ri in 0..5 {
                                let ratio = parameters[6 + ri * 3];
                                let (wp, wb) = (parameters[7 + ri * 3], parameters[8 + ri * 3]);
                                let aa = pa + ratio * pb;
                                let qq = pq + 2.0 * ratio * px + ratio * ratio * pr;
                                let ep = positive_amplitude_evidence(aa, qq, tau_p) + wp.ln() + logwp;
                                let diluted = positive_amplitude_evidence(aa, qq, tau_d);
                                let eb = log_add(positive_amplitude_evidence(aa, qq, tau_b), diluted) + (0.25 * wb).ln() + logwb;
                                lp = log_add(lp, ep);
                                lb = log_add(lb, eb);
                                if eb > component_best {
                                    sec_ratio = ratio;
                                    sec_phase = phase;
                                    component_best = eb;
                                }
                            }
                        } else {
                            omitted += 1;
                            if raw_upper > omitted_upper {
                                omitted_upper = raw_upper;
                            }
                        }
                    }
                    let (ba, bb, bq, br, bx) = (ae / rho2, ao / rho2, qe / rho2, qo / rho2, qeo / rho2);
                    let alias_mass = if k < nb2 / 8 { 0.0 } else { NEG_INF };
                    let raw_upper = subspace_upper(ba, bb, bq, br, bx, 1.0) + 1e-6 + alias_mass;
                    let upper = raw_upper + dlm;
                    unsafe {
                        S_SUM[lane] = if j < nb { upper } else { NEG_INF };
                    }
                    thread::sync_threads();
                    let mut tile_max = NEG_INF;
                    for l in 0..EV_TILE {
                        let v = unsafe { S_SUM[l] };
                        if v > tile_max {
                            tile_max = v;
                        }
                    }
                    thread::sync_threads();
                    if tile_max >= floor_b {
                        for ri in 0..7 {
                            let ratio = parameters[21 + ri];
                            let aa = ba + ratio * bb;
                            let qq = bq + 2.0 * ratio * bx + ratio * ratio * br;
                            let diluted = positive_amplitude_evidence(aa, qq, tau_d);
                            let eb = log_add(positive_amplitude_evidence(aa, qq, tau_b), diluted) + alias_mass + (0.25 / 7.0f64).ln();
                            lb = log_add(lb, eb);
                            if eb > component_best {
                                sec_ratio = 0.0;
                                sec_phase = 0.5;
                                odd_ratio = ratio;
                                binary_multiple = 2.0;
                                component_best = eb;
                            }
                        }
                    } else {
                        omitted += 1;
                        if raw_upper > omitted_upper {
                            omitted_upper = raw_upper;
                        }
                    }
                    let certain = component_best > omitted_upper;
                    if !certain {
                        sec_ratio = f64::NAN;
                        sec_phase = f64::NAN;
                        odd_ratio = f64::NAN;
                        binary_multiple = f64::NAN;
                    }
                    lp += shape_priors[si * 2] + dlm;
                    lb += shape_priors[si * 2 + 1] + dlm;
                    if j < nb {
                        sum_p = log_add(sum_p, lp);
                        sum_b = log_add(sum_b, lb);
                    }
                    let joint = lp;
                    let key = ((si * nd + di) as u64) * nb as u64 + jj as u64;
                    let bj = valid && (joint > best_joint || (joint == best_joint && key < joint_key));
                    let bs = valid && (snr > best_signal || (snr == best_signal && key < signal_key));
                    if bj {
                        j_out = [snr, depth, lp, lb, sec_ratio, sec_phase, odd_ratio, binary_multiple, joint];
                        best_joint = joint;
                        joint_key = key;
                    }
                    if bs {
                        s_out = [snr, depth, lp, lb, sec_ratio, sec_phase, odd_ratio, binary_multiple, joint];
                        best_signal = snr;
                        signal_key = key;
                    }
                }
            }
        }
        for selection in 0..2 {
            let (merit, key) = if selection == 0 { (best_joint, joint_key) } else { (best_signal, signal_key) };
            unsafe {
                S_VAL[lane] = merit;
                S_KEY[lane] = key;
                S_SUM[lane] = if selection == 0 { sum_p } else { sum_b };
            }
            thread::sync_threads();
            if lane == 0 {
                let mut maximum = NEG_INF;
                for l in 0..EV_TILE {
                    let v = unsafe { S_VAL[l] };
                    if v > maximum {
                        maximum = v;
                    }
                }
                let mut best_key = KEY_MAX;
                let mut chosen = EV_TILE;
                for l in 0..EV_TILE {
                    let (v, k) = unsafe { (S_VAL[l], S_KEY[l]) };
                    if v == maximum && k < best_key {
                        best_key = k;
                        chosen = l;
                    }
                }
                let mut high = NEG_INF;
                for l in 0..EV_TILE {
                    let v = unsafe { S_SUM[l] };
                    if v > high {
                        high = v;
                    }
                }
                let log_sum = if high > NEG_INF {
                    let mut acc = 0.0;
                    for l in 0..EV_TILE {
                        acc += (unsafe { S_SUM[l] } - high).exp();
                    }
                    high + acc.ln()
                } else {
                    high
                };
                unsafe {
                    S_OUT[0] = best_key as f64;
                    S_OUT[1] = maximum;
                    S_OUT[2] = chosen as f64;
                    S_OUT[3] = log_sum;
                }
            }
            thread::sync_threads();
            let chosen = unsafe { S_OUT[2] } as usize;
            if lane == chosen.min(EV_TILE - 1) {
                let vals = if selection == 0 { j_out } else { s_out };
                let out = (tile * 2 + selection) * PART;
                unsafe {
                    *parts.get_unchecked_mut(out) = S_OUT[0];
                    *parts.get_unchecked_mut(out + 1) = S_OUT[1];
                    *parts.get_unchecked_mut(out + 2) = if selection == 0 { vals[0] } else { best_signal };
                    *parts.get_unchecked_mut(out + 3) = vals[1];
                    *parts.get_unchecked_mut(out + 4) = vals[2];
                    *parts.get_unchecked_mut(out + 5) = vals[3];
                    *parts.get_unchecked_mut(out + 6) = vals[4];
                    *parts.get_unchecked_mut(out + 7) = vals[5];
                    *parts.get_unchecked_mut(out + 8) = vals[6];
                    *parts.get_unchecked_mut(out + 9) = vals[7];
                    *parts.get_unchecked_mut(out + 10) = if selection == 0 { best_joint } else { vals[8] };
                    *parts.get_unchecked_mut(out + 13) = omitted as f64;
                }
            }
            if lane == 0 {
                let out = (tile * 2 + selection) * PART;
                unsafe {
                    *parts.get_unchecked_mut(out + 11 + selection) = S_OUT[3];
                }
            }
            thread::sync_threads();
        }
        if lane == 0 {
            // Both selections carry both sums, as the merge reads them per selection.
            let out0 = tile * 2 * PART;
            unsafe {
                let sp = *parts.get_unchecked_mut(out0 + 11);
                let sb = *parts.get_unchecked_mut(out0 + PART + 12);
                *parts.get_unchecked_mut(out0 + 12) = sb;
                *parts.get_unchecked_mut(out0 + PART + 11) = sp;
            }
        }
    }
}

/// Evidence row: the 14 table columns of the joint (selection 1) or
/// amplitude (selection 2) winner at one period; flags 3 when both coincide.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct EvidenceRow {
    pub period: f64,
    pub epoch: f64,
    pub duration: f64,
    pub snr: f64,
    pub depth: f64,
    pub planet_log_odds: f64,
    pub log_bf_planet_noise: f64,
    pub log_bf_binary_noise: f64,
    pub binary_secondary_ratio: f64,
    pub binary_secondary_phase: f64,
    pub binary_odd_even_ratio: f64,
    pub binary_period_days: f64,
    pub ingress_fraction: f64,
    pub selection_flags: u32,
}

fn logsumexp(values: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = values.collect();
    let high = v.iter().cloned().fold(NEG_INF, f64::max);
    if high == NEG_INF { high } else { high + v.iter().map(|x| (x - high).exp()).sum::<f64>().ln() }
}

pub fn parameters(cfg: &SearchConfig) -> Vec<f64> {
    let dilution = cfg.target_flux_fraction;
    let mut planet_scale = cfg.planet_depth_scale;
    if let Some(radius) = cfg.stellar_radius_solar {
        planet_scale = (0.10045 * cfg.planet_radius_scale_rj / radius).powi(2);
    }
    let mut p = vec![
        planet_scale * dilution,
        cfg.binary_depth_scale * dilution,
        cfg.diluted_binary_depth_scale,
        cfg.planet_prior.ln(),
        cfg.binary_prior.ln(),
        (1.0 - cfg.planet_prior - cfg.binary_prior).ln(),
    ];
    let ratios = [0.0, 0.01, 0.1, 0.5, 1.0];
    let wp = [0.75, 0.15, 0.08, 0.015, 0.005];
    let wb = [0.10, 0.10, 0.20, 0.25, 0.35];
    for i in 0..5 {
        p.extend([ratios[i], wp[i], wb[i]]);
    }
    p.extend([0.25, 0.5, 0.75, 1.0, 4.0 / 3.0, 2.0, 4.0]);
    p.extend([0.001, 0.05]);
    assert_eq!(p.len(), NPARAM);
    p
}

/// Joint evidence at one trial period for one family. Returns the rows of
/// the joint and amplitude winners (merged when identical).
pub fn compare_search(gpu: &Gpu, state: &FamilyState, period: f64, cfg: &SearchConfig) -> Result<Vec<EvidenceRow>, Box<dyn std::error::Error>> {
    let mut ds: Vec<(f64, f64)> = state.durations.iter().cloned().zip(state.rho.iter().cloned()).filter(|d| d.0 < period / 4.0).collect();
    ds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    if ds.is_empty() {
        return Ok(Vec::new());
    }
    let durations: Vec<f64> = ds.iter().map(|d| d.0).collect();
    let rho: Vec<f64> = ds.iter().map(|d| d.1).collect();
    let nd = durations.len();
    let ns = cfg.ingress_fractions.len();
    let layers = cfg.shape_layers;
    let ready = &state.ready;
    let n = ready.time.len();
    let dmin = durations[0];
    let nb = (4 * ((cfg.phase_oversample as f64 * period / dmin / 4.0).ceil() as usize)).max(8);
    let nb2 = 2 * nb;
    let scale = nb as f64 / period;

    let mut dlm: Vec<f64> = rho.iter().map(|r| -0.5 * ((n as f64 - 1.0) * (r * r).ln() + ready.residual_energy / (r * r))).collect();
    let dmax = dlm.iter().cloned().fold(NEG_INF, f64::max);
    dlm.iter_mut().for_each(|x| *x -= dmax);

    let logits: Vec<(f64, f64)> = cfg.ingress_fractions.iter().map(|&i| (-4.0 * i, 2.0 * i)).collect();
    let (lp0, lb0) = (logsumexp(logits.iter().map(|l| l.0)), logsumexp(logits.iter().map(|l| l.1)));
    let shape_priors: Vec<f64> = logits.iter().flat_map(|l| [l.0 - lp0, l.1 - lb0]).collect();
    let mut shapes = Vec::with_capacity(ns * layers * 3);
    for &ing in &cfg.ingress_fractions {
        let mut previous = 0.0;
        for layer in 0..layers {
            let fraction = 1.0 - 2.0 * ing * layer as f64 / (layers.max(2) - 1) as f64;
            let inc = 1.0 / layers as f64;
            shapes.extend([fraction, inc, inc * (2.0 * previous + inc)]);
            previous += inc;
        }
    }
    let params = parameters(cfg);

    // Orbit bank without extended physics: secondary phases 0.25, 0.5, 0.75.
    let phases = [0.25, 0.5, 0.75];
    let logw = [0.1f64.ln(), 0.8f64.ln(), 0.1f64.ln()];
    let mut lp_bank = vec![NEG_INF; nd * NORB];
    let mut lb_bank = vec![NEG_INF; nd * NORB];
    let mut valid_count = 0usize;
    for di in 0..nd {
        let k = (durations[di] * scale + 0.5).floor().max(1.0);
        if k < (nb / 4) as f64 {
            valid_count += 1;
        }
        for o in 0..NORB {
            let ks = (k + 0.5).floor().max(1.0);
            let shift = (phases[o] * nb as f64 + 0.5).floor();
            let start = shift + ((k - ks) / 2.0).floor();
            let valid = k < (nb / 4) as f64 && ks < (nb / 2) as f64 && start >= k && start + ks <= nb as f64;
            if valid {
                lp_bank[di * NORB + o] = logw[o];
                lb_bank[di * NORB + o] = logw[o];
            }
        }
    }
    let count = (valid_count as f64).ln();
    let norm_p = count - logsumexp(lp_bank.iter().cloned());
    let norm_b = count - logsumexp(lb_bank.iter().cloned());
    let mut orbit = Vec::with_capacity(nd * NORB * 4);
    for di in 0..nd {
        for o in 0..NORB {
            orbit.extend([phases[o], 1.0, lp_bank[di * NORB + o] + norm_p, lb_bank[di * NORB + o] + norm_b]);
        }
    }

    let dev = DeviceReady::upload(gpu, ready)?;
    let stream = &gpu.stream;
    let d_dur = DeviceBuffer::from_host(stream, &durations)?;
    let d_rho = DeviceBuffer::from_host(stream, &rho)?;
    let d_dlm = DeviceBuffer::from_host(stream, &dlm)?;
    let d_shapes = DeviceBuffer::from_host(stream, &shapes)?;
    let d_priors = DeviceBuffer::from_host(stream, &shape_priors)?;
    let d_params = DeviceBuffer::from_host(stream, &params)?;
    let d_orbit = DeviceBuffer::from_host(stream, &orbit)?;
    let folded = DeviceBuffer::<f64>::zeroed(stream, NF * nb2)?.wrap_elem::<DeviceAtomicF64>();
    let mut cum = DeviceBuffer::<f64>::zeroed(stream, NF * (nb2 + 1))?;
    let tiles = nb.div_ceil(EV_TILE);
    let mut screen = DeviceBuffer::<f64>::zeroed(stream, tiles * 4)?;
    let module = &gpu.evidence;
    // SAFETY: grid shapes match each kernel's indexing and every buffer covers the kernel's accesses.
    unsafe {
        module.fold_two(stream, LaunchConfig { grid_dim: ((n as u32).div_ceil(FOLD_BLOCK), 1, 1), block_dim: (FOLD_BLOCK, 1, 1), shared_mem_bytes: 0 }, &dev.time, &dev.fields, scale, &folded, n as u32, nb2 as u32)?;
        module.prefix_fields(stream, LaunchConfig { grid_dim: (NF as u32, 1, 1), block_dim: (CUM_BLOCK as u32, 1, 1), shared_mem_bytes: 0 }, &folded, &mut cum, nb2 as u32)?;
        module.comparison_screen(
            stream,
            LaunchConfig { grid_dim: (tiles as u32, 1, 1), block_dim: (EV_TILE as u32, 1, 1), shared_mem_bytes: 0 },
            &cum,
            scale,
            &d_dur,
            &d_rho,
            &d_dlm,
            &d_shapes,
            &d_priors,
            &d_params,
            ready.baseline_norm,
            n as u32,
            nb2 as u32,
            nd as u32,
            ns as u32,
            layers as u32,
            &mut screen,
        )?;
    }
    let screen = screen.to_host_vec(stream)?;
    let maximum = (0..tiles).map(|t| screen[t * 4]).fold(NEG_INF, f64::max);
    if !(maximum >= cfg.threshold * (1.0 - 1e-12)) {
        return Ok(Vec::new());
    }
    let winner = (0..tiles).max_by(|&a, &b| screen[a * 4 + 1].partial_cmp(&screen[b * 4 + 1]).unwrap()).unwrap();
    let pilot_key = screen[winner * 4 + 3];
    let margin = if cfg.evidence_rtol > 0.0 { cfg.evidence_rtol.ln() - (((NORB + 1) * nb * nd * ns) as f64).ln() } else { NEG_INF };

    let mut parts = DeviceBuffer::<f64>::zeroed(stream, tiles * 2 * PART)?;
    let d_floors = DeviceBuffer::from_host(stream, &[NEG_INF, NEG_INF, pilot_key])?;
    unsafe {
        module.joint_tiles(
            stream,
            LaunchConfig { grid_dim: (1, 1, 1), block_dim: (EV_TILE as u32, 1, 1), shared_mem_bytes: 0 },
            &cum,
            scale,
            &d_dur,
            &d_rho,
            &d_dlm,
            &d_shapes,
            &d_priors,
            &d_params,
            &d_orbit,
            &d_floors,
            ready.baseline_norm,
            cfg.threshold,
            n as u32,
            nb2 as u32,
            nd as u32,
            ns as u32,
            layers as u32,
            1,
            &mut parts,
        )?;
    }
    let pilot = parts.to_host_vec(stream)?;
    let floors = [pilot[11] + margin, pilot[12] + margin, pilot_key];
    let d_floors = DeviceBuffer::from_host(stream, &floors)?;
    unsafe {
        module.joint_tiles(
            stream,
            LaunchConfig { grid_dim: (tiles as u32, 1, 1), block_dim: (EV_TILE as u32, 1, 1), shared_mem_bytes: 0 },
            &cum,
            scale,
            &d_dur,
            &d_rho,
            &d_dlm,
            &d_shapes,
            &d_priors,
            &d_params,
            &d_orbit,
            &d_floors,
            ready.baseline_norm,
            cfg.threshold,
            n as u32,
            nb2 as u32,
            nd as u32,
            ns as u32,
            layers as u32,
            0,
            &mut parts,
        )?;
    }
    let parts = parts.to_host_vec(stream)?;

    let in_bank: Vec<bool> = durations.iter().map(|d| (d * scale + 0.5).floor().max(1.0) < (nb / 4) as f64).collect();
    let log_mass = (nb as f64).ln() + logsumexp((0..nd).filter(|&i| in_bank[i]).map(|i| dlm[i]));
    let mut rows = Vec::new();
    for selection in 0..2 {
        let value = |t: usize, i: usize| parts[((t * 2 + selection) * PART) + i];
        let maximum = (0..tiles).map(|t| value(t, 1)).fold(NEG_INF, f64::max);
        let key = (0..tiles).filter(|&t| value(t, 1) == maximum).map(|t| value(t, 0)).fold(f64::INFINITY, f64::min);
        let tile = (0..tiles).find(|&t| value(t, 1) == maximum && value(t, 0) == key).unwrap_or(0);
        let lp = logsumexp((0..tiles).map(|t| value(t, 11))) - log_mass;
        let lb = logsumexp((0..tiles).map(|t| value(t, 12))) - log_mass;
        let joint = params[3] + lp - log_add(params[5], params[4] + lb);
        let decoded = if maximum > NEG_INF { key as u64 } else { 0 };
        let j = (decoded % nb as u64) as usize;
        let di = ((decoded / nb as u64) % nd as u64) as usize;
        let si = (decoded / (nb as u64 * nd as u64)) as usize;
        let width = (durations[di] * scale + 0.5).floor().max(1.0);
        let snr = if maximum > NEG_INF { value(tile, 2) } else { NEG_INF };
        rows.push(EvidenceRow {
            period,
            epoch: (j as f64 + width / 2.0) / scale + ready.origin,
            duration: width / scale,
            snr,
            depth: value(tile, 3),
            planet_log_odds: joint,
            log_bf_planet_noise: lp,
            log_bf_binary_noise: lb,
            binary_secondary_ratio: value(tile, 6),
            binary_secondary_phase: value(tile, 7),
            binary_odd_even_ratio: value(tile, 8),
            binary_period_days: value(tile, 9) * period,
            ingress_fraction: cfg.ingress_fractions[si.min(ns - 1)],
            selection_flags: if selection == 0 { 1 } else { 2 },
        });
    }
    let hit: Vec<bool> = rows.iter().map(|r| r.snr.is_finite()).collect();
    let same = hit[0]
        && hit[1]
        && rows[0].period == rows[1].period
        && rows[0].epoch == rows[1].epoch
        && rows[0].duration == rows[1].duration
        && rows[0].snr == rows[1].snr
        && rows[0].depth == rows[1].depth
        && rows[0].ingress_fraction == rows[1].ingress_fraction;
    let mut out = Vec::new();
    if hit[0] {
        let mut r = rows[0];
        if same {
            r.selection_flags = 3;
        }
        out.push(r);
    }
    if hit[1] && !same {
        out.push(rows[1]);
    }
    Ok(out)
}
