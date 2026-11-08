//! CUDA kernels of the fused diagonal scan.
//!
//! Memory layouts, all row-major:
//! - `fields`: NF x n, field-major (v, w, 1); the baseline projection p equals
//!   w for the diagonal noise model, so one field serves both.
//! - `folded`: periods x nb x NF, interleaved fields per phase bin.
//! - `cum`: periods x NF x (2 nb + 1), field-major exclusive prefix sums over
//!   the bins repeated twice, so any interval [start, start + width) with
//!   start < nb and width <= nb reads without wrap-around and consecutive
//!   lanes read consecutive words.
//! - `cum32`: the same sums as (hi, lo) f32 pairs, hi = f32(cum),
//!   lo = f32(cum - hi). The difference of two nearby prefix sums is
//!   (hi_b - hi_a) + (lo_b - lo_a): the hi difference is exact (Sterbenz) and
//!   the lo difference carries the rounding, so the f32 pre-ranking reads
//!   interval sums at f32 precision without f64 arithmetic.
//! - `parts`: periods x tiles x 9 (phase, width, a, norm, d, ingress, rho, score, depth).
//!
//! Scores are ranked in f32 by the merit a^2 / (rho^2 (c norm - hp^2)) and the
//! winner is recomputed in f64. Tiles with near ties or out-of-range merits are
//! rescanned in f64.

use cuda_device::atomic::{AtomicOrdering, BlockAtomicF64, DeviceAtomicF64, DeviceAtomicU32};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

pub const TILE: usize = 128;
/// Folded fields: v, w (also the baseline projection), count.
pub const NF: usize = 3;
pub const F_V: usize = 0;
pub const F_W: usize = 1;
pub const F_N: usize = 2;
pub const FOLD_BLOCK: u32 = 256;
/// Periods folded per block in the global-atomic fold.
pub const FOLD_PERIODS: usize = 4;
pub const CUM_BLOCK: usize = 256;
/// Largest nb whose NF-field f64 histogram fits the 48 KiB static shared memory.
pub const FOLD_SHARED_BINS: usize = 2048;
pub const PART_STRIDE: usize = 9;
pub const RECORD_STRIDE: usize = 9;
pub const EVENT_STRIDE: usize = 4;
pub const RESCORE_BLOCK: usize = 256;
/// Points per block and shared-memory window of the biweight detrend kernel.
pub const BIWEIGHT_BLOCK: usize = 128;
pub const BIWEIGHT_WINDOW: usize = 4096;

const NEG_INF: f64 = f64::NEG_INFINITY;
const KEY_MAX: u64 = u64::MAX;

/// `row` points at field 0 of the period; `stride` is 2 nb + 1.
#[inline(always)]
fn interval(
    cum: &[f64],
    row: usize,
    stride: usize,
    field: usize,
    start: usize,
    width: usize,
) -> f64 {
    let base = row + field * stride;
    unsafe { *cum.get_unchecked(base + start + width) - *cum.get_unchecked(base + start) }
}

/// f32 interval sum from the (hi, lo) pairs stored at twice the f64 index.
#[inline(always)]
fn interval32(
    cum32: &[f32],
    row: usize,
    stride: usize,
    field: usize,
    start: usize,
    width: usize,
) -> f32 {
    let base = 2 * (row + field * stride + start);
    let end = base + 2 * width;
    unsafe {
        (*cum32.get_unchecked(end) - *cum32.get_unchecked(base))
            + (*cum32.get_unchecked(end + 1) - *cum32.get_unchecked(base + 1))
    }
}

/// Numerator, baseline projection and diagonal norm of a staircase template
/// with `layers` nested intervals of fractional widths `factors[si * layers + l]`.
#[inline(always)]
fn shape_sums(
    cum: &[f64],
    row: usize,
    stride: usize,
    j: usize,
    k: usize,
    outer: (f64, f64, f64),
    ingress: f64,
    factors: &[f64],
    si: usize,
    layers: usize,
) -> (f64, f64, f64) {
    if ingress == 0.0 {
        return outer;
    }
    let l = layers as f64;
    let mut a = outer.0 / l;
    let mut hp = outer.1 / l;
    let mut norm = outer.2 / (l * l);
    for layer in 1..layers {
        let factor = factors[si * layers + layer];
        let width = ((k as f64 * factor + 0.5).floor() as usize).max(1);
        let start = j + (k - width) / 2;
        let dw = interval(cum, row, stride, F_W, start, width);
        a += interval(cum, row, stride, F_V, start, width) / l;
        hp += dw / l;
        norm += dw * ((2.0 * layer as f64 + 1.0) / (l * l));
    }
    (a, hp, norm)
}

/// Same sums in f32 from the (hi, lo) prefix pairs, for the pre-ranking.
#[inline(always)]
fn shape_sums_f32(
    cum32: &[f32],
    row: usize,
    stride: usize,
    j: usize,
    k: usize,
    outer: (f32, f32, f32),
    ingress: f64,
    factors: &[f64],
    si: usize,
    layers: usize,
) -> (f32, f32, f32) {
    if ingress == 0.0 {
        return outer;
    }
    let l = layers as f32;
    let mut a = outer.0 / l;
    let mut hp = outer.1 / l;
    let mut norm = outer.2 / (l * l);
    for layer in 1..layers {
        let factor = factors[si * layers + layer];
        let width = ((k as f64 * factor + 0.5).floor() as usize).max(1);
        let start = j + (k - width) / 2;
        let dw = interval32(cum32, row, stride, F_W, start, width);
        a += interval32(cum32, row, stride, F_V, start, width) / l;
        hp += dw / l;
        norm += dw * ((2.0 * layer as f32 + 1.0) / (l * l));
    }
    (a, hp, norm)
}

/// Block-wide winner over TILE lanes: largest value, then smallest key, then
/// lowest lane. Also folds the lane flags: unsafe (bit 0), near-top-and-
/// sensitive (bit 1), bad (bit 2) and the count of near-top lanes.
/// Warp shuffles reduce each warp; the four warp results are combined from
/// shared memory. Returns (value, key, lane, near_top_count, flags).
#[inline(always)]
fn block_winner(
    value: f64,
    key: u64,
    lane: usize,
    scratch_v: &mut SharedArray<u64, 16>,
) -> (f64, u64, u32) {
    use cuda_device::warp::{shuffle_xor_f64, shuffle_xor_u64};
    let mut v = value;
    let mut k = key;
    let mut l = lane as u64;
    let mut offset = 16u32;
    while offset > 0 {
        let ov = shuffle_xor_f64(v, offset);
        let ok = shuffle_xor_u64(k, offset);
        let ol = shuffle_xor_u64(l, offset);
        if ov > v || (ov == v && (ok < k || (ok == k && ol < l))) {
            v = ov;
            k = ok;
            l = ol;
        }
        offset /= 2;
    }
    let warp = lane / 32;
    if lane.is_multiple_of(32) {
        scratch_v[warp * 3] = v.to_bits();
        scratch_v[warp * 3 + 1] = k;
        scratch_v[warp * 3 + 2] = l;
    }
    thread::sync_threads();
    let mut best = (f64::from_bits(scratch_v[0]), scratch_v[1], scratch_v[2]);
    for w in 1..TILE / 32 {
        let cand = (
            f64::from_bits(scratch_v[w * 3]),
            scratch_v[w * 3 + 1],
            scratch_v[w * 3 + 2],
        );
        if cand.0 > best.0
            || (cand.0 == best.0 && (cand.1 < best.1 || (cand.1 == best.1 && cand.2 < best.2)))
        {
            best = cand;
        }
    }
    thread::sync_threads();
    (best.0, best.1, best.2 as u32)
}

/// Block-wide sum of small counters packed in a u32 (fields of 8 bits).
#[inline(always)]
fn block_sum_u32(value: u32, lane: usize, scratch: &mut SharedArray<u32, 4>) -> u32 {
    use cuda_device::warp::shuffle_xor;
    let mut v = value;
    let mut offset = 16u32;
    while offset > 0 {
        v += shuffle_xor(v, offset);
        offset /= 2;
    }
    if lane.is_multiple_of(32) {
        scratch[lane / 32] = v;
    }
    thread::sync_threads();
    let total = scratch[0] + scratch[1] + scratch[2] + scratch[3];
    thread::sync_threads();
    total
}

/// Inclusive scan of 32 values across a warp.
#[inline(always)]
fn warp_scan(mut v: f64, lane32: u32) -> f64 {
    use cuda_device::warp::shuffle_up_f64;
    let mut offset = 1u32;
    while offset < 32 {
        let other = shuffle_up_f64(v, offset);
        if lane32 >= offset {
            v += other;
        }
        offset *= 2;
    }
    v
}

/// Writes the doubled prefix sums of one period (f64 and (hi, lo) f32 pairs)
/// from a histogram accessed through `load(bin, field)`.
#[inline(always)]
fn prefix_block<F: Fn(usize, usize) -> f64>(
    load: &F,
    cum: &mut DisjointSlice<f64>,
    cum32: &mut DisjointSlice<f32>,
    f: usize,
    nb: usize,
    lane: usize,
    warp_totals: &mut SharedArray<f64, { NF * (CUM_BLOCK / 32) }>,
) {
    let row_stride = 2 * nb + 1;
    let row_out = f * NF * row_stride;
    let lane32 = (lane % 32) as u32;
    let warp = lane / 32;
    let warps = CUM_BLOCK / 32;
    let store =
        |cum: &mut DisjointSlice<f64>, cum32: &mut DisjointSlice<f32>, at: usize, value: f64| {
            let hi = value as f32;
            unsafe {
                *cum.get_unchecked_mut(at) = value;
                *cum32.get_unchecked_mut(2 * at) = hi;
                *cum32.get_unchecked_mut(2 * at + 1) = (value - hi as f64) as f32;
            }
        };
    if lane == 0 {
        for field in 0..NF {
            store(cum, cum32, row_out + field * row_stride, 0.0);
        }
    }
    let mut carry = [0.0f64; NF];
    let segments = nb.div_ceil(CUM_BLOCK);
    for seg in 0..segments {
        let b = seg * CUM_BLOCK + lane;
        let mut local = [0.0f64; NF];
        for field in 0..NF {
            let v = if b < nb { load(b, field) } else { 0.0 };
            local[field] = warp_scan(v, lane32);
            if lane32 == 31 {
                warp_totals[field * warps + warp] = local[field];
            }
        }
        thread::sync_threads();
        for field in 0..NF {
            let mut offset = carry[field];
            for w in 0..warp {
                offset += warp_totals[field * warps + w];
            }
            if b < nb {
                store(
                    cum,
                    cum32,
                    row_out + field * row_stride + b + 1,
                    local[field] + offset,
                );
            }
            let mut total = 0.0;
            for w in 0..warps {
                total += warp_totals[field * warps + w];
            }
            carry[field] += total;
        }
        thread::sync_threads();
    }
    let totals = carry;
    let mut b = lane;
    while b < nb {
        for field in 0..NF {
            let at = row_out + field * row_stride + b + 1;
            let value = unsafe { *cum.get_unchecked_mut(at) } + totals[field];
            store(cum, cum32, at + nb, value);
        }
        b += CUM_BLOCK;
    }
}

/// floor(t * scale) mod nb for t >= 0 without an integer division: the
/// quotient from the f64 reciprocal is corrected by at most one step.
#[inline(always)]
fn phase_cell(t: f64, scale: f64, nb: u32, inv_nb: f64) -> usize {
    let v = (t * scale).floor() as u32;
    let mut q = (v as f64 * inv_nb) as u32;
    let mut r = v.wrapping_sub(q.wrapping_mul(nb));
    if r >= nb {
        if (r as i32) < 0 {
            q = q.wrapping_sub(1);
        } else {
            q = q.wrapping_add(1);
        }
        r = v.wrapping_sub(q.wrapping_mul(nb));
    }
    r as usize
}

/// Lower median (k-th smallest, k = (e - a - 1) / 2) of the window values in
/// shared memory, or of their absolute deviation from `center`, by bisection
/// on the value range followed by the smallest element at or above the bound.
#[inline(always)]
fn window_median(
    win: *const f64,
    a: usize,
    e: usize,
    k: usize,
    center: f64,
    absolute: bool,
) -> f64 {
    let value = |j: usize| {
        let x = unsafe { *win.add(j) };
        if absolute { (x - center).abs() } else { x }
    };
    let mut vlo = f64::INFINITY;
    let mut vhi = f64::NEG_INFINITY;
    for j in a..e {
        let v = value(j);
        vlo = vlo.min(v);
        vhi = vhi.max(v);
    }
    let mut it = 0;
    while it < 64 && vhi > vlo {
        let mid = 0.5 * (vlo + vhi);
        if mid <= vlo || mid >= vhi {
            break;
        }
        let mut below = 0usize;
        for j in a..e {
            below += (value(j) < mid) as usize;
        }
        if below <= k {
            vlo = mid;
        } else {
            vhi = mid;
        }
        it += 1;
    }
    let mut best = f64::INFINITY;
    for j in a..e {
        let v = value(j);
        if v >= vlo && v < best {
            best = v;
        }
    }
    best
}

struct Lane {
    best: f64,
    key: u64,
    di: u32,
    bad: bool,
    sensitive: bool,
    unsafe_merit: bool,
}

#[inline(always)]
fn rank_fast(
    lane: &mut Lane,
    a32: f32,
    hp32: f32,
    norm32: f32,
    rho: f64,
    baseline: f64,
    valid: bool,
    key: u64,
    di: u32,
) -> f32 {
    let cn = baseline as f32 * norm32;
    let qq = (rho * rho) as f32 * (cn - hp32 * hp32);
    let merit = a32 * a32 / qq;
    let safe = cn > 0.0 && qq >= 0.001 * cn && (1e-30..=1e30).contains(&merit);
    lane.unsafe_merit |= valid && !safe;
    let best = lane.best as f32;
    let guard = 1e-5f32;
    let near = best > 0.0 && (merit - best).abs() <= guard * merit.max(best);
    let far_better = merit > best * (1.0 + 4.0 * guard);
    lane.sensitive = if valid && safe && far_better {
        false
    } else {
        lane.sensitive || (valid && safe && near)
    };
    let better = valid && safe && (merit > best || (merit == best && key < lane.key));
    if better {
        lane.best = merit as f64;
        lane.key = key;
        lane.di = di;
    }
    if valid && safe {
        merit
    } else {
        f32::NEG_INFINITY
    }
}

#[inline(always)]
fn rank_exact(
    lane: &mut Lane,
    a: f64,
    hp: f64,
    norm: f64,
    rho: f64,
    baseline: f64,
    valid: bool,
    key: u64,
    di: u32,
) -> f64 {
    let d = hp * hp / baseline;
    let q = norm - d;
    lane.bad |= valid && q <= 0.0;
    let candidate = (a / rho) / q.max(1e-300).sqrt();
    let merit = if valid && q > 0.0 && candidate < f64::INFINITY {
        candidate
    } else {
        NEG_INF
    };
    let better = merit > lane.best || (merit == lane.best && key < lane.key);
    if better {
        lane.best = merit;
        lane.key = key;
        lane.di = di;
    }
    merit
}

#[cuda_module]
pub mod gpu {
    use super::*;

    /// Grid (ceil(n / 256), ceil(periods / FOLD_PERIODS)), block 256. One
    /// thread per observation; each observation is read once and scattered
    /// into FOLD_PERIODS period histograms with global f64 atomics.
    #[kernel]
    pub fn fold(
        time: &[f64],
        fields: &[f64],
        scales: &[f64],
        folded: &[DeviceAtomicF64],
        n: u32,
        nb: u32,
        periods: u32,
    ) {
        let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let group = thread::blockIdx_y() as usize;
        let n = n as usize;
        if i >= n {
            return;
        }
        let t = time[i];
        let v = [fields[i], fields[n + i], fields[2 * n + i]];
        let inv_nb = 1.0 / nb as f64;
        for q in 0..FOLD_PERIODS {
            let f = group * FOLD_PERIODS + q;
            if f >= periods as usize {
                break;
            }
            let cell = phase_cell(t, scales[f], nb, inv_nb);
            let base = (f * nb as usize + cell) * NF;
            for k in 0..NF {
                folded[base + k].fetch_add(v[k], AtomicOrdering::Relaxed);
            }
        }
    }

    /// One block per group of `per_block` periods whose histograms share the
    /// 48 KiB of static shared memory: every observation is read once per
    /// group and scattered into `per_block` histograms, then written out.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn fold_shared(
        time: &[f64],
        fields: &[f64],
        scales: &[f64],
        mut folded: DisjointSlice<f64>,
        n: u32,
        nb: u32,
        periods: u32,
        per_block: u32,
    ) {
        static mut HIST: SharedArray<f64, { NF * FOLD_SHARED_BINS }> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let block = thread::blockDim_x() as usize;
        let group = thread::blockIdx_x() as usize;
        let n = n as usize;
        let nb = nb as usize;
        let per_block = per_block as usize;
        let first = group * per_block;
        let count = per_block.min(periods as usize - first);
        let words = count * nb * NF;
        let hist = (&raw mut HIST) as *mut f64;
        let mut b = lane;
        while b < words {
            unsafe {
                *hist.add(b) = 0.0;
            }
            b += block;
        }
        thread::sync_threads();
        let inv_nb = 1.0 / nb as f64;
        let mut i = lane;
        while i < n {
            let t = time[i];
            let v = [fields[i], fields[n + i], fields[2 * n + i]];
            for q in 0..count {
                let cell = phase_cell(t, scales[first + q], nb as u32, inv_nb);
                let base = (q * nb + cell) * NF;
                for k in 0..NF {
                    let slot = unsafe { BlockAtomicF64::from_ptr(hist.add(base + k)) };
                    slot.fetch_add(v[k], AtomicOrdering::Relaxed);
                }
            }
            i += block;
        }
        thread::sync_threads();
        let mut b = lane;
        while b < words {
            unsafe {
                *folded.get_unchecked_mut(first * nb * NF + b) = *hist.add(b);
            }
            b += block;
        }
    }

    #[kernel]
    pub fn zero_folded(folded: &[DeviceAtomicF64]) {
        let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if i < folded.len() {
            folded[i].store(0.0, AtomicOrdering::Relaxed);
        }
    }

    /// One block of CUM_BLOCK threads per period. Each segment of CUM_BLOCK
    /// bins is scanned with warp shuffles (one shared exchange of the warp
    /// totals), the running carry links the segments, and the second copy is
    /// the first plus the period total.
    #[kernel]
    pub fn cumulative(
        folded: &[DeviceAtomicF64],
        mut cum: DisjointSlice<f64>,
        mut cum32: DisjointSlice<f32>,
        nb: u32,
    ) {
        static mut WARP_TOTALS: SharedArray<f64, { NF * (CUM_BLOCK / 32) }> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let f = thread::blockIdx_x() as usize;
        let nb = nb as usize;
        let row_in = f * nb * NF;
        let load =
            |b: usize, field: usize| folded[row_in + b * NF + field].load(AtomicOrdering::Relaxed);
        prefix_block(&load, &mut cum, &mut cum32, f, nb, lane, unsafe {
            &mut *core::ptr::addr_of_mut!(WARP_TOTALS)
        });
    }

    /// Same scan over a plain f64 histogram.
    #[kernel]
    pub fn cumulative_plain(
        folded: &[f64],
        mut cum: DisjointSlice<f64>,
        mut cum32: DisjointSlice<f32>,
        nb: u32,
    ) {
        static mut WARP_TOTALS_P: SharedArray<f64, { NF * (CUM_BLOCK / 32) }> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let f = thread::blockIdx_x() as usize;
        let nb = nb as usize;
        let row_in = f * nb * NF;
        let load =
            |b: usize, field: usize| unsafe { *folded.get_unchecked(row_in + b * NF + field) };
        prefix_block(&load, &mut cum, &mut cum32, f, nb, lane, unsafe {
            &mut *core::ptr::addr_of_mut!(WARP_TOTALS_P)
        });
    }

    /// Grid (tiles, periods), block TILE. Each lane owns one phase bin and
    /// ranks every (duration, shape) template; the block keeps its winner.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn scan_tiles(
        cum: &[f64],
        cum32: &[f32],
        scales: &[f64],
        durations: &[f64],
        rho: &[f64],
        ingress: &[f64],
        factors: &[f64],
        baseline: f64,
        gate: f64,
        n: u32,
        nb: u32,
        nd: u32,
        ns: u32,
        layers: u32,
        mut parts: DisjointSlice<f64>,
        mut keys: DisjointSlice<u64>,
        mut flags: DisjointSlice<u32>,
    ) {
        static mut S_FLAG: SharedArray<u32, 4> = SharedArray::UNINIT;
        static mut S_WIN: SharedArray<u64, 16> = SharedArray::UNINIT;
        static mut S_CNT: SharedArray<u32, 4> = SharedArray::UNINIT;
        static mut S_OUT: SharedArray<u64, 4> = SharedArray::UNINIT;

        let lane = thread::threadIdx_x() as usize;
        let tile = thread::blockIdx_x() as usize;
        let f = thread::blockIdx_y() as usize;
        let nb = nb as usize;
        let (nd, ns, layers) = (nd as usize, ns as usize, layers as usize);
        let j = tile * TILE + lane;
        let jj = j.min(nb - 1);
        let scale = scales[f];
        let stride = 2 * nb + 1;
        let row = f * NF * stride;
        let n = n as f64;
        let nb1 = (nb + 1) as u64;

        let mut state = Lane {
            best: NEG_INF,
            key: KEY_MAX,
            di: 0,
            bad: false,
            sensitive: false,
            unsafe_merit: false,
        };
        let mut fast = true;
        let mut retry = false;
        let mut winner_key = KEY_MAX;
        let mut winner_di = 0u32;
        let mut winner_value = NEG_INF;
        let mut bad_any = false;

        for pass in 0..2 {
            if pass == 1 {
                if !retry {
                    break;
                }
                fast = false;
                state = Lane {
                    best: NEG_INF,
                    key: KEY_MAX,
                    di: 0,
                    bad: false,
                    sensitive: false,
                    unsafe_merit: false,
                };
            }
            let gate_value = if fast { gate * gate / baseline } else { gate };
            for di in 0..nd {
                let raw_k = ((durations[di] * scale + 0.5).floor() as usize).max(1);
                let k = raw_k.min(nb / 2);
                let r = rho[di];
                let key = (k as u64) * nb1 + jj as u64;
                let (merit, usable, valid, outer32) = if fast {
                    let count = interval32(cum32, row, stride, F_N, jj, k);
                    let w = interval32(cum32, row, stride, F_W, jj, k);
                    let outer32 = (interval32(cum32, row, stride, F_V, jj, k), w, w);
                    let usable =
                        j < nb && raw_k < nb / 2 && count >= 3.0 && count as f64 <= n - 3.0;
                    let valid = usable && outer32.0 > 0.0;
                    (
                        rank_fast(
                            &mut state, outer32.0, outer32.1, outer32.2, r, baseline, valid, key,
                            di as u32,
                        ) as f64,
                        usable,
                        valid,
                        outer32,
                    )
                } else {
                    let count = interval(cum, row, stride, F_N, jj, k);
                    let w = interval(cum, row, stride, F_W, jj, k);
                    let outer = (interval(cum, row, stride, F_V, jj, k), w, w);
                    let usable = j < nb && raw_k < nb / 2 && count >= 3.0 && count <= n - 3.0;
                    let valid = usable && outer.0 > 0.0;
                    (
                        rank_exact(
                            &mut state, outer.0, outer.1, outer.2, r, baseline, valid, key,
                            di as u32,
                        ),
                        usable,
                        valid,
                        (0.0, 0.0, 0.0),
                    )
                };
                // Tile-level shape gate: shapes are evaluated in a tile when
                // any lane's box reaches the gate.
                let warp_pass = cuda_device::warp::any(valid && merit >= gate_value);
                if lane.is_multiple_of(32) {
                    unsafe {
                        S_FLAG[lane / 32] = warp_pass as u32;
                    }
                }
                thread::sync_threads();
                let passing = unsafe { S_FLAG[0] | S_FLAG[1] | S_FLAG[2] | S_FLAG[3] };
                thread::sync_threads();
                if passing > 0 {
                    for si in 1..ns {
                        let ing = ingress[si];
                        let key = ((si as u64) * nb1 + k as u64) * nb1 + jj as u64;
                        if fast {
                            let (a, hp, norm) = shape_sums_f32(
                                cum32, row, stride, jj, k, outer32, ing, factors, si, layers,
                            );
                            rank_fast(
                                &mut state,
                                a,
                                hp,
                                norm,
                                r,
                                baseline,
                                usable && a > 0.0,
                                key,
                                di as u32,
                            );
                        } else {
                            let w = interval(cum, row, stride, F_W, jj, k);
                            let outer = (interval(cum, row, stride, F_V, jj, k), w, w);
                            let (a, hp, norm) = shape_sums(
                                cum, row, stride, jj, k, outer, ing, factors, si, layers,
                            );
                            rank_exact(
                                &mut state,
                                a,
                                hp,
                                norm,
                                r,
                                baseline,
                                usable && a > 0.0,
                                key,
                                di as u32,
                            );
                        }
                    }
                }
            }
            if fast && state.best <= 0.0 {
                state.key = nb1 + jj as u64;
            }
            let (value, key, chosen) = block_winner(state.best, state.key, lane, unsafe {
                &mut *core::ptr::addr_of_mut!(S_WIN)
            });
            let guard = if fast { 1e-5 } else { 1e-10 };
            let near_top = state.best > 0.0 && state.best >= value * (1.0 - guard);
            let packed = (near_top as u32)
                | ((near_top && state.sensitive) as u32) << 8
                | (state.unsafe_merit as u32) << 16
                | (state.bad as u32) << 24;
            let sums = block_sum_u32(packed, lane, unsafe {
                &mut *core::ptr::addr_of_mut!(S_CNT)
            });
            let near_count = sums & 0xff;
            let near_sensitive = (sums >> 8) & 0xff > 0;
            let unsafe_any = (sums >> 16) & 0xff > 0;
            let bad = (sums >> 24) & 0xff > 0;
            if lane == chosen as usize {
                unsafe {
                    S_OUT[0] = state.di as u64;
                }
            }
            thread::sync_threads();
            let di = unsafe { S_OUT[0] } as u32;
            thread::sync_threads();
            winner_key = key;
            winner_value = value;
            winner_di = di;
            retry = fast && (unsafe_any || near_count > 1 || near_sensitive);
            bad_any |= bad;
        }

        if lane == 0 {
            let phase = (winner_key % nb1) as usize;
            let width = ((winner_key / nb1) % nb1) as usize;
            let si = (winner_key / (nb1 * nb1)) as usize;
            let ing = ingress[si.min(ns - 1)];
            let r = rho[winner_di as usize];
            let w = interval(cum, row, stride, F_W, phase, width);
            let outer = (interval(cum, row, stride, F_V, phase, width), w, w);
            let (a0, hp, norm) = shape_sums(
                cum,
                row,
                stride,
                phase,
                width,
                outer,
                ing,
                factors,
                si.min(ns - 1),
                layers,
            );
            let a = a0 / r;
            let d = hp * hp / baseline;
            let q = (norm - d).max(1e-300);
            let value = if winner_value > NEG_INF {
                a / q.sqrt()
            } else {
                NEG_INF
            };
            let depth = a / q * r;
            let out = (f * (thread::gridDim_x() as usize) + tile) * PART_STRIDE;
            unsafe {
                *parts.get_unchecked_mut(out) = phase as f64;
                *parts.get_unchecked_mut(out + 1) = width as f64;
                *parts.get_unchecked_mut(out + 2) = a;
                *parts.get_unchecked_mut(out + 3) = norm;
                *parts.get_unchecked_mut(out + 4) = d;
                *parts.get_unchecked_mut(out + 5) = ing;
                *parts.get_unchecked_mut(out + 6) = r;
                *parts.get_unchecked_mut(out + 7) = value;
                *parts.get_unchecked_mut(out + 8) = depth;
                *keys.get_unchecked_mut(f * (thread::gridDim_x() as usize) + tile) = winner_key;
                *flags.get_unchecked_mut(f * (thread::gridDim_x() as usize) + tile) =
                    (retry as u32) | ((bad_any as u32) << 1);
            }
        }
    }

    /// Grid (tiles, 1), block TILE. Appends every template whose exact score
    /// reaches the threshold: (phase, width, score, depth) per record.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn scan_all(
        cum: &[f64],
        durations: &[f64],
        rho: &[f64],
        ingress: &[f64],
        factors: &[f64],
        scale: f64,
        baseline: f64,
        threshold: f64,
        n: u32,
        nb: u32,
        nd: u32,
        ns: u32,
        layers: u32,
        capacity: u32,
        counter: &[DeviceAtomicU32],
        mut records: DisjointSlice<f64>,
    ) {
        let lane = thread::threadIdx_x() as usize;
        let tile = thread::blockIdx_x() as usize;
        let nb = nb as usize;
        let (nd, ns, layers) = (nd as usize, ns as usize, layers as usize);
        let j = tile * TILE + lane;
        if j >= nb {
            return;
        }
        let n = n as f64;
        let stride = 2 * nb + 1;
        for di in 0..nd {
            let raw_k = ((durations[di] * scale + 0.5).floor() as usize).max(1);
            let k = raw_k.min(nb / 2);
            let r = rho[di];
            let count = interval(cum, 0, stride, F_N, j, k);
            let w = interval(cum, 0, stride, F_W, j, k);
            let outer = (interval(cum, 0, stride, F_V, j, k), w, w);
            let usable = raw_k < nb / 2 && count >= 3.0 && count <= n - 3.0;
            if !usable {
                continue;
            }
            for si in 0..ns {
                let (a, hp, norm) = shape_sums(
                    cum,
                    0,
                    stride,
                    j,
                    k,
                    outer,
                    ingress[si],
                    factors,
                    si,
                    layers,
                );
                let q = norm - hp * hp / baseline;
                if !(a > 0.0 && q > 0.0) {
                    continue;
                }
                let score = (a / r) / q.sqrt();
                if score >= threshold {
                    let slot = counter[0].fetch_add(1, AtomicOrdering::Relaxed) as usize;
                    if slot < capacity as usize {
                        unsafe {
                            *records.get_unchecked_mut(slot * EVENT_STRIDE) = j as f64;
                            *records.get_unchecked_mut(slot * EVENT_STRIDE + 1) = k as f64;
                            *records.get_unchecked_mut(slot * EVENT_STRIDE + 2) = score;
                            *records.get_unchecked_mut(slot * EVENT_STRIDE + 3) = a / q;
                        }
                    }
                }
            }
        }
    }

    #[kernel]
    pub fn read_counter(counter: &[DeviceAtomicU32], mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = counter[0].load(AtomicOrdering::Relaxed);
            }
        }
    }

    /// One block per row: box-template sums (a, hp, norm) over the family's
    /// observations in transit and not excluded.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn rescore_rows(
        time: &[f64],
        fields: &[f64],
        excluded: &[u8],
        rows: &[f64],
        origin: f64,
        n: u32,
        mut sums: DisjointSlice<f64>,
    ) {
        static mut RED: SharedArray<f64, { 3 * RESCORE_BLOCK }> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let r = thread::blockIdx_x() as usize;
        let n = n as usize;
        let (period, epoch, duration) = (rows[r * 3], rows[r * 3 + 1], rows[r * 3 + 2]);
        let inverse = 1.0 / period;
        let half = duration / 2.0;
        let (mut a, mut hp, mut norm) = (0.0, 0.0, 0.0);
        let mut i = lane;
        while i < n {
            if excluded[i] == 0 {
                let x = (time[i] + origin - epoch) * inverse + 0.5;
                let phase = (x - x.floor() - 0.5) * period;
                if phase.abs() <= half {
                    a += fields[i];
                    hp += fields[n + i];
                    norm += fields[n + i];
                }
            }
            i += RESCORE_BLOCK;
        }
        unsafe {
            RED[lane] = a;
            RED[RESCORE_BLOCK + lane] = hp;
            RED[2 * RESCORE_BLOCK + lane] = norm;
        }
        thread::sync_threads();
        let mut stride = RESCORE_BLOCK / 2;
        while stride > 0 {
            if lane < stride {
                unsafe {
                    RED[lane] += RED[lane + stride];
                    RED[RESCORE_BLOCK + lane] += RED[RESCORE_BLOCK + lane + stride];
                    RED[2 * RESCORE_BLOCK + lane] += RED[2 * RESCORE_BLOCK + lane + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if lane == 0 {
            unsafe {
                *sums.get_unchecked_mut(r * 3) = RED[0];
                *sums.get_unchecked_mut(r * 3 + 1) = RED[RESCORE_BLOCK];
                *sums.get_unchecked_mut(r * 3 + 2) = RED[2 * RESCORE_BLOCK];
            }
        }
    }

    /// One block per row: in-transit observation count and the count not excluded.
    #[kernel]
    pub fn count_rows(
        time: &[f64],
        excluded: &[u8],
        rows: &[f64],
        n: u32,
        mut counts: DisjointSlice<u32>,
    ) {
        static mut CNT: SharedArray<u32, { 2 * RESCORE_BLOCK }> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let r = thread::blockIdx_x() as usize;
        let n = n as usize;
        let (period, epoch, duration) = (rows[r * 3], rows[r * 3 + 1], rows[r * 3 + 2]);
        let inverse = 1.0 / period;
        let half = duration / 2.0;
        let (mut total, mut unshared) = (0u32, 0u32);
        let mut i = lane;
        while i < n {
            let x = (time[i] - epoch) * inverse + 0.5;
            let phase = (x - x.floor() - 0.5) * period;
            if phase.abs() <= half {
                total += 1;
                unshared += (excluded[i] == 0) as u32;
            }
            i += RESCORE_BLOCK;
        }
        unsafe {
            CNT[lane] = total;
            CNT[RESCORE_BLOCK + lane] = unshared;
        }
        thread::sync_threads();
        let mut stride = RESCORE_BLOCK / 2;
        while stride > 0 {
            if lane < stride {
                unsafe {
                    CNT[lane] += CNT[lane + stride];
                    CNT[RESCORE_BLOCK + lane] += CNT[RESCORE_BLOCK + lane + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if lane == 0 {
            unsafe {
                *counts.get_unchecked_mut(r * 2) = CNT[0];
                *counts.get_unchecked_mut(r * 2 + 1) = CNT[RESCORE_BLOCK];
            }
        }
    }

    /// Tukey biweight location in a sliding window, one thread per point.
    /// `blocks` holds (first point, point count, union start, union end) per
    /// block; the union of the block's windows is staged in shared memory.
    /// Medians are exact lower medians found by bisection on the value range
    /// (the k-th smallest element, k = (W - 1) / 2).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn biweight(
        flux: &[f64],
        lo: &[u32],
        hi: &[u32],
        blocks: &[u32],
        c: f64,
        iterations: u32,
        mut out: DisjointSlice<f64>,
    ) {
        static mut WIN: SharedArray<f64, BIWEIGHT_WINDOW> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let b = thread::blockIdx_x() as usize * 4;
        let (first, count, ulo, uhi) = (
            blocks[b] as usize,
            blocks[b + 1] as usize,
            blocks[b + 2] as usize,
            blocks[b + 3] as usize,
        );
        let win = (&raw mut WIN) as *mut f64;
        let mut q = lane;
        while q < uhi - ulo {
            unsafe {
                *win.add(q) = flux[ulo + q];
            }
            q += BIWEIGHT_BLOCK;
        }
        thread::sync_threads();
        if lane >= count {
            return;
        }
        let p = first + lane;
        let (a, e) = (lo[p] as usize - ulo, hi[p] as usize - ulo);
        let w = e - a;
        let k = (w - 1) / 2;
        let at = |j: usize| unsafe { *win.add(j) };
        let mut location = window_median(win, a, e, k, 0.0, false);
        let mut n = 0;
        while n < iterations {
            let loc = location;
            let scale = (c * window_median(win, a, e, k, loc, true)).max(1e-12);
            let mut total = 0.0;
            let mut acc = 0.0;
            for j in a..e {
                let v = at(j);
                let u = (v - loc) / scale;
                if u.abs() < 1.0 {
                    let wt = (1.0 - u * u) * (1.0 - u * u);
                    total += wt;
                    acc += wt * v;
                }
            }
            if total > 0.0 {
                location = acc / total;
            }
            n += 1;
        }
        unsafe {
            *out.get_unchecked_mut(p) = at(p - ulo) / location - 1.0;
        }
    }

    /// One thread per period: the best tile by (score, lowest key).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn merge_tiles(
        parts: &[f64],
        keys: &[u64],
        periods_in: &[f64],
        scales: &[f64],
        periods: u32,
        tiles: u32,
        mut records: DisjointSlice<f64>,
        mut score: DisjointSlice<f64>,
        mut depth: DisjointSlice<f64>,
    ) {
        let f = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if f >= periods as usize {
            return;
        }
        let tiles = tiles as usize;
        let mut best = NEG_INF;
        let mut key = KEY_MAX;
        let mut chosen = 0usize;
        for t in 0..tiles {
            let idx = f * tiles + t;
            let v = parts[idx * PART_STRIDE + 7];
            let k = keys[idx];
            if v > best || (v == best && k < key) {
                best = v;
                key = k;
                chosen = t;
            }
        }
        let part = (f * tiles + chosen) * PART_STRIDE;
        let out = f * RECORD_STRIDE;
        unsafe {
            *records.get_unchecked_mut(out) = periods_in[f];
            *records.get_unchecked_mut(out + 1) = scales[f];
            for i in 0..7 {
                *records.get_unchecked_mut(out + 2 + i) = parts[part + i];
            }
            *score.get_unchecked_mut(f) = best;
            *depth.get_unchecked_mut(f) = parts[part + 8];
        }
    }
}
