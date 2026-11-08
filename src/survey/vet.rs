//! Flux-level false-positive tests of one signal (period, epoch, duration)
//! on the detrended curve of the detector family closest to its duration,
//! with the other accepted signals masked.
//!
//! Tests: depth and SNR against the out-of-transit noise at the transit time
//! scale, odd/even depth difference, secondary eclipse at every phase,
//! trapezoid shape fit and circular-orbit geometry, duration versus stellar
//! density (too long and too short), centroid shift versus the target
//! position, crowding, and the flux tests of the Kepler Robovetter as adapted
//! for TESS (uniqueness, sinusoidal variability, event consistency, shape,
//! gaps, model comparison). Thresholds follow SPOC conventions: detection at
//! 7.1 sigma, at least two transits. A warning never passes a signal that a
//! failure would reject.

use super::catalog::wrap;
use super::numeric::{
    Rng, chi2_sigma_2, erfcinv, fit_errors, least_squares, lstsq, mad, mean, median, percentile,
    std,
};
use super::star::Sidecar;
use crate::io::fits::StarParameters;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const G: f64 = 6.674e-11;
pub const DAY: f64 = 86400.0;
pub const REARTH_PER_RSUN: f64 = 109.1;
pub const RSUN_AU: f64 = 0.00465047;
pub const PIXEL_ARCSEC: f64 = 21.0;
pub const MIN_SNR: f64 = 7.1;
pub const MIN_TRANSITS: usize = 2;
pub const ODD_EVEN_SIGMA: f64 = 4.0;
pub const CENTROID_SIGMA: f64 = 4.0;
pub const SECONDARY_SIGMA: f64 = 5.0;
pub const MAX_RADIUS_EARTH: f64 = 25.0;
/// t14 / P of a grazing transit at a/R* = 2; no stellar transit lasts longer.
pub const MAX_DURATION_FRACTION: f64 = 0.15;
/// Fitted duration more than 1.7x the longest the TIC density allows.
pub const MIN_DENSITY_RATIO: f64 = 0.2;
/// Duration shorter than a fifth of a central transit: b > 0.98 or not a transit.
pub const MAX_DENSITY_RATIO: f64 = 100.0;
pub const VETTING_VERSION: i64 = 2;
pub const FAIL: &str = "fail";
pub const WARN: &str = "warn";

/// Phase-folded detrended curve of one signal.
pub struct Folded {
    pub time: Vec<f64>,
    pub phase: Vec<f64>,
    pub corrected: Vec<f64>,
    pub valid: Vec<bool>,
    pub outside: Vec<bool>,
    pub cadence: f64,
    pub period: f64,
    pub epoch: f64,
    pub duration: f64,
    pub sector: Vec<i64>,
}

pub fn fold(
    time: &[f64],
    corrected: &[f64],
    sector: &[i64],
    period: f64,
    epoch: f64,
    duration_hours: f64,
    masks: &[(f64, f64, f64)],
) -> Result<Folded, String> {
    let duration = duration_hours / 24.0;
    if time.len() < 200 || !(0.0 < duration && duration < period / 2.0) {
        return Err("invalid coverage or duration".into());
    }
    let phase: Vec<f64> = time.iter().map(|&t| wrap(t - epoch, period)).collect();
    let mut valid = vec![false; time.len()];
    let mut outside = vec![false; time.len()];
    let (mut inside_valid, mut outside_count) = (0, 0);
    for i in 0..time.len() {
        let mut other = masks
            .iter()
            .any(|&(p, e, d)| wrap(time[i] - e, p).abs() < d / 24.0);
        other &= phase[i].abs() >= duration;
        valid[i] = corrected[i].is_finite() && !other;
        if valid[i] && phase[i].abs() < duration / 2.0 {
            inside_valid += 1;
        }
        outside[i] = phase[i].abs() > duration && valid[i];
        outside_count += outside[i] as usize;
    }
    if inside_valid < 5 || outside_count < 100 {
        return Err("not enough coverage of the predicted transit".into());
    }
    let steps: Vec<f64> = time
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&s| s > 0.0)
        .collect();
    let cadence = median(&steps);
    Ok(Folded {
        time: time.to_vec(),
        phase,
        corrected: corrected.to_vec(),
        valid,
        outside,
        cadence,
        period,
        epoch,
        duration,
        sector: sector.to_vec(),
    })
}

/// Noise of a mean over `width` days, measured out of transit (includes
/// correlated noise). Returns (noise, expected points per window).
pub fn window_noise(
    time: &[f64],
    values: &[f64],
    mask: &[bool],
    width: f64,
    cadence: f64,
) -> (f64, f64) {
    let idx: Vec<usize> = (0..time.len()).filter(|&i| mask[i]).collect();
    let expected = width / cadence;
    if idx.len() < 10 {
        return (f64::NAN, expected);
    }
    let t0 = time[idx[0]];
    let mut bins: BTreeMap<i64, (f64, usize)> = BTreeMap::new();
    for &i in &idx {
        let e = bins
            .entry(((time[i] - t0) / width).floor() as i64)
            .or_insert((0.0, 0));
        e.0 += values[i];
        e.1 += 1;
    }
    let v: Vec<f64> = idx.iter().map(|&i| values[i]).collect();
    let white = 1.4826 * mad(&v) / expected.sqrt();
    let full: Vec<f64> = bins
        .values()
        .filter(|(_, c)| *c as f64 >= expected / 2.0)
        .map(|(s, c)| s / *c as f64)
        .collect();
    if full.len() < 5 {
        return (white, expected);
    }
    let red = 1.4826 * mad(&full);
    (red.max(white), expected)
}

pub fn trapezoid(x: f64, depth: f64, total: f64, ingress: f64) -> f64 {
    let half = total / 2.0;
    let ramp = (ingress * total).max(1e-6);
    -depth * ((half - x.abs()) / ramp).clamp(0.0, 1.0)
}

/// Trapezoid fit of the binned folded profile: (ingress fraction, total duration, their errors).
pub fn fit_shape(prep: &Folded) -> (f64, f64, f64, f64) {
    let mut centers = Vec::new();
    let mut profile = Vec::new();
    let edges: Vec<f64> = (0..=80).map(|i| -2.0 + 4.0 * i as f64 / 80.0).collect();
    for w in edges.windows(2) {
        let vals: Vec<f64> = (0..prep.time.len())
            .filter(|&i| prep.valid[i])
            .map(|i| (prep.phase[i] / prep.duration, prep.corrected[i]))
            .filter(|(x, _)| x.abs() < 2.0 && *x >= w[0] && *x < w[1])
            .map(|(_, c)| c)
            .collect();
        if vals.len() >= 3 {
            centers.push((w[0] + w[1]) / 2.0);
            profile.push(median(&vals));
        }
    }
    if centers.len() < 20 {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let lowest = profile.iter().cloned().fold(f64::INFINITY, f64::min);
    let start = [(-lowest).clamp(1e-6, 0.99), 1.0, 0.2];
    let (p, r, jac) = least_squares(
        |p| {
            centers
                .iter()
                .zip(&profile)
                .map(|(&x, &y)| trapezoid(x, p[0], p[1], p[2]) - y)
                .collect()
        },
        &start,
        &[0.0, 0.3, 0.01],
        &[1.0, 2.5, 0.5],
    );
    let err = fit_errors(&r, &jac);
    (p[2], p[1] * prep.duration, err[2], err[1] * prep.duration)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Geometry {
    pub impact: Option<f64>,
    pub impact_err: Option<f64>,
    pub inclination_deg: Option<f64>,
    pub inclination_err: Option<f64>,
    pub a_over_rstar_fit: Option<f64>,
    pub a_over_rstar_fit_err: Option<f64>,
    pub fitted_density_gcc: Option<f64>,
    pub fitted_density_err: Option<f64>,
    pub insolation_earth: Option<f64>,
    pub insolation_err: Option<f64>,
    pub teq_k: Option<f64>,
    pub teq_err: Option<f64>,
    pub reliable: bool,
    pub unreliable: Vec<String>,
}

fn summary(values: &[f64]) -> (f64, f64) {
    let (lo, mid, hi) = (
        percentile(values, 16.0),
        percentile(values, 50.0),
        percentile(values, 84.0),
    );
    (mid, (hi - lo) / 2.0)
}

/// Circular-orbit geometry from the trapezoid fit with Monte Carlo
/// propagation of the depth, duration and ingress uncertainties. Impact
/// parameter, inclination, a/R* and density are reported only when the
/// ingress is resolved and measured to better than 50%, the duration to
/// better than 20%, the depth is significant and most samples give a valid
/// geometry. Insolation and equilibrium temperature use the fitted a/R* when
/// reliable and the one from the TIC density otherwise.
#[allow(clippy::too_many_arguments)]
pub fn geometry(
    period: f64,
    depth: f64,
    depth_err: f64,
    t14: f64,
    t14_err: f64,
    ingress: f64,
    ingress_err: f64,
    cadence: f64,
    star: &StarParameters,
    a_over_r_tic: f64,
) -> Geometry {
    let samples = 4000;
    let finite = |x: f64| x.is_finite();
    let mut out = Geometry::default();
    let mut reasons = Vec::new();
    if !(finite(depth) && finite(depth_err) && depth_err > 0.0 && depth / depth_err >= MIN_SNR) {
        reasons.push("depth".to_string());
    }
    if !(finite(t14) && finite(t14_err) && t14_err < 0.2 * t14) {
        reasons.push("duration".to_string());
    }
    if !(finite(ingress)
        && finite(ingress_err)
        && ingress_err < 0.5 * ingress
        && ingress * t14 >= 2.0 * cadence)
    {
        reasons.push("ingress".to_string());
    }
    let mut a_over_r: Option<(f64, f64)> = None;
    if reasons.is_empty() {
        let mut rng = Rng::new(0);
        let (mut bs, mut ars) = (Vec::new(), Vec::new());
        let mut ok_count = 0;
        for _ in 0..samples {
            let d = rng.normal(depth, depth_err);
            let t = rng.normal(t14, t14_err);
            let f = rng.normal(ingress, ingress_err);
            let mut ok = d > 0.0 && t > 0.0 && t < period / 2.0 && f > 0.0 && f < 0.5;
            if ok {
                let k = d.sqrt();
                let r = 1.0 - 2.0 * f;
                let b2 = ((1.0 - k).powi(2) - r * r * (1.0 + k).powi(2)) / (1.0 - r * r);
                let s2 = (std::f64::consts::PI * t / period).sin().powi(2);
                ok &= b2 >= 0.0 && b2 < (1.0 + k).powi(2);
                if ok {
                    let b = b2.sqrt();
                    let ar = (((1.0 + k).powi(2) - b2 * (1.0 - s2)) / s2).sqrt();
                    ok &= ar.is_finite() && ar > b;
                    if ok {
                        bs.push(b);
                        ars.push(ar);
                    }
                }
            }
            ok_count += ok as usize;
        }
        if ok_count as f64 / samples as f64 >= 0.68 {
            let (b, be) = summary(&bs);
            let incl: Vec<f64> = bs
                .iter()
                .zip(&ars)
                .map(|(b, ar)| (b / ar).acos().to_degrees())
                .collect();
            let (i, ie) = summary(&incl);
            let (ar, are) = summary(&ars);
            let dens: Vec<f64> = ars
                .iter()
                .map(|ar| {
                    3.0 * std::f64::consts::PI * ar.powi(3) / (G * (period * DAY).powi(2)) / 1000.0
                })
                .collect();
            let (de, dee) = summary(&dens);
            out.impact = Some(b);
            out.impact_err = Some(be);
            out.inclination_deg = Some(i);
            out.inclination_err = Some(ie);
            out.a_over_rstar_fit = Some(ar);
            out.a_over_rstar_fit_err = Some(are);
            out.fitted_density_gcc = Some(de);
            out.fitted_density_err = Some(dee);
            a_over_r = Some((ar, are));
        } else {
            reasons.push("geometry".to_string());
        }
    }
    if a_over_r.is_none() && a_over_r_tic.is_finite() {
        a_over_r = Some((a_over_r_tic, 0.1 * a_over_r_tic));
    }
    if let (Some((ar0, are)), Some(teff)) = (a_over_r, star.teff.filter(|t| *t > 0.0)) {
        let mut rng = Rng::new(1);
        let (mut teq, mut ins) = (Vec::new(), Vec::new());
        for _ in 0..samples {
            let ar = rng.normal(ar0, are);
            let te = rng.normal(teff, 150.0);
            if ar > 1.0 && te > 0.0 {
                teq.push(te * (1.0 / (2.0 * ar)).sqrt());
                ins.push((te / 5772.0).powi(4) / (ar * RSUN_AU).powi(2));
            }
        }
        let (t, te) = summary(&teq);
        let (i, ie) = summary(&ins);
        out.teq_k = Some(t);
        out.teq_err = Some(te);
        out.insolation_earth = Some(i);
        out.insolation_err = Some(ie);
    }
    out.reliable = reasons.is_empty();
    out.unreliable = reasons;
    out
}

/// Source position from the flux-weighted centroid shift against the target
/// position: a source at x_s dimming a fraction delta of the aperture flux
/// moves the centroid by delta (c_out - x_s) / (1 - delta). Sectors combine
/// by weighted mean on each axis. Returns (sigma, offset in arcsec).
pub fn centroid_test(
    prep: &Folded,
    centr1: &[f64],
    centr2: &[f64],
    depth: f64,
    meta: &BTreeMap<i64, Sidecar>,
) -> (f64, f64) {
    let n = prep.time.len();
    let core: Vec<bool> = (0..n)
        .map(|i| prep.phase[i].abs() < prep.duration / 4.0)
        .collect();
    let near: Vec<bool> = (0..n)
        .map(|i| prep.outside[i] && prep.phase[i].abs() < 3.0 * prep.duration)
        .collect();
    let mut offsets: Vec<[(f64, f64); 2]> = Vec::new();
    let mut sectors: Vec<i64> = prep.sector.clone();
    sectors.sort();
    sectors.dedup();
    for s in sectors {
        let Some(info) = meta.get(&s) else { continue };
        let Some(target) = info.target_pixel else {
            continue;
        };
        if !(depth > 0.0) {
            continue;
        }
        let aperture_depth = depth * info.crowdsap.unwrap_or(1.0);
        let mut shifts: Vec<(f64, f64)> = Vec::new();
        for (values, tgt) in [(centr1, target.0), (centr2, target.1)] {
            let good: Vec<bool> = (0..n)
                .map(|i| prep.sector[i] == s && values[i].is_finite())
                .collect();
            let inside: Vec<usize> = (0..n).filter(|&i| good[i] && core[i]).collect();
            let reference: Vec<bool> = (0..n).map(|i| good[i] && near[i]).collect();
            let ref_count = reference.iter().filter(|&&x| x).count();
            if inside.len() < 3 || ref_count < 10 {
                break;
            }
            let ref_vals: Vec<f64> = (0..n)
                .filter(|&i| reference[i])
                .map(|i| values[i])
                .collect();
            let baseline = median(&ref_vals);
            let residual: Vec<f64> = values.iter().map(|v| v - baseline).collect();
            let (noise, expected) = window_noise(
                &prep.time,
                &residual,
                &reference,
                prep.duration / 2.0,
                prep.cadence,
            );
            if !noise.is_finite() {
                break;
            }
            let shift = mean(&inside.iter().map(|&i| residual[i]).collect::<Vec<_>>())
                - median(
                    &(0..n)
                        .filter(|&i| reference[i])
                        .map(|i| residual[i])
                        .collect::<Vec<_>>(),
                );
            let factor = (1.0 - aperture_depth) / aperture_depth;
            let source = baseline - shift * factor;
            let error = (noise * (expected / inside.len() as f64).sqrt() * factor).hypot(0.1);
            shifts.push((source - tgt, error));
        }
        if shifts.len() == 2 {
            offsets.push([shifts[0], shifts[1]]);
        }
    }
    if offsets.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut total = Vec::new();
    for axis in 0..2 {
        let w: f64 = offsets.iter().map(|o| 1.0 / (o[axis].1 * o[axis].1)).sum();
        let v: f64 = offsets
            .iter()
            .map(|o| o[axis].0 / (o[axis].1 * o[axis].1))
            .sum();
        total.push((v / w, 1.0 / w.sqrt()));
    }
    let chi2: f64 = total.iter().map(|(o, e)| (o / e) * (o / e)).sum();
    (
        chi2_sigma_2(chi2),
        total[0].0.hypot(total[1].0) * PIXEL_ARCSEC,
    )
}

/// Mean stellar density (kg/m^3) from log g and radius.
pub fn stellar_density(star: &StarParameters) -> f64 {
    match (star.logg, star.radius) {
        (Some(logg), Some(r)) if r > 0.0 => {
            3.0 * (10f64.powf(logg) / 100.0) / (4.0 * std::f64::consts::PI * G * r * 6.957e8)
        }
        _ => f64::NAN,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub period_days: Option<f64>,
    pub epoch_bjd: f64,
    pub duration_hours: f64,
    pub depth_ppm: Option<f64>,
    pub depth_err_ppm: Option<f64>,
    pub snr: Option<f64>,
    pub transits: usize,
    pub odd_depth_ppm: Option<f64>,
    pub even_depth_ppm: Option<f64>,
    pub odd_even_sigma: Option<f64>,
    pub secondary_depth_ppm: Option<f64>,
    pub secondary_sigma: Option<f64>,
    pub secondary_phase: Option<f64>,
    pub secondary_albedo: Option<f64>,
    pub secondary_threshold_sigma: Option<f64>,
    pub brightening_sigma: Option<f64>,
    pub ingress_fraction: Option<f64>,
    pub fitted_duration_hours: Option<f64>,
    pub fitted_duration_err_hours: Option<f64>,
    pub radius_earth: Option<f64>,
    pub a_over_rstar: Option<f64>,
    pub density_ratio: Option<f64>,
    pub geometry: Option<Geometry>,
    pub centroid_sigma: Option<f64>,
    pub source_offset_arcsec: Option<f64>,
    pub crowdsap: Option<f64>,
    pub star: StarParameters,
    pub robovetter: Option<RoboMetrics>,
    pub sectors: Vec<i64>,
    pub flags: BTreeMap<String, String>,
    pub verdict: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pixel_test: Option<serde_json::Value>,
}

fn num(v: f64) -> Option<f64> {
    if v.is_finite() { Some(v) } else { None }
}

/// Vetting metrics and flags of one folded signal; `raw` is the normalised
/// PDCSAP flux on the same cadences for the tests that must not see the
/// detrending.
pub fn measure(
    prep: &Folded,
    star: &StarParameters,
    centr1: Option<&[f64]>,
    centr2: Option<&[f64]>,
    crowding: Option<f64>,
    meta: &BTreeMap<i64, Sidecar>,
    raw: Option<&[f64]>,
) -> Metrics {
    let n = prep.time.len();
    let (period, duration) = (prep.period, prep.duration);
    let (noise, expected) = window_noise(
        &prep.time,
        &prep.corrected,
        &prep.outside,
        duration / 2.0,
        prep.cadence,
    );
    let (box_noise, box_expected) = window_noise(
        &prep.time,
        &prep.corrected,
        &prep.outside,
        0.8 * duration,
        prep.cadence,
    );
    let number: Vec<f64> = prep
        .time
        .iter()
        .map(|t| ((t - prep.epoch) / period).round())
        .collect();
    let odd: Vec<bool> = number.iter().map(|k| k.rem_euclid(2.0) == 1.0).collect();

    let depth = |offset: f64, select: Option<&Vec<bool>>, half: f64| -> (f64, f64, usize) {
        let chosen: Vec<usize> = (0..n)
            .filter(|&i| {
                prep.valid[i]
                    && select.is_none_or(|s| s[i])
                    && wrap(prep.phase[i] - offset * period, period).abs() < half
            })
            .collect();
        if chosen.is_empty() {
            return (f64::NAN, f64::NAN, 0);
        }
        let (sigma, points) = if half == duration / 4.0 {
            (noise, expected)
        } else {
            (box_noise, box_expected)
        };
        let mut error = sigma * (points / chosen.len() as f64).sqrt();
        let mut cycles: BTreeMap<i64, (f64, usize)> = BTreeMap::new();
        for &i in &chosen {
            let c = ((prep.time[i] - prep.epoch - offset * period) / period).round() as i64;
            let e = cycles.entry(c).or_insert((0.0, 0));
            e.0 += prep.corrected[i];
            e.1 += 1;
        }
        let full: Vec<f64> = cycles
            .values()
            .filter(|(_, c)| *c as f64 >= (points / 4.0).max(3.0))
            .map(|(s, c)| -s / *c as f64)
            .collect();
        if full.len() >= 5 {
            error = error.max(1.4826 * mad(&full) / (full.len() as f64).sqrt());
        }
        (
            -mean(
                &chosen
                    .iter()
                    .map(|&i| prep.corrected[i])
                    .collect::<Vec<_>>(),
            ),
            error,
            full.len(),
        )
    };

    let (d, d_err, transits) = depth(0.0, None, duration / 4.0);
    let (bx, box_err, _) = depth(0.0, None, 0.4 * duration);
    let (d_odd, e_odd, n_odd) = depth(0.0, Some(&odd), duration / 4.0);
    let even: Vec<bool> = odd.iter().map(|o| !o).collect();
    let (d_even, e_even, n_even) = depth(0.0, Some(&even), duration / 4.0);
    let odd_even = if n_odd.min(n_even) >= 2 {
        (d_odd - d_even).abs() / e_odd.hypot(e_even)
    } else {
        f64::NAN
    };

    // Secondary eclipse search over the whole orbit.
    let nbins = ((8.0 * period / duration).ceil() as usize).max(64);
    let mut sums = vec![0.0; nbins];
    let mut hits = vec![0.0; nbins];
    for i in 0..n {
        if prep.valid[i] {
            let cell = (((prep.phase[i] / period).rem_euclid(1.0)) * nbins as f64).floor() as usize
                % nbins;
            sums[cell] += prep.corrected[i];
            hits[cell] += 1.0;
        }
    }
    let width = ((nbins as f64 * duration / (2.0 * period)).round() as usize).max(1);
    let (mut sec_depth, mut sec_sigma, mut sec_phase, mut brightening, mut sec_threshold) =
        (f64::NAN, 0.0, f64::NAN, 0.0, f64::INFINITY);
    {
        let mut window_sum = vec![0.0; nbins];
        let mut window_hits = vec![0.0; nbins];
        for b in 0..nbins {
            for k in 0..width {
                window_sum[b] += sums[(b + k) % nbins];
                window_hits[b] += hits[(b + k) % nbins];
            }
        }
        let center: Vec<f64> = (0..nbins)
            .map(|b| ((b as f64 + width as f64 / 2.0) / nbins as f64).rem_euclid(1.0))
            .collect();
        let allowed: Vec<bool> = (0..nbins)
            .map(|b| {
                (center[b] - 0.5).abs() < 0.5 - 1.5 * duration / period
                    && window_hits[b] >= (expected / 4.0).max(3.0)
            })
            .collect();
        let count = allowed.iter().filter(|&&a| a).count();
        if count > 0 {
            let sigmas: Vec<f64> = (0..nbins)
                .map(|b| {
                    (-window_sum[b] / window_hits[b].max(1.0))
                        / (noise * (expected / window_hits[b].max(1.0)).sqrt())
                })
                .collect();
            let best = (0..nbins)
                .filter(|&b| allowed[b])
                .max_by(|&a, &b| sigmas[a].partial_cmp(&sigmas[b]).unwrap())
                .unwrap();
            sec_phase = center[best];
            let (sd, se, _) = depth(sec_phase, None, duration / 4.0);
            sec_depth = sd;
            sec_sigma = if se > 0.0 { (sd / se).max(0.0) } else { 0.0 };
            brightening = (-(0..nbins)
                .filter(|&b| allowed[b])
                .map(|b| sigmas[b])
                .fold(f64::INFINITY, f64::min))
            .max(0.0);
            sec_threshold =
                super::numeric::norm_isf(0.01 / (count as f64 / width as f64).max(1.0)).max(3.0);
        }
    }

    let (ingress, total, ingress_err, total_err) = fit_shape(prep);
    let density = stellar_density(star);
    let k = d.max(0.0).sqrt();
    let (a_over_r, albedo) = if density.is_finite() {
        let ar =
            (G * density * (period * DAY).powi(2) / (3.0 * std::f64::consts::PI)).powf(1.0 / 3.0);
        (
            ar,
            if d > 0.0 && sec_depth.is_finite() {
                sec_depth * ar * ar / d
            } else {
                f64::NAN
            },
        )
    } else {
        (f64::NAN, f64::NAN)
    };
    let t14 = if total.is_finite() { total } else { duration };
    let max_density =
        3.0 * std::f64::consts::PI * (period / (std::f64::consts::PI * t14) * (1.0 + k)).powi(3)
            / (G * (period * DAY).powi(2));
    let density_ratio = if density.is_finite() {
        max_density / density
    } else {
        f64::NAN
    };
    let radius = star
        .radius
        .map(|r| k * r * REARTH_PER_RSUN)
        .unwrap_or(f64::NAN);
    let (centroid_sigma, centroid_arcsec) = match (centr1, centr2) {
        (Some(c1), Some(c2)) if !meta.is_empty() => centroid_test(prep, c1, c2, d, meta),
        _ => (f64::NAN, f64::NAN),
    };
    let snr = if box_err.is_finite() && box_err > 0.0 {
        bx / box_err
    } else {
        f64::NAN
    };
    let shape = geometry(
        period,
        d,
        d_err,
        total,
        total_err,
        ingress,
        ingress_err,
        prep.cadence,
        star,
        a_over_r,
    );

    let chosen: Vec<usize> = (0..n).filter(|&i| prep.valid[i]).collect();
    let time: Vec<f64> = chosen.iter().map(|&i| prep.time[i]).collect();
    let flux: Vec<f64> = chosen.iter().map(|&i| prep.corrected[i]).collect();
    let raw_flux: Vec<f64> = chosen
        .iter()
        .map(|&i| raw.map(|r| r[i]).unwrap_or(f64::NAN))
        .collect();
    let robo = robo_metrics(&time, &flux, &raw_flux, period, prep.epoch, duration);
    let mut flags = robo_flags(
        &robo,
        period,
        d,
        num(radius),
        num(albedo),
        shape.impact,
        if d > 0.0 { Some(k) } else { None },
        MAX_RADIUS_EARTH,
    );
    if !(snr >= MIN_SNR) {
        flags.insert("low_snr".into(), FAIL.into());
    }
    if transits < MIN_TRANSITS {
        flags.insert("single_transit".into(), FAIL.into());
    }
    if odd_even > ODD_EVEN_SIGMA {
        flags.insert("odd_even".into(), FAIL.into());
    } else if odd_even > 3.0 {
        flags.insert("odd_even".into(), WARN.into());
    }
    if sec_sigma > sec_threshold
        && sec_sigma - brightening > 1.0
        && (if albedo.is_finite() {
            albedo > 1.0
        } else {
            sec_depth > 0.1 * d
        })
    {
        flags.insert(
            "secondary_eclipse".into(),
            if sec_sigma > SECONDARY_SIGMA {
                FAIL
            } else {
                WARN
            }
            .into(),
        );
    }
    if radius > MAX_RADIUS_EARTH {
        flags.insert("too_large".into(), FAIL.into());
    }
    if centroid_sigma > CENTROID_SIGMA && centroid_arcsec > PIXEL_ARCSEC {
        flags.insert("centroid_offset".into(), FAIL.into());
    } else if centroid_sigma > 3.0 && centroid_arcsec > PIXEL_ARCSEC / 2.0 {
        flags.insert("centroid_offset".into(), WARN.into());
    }
    if crowding.is_some_and(|c| c < 0.8) {
        flags.insert("crowded".into(), WARN.into());
    }
    if t14 / period > MAX_DURATION_FRACTION || density_ratio < MIN_DENSITY_RATIO {
        flags.insert("duration_too_long".into(), FAIL.into());
    } else if density_ratio < 2.0 * MIN_DENSITY_RATIO {
        flags.insert("duration_too_long".into(), WARN.into());
    }
    if density_ratio > MAX_DENSITY_RATIO {
        flags.insert("duration_too_short".into(), FAIL.into());
    } else if density_ratio > MAX_DENSITY_RATIO / 5.0 {
        flags.insert("duration_too_short".into(), WARN.into());
    }
    if ingress > 0.4 {
        flags.entry("v_shape".into()).or_insert(WARN.into());
    }
    let verdict = if flags.values().any(|v| v == FAIL) {
        FAIL
    } else {
        "pass"
    }
    .to_string();
    let mut sectors = prep.sector.clone();
    sectors.sort();
    sectors.dedup();
    Metrics {
        period_days: Some(period),
        epoch_bjd: prep.epoch,
        duration_hours: duration * 24.0,
        depth_ppm: num(d * 1e6),
        depth_err_ppm: num(d_err * 1e6),
        snr: num(snr),
        transits,
        odd_depth_ppm: num(d_odd * 1e6),
        even_depth_ppm: num(d_even * 1e6),
        odd_even_sigma: num(odd_even),
        secondary_depth_ppm: num(sec_depth * 1e6),
        secondary_sigma: num(sec_sigma),
        secondary_phase: num(sec_phase),
        secondary_albedo: num(albedo),
        secondary_threshold_sigma: num(sec_threshold),
        brightening_sigma: num(brightening),
        ingress_fraction: num(ingress),
        fitted_duration_hours: num(total * 24.0),
        fitted_duration_err_hours: num(total_err * 24.0),
        radius_earth: num(radius),
        a_over_rstar: num(a_over_r),
        density_ratio: num(density_ratio),
        geometry: Some(shape),
        centroid_sigma: num(centroid_sigma),
        source_offset_arcsec: num(centroid_arcsec),
        crowdsap: crowding,
        star: star.clone(),
        robovetter: Some(robo),
        sectors,
        flags,
        verdict,
        pixel_test: None,
    }
}

// ---------------------------------------------------------------------------
// Robovetter-style flux tests (Kepler Robovetter as adapted for TESS).

pub const N_TCE: f64 = 20000.0;
pub const SWEET_RAW_SNR: f64 = 50.0;
pub const GAP_DAYS: f64 = 0.3;
pub const RUBBLE: f64 = 0.75;
pub const CHASES_BAD: f64 = 0.01;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoboMetrics {
    pub mes: f64,
    pub shp: f64,
    pub sig_w: f64,
    pub sig_r: f64,
    pub n_tran: usize,
    pub n_in: usize,
    pub event_ses: Vec<f64>,
    pub event_depth: Vec<f64>,
    pub chases: Vec<f64>,
    pub rubble: Vec<f64>,
    pub mean_chases: f64,
    pub chi: f64,
    pub dmm: f64,
    pub max_ses: f64,
    pub n_gap: usize,
    pub new_n_transit: usize,
    pub new_mes: f64,
    pub fa1: f64,
    pub fa2: f64,
    pub sig_pri: f64,
    pub sig_sec: f64,
    pub sig_ter: f64,
    pub sig_pos: f64,
    pub dep_sec: f64,
    pub err_sec: f64,
    pub phs_sec: f64,
    pub phs_pos: f64,
    pub fred: f64,
    pub sine_amp: f64,
    pub sine_amp_err: f64,
    pub sine_sig: f64,
    pub raw_sine_amp: f64,
    pub raw_sine_sig: f64,
    pub line_aic: f64,
    pub trap_aic: f64,
    pub line_chisqr: f64,
    pub trap_chisqr: f64,
    pub trap_total_left: f64,
    pub trap_total_right: f64,
    pub asym_sigma: f64,
}

fn phase_of(t: f64, period: f64, epoch: f64) -> f64 {
    let x = (t - epoch) / period;
    x - x.round()
}

fn lower_bound(v: &[f64], x: f64) -> usize {
    v.partition_point(|&a| a < x)
}
fn upper_bound(v: &[f64], x: f64) -> usize {
    v.partition_point(|&a| a <= x)
}

struct Series {
    q: f64,
    phase: Vec<f64>,
    in_tran: Vec<bool>,
    near: Vec<bool>,
    epochs: Vec<i64>,
    tran_epochs: Vec<i64>,
    n_tran: usize,
    n_in: usize,
    zpt: f64,
    dep: f64,
    sig_w: f64,
    sig_r: f64,
    mes: f64,
    shp: f64,
    ses_series: Vec<f64>,
    grid_phase: Vec<f64>,
    dep_mes: Vec<f64>,
    err_mes: Vec<f64>,
    mes_series: Vec<f64>,
}

/// SES series per cadence, MES series on a phase grid (10 bins per duration),
/// white and red noise, MES and SHP.
fn ses_mes(time: &[f64], flux: &[f64], period: f64, epoch: f64, duration: f64) -> Series {
    let n = time.len();
    let q = duration / period;
    let phase: Vec<f64> = time.iter().map(|&t| phase_of(t, period, epoch)).collect();
    let in_tran: Vec<bool> = phase.iter().map(|p| p.abs() < q / 2.0).collect();
    let near: Vec<bool> = phase.iter().map(|p| p.abs() < q).collect();
    let epochs: Vec<i64> = time
        .iter()
        .map(|&t| ((t - epoch) / period).round() as i64)
        .collect();
    let mut tran_epochs: Vec<i64> = (0..n).filter(|&i| in_tran[i]).map(|i| epochs[i]).collect();
    tran_epochs.sort();
    tran_epochs.dedup();
    let n_tran = tran_epochs.len();
    let n_in = in_tran.iter().filter(|&&x| x).count();
    let zpt = mean(
        &(0..n)
            .filter(|&i| !near[i])
            .map(|i| flux[i])
            .collect::<Vec<_>>(),
    );
    let dep = zpt
        - mean(
            &(0..n)
                .filter(|&i| in_tran[i])
                .map(|i| flux[i])
                .collect::<Vec<_>>(),
        );
    let mut cum = vec![0.0; n + 1];
    for i in 0..n {
        cum[i + 1] = cum[i] + flux[i];
    }
    let (mut mean_t, mut n_t) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let lo = lower_bound(time, time[i] - duration / 2.0);
        let hi = upper_bound(time, time[i] + duration / 2.0);
        n_t[i] = (hi - lo) as f64;
        mean_t[i] = (cum[hi] - cum[lo]) / n_t[i];
    }
    let mut cum_out = vec![0.0; n + 1];
    let mut cnt_out = vec![0.0; n + 1];
    for i in 0..n {
        cum_out[i + 1] = cum_out[i] + if near[i] { 0.0 } else { flux[i] };
        cnt_out[i + 1] = cnt_out[i] + if near[i] { 0.0 } else { 1.0 };
    }
    let mut bin_means = Vec::new();
    let mut inv_counts = Vec::new();
    let mut out_vals = Vec::new();
    for i in 0..n {
        let lo = lower_bound(time, time[i] - duration / 2.0);
        let hi = upper_bound(time, time[i] + duration / 2.0);
        let c = cnt_out[hi] - cnt_out[lo];
        if !near[i] && c > 0.0 {
            bin_means.push((cum_out[hi] - cum_out[lo]) / c);
            inv_counts.push(1.0 / c);
            out_vals.push(flux[i]);
        }
    }
    let sig_w = std(&out_vals, 0);
    let bin_std = std(&bin_means, 0);
    let expected = sig_w * mean(&inv_counts).sqrt();
    let sig_r = (bin_std * bin_std - expected * expected).max(0.0).sqrt();
    let nbins = (10.0 / q).clamp(50.0, 20000.0) as usize;
    let width = ((nbins as f64 * q / 2.0).round() as usize).max(1);
    let cell: Vec<usize> = phase
        .iter()
        .map(|p| (((p + 0.5) * nbins as f64).floor() as i64).rem_euclid(nbins as i64) as usize)
        .collect();
    let mut sums = vec![0.0; nbins];
    let mut hits = vec![0.0; nbins];
    for i in 0..n {
        sums[cell[i]] += flux[i];
        hits[cell[i]] += 1.0;
    }
    let ring = |a: &[f64]| -> Vec<f64> {
        (0..nbins)
            .map(|b| {
                (0..=2 * width)
                    .map(|k| a[(b + nbins + k - width) % nbins])
                    .sum()
            })
            .collect()
    };
    let grid_sum = ring(&sums);
    let grid_n = ring(&hits);
    let mut per_epoch: Vec<Vec<f64>> = vec![vec![0.0; nbins]; n_tran];
    for i in 0..n {
        if let Ok(k) = tran_epochs.binary_search(&epochs[i]) {
            per_epoch[k][cell[i]] += 1.0;
        }
    }
    let mut n_transit_mes = vec![0.0; nbins];
    for row in &per_epoch {
        let r = ring(row);
        for b in 0..nbins {
            n_transit_mes[b] += (r[b] > 0.0) as i32 as f64;
        }
    }
    let dep_mes: Vec<f64> = (0..nbins).map(|b| zpt - grid_sum[b] / grid_n[b]).collect();
    let err_mes: Vec<f64> = (0..nbins)
        .map(|b| (sig_w * sig_w / grid_n[b] + sig_r * sig_r / n_transit_mes[b].max(1.0)).sqrt())
        .collect();
    let mes_series: Vec<f64> = (0..nbins).map(|b| dep_mes[b] / err_mes[b]).collect();
    let ses_series: Vec<f64> = (0..n)
        .map(|i| (zpt - mean_t[i]) / (sig_w * sig_w / n_t[i] + sig_r * sig_r).sqrt())
        .collect();
    let grid_phase: Vec<f64> = (0..nbins)
        .map(|b| (b as f64 + 0.5) / nbins as f64 - 0.5)
        .collect();
    let err = (sig_w * sig_w / n_in.max(1) as f64 + sig_r * sig_r / n_tran.max(1) as f64).sqrt();
    let finite: Vec<f64> = dep_mes
        .iter()
        .cloned()
        .filter(|v| v.is_finite())
        .map(|v| -v)
        .collect();
    let (f_min, f_max) = (
        finite.iter().cloned().fold(f64::INFINITY, f64::min),
        finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    Series {
        q,
        phase,
        in_tran,
        near,
        epochs,
        tran_epochs,
        n_tran,
        n_in,
        zpt,
        dep,
        sig_w,
        sig_r,
        mes: if err > 0.0 { dep / err } else { f64::NAN },
        shp: if f_max > f_min {
            f_max / (f_max - f_min)
        } else {
            f64::NAN
        },
        ses_series,
        grid_phase,
        dep_mes,
        err_mes,
        mes_series,
    }
}

fn individual_events(
    time: &[f64],
    flux: &[f64],
    period: f64,
    epoch: f64,
    duration: f64,
    s: &Series,
    out: &mut RoboMetrics,
) {
    let n = time.len();
    let q = s.q;
    let steps: Vec<f64> = time.windows(2).map(|w| w[1] - w[0]).collect();
    let cadence = median(
        &steps
            .iter()
            .cloned()
            .filter(|&x| x > 0.0)
            .collect::<Vec<_>>(),
    );
    let mut at_edge = vec![false; n];
    for (i, &st) in steps.iter().enumerate() {
        if st > GAP_DAYS {
            at_edge[i] = true;
            at_edge[i + 1] = true;
        }
    }
    let (mut deps, mut errs, mut ses, mut chases, mut rubble) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut gapped = 0;
    for &e in &s.tran_epochs {
        let inside: Vec<usize> = (0..n)
            .filter(|&i| s.in_tran[i] && s.epochs[i] == e)
            .collect();
        let dep = s.zpt - mean(&inside.iter().map(|&i| flux[i]).collect::<Vec<_>>());
        let err = (s.sig_w * s.sig_w / inside.len() as f64 + s.sig_r * s.sig_r).sqrt();
        let se = dep / err;
        let centre = epoch + period * e as f64;
        let nearby: Vec<f64> = (0..n)
            .filter(|&i| {
                s.phase[i].abs() > q
                    && s.phase[i].abs() < 0.1
                    && s.epochs[i] == e
                    && s.ses_series[i].abs() > 0.7 * se
            })
            .map(|i| (time[i] - centre).abs())
            .collect();
        let ch = if nearby.is_empty() {
            1.0
        } else {
            nearby.iter().cloned().fold(f64::INFINITY, f64::min) / (0.1 * period)
        };
        let fit: Vec<usize> = (0..n)
            .filter(|&i| s.phase[i].abs() < 2.0 * q && s.epochs[i] == e)
            .collect();
        let rb = fit.len() as f64 / (4.0 * duration / cadence);
        gapped += fit.iter().any(|&i| at_edge[i]) as usize;
        deps.push(dep);
        errs.push(err);
        ses.push(se);
        chases.push(ch);
        rubble.push(rb);
    }
    let n_tran = s.n_tran;
    let (chi, chi2) = if n_tran > 1 {
        let chi2: f64 = (0..n_tran)
            .map(|i| {
                let exp = s.dep / errs[i];
                (ses[i] - exp).powi(2) / exp
            })
            .sum();
        (
            if chi2 > 0.0 {
                s.mes / (chi2 / (n_tran - 1) as f64).sqrt()
            } else {
                f64::NAN
            },
            chi2,
        )
    } else {
        (f64::NAN, f64::NAN)
    };
    let _ = chi2;
    let bad: Vec<bool> = (0..n_tran)
        .map(|i| rubble[i] <= RUBBLE || ses[i] < 0.0 || (n_tran <= 5 && chases[i] < CHASES_BAD))
        .collect();
    let good_epochs: Vec<i64> = (0..n_tran)
        .filter(|&i| !bad[i])
        .map(|i| s.tran_epochs[i])
        .collect();
    let keep: Vec<usize> = (0..n)
        .filter(|&i| s.in_tran[i] && good_epochs.binary_search(&s.epochs[i]).is_ok())
        .collect();
    let (new_n, new_mes) = if keep.is_empty() {
        (0, 0.0)
    } else {
        let new_n = good_epochs.len();
        (
            (new_n),
            (s.zpt - mean(&keep.iter().map(|&i| flux[i]).collect::<Vec<_>>()))
                / (s.sig_w * s.sig_w / keep.len() as f64 + s.sig_r * s.sig_r / new_n as f64).sqrt(),
        )
    };
    let med = median(&deps);
    out.event_ses = ses.clone();
    out.event_depth = deps.clone();
    out.chases = chases.clone();
    out.rubble = rubble;
    out.mean_chases = mean(&chases);
    out.chi = chi;
    out.dmm = if med != 0.0 {
        mean(&deps) / med
    } else {
        f64::NAN
    };
    out.max_ses = ses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    out.n_gap = gapped;
    out.new_n_transit = new_n;
    out.new_mes = new_mes;
}

fn dist(a: f64, b: f64) -> f64 {
    ((a - b + 0.5).rem_euclid(1.0) - 0.5).abs()
}

fn uniqueness(flux: &[f64], s: &Series, out: &mut RoboMetrics) {
    let q = s.q;
    out.fa1 = std::f64::consts::SQRT_2 * erfcinv(q / N_TCE);
    out.fa2 = std::f64::consts::SQRT_2 * erfcinv(q);
    out.sig_pri = f64::NAN;
    out.sig_sec = f64::NAN;
    out.sig_ter = f64::NAN;
    out.sig_pos = f64::NAN;
    out.dep_sec = f64::NAN;
    out.err_sec = f64::NAN;
    out.phs_sec = f64::NAN;
    out.phs_pos = f64::NAN;
    out.fred = f64::NAN;
    let nb = s.grid_phase.len();
    let finite: Vec<bool> = s.mes_series.iter().map(|v| v.is_finite()).collect();
    let inside: Vec<usize> = (0..nb)
        .filter(|&b| finite[b] && s.grid_phase[b].abs() < q / 2.0)
        .collect();
    let Some(&i) = inside
        .iter()
        .max_by(|&&a, &&b| s.mes_series[a].partial_cmp(&s.mes_series[b]).unwrap())
    else {
        return;
    };
    out.sig_pri = s.mes_series[i];
    let pri = s.grid_phase[i];
    let free: Vec<usize> = (0..nb)
        .filter(|&b| finite[b] && dist(s.grid_phase[b], pri) >= 2.0 * q)
        .collect();
    let Some(&j) = free
        .iter()
        .max_by(|&&a, &&b| s.mes_series[a].partial_cmp(&s.mes_series[b]).unwrap())
    else {
        return;
    };
    out.sig_sec = s.mes_series[j];
    out.dep_sec = s.dep_mes[j];
    out.err_sec = s.err_mes[j];
    out.phs_sec = s.grid_phase[j];
    let sec = s.grid_phase[j];
    let rest: Vec<f64> = free
        .iter()
        .filter(|&&b| dist(s.grid_phase[b], sec) >= q)
        .map(|&b| s.dep_mes[b])
        .collect();
    let points: Vec<f64> = (0..flux.len())
        .filter(|&i| dist(s.phase[i], pri) >= 2.0 * q && dist(s.phase[i], sec) >= q)
        .map(|i| flux[i])
        .collect();
    if !rest.is_empty() && !points.is_empty() && std(&points, 0) > 0.0 {
        out.fred = (s.n_in as f64).sqrt() * std(&rest, 0) / std(&points, 0);
    }
    let free2: Vec<usize> = free
        .iter()
        .cloned()
        .filter(|&b| dist(s.grid_phase[b], sec) >= 2.0 * q)
        .collect();
    if !free2.is_empty() {
        out.sig_ter = free2
            .iter()
            .map(|&b| s.mes_series[b])
            .fold(f64::NEG_INFINITY, f64::max);
    }
    let far: Vec<usize> = (0..nb)
        .filter(|&b| {
            finite[b]
                && dist(s.grid_phase[b], pri) >= 3.0 * q
                && dist(s.grid_phase[b], sec) >= 3.0 * q
        })
        .collect();
    if let Some(&k) = far
        .iter()
        .max_by(|&&a, &&b| (-s.mes_series[a]).partial_cmp(&-s.mes_series[b]).unwrap())
    {
        out.sig_pos = -s.mes_series[k];
        out.phs_pos = s.grid_phase[k];
    }
}

/// Strongest sinusoid at half, once and twice the period in the flux outside
/// `exclude`: (amplitude, amplitude error, significance).
fn sweet(time: &[f64], flux: &[f64], period: f64, epoch: f64, exclude: &[bool]) -> (f64, f64, f64) {
    let idx: Vec<usize> = (0..time.len()).filter(|&i| !exclude[i]).collect();
    let t: Vec<f64> = idx.iter().map(|&i| time[i]).collect();
    let f: Vec<f64> = idx.iter().map(|&i| flux[i]).collect();
    let mut best = (0.0, f64::NAN, 0.0);
    for per in [period / 2.0, period, 2.0 * period] {
        let w: Vec<f64> = t
            .iter()
            .map(|x| 2.0 * std::f64::consts::PI * (x - epoch) / per)
            .collect();
        let design = vec![
            vec![1.0; t.len()],
            w.iter().map(|x| x.sin()).collect(),
            w.iter().map(|x| x.cos()).collect(),
        ];
        let Some((coef, cov, _)) = lstsq(&design, &f) else {
            continue;
        };
        let amp = coef[1].hypot(coef[2]);
        if amp == 0.0 {
            continue;
        }
        let g = [coef[1] / amp, coef[2] / amp];
        let var = g[0] * g[0] * cov[1][1] + 2.0 * g[0] * g[1] * cov[1][2] + g[1] * g[1] * cov[2][2];
        let err = var.max(1e-30).sqrt();
        if amp / err > best.2 {
            best = (amp, err, amp / err);
        }
    }
    best
}

fn trapezoid_model(x: f64, depth: f64, total: f64, ingress: f64, zpt: f64) -> f64 {
    let ramp = (ingress * total).max(1e-6);
    zpt - depth * ((total / 2.0 - x.abs()) / ramp).clamp(0.0, 1.0)
}

fn aic(chi2: f64, n: usize, k: usize) -> f64 {
    n as f64 * (chi2 / n as f64).max(1e-300).ln() + 2.0 * k as f64
}

/// Trapezoid against a straight line within two durations of the event (AIC),
/// and the half-durations fitted on each side with the trapezoid otherwise fixed.
fn model_comparison(flux: &[f64], period: f64, duration: f64, s: &Series, out: &mut RoboMetrics) {
    let (q, sig) = (s.q, s.sig_w);
    let zone: Vec<usize> = (0..flux.len())
        .filter(|&i| s.phase[i].abs() < 2.0 * q)
        .collect();
    let x: Vec<f64> = zone.iter().map(|&i| s.phase[i] * period).collect();
    let f: Vec<f64> = zone.iter().map(|&i| flux[i]).collect();
    let n = f.len();
    out.line_aic = f64::NAN;
    out.trap_aic = f64::NAN;
    out.line_chisqr = f64::NAN;
    out.trap_chisqr = f64::NAN;
    out.trap_total_left = f64::NAN;
    out.trap_total_right = f64::NAN;
    out.asym_sigma = f64::NAN;
    if n < 20 || sig <= 0.0 {
        return;
    }
    let fm = mean(&f);
    let chi_line: f64 = f.iter().map(|v| (v - fm).powi(2)).sum::<f64>() / (sig * sig);
    out.line_aic = aic(chi_line, n, 1);
    out.line_chisqr = chi_line / (n - 1) as f64;
    let start = [s.dep.max(1e-6), duration, 0.2, s.zpt, 0.0];
    let lower = [0.0, 0.3 * duration, 0.01, f64::NEG_INFINITY, -0.1 * period];
    let upper = [
        1.0,
        (2.5 * duration).min(period / 2.0),
        0.5,
        f64::INFINITY,
        0.1 * period,
    ];
    let (p, r, _) = least_squares(
        |p| {
            x.iter()
                .zip(&f)
                .map(|(&xi, &fi)| (trapezoid_model(xi - p[4], p[0], p[1], p[2], p[3]) - fi) / sig)
                .collect()
        },
        &start,
        &lower,
        &upper,
    );
    let chi_trap: f64 = r.iter().map(|v| v * v).sum();
    out.trap_aic = aic(chi_trap, n, 5);
    out.trap_chisqr = chi_trap / (n.saturating_sub(5)).max(1) as f64;
    let (depth, total, ingress, zpt, shift) = (p[0], p[1], p[2], p[3], p[4]);
    let mut sides: Vec<(f64, f64)> = Vec::new();
    for left in [true, false] {
        let sel: Vec<usize> = (0..n)
            .filter(|&i| {
                if left {
                    x[i] - shift < 0.0
                } else {
                    x[i] - shift > 0.0
                }
            })
            .collect();
        if sel.len() < 10 {
            break;
        }
        let xs: Vec<f64> = sel.iter().map(|&i| x[i] - shift).collect();
        let fs: Vec<f64> = sel.iter().map(|&i| f[i]).collect();
        let (ph, rh, jh) = least_squares(
            |p| {
                xs.iter()
                    .zip(&fs)
                    .map(|(&xi, &fi)| (trapezoid_model(xi, depth, p[0], ingress, zpt) - fi) / sig)
                    .collect()
            },
            &[total],
            &[0.3 * duration],
            &[(2.5 * duration).min(period / 2.0)],
        );
        let curvature: f64 = jh[0].iter().map(|v| v * v).sum();
        let cost: f64 = rh.iter().map(|v| v * v).sum::<f64>() / 2.0;
        let err = if curvature > 0.0 {
            (2.0 * cost / (xs.len() - 1).max(1) as f64 / curvature).sqrt()
        } else {
            f64::NAN
        };
        sides.push((ph[0], err));
    }
    if sides.len() == 2 {
        let ((tl, el), (tr, er)) = (sides[0], sides[1]);
        out.trap_total_left = tl;
        out.trap_total_right = tr;
        out.asym_sigma = if el > 0.0 && er > 0.0 {
            (tl - tr).abs() / el.hypot(er)
        } else {
            f64::NAN
        };
    }
}

/// Every flux metric of one signal; `flux` is the detrended curve, `raw` the
/// normalised PDCSAP flux on the same cadences (NaN where missing).
pub fn robo_metrics(
    time: &[f64],
    flux: &[f64],
    raw: &[f64],
    period: f64,
    epoch: f64,
    duration: f64,
) -> RoboMetrics {
    let mut order: Vec<usize> = (0..time.len()).collect();
    order.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap());
    let time: Vec<f64> = order.iter().map(|&i| time[i]).collect();
    let flux: Vec<f64> = order.iter().map(|&i| flux[i]).collect();
    let raw: Vec<f64> = order.iter().map(|&i| raw[i]).collect();
    let s = ses_mes(&time, &flux, period, epoch, duration);
    let mut out = RoboMetrics {
        mes: s.mes,
        shp: s.shp,
        sig_w: s.sig_w,
        sig_r: s.sig_r,
        n_tran: s.n_tran,
        n_in: s.n_in,
        ..Default::default()
    };
    individual_events(&time, &flux, period, epoch, duration, &s, &mut out);
    uniqueness(&flux, &s, &mut out);
    let (a, e, sg) = sweet(&time, &flux, period, epoch, &s.near);
    out.sine_amp = a;
    out.sine_amp_err = e;
    out.sine_sig = sg;
    let good: Vec<usize> = (0..raw.len()).filter(|&i| raw[i].is_finite()).collect();
    if good.len() > 100 {
        let t: Vec<f64> = good.iter().map(|&i| time[i]).collect();
        let r: Vec<f64> = good.iter().map(|&i| raw[i]).collect();
        let ex: Vec<bool> = good.iter().map(|&i| s.near[i]).collect();
        let (a, _, sg) = sweet(&t, &r, period, epoch, &ex);
        out.raw_sine_amp = a;
        out.raw_sine_sig = sg;
    } else {
        out.raw_sine_amp = f64::NAN;
        out.raw_sine_sig = f64::NAN;
    }
    model_comparison(&flux, period, duration, &s, &mut out);
    out
}

pub struct Thresholds;
impl Thresholds {
    pub const MES: f64 = 6.2;
    pub const SHP: f64 = 0.6;
    pub const MS1: f64 = 0.2;
    pub const MS2: f64 = 0.8;
    pub const MS3: f64 = 0.8;
    pub const CHASES: f64 = 0.78;
    pub const DMM: f64 = 1.5;
    pub const MAX_SES_TO_MES: f64 = 0.8;
    pub const AIC1: f64 = -60.0;
    pub const AIC2: f64 = -30.0;
    pub const SWEET: f64 = 15.0;
    pub const ASYM: f64 = 8.0;
    pub const CHI: f64 = 7.8;
    pub const FRAC_GAP: f64 = 0.5;
    pub const V_SHAPE: f64 = 1.5;
    pub const MS4: f64 = 0.0;
    pub const MS5: f64 = -1.0;
    pub const MS6: f64 = -1.0;
}

/// Verdicts from the flux metrics: flag name -> fail/warn.
#[allow(clippy::too_many_arguments)]
pub fn robo_flags(
    m: &RoboMetrics,
    period: f64,
    depth: f64,
    radius_earth: Option<f64>,
    albedo: Option<f64>,
    impact: Option<f64>,
    radius_ratio: Option<f64>,
    max_radius: f64,
) -> BTreeMap<String, String> {
    let mut f = BTreeMap::new();
    let nan = |v: f64| v.is_nan();
    let fail = |f: &mut BTreeMap<String, String>, k: &str| {
        f.insert(k.to_string(), FAIL.to_string());
    };
    if m.mes < Thresholds::MES {
        fail(&mut f, "weak");
    }
    if m.new_mes < Thresholds::MES || m.new_n_transit < 2 {
        fail(&mut f, "invalid_transits");
    }
    if m.shp > Thresholds::SHP {
        fail(&mut f, "bad_shape");
    }
    if !nan(m.sig_pri) {
        let ms1 = !nan(m.fred) && m.fred > 0.0 && (m.sig_pri / m.fred - m.fa1) < Thresholds::MS1;
        let ms2 = !nan(m.sig_ter) && (m.sig_pri - m.sig_ter - m.fa2) < Thresholds::MS2;
        let ms3 = !nan(m.sig_pos) && (m.sig_pri - m.sig_pos - m.fa2) < Thresholds::MS3;
        if ms1 || ms2 || ms3 {
            fail(&mut f, "not_unique");
        }
    }
    if m.n_tran <= 5 && m.mean_chases < Thresholds::CHASES {
        fail(&mut f, "chases");
    }
    if m.dmm > Thresholds::DMM {
        fail(&mut f, "inconsistent_depths");
    }
    if m.n_tran <= 10 && m.mes > 0.0 && m.max_ses / m.mes > Thresholds::MAX_SES_TO_MES {
        fail(&mut f, "single_event_dominates");
    }
    if nan(m.trap_aic)
        || m.trap_chisqr > m.line_chisqr
        || (m.trap_aic - m.line_aic)
            > if m.n_tran <= 10 {
                Thresholds::AIC1
            } else {
                Thresholds::AIC2
            }
    {
        fail(&mut f, "bad_fit");
    }
    if m.sine_sig > Thresholds::SWEET && period < 10.0 {
        fail(&mut f, "sinusoidal");
    }
    if !nan(m.raw_sine_sig) && m.raw_sine_sig > SWEET_RAW_SNR && m.raw_sine_amp > depth {
        if period < 10.0 {
            fail(&mut f, "sinusoidal");
        } else {
            f.entry("sinusoidal".into()).or_insert(WARN.into());
        }
    }
    if m.asym_sigma > Thresholds::ASYM {
        fail(&mut f, "asymmetric");
    }
    if m.chi < Thresholds::CHI {
        fail(&mut f, "inconsistent_snr");
    }
    if m.n_tran > 0 && m.n_gap as f64 / m.n_tran as f64 >= Thresholds::FRAC_GAP {
        fail(&mut f, "gapped");
    }
    if !nan(m.sig_sec) && !nan(m.fred) && m.fred > 0.0 {
        let ms4 = (m.sig_sec / m.fred - m.fa1) > Thresholds::MS4 || m.fred > 1.8;
        let ms5 = nan(m.sig_ter) || (m.sig_sec - m.sig_ter - m.fa2) > Thresholds::MS5;
        let ms6 = nan(m.sig_pos) || (m.sig_sec - m.sig_pos - m.fa2) > Thresholds::MS6;
        let planet_like = radius_earth.is_some_and(|r| r < max_radius)
            && albedo.is_some_and(|a| a < 1.0)
            && m.dep_sec < 0.1 * depth
            && impact.is_none_or(|b| b < 0.95);
        if ms4 && (ms5 || ms6) && !planet_like {
            fail(&mut f, "secondary");
        }
    }
    if let (Some(b), Some(k)) = (impact, radius_ratio)
        && b + k > Thresholds::V_SHAPE
    {
        fail(&mut f, "v_shape");
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(depth: f64, period: f64, duration: f64, noise: f64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Rng::new(7);
        let t: Vec<f64> = (0..20000)
            .map(|i| 2458000.0 + i as f64 * 2.0 / 1440.0)
            .collect();
        let f: Vec<f64> = t
            .iter()
            .map(|&x| {
                let ph = wrap(x - 2458001.3, period);
                rng.normal(0.0, noise)
                    - if ph.abs() < duration / 2.0 {
                        depth
                    } else {
                        0.0
                    }
            })
            .collect();
        (t, f)
    }

    #[test]
    fn box_transit_passes_and_noise_fails() {
        let (t, f) = synthetic(0.003, 2.5, 0.1, 0.0005);
        let sector = vec![1i64; t.len()];
        let prep = fold(&t, &f, &sector, 2.5, 2458001.3, 2.4, &[]).unwrap();
        let star = StarParameters {
            radius: Some(1.0),
            teff: Some(5700.0),
            logg: Some(4.4),
            tmag: Some(10.0),
        };
        let m = measure(&prep, &star, None, None, Some(0.95), &BTreeMap::new(), None);
        assert!(m.snr.unwrap() > 7.1, "snr {:?}", m.snr);
        assert!(
            (m.depth_ppm.unwrap() - 3000.0).abs() < 300.0,
            "depth {:?}",
            m.depth_ppm
        );
        assert!(m.transits >= 5);
        assert_eq!(m.verdict, "pass", "{:?}", m.flags);
        let (t, f) = synthetic(0.0, 2.5, 0.1, 0.0005);
        let prep = fold(&t, &f, &sector, 2.5, 2458001.3, 2.4, &[]).unwrap();
        let m = measure(&prep, &star, None, None, Some(0.95), &BTreeMap::new(), None);
        assert_eq!(m.verdict, FAIL);
        assert!(m.flags.contains_key("low_snr"));
    }

    #[test]
    fn odd_even_difference_is_flagged() {
        let mut rng = Rng::new(3);
        let t: Vec<f64> = (0..30000)
            .map(|i| 2458000.0 + i as f64 * 2.0 / 1440.0)
            .collect();
        let f: Vec<f64> = t
            .iter()
            .map(|&x| {
                let ph = wrap(x - 2458001.0, 2.0);
                let k = ((x - 2458001.0) / 2.0).round() as i64;
                rng.normal(0.0, 0.0003)
                    - if ph.abs() < 0.05 {
                        if k % 2 == 0 { 0.004 } else { 0.001 }
                    } else {
                        0.0
                    }
            })
            .collect();
        let prep = fold(&t, &f, &vec![1; t.len()], 2.0, 2458001.0, 2.4, &[]).unwrap();
        let m = measure(
            &prep,
            &StarParameters::default(),
            None,
            None,
            None,
            &BTreeMap::new(),
            None,
        );
        assert_eq!(
            m.flags.get("odd_even").map(String::as_str),
            Some(FAIL),
            "{:?}",
            m.flags
        );
    }
}
