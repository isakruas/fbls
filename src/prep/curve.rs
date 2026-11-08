//! Light-curve preparation on the host: per-group median normalization, time
//! binning, robust error inflation, duration families, biweight detrending and
//! the diagonal noise model consumed by the GPU search.

use crate::prep::stats::{argsort, lower_bound, mad_scale, median_lower, upper_bound};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct Observations {
    pub time: Vec<f64>,
    pub flux: Vec<f64>,
    pub err: Option<Vec<f64>>,
    /// Normalization group (input file, sector) as a dense index.
    pub group: Vec<i64>,
}

/// Diagonal noise model: times relative to `origin`, flux deficit (-relative flux)
/// and inflated variances.
#[derive(Debug, Clone)]
pub struct DiagonalNoise {
    pub time: Vec<f64>,
    pub deficit: Vec<f64>,
    pub variance: Vec<f64>,
    pub origin: f64,
    pub inflation: f64,
    pub segments: usize,
}

#[derive(Debug, Clone)]
pub struct PreparedCurve {
    /// Absolute BJD.
    pub time: Vec<f64>,
    /// Detrended relative flux; transits are negative.
    pub flux: Vec<f64>,
    pub sigma: Vec<f64>,
    pub window_days: f64,
    pub clipped: usize,
}

/// Precision-weighted fields of the generalized least-squares amplitude statistic.
/// v = p*y - p*(sum p y)/c, p = w = 1/variance, c = sum p.
#[derive(Debug, Clone)]
pub struct Ready {
    pub time: Vec<f64>,
    pub origin: f64,
    pub v: Vec<f64>,
    pub p: Vec<f64>,
    pub w: Vec<f64>,
    pub baseline_norm: f64,
    pub residual_energy: f64,
}

#[derive(Debug, Clone)]
pub struct FamilyState {
    pub curve: PreparedCurve,
    pub durations: Vec<f64>,
    pub indices: Vec<usize>,
    pub ready: Ready,
    pub rho: Vec<f64>,
    pub noise: NoiseSummary,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct NoiseSummary {
    pub inflation: f64,
    pub segments: usize,
}

fn unique_inverse(keys: &[(i64, i64)]) -> (Vec<(i64, i64)>, Vec<usize>) {
    let mut map: BTreeMap<(i64, i64), usize> = BTreeMap::new();
    for k in keys {
        map.entry(*k).or_insert(0);
    }
    let unique: Vec<(i64, i64)> = map.keys().copied().collect();
    for (i, k) in unique.iter().enumerate() {
        map.insert(*k, i);
    }
    let inverse = keys.iter().map(|k| map[k]).collect();
    (unique, inverse)
}

/// Normalizes each group by its median, optionally bins with inverse-variance
/// weights (never across groups), and inflates errors by the robust scatter of
/// standardized adjacent differences. Without input errors a white-noise level
/// is estimated from adjacent differences instead.
pub fn fit_segment_noise(
    time: &[f64],
    flux: &[f64],
    err: Option<&[f64]>,
    group: Option<&[i64]>,
    bin_minutes: f64,
    gap_days: f64,
) -> Result<DiagonalNoise, String> {
    let n = time.len();
    let ones = vec![1.0; n];
    let err_in: &[f64] = err.unwrap_or(&ones);
    let zeros = vec![0i64; n];
    let group: &[i64] = group.unwrap_or(&zeros);
    let mut keep: Vec<usize> = (0..n)
        .filter(|&i| {
            time[i].is_finite() && flux[i].is_finite() && err_in[i].is_finite() && err_in[i] > 0.0
        })
        .collect();
    let t_keep: Vec<f64> = keep.iter().map(|&i| time[i]).collect();
    let order = argsort(&t_keep);
    keep = order.iter().map(|&j| keep[j]).collect();
    let mut t: Vec<f64> = keep.iter().map(|&i| time[i]).collect();
    let mut f: Vec<f64> = keep.iter().map(|&i| flux[i]).collect();
    let mut e: Vec<f64> = keep.iter().map(|&i| err_in[i]).collect();
    let mut sec: Vec<i64> = keep.iter().map(|&i| group[i]).collect();
    if t.len() < 6 {
        return Err("insufficient usable observations".into());
    }
    let mut labels = sec.clone();
    labels.sort();
    labels.dedup();
    for label in labels {
        let idx: Vec<usize> = (0..t.len()).filter(|&i| sec[i] == label).collect();
        let vals: Vec<f64> = idx.iter().map(|&i| f[i]).collect();
        let m = median_lower(&vals);
        if !(m > 0.0) {
            return Err("mission flux must have a positive median".into());
        }
        for &i in &idx {
            f[i] = f[i] / m - 1.0;
            e[i] /= m;
        }
    }
    let origin = t[0];
    for x in t.iter_mut() {
        *x -= origin;
    }
    if bin_minutes > 0.0 {
        let width = bin_minutes / 1440.0;
        let keys: Vec<(i64, i64)> = t
            .iter()
            .zip(&sec)
            .map(|(x, s)| ((x / width).floor() as i64, *s))
            .collect();
        let (unique, inv) = unique_inverse(&keys);
        let m = unique.len();
        let mut den = vec![0.0; m];
        let mut st = vec![0.0; m];
        let mut sf = vec![0.0; m];
        for i in 0..t.len() {
            let w = 1.0 / (e[i] * e[i]);
            den[inv[i]] += w;
            st[inv[i]] += t[i] * w;
            sf[inv[i]] += f[i] * w;
        }
        t = (0..m).map(|k| st[k] / den[k]).collect();
        f = (0..m).map(|k| sf[k] / den[k]).collect();
        e = (0..m).map(|k| 1.0 / den[k].sqrt()).collect();
        sec = unique.iter().map(|k| k.1).collect();
        let order = argsort(&t);
        t = order.iter().map(|&i| t[i]).collect();
        f = order.iter().map(|&i| f[i]).collect();
        e = order.iter().map(|&i| e[i]).collect();
        sec = order.iter().map(|&i| sec[i]).collect();
    }
    let n = t.len();
    let mut standardized = Vec::new();
    let mut deltas = Vec::new();
    for i in 1..n {
        if t[i] - t[i - 1] < gap_days && sec[i] == sec[i - 1] {
            let d = f[i] - f[i - 1];
            deltas.push(d);
            standardized.push(d / (e[i] * e[i] + e[i - 1] * e[i - 1]).sqrt());
        }
    }
    if deltas.is_empty() {
        return Err("no adjacent observations available for noise estimation".into());
    }
    let inflation = mad_scale(&standardized).max(1.0);
    let variance: Vec<f64> = if err.is_some() {
        e.iter().map(|x| (x * inflation).powi(2)).collect()
    } else {
        let m = median_lower(&deltas);
        let dev: Vec<f64> = deltas.iter().map(|d| (d - m).abs()).collect();
        let white = (1.4826 / 2f64.sqrt() * median_lower(&dev)).max(1e-12);
        vec![white * white; n]
    };
    let segments = 1
        + (1..n)
            .filter(|&i| t[i] - t[i - 1] > gap_days || sec[i] != sec[i - 1])
            .count();
    Ok(DiagonalNoise {
        time: t,
        deficit: f.iter().map(|x| -x).collect(),
        variance,
        origin,
        inflation,
        segments,
    })
}

/// Tukey biweight location in a sliding time window per gap-separated segment.
/// Returns (1 + f) / trend - 1.
pub fn biweight_flatten(
    time: &[f64],
    relative_flux: &[f64],
    window: f64,
    gap: f64,
    c: f64,
    iterations: usize,
) -> Vec<f64> {
    let n = time.len();
    let flux: Vec<f64> = relative_flux.iter().map(|x| 1.0 + x).collect();
    let mut out = vec![0.0; n];
    let mut bounds = vec![0usize];
    for i in 1..n {
        if time[i] - time[i - 1] > gap {
            bounds.push(i);
        }
    }
    bounds.push(n);
    for pair in bounds.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let lt = &time[a..b];
        let lf = &flux[a..b];
        for i in 0..(b - a) {
            let lo = lower_bound(lt, lt[i] - window / 2.0);
            let hi = upper_bound(lt, lt[i] + window / 2.0);
            let values = &lf[lo..hi];
            let mut location = crate::prep::stats::median_lower(values);
            for _ in 0..iterations {
                let dev: Vec<f64> = values.iter().map(|v| (v - location).abs()).collect();
                let scale = (c * crate::prep::stats::median_lower(&dev)).max(1e-12);
                let mut total = 0.0;
                let mut acc = 0.0;
                for &v in values {
                    let u: f64 = (v - location) / scale;
                    if u.abs() < 1.0 {
                        let w = (1.0 - u * u).powi(2);
                        total += w;
                        acc += w * v;
                    }
                }
                if total > 0.0 {
                    location = acc / total;
                }
            }
            out[a + i] = lf[i] / location - 1.0;
        }
    }
    out
}

/// Distance of every observation to the nearest edge of its gap-separated segment.
pub fn segment_edges(time: &[f64], gap: f64) -> Vec<f64> {
    let n = time.len();
    let mut out = vec![0.0; n];
    let mut start = 0;
    for i in 1..=n {
        if i == n || time[i] - time[i - 1] > gap {
            for j in start..i {
                out[j] = (time[j] - time[start]).min(time[i - 1] - time[j]);
            }
            start = i;
        }
    }
    out
}

/// Groups durations by detrending window clamp(factor * D, minimum, maximum),
/// rounded to 1e-6 days. Returns (window, duration indices) sorted by window.
pub fn detrending_windows(
    durations: &[f64],
    minimum: f64,
    factor: f64,
    maximum: f64,
) -> Vec<(f64, Vec<usize>)> {
    let windows: Vec<f64> = durations
        .iter()
        .map(|d| ((factor * d).clamp(minimum, maximum) * 1e6).round() / 1e6)
        .collect();
    let mut uniq = windows.clone();
    uniq.sort_by(|a, b| a.partial_cmp(b).unwrap());
    uniq.dedup();
    uniq.into_iter()
        .map(|w| {
            (
                w,
                (0..durations.len()).filter(|&i| windows[i] == w).collect(),
            )
        })
        .collect()
}

pub struct PrepConfig {
    pub bin_minutes: f64,
    pub window_min: f64,
    pub window_factor: f64,
    pub window_max: f64,
    pub clip_sigma: f64,
    pub edge_trim: f64,
}

/// One detrended curve per duration family. Positive outliers above
/// `clip_sigma` robust scatter and observations within `edge_trim` windows of a
/// segment edge are dropped.
pub fn prepare_families(
    obs: &Observations,
    durations: &[f64],
    cfg: &PrepConfig,
) -> Result<Vec<(PreparedCurve, Vec<usize>)>, String> {
    let diag = fit_segment_noise(
        &obs.time,
        &obs.flux,
        obs.err.as_deref(),
        Some(&obs.group),
        cfg.bin_minutes,
        0.75,
    )?;
    let sigma: Vec<f64> = diag.variance.iter().map(|v| v.sqrt()).collect();
    let edge = segment_edges(&diag.time, 0.5);
    let relative: Vec<f64> = diag.deficit.iter().map(|d| -d).collect();
    let mut out = Vec::new();
    for (window, indices) in
        detrending_windows(durations, cfg.window_min, cfg.window_factor, cfg.window_max)
    {
        let flat = biweight_flatten(&diag.time, &relative, window, 0.5, 5.0, 10);
        let scale = mad_scale(&flat);
        let keep: Vec<usize> = (0..flat.len())
            .filter(|&i| {
                flat[i].is_finite()
                    && flat[i] <= cfg.clip_sigma * scale
                    && edge[i] >= cfg.edge_trim * window
            })
            .collect();
        let curve = PreparedCurve {
            time: keep.iter().map(|&i| diag.time[i] + diag.origin).collect(),
            flux: keep.iter().map(|&i| flat[i]).collect(),
            sigma: keep.iter().map(|&i| sigma[i]).collect(),
            window_days: window,
            clipped: flat.len() - keep.len(),
        };
        out.push((curve, indices));
    }
    Ok(out)
}

pub fn prepare_ready(model: &DiagonalNoise) -> Result<Ready, String> {
    let n = model.time.len();
    if n < 6 {
        return Err("at least six observations are required".into());
    }
    let p: Vec<f64> = model.variance.iter().map(|d| 1.0 / d).collect();
    let c: f64 = p.iter().sum();
    if !(c.is_finite() && c > 0.0) {
        return Err("invalid baseline precision".into());
    }
    let py: Vec<f64> = (0..n).map(|i| p[i] * model.deficit[i]).collect();
    let mean = py.iter().sum::<f64>() / c;
    let v: Vec<f64> = (0..n).map(|i| py[i] - p[i] * mean).collect();
    let residual_energy = (0..n).map(|i| model.deficit[i] * v[i]).sum();
    Ok(Ready {
        time: model.time.clone(),
        origin: model.origin,
        w: p.clone(),
        p,
        v,
        baseline_norm: c,
        residual_energy,
    })
}

/// Per-duration correlated-noise factor rho_D >= 1: robust scatter of the
/// standardized inverse-variance means of residuals in contiguous bins of
/// length D. Under-sampled durations inherit sqrt(D/D_ref) scaling from the
/// nearest supported duration, floored by their own measured scatter.
pub fn correlated_noise_factors(
    time: &[f64],
    residual: &[f64],
    sigma: &[f64],
    durations: &[f64],
) -> Vec<f64> {
    let min_fill = 0.5;
    let minimum_bins = 20usize;
    let n = time.len();
    let weights: Vec<f64> = sigma.iter().map(|s| 1.0 / (s * s)).collect();
    let steps: Vec<f64> = (1..n)
        .map(|i| time[i] - time[i - 1])
        .filter(|&s| s > 0.0)
        .collect();
    let cadence = median_lower(&steps);
    let mut measured = Vec::new();
    let mut counts = Vec::new();
    for &d in durations {
        let mut map: BTreeMap<i64, (f64, f64, usize)> = BTreeMap::new();
        for i in 0..n {
            let cell = ((time[i] - time[0]) / d).floor() as i64;
            let e = map.entry(cell).or_insert((0.0, 0.0, 0));
            e.0 += weights[i];
            e.1 += weights[i] * residual[i];
            e.2 += 1;
        }
        let z: Vec<f64> = map
            .values()
            .filter(|(_, _, cnt)| (*cnt as f64) >= min_fill * d / cadence)
            .map(|(total, acc, _)| (acc / total) * total.sqrt())
            .collect();
        counts.push(z.len());
        if std::env::var_os("FBLS_DEBUG").is_some() {
            eprintln!(
                "rho debug: duration={d} cadence={cadence} bins={} usable={} fill_threshold={}",
                map.len(),
                z.len(),
                min_fill * d / cadence
            );
        }
        measured.push(if z.len() >= 3 {
            mad_scale(&z).max(1.0)
        } else {
            1.0
        });
    }
    let supported: Vec<bool> = counts.iter().map(|&c| c >= minimum_bins).collect();
    let inherited: Vec<f64> = if supported.iter().any(|&s| s) {
        let valid: Vec<usize> = (0..durations.len()).filter(|&i| supported[i]).collect();
        durations
            .iter()
            .map(|&d| {
                let anchor = *valid
                    .iter()
                    .min_by(|&&a, &&b| {
                        (d.ln() - durations[a].ln())
                            .abs()
                            .partial_cmp(&(d.ln() - durations[b].ln()).abs())
                            .unwrap()
                    })
                    .unwrap();
                measured[anchor] * (d / durations[anchor]).max(1.0).sqrt()
            })
            .collect()
    } else {
        let standardized: Vec<f64> = (0..n).map(|i| residual[i] / sigma[i]).collect();
        let scatter = mad_scale(&standardized).max(1.0);
        durations
            .iter()
            .map(|&d| scatter * (d / cadence).max(1.0).sqrt())
            .collect()
    };
    (0..durations.len())
        .map(|i| {
            if supported[i] {
                measured[i]
            } else {
                measured[i].max(inherited[i])
            }
        })
        .collect()
}

pub fn family_state(
    curve: PreparedCurve,
    durations: &[f64],
    indices: &[usize],
    correlated: bool,
) -> Result<FamilyState, String> {
    let flux1: Vec<f64> = curve.flux.iter().map(|f| 1.0 + f).collect();
    let model = fit_segment_noise(&curve.time, &flux1, Some(&curve.sigma), None, 0.0, 0.75)?;
    let ready = prepare_ready(&model)?;
    let rho = if correlated {
        let resid: Vec<f64> = model.deficit.iter().map(|d| -d).collect();
        let sig: Vec<f64> = model.variance.iter().map(|v| v.sqrt()).collect();
        correlated_noise_factors(&model.time, &resid, &sig, durations)
    } else {
        vec![1.0; durations.len()]
    };
    let noise = NoiseSummary {
        inflation: model.inflation,
        segments: model.segments,
    };
    Ok(FamilyState {
        curve,
        durations: durations.to_vec(),
        indices: indices.to_vec(),
        ready,
        rho,
        noise,
    })
}

/// Observations within (1 + margin) * duration / 2 of predicted transit centers.
pub fn transit_mask(
    time: &[f64],
    period: f64,
    epoch: f64,
    duration: f64,
    margin: f64,
) -> Vec<bool> {
    time.iter()
        .map(|&t| {
            let phase = (t - epoch + period / 2.0).rem_euclid(period) - period / 2.0;
            phase.abs() <= (1.0 + margin) * duration / 2.0
        })
        .collect()
}

pub fn mask_curve(
    curve: &PreparedCurve,
    period: f64,
    epoch: f64,
    duration: f64,
    margin: f64,
) -> PreparedCurve {
    let m = transit_mask(&curve.time, period, epoch, duration, margin);
    let keep: Vec<usize> = (0..m.len()).filter(|&i| !m[i]).collect();
    PreparedCurve {
        time: keep.iter().map(|&i| curve.time[i]).collect(),
        flux: keep.iter().map(|&i| curve.flux[i]).collect(),
        sigma: keep.iter().map(|&i| curve.sigma[i]).collect(),
        window_days: curve.window_days,
        clipped: curve.clipped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_group_equal_clamped_values() {
        let d = [0.5 / 24.0, 1.0 / 24.0, 2.0 / 24.0, 12.0 / 24.0, 40.0 / 24.0];
        let w = detrending_windows(&d, 0.4, 3.0, 3.5);
        assert_eq!(w[0], (0.4, vec![0, 1, 2]));
        assert_eq!(w[1].0, 1.5);
        assert_eq!(w[2].0, 3.5);
    }

    #[test]
    fn edges_follow_gap_separated_segments() {
        let t = [0.0, 1.0, 2.0, 10.0, 10.2];
        assert_eq!(segment_edges(&t, 1.5), vec![0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn biweight_removes_a_smooth_trend() {
        let t: Vec<f64> = (0..2000).map(|i| i as f64 * 0.01).collect();
        let trend: Vec<f64> = t.iter().map(|x| 0.01 * (x / 5.0).sin()).collect();
        let flat = biweight_flatten(&t, &trend, 0.5, 0.5, 5.0, 10);
        let worst = flat[100..1900]
            .iter()
            .cloned()
            .fold(0.0, |m: f64, x| m.max(x.abs()));
        assert!(worst < 5e-5, "residual trend {worst}");
    }

    #[test]
    fn noise_model_normalizes_and_inflates() {
        let n = 500;
        let t: Vec<f64> = (0..n).map(|i| 1000.0 + i as f64 * 0.01).collect();
        let f: Vec<f64> = (0..n)
            .map(|i| 2000.0 + 3.0 * (i as f64 * 1.7).sin())
            .collect();
        let e = vec![1.0; n];
        let m = fit_segment_noise(&t, &f, Some(&e), None, 0.0, 0.75).unwrap();
        assert_eq!(m.origin, 1000.0);
        assert!(m.time[0] == 0.0);
        let mut sorted = f.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[(n - 1) / 2];
        assert!(
            (m.deficit[0] - (1.0 - f[0] / median)).abs() < 1e-12,
            "deficit[0] = {}",
            m.deficit[0]
        );
        assert!(
            m.inflation > 2.0,
            "adjacent differences of several units must inflate the unit errors: {}",
            m.inflation
        );
        assert_eq!(m.segments, 1);
    }

    #[test]
    fn transit_mask_and_cycles() {
        let t: Vec<f64> = (0..1000).map(|i| i as f64 * 0.01).collect();
        let m = transit_mask(&t, 2.0, 1.0, 0.2, 0.0);
        let direct = t
            .iter()
            .filter(|&&x| ((x - 1.0 + 1.0).rem_euclid(2.0) - 1.0).abs() <= 0.1)
            .count();
        assert_eq!(m.iter().filter(|&&x| x).count(), direct);
        assert!((100..=110).contains(&direct));
    }
}
