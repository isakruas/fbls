//! Numerical helpers of the vetting stage: robust statistics, least squares,
//! normal and chi-square tails, a deterministic generator.

pub fn median(v: &[f64]) -> f64 {
    let mut x: Vec<f64> = v.iter().cloned().filter(|a| a.is_finite()).collect();
    if x.is_empty() {
        return f64::NAN;
    }
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = x.len();
    if n % 2 == 1 { x[n / 2] } else { 0.5 * (x[n / 2 - 1] + x[n / 2]) }
}

pub fn mad(v: &[f64]) -> f64 {
    let m = median(v);
    median(&v.iter().map(|x| (x - m).abs()).collect::<Vec<_>>())
}

pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
}

pub fn std(v: &[f64], ddof: usize) -> f64 {
    if v.len() <= ddof {
        return f64::NAN;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - ddof) as f64).sqrt()
}

pub fn percentile(v: &[f64], q: f64) -> f64 {
    let mut x: Vec<f64> = v.iter().cloned().filter(|a| a.is_finite()).collect();
    if x.is_empty() {
        return f64::NAN;
    }
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = q / 100.0 * (x.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    x[lo] + (x[hi] - x[lo]) * (pos - lo as f64)
}

/// Inverse of the standard normal survival function (Acklam's rational
/// approximation refined by one Halley step), for 0 < p < 1.
pub fn norm_isf(p: f64) -> f64 {
    -norm_ppf(p)
}

pub fn norm_ppf(p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) {
        return if p <= 0.0 { f64::NEG_INFINITY } else if p >= 1.0 { f64::INFINITY } else { f64::NAN };
    }
    const A: [f64; 6] = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const B: [f64; 5] = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const C: [f64; 6] = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const D: [f64; 4] = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    let x = if p < 0.02425 {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p > 1.0 - 0.02425 {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    };
    // Halley refinement with the complementary error function.
    let e = 0.5 * crate::search::math::erfc(-x / std::f64::consts::SQRT_2) - p;
    let u = e * (2.0 * std::f64::consts::PI).sqrt() * (x * x / 2.0).exp();
    x - u / (1.0 + x * u / 2.0)
}

/// Inverse complementary error function.
pub fn erfcinv(p: f64) -> f64 {
    -norm_ppf(p / 2.0) / std::f64::consts::SQRT_2
}

/// Survival function of chi-square with two degrees of freedom.
pub fn chi2_sf_2(x: f64) -> f64 {
    (-x / 2.0).exp()
}

/// Gaussian significance of a chi-square value with two degrees of freedom,
/// two-sided, capped at 40 sigma.
pub fn chi2_sigma_2(chi2: f64) -> f64 {
    norm_isf(chi2_sf_2(chi2).max(1e-300) / 2.0).min(40.0)
}

/// Solves the normal equations of a small dense least-squares problem
/// (columns of `design`, length n each) and returns (coefficients, covariance
/// scaled by the residual variance, residuals).
pub fn lstsq(design: &[Vec<f64>], y: &[f64]) -> Option<(Vec<f64>, Vec<Vec<f64>>, Vec<f64>)> {
    let k = design.len();
    let n = y.len();
    let mut ata = vec![vec![0.0; k]; k];
    let mut aty = vec![0.0; k];
    for i in 0..k {
        for j in 0..k {
            ata[i][j] = (0..n).map(|r| design[i][r] * design[j][r]).sum();
        }
        aty[i] = (0..n).map(|r| design[i][r] * y[r]).sum();
    }
    let inv = invert(&ata)?;
    let coef: Vec<f64> = (0..k).map(|i| (0..k).map(|j| inv[i][j] * aty[j]).sum()).collect();
    let resid: Vec<f64> = (0..n).map(|r| y[r] - (0..k).map(|i| coef[i] * design[i][r]).sum::<f64>()).collect();
    let dof = n.saturating_sub(k).max(1) as f64;
    let s2 = resid.iter().map(|r| r * r).sum::<f64>() / dof;
    let cov = inv.iter().map(|row| row.iter().map(|v| v * s2).collect()).collect();
    Some((coef, cov, resid))
}

/// Gauss-Jordan inverse of a small matrix.
pub fn invert(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    let mut a: Vec<Vec<f64>> = m.iter().enumerate().map(|(i, row)| {
        let mut r = row.clone();
        r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
        r
    }).collect();
    if a.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    for c in 0..n {
        let pivot = (c..n).max_by(|&i, &j| a[i][c].abs().partial_cmp(&a[j][c].abs()).unwrap())?;
        if a[pivot][c].abs() < 1e-300 {
            return None;
        }
        a.swap(c, pivot);
        let p = a[c][c];
        for v in a[c].iter_mut() {
            *v /= p;
        }
        for r in 0..n {
            if r != c {
                let f = a[r][c];
                if f != 0.0 {
                    for j in 0..2 * n {
                        a[r][j] -= f * a[c][j];
                    }
                }
            }
        }
    }
    Some(a.into_iter().map(|row| row[n..].to_vec()).collect())
}

/// Bounded Levenberg-Marquardt least squares with a forward-difference
/// Jacobian. `residuals(p)` returns the residual vector; parameters are kept
/// inside [lower, upper]. Returns (parameters, residuals, jacobian).
pub fn least_squares<F: Fn(&[f64]) -> Vec<f64>>(residuals: F, start: &[f64], lower: &[f64], upper: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let k = start.len();
    let clamp = |p: &mut Vec<f64>| {
        for i in 0..k {
            p[i] = p[i].clamp(lower[i], upper[i]);
        }
    };
    let mut p = start.to_vec();
    clamp(&mut p);
    let mut r = residuals(&p);
    let mut cost: f64 = r.iter().map(|x| x * x).sum();
    let mut lambda = 1e-3;
    let jacobian = |p: &[f64], r0: &[f64]| -> Vec<Vec<f64>> {
        (0..k)
            .map(|i| {
                let span = upper[i] - lower[i];
                let scale = if span.is_finite() { 1e-3 * span.abs() } else { 1e-6 };
                let h = 1e-6 * p[i].abs().max(scale).max(1e-9);
                let mut q = p.to_vec();
                q[i] = (q[i] + h).min(upper[i]);
                let step = q[i] - p[i];
                if step == 0.0 {
                    return vec![0.0; r0.len()];
                }
                let r1 = residuals(&q);
                r1.iter().zip(r0).map(|(a, b)| (a - b) / step).collect()
            })
            .collect()
    };
    let mut jac = jacobian(&p, &r);
    for _ in 0..200 {
        let n = r.len();
        let mut jtj = vec![vec![0.0; k]; k];
        let mut jtr = vec![0.0; k];
        for i in 0..k {
            for j in 0..k {
                jtj[i][j] = (0..n).map(|t| jac[i][t] * jac[j][t]).sum();
            }
            jtr[i] = (0..n).map(|t| jac[i][t] * r[t]).sum();
        }
        let mut improved = false;
        for _ in 0..20 {
            let mut m = jtj.clone();
            for i in 0..k {
                m[i][i] += lambda * (jtj[i][i].max(1e-12));
            }
            let Some(inv) = invert(&m) else {
                lambda *= 10.0;
                continue;
            };
            let mut q = p.clone();
            for i in 0..k {
                q[i] -= (0..k).map(|j| inv[i][j] * jtr[j]).sum::<f64>();
            }
            clamp(&mut q);
            let rq = residuals(&q);
            let cq: f64 = rq.iter().map(|x| x * x).sum();
            if cq.is_finite() && cq < cost {
                let change = (cost - cq) / cost.max(1e-300);
                p = q;
                r = rq;
                cost = cq;
                lambda = (lambda / 10.0).max(1e-12);
                improved = true;
                if change < 1e-12 {
                    jac = jacobian(&p, &r);
                    return (p, r, jac);
                }
                break;
            }
            lambda *= 10.0;
            if lambda > 1e12 {
                break;
            }
        }
        if !improved {
            break;
        }
        jac = jacobian(&p, &r);
    }
    (p, r, jac)
}

/// Parameter standard errors from the Jacobian with unit weights (no
/// residual scaling).
pub fn fit_errors_unit(jac: &[Vec<f64>]) -> Vec<f64> {
    let k = jac.len();
    let n = jac.first().map_or(0, |c| c.len());
    let mut jtj = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            jtj[i][j] = (0..n).map(|t| jac[i][t] * jac[j][t]).sum();
        }
    }
    match invert(&jtj) {
        Some(inv) => (0..k).map(|i| if inv[i][i] > 0.0 { inv[i][i].sqrt() } else { f64::NAN }).collect(),
        None => vec![f64::NAN; k],
    }
}

/// Parameter standard errors from the Jacobian, scaled by the residual variance.
pub fn fit_errors(residuals: &[f64], jac: &[Vec<f64>]) -> Vec<f64> {
    let k = jac.len();
    let n = residuals.len();
    let mut jtj = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            jtj[i][j] = (0..n).map(|t| jac[i][t] * jac[j][t]).sum();
        }
    }
    let scale = residuals.iter().map(|x| x * x).sum::<f64>() / n.saturating_sub(k).max(1) as f64;
    match invert(&jtj) {
        Some(inv) => (0..k).map(|i| if inv[i][i] > 0.0 { (inv[i][i] * scale).sqrt() } else { f64::NAN }).collect(),
        None => vec![f64::NAN; k],
    }
}

/// Deterministic xorshift generator with normal deviates (Box-Muller).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    pub fn uniform(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn normal(&mut self, mean: f64, sigma: f64) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        mean + sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_tails() {
        assert!((norm_isf(0.05) - 1.6448536269514722).abs() < 1e-9);
        assert!((norm_isf(1e-6) - 4.753424308822899).abs() < 1e-7);
        assert!((erfcinv(0.5) - 0.4769362762044699).abs() < 1e-9);
    }

    #[test]
    fn levenberg_marquardt_fits_a_line() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|v| 3.0 * v - 1.0).collect();
        let (p, r, _) = least_squares(|p| x.iter().zip(&y).map(|(a, b)| p[0] * a + p[1] - b).collect(), &[0.0, 0.0], &[-10.0, -10.0], &[10.0, 10.0]);
        assert!((p[0] - 3.0).abs() < 1e-6 && (p[1] + 1.0).abs() < 1e-6, "{p:?}");
        assert!(r.iter().all(|v| v.abs() < 1e-5));
    }
}
