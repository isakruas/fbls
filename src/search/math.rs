//! Scalar functions shared by host and device code.

const NEG_INF: f64 = f64::NEG_INFINITY;

/// Complementary error function (Cody's rational approximations, relative
/// error below 1e-16 in double precision).
#[inline(always)]
pub fn erfc(x: f64) -> f64 {
    const A: [f64; 5] = [3.16112374387056560e0, 1.13864154151050156e2, 3.77485237685302021e2, 3.20937758913846947e3, 1.85777706184603153e-1];
    const B: [f64; 4] = [2.36012909523441209e1, 2.44024637934444173e2, 1.28261652607737228e3, 2.84423683343917062e3];
    const C: [f64; 9] = [
        5.64188496988670089e-1,
        8.88314979438837594e0,
        6.61191906371416295e1,
        2.98635138197400131e2,
        8.81952221241769090e2,
        1.71204761263407058e3,
        2.05107837782607147e3,
        1.23033935479799725e3,
        2.15311535474403846e-8,
    ];
    const D: [f64; 8] = [
        1.57449261107098347e1,
        1.17693950891312499e2,
        5.37181101862009858e2,
        1.62138957456669019e3,
        3.29079923573345963e3,
        4.36261909014324716e3,
        3.43936767414372164e3,
        1.23033935480374942e3,
    ];
    const P: [f64; 6] = [3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1, 1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2];
    const Q: [f64; 5] = [2.56852019228982242e0, 1.87295284992346725e0, 5.27905102951428412e-1, 6.05183413124413191e-2, 2.33520497626869185e-3];
    const SQRPI: f64 = 5.6418958354775628695e-1;
    const THRESH: f64 = 0.46875;
    const XBIG: f64 = 26.543;
    let y = x.abs();
    let result;
    if y <= THRESH {
        let ysq = if y > 1.11e-16 { y * y } else { 0.0 };
        let mut xnum = A[4] * ysq;
        let mut xden = ysq;
        for i in 0..3 {
            xnum = (xnum + A[i]) * ysq;
            xden = (xden + B[i]) * ysq;
        }
        return 1.0 - x * (xnum + A[3]) / (xden + B[3]);
    } else if y <= 4.0 {
        let mut xnum = C[8] * y;
        let mut xden = y;
        for i in 0..7 {
            xnum = (xnum + C[i]) * y;
            xden = (xden + D[i]) * y;
        }
        let r = (xnum + C[7]) / (xden + D[7]);
        let ysq = (y * 16.0).trunc() / 16.0;
        let del = (y - ysq) * (y + ysq);
        result = (-ysq * ysq).exp() * (-del).exp() * r;
    } else if y < XBIG {
        let ysq = 1.0 / (y * y);
        let mut xnum = P[5] * ysq;
        let mut xden = ysq;
        for i in 0..4 {
            xnum = (xnum + P[i]) * ysq;
            xden = (xden + Q[i]) * ysq;
        }
        let r = ysq * (xnum + P[4]) / (xden + Q[4]);
        let r = (SQRPI - r) / y;
        let ysq = (y * 16.0).trunc() / 16.0;
        let del = (y - ysq) * (y + ysq);
        result = (-ysq * ysq).exp() * (-del).exp() * r;
    } else {
        result = 0.0;
    }
    if x < 0.0 { 2.0 - result } else { result }
}

#[inline(always)]
pub fn log_add(x: f64, y: f64) -> f64 {
    let high = x.max(y);
    if high == NEG_INF { high } else { high + (1.0 + (x.min(y) - high).exp()).ln() }
}

/// Log evidence against zero amplitude for a normalized half-normal depth
/// prior of scale `tau`, given numerator `a` and precision `q`.
#[inline(always)]
pub fn positive_amplitude_evidence(a: f64, q: f64, tau: f64) -> f64 {
    let q = q.max(0.0);
    let precision = 1.0 + tau * tau * q;
    let z = tau * a / precision.sqrt();
    if z < -26.0 {
        let zz = (z * z).max(1.0);
        let inverse = 1.0 / zz;
        let series = 1.0 + inverse * (-1.0 + inverse * (3.0 + inverse * (-15.0 + inverse * (105.0 + inverse * (-945.0 + inverse * (10395.0 - inverse * 135135.0))))));
        -0.5 * precision.ln() - (-z).max(1.0).ln() - 0.22579135264472743236 + series.ln()
    } else {
        -0.5 * precision.ln() + 0.5 * z * z + erfc(-z * 0.7071067811865475244).max(1e-300).ln()
    }
}

/// Upper bound of the log evidence of every positive-depth mixture of two
/// templates with numerators (a, b), precisions (q, r) and cross term.
#[inline(always)]
pub fn subspace_upper(a: f64, b: f64, q: f64, r: f64, cross: f64, rho2: f64) -> f64 {
    let product = q * r;
    let det = q * r - cross * cross;
    let energy = (a * a * r + b * b * q - 2.0 * a * b * cross) / (det * rho2).max(1e-300);
    let safe = q > 0.0 && r > 0.0 && det > 1e-8 * product;
    let (pa, pb) = (a.max(0.0), b.max(0.0));
    let boundary = 0.5 * (pa * pa / (q * rho2).max(1e-300)).max(pb * pb / (r * rho2).max(1e-300));
    let interior = a * r - b * cross > 0.0 && b * q - a * cross > 0.0;
    let mut upper = if safe { if interior { 0.5 * energy.max(0.0) } else { boundary } } else { f64::INFINITY };
    let only_a = r == 0.0 && b == 0.0 && cross == 0.0;
    let only_b = q == 0.0 && a == 0.0 && cross == 0.0;
    if only_a && q > 0.0 {
        upper = 0.5 * a * a / (q * rho2);
    }
    if only_b && r > 0.0 {
        upper = 0.5 * b * b / (r * rho2);
    }
    if only_a && only_b {
        upper = 0.0;
    }
    if a <= 0.0 && b <= 0.0 {
        upper = 0.0;
    }
    upper * 1.000001 + 1e-6
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erfc_reference_values() {
        let cases = [
            (0.0, 1.0),
            (0.5, 0.4795001221869535),
            (1.0, 0.15729920705028513),
            (2.0, 0.004677734981047266),
            (3.0, 2.209049699858544e-05),
            (5.0, 1.5374597944280347e-12),
            (10.0, 2.088487583762545e-45),
            (-1.0, 1.8427007929497148),
            (-3.0, 1.9999779095030014),
            (0.3, 0.6713732405408726),
            (0.46, 0.5153446099983203),
            (0.47, 0.5062549491139179),
            (3.99, 1.6739211364520824e-08),
            (4.01, 1.4196939421371057e-08),
            (15.0, 7.212994172451209e-100),
            (26.0, 5.663192408856145e-296),
        ];
        for (x, want) in cases {
            let got = erfc(x);
            assert!(((got - want) / want).abs() < 1e-14, "erfc({x}) = {got}, want {want}");
        }
    }
}
