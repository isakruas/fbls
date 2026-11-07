//! Period grid with a uniform accumulated-phase-drift bound.

/// Logarithmic grid such that baseline * (P_next / P - 1) <= duration / oversample.
/// The first and last values are exactly `minimum` and `maximum`.
pub fn period_grid(minimum: f64, maximum: f64, baseline: f64, minimum_duration: f64, oversample: f64) -> Vec<f64> {
    let log_step = (minimum_duration / (oversample * baseline)).ln_1p();
    let count = ((maximum / minimum).ln() / log_step).ceil() as usize + 1;
    let (a, b) = (minimum.ln(), maximum.ln());
    let mut grid: Vec<f64> = (0..count)
        .map(|i| if count == 1 { a } else { a + (b - a) * (i as f64) / ((count - 1) as f64) })
        .map(f64::exp)
        .collect();
    grid[0] = minimum;
    *grid.last_mut().unwrap() = maximum;
    grid
}

#[cfg(test)]
mod tests {
    use super::period_grid;

    #[test]
    fn grid_bounds_and_drift() {
        let g = period_grid(0.5, 20.0, 50.0, 0.02, 3.0);
        assert_eq!(g[0], 0.5);
        assert_eq!(*g.last().unwrap(), 20.0);
        for w in g.windows(2) {
            assert!(50.0 * (w[1] / w[0] - 1.0) <= 0.02 / 3.0 * (1.0 + 1e-9));
        }
    }
}
