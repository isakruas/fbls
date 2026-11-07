//! Robust host-side statistics shared by the preparation stages.

/// Median with the lower-element convention of `torch.median` for even counts.
pub fn median_lower(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[(v.len() - 1) / 2]
}

/// 1.4826 * median(|x - median(x)|), lower-median convention.
pub fn mad_scale(values: &[f64]) -> f64 {
    let m = median_lower(values);
    let dev: Vec<f64> = values.iter().map(|x| (x - m).abs()).collect();
    1.4826 * median_lower(&dev)
}

pub fn argsort(values: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

/// First index with v[i] >= x in a sorted slice.
pub fn lower_bound(v: &[f64], x: f64) -> usize {
    v.partition_point(|&a| a < x)
}

/// First index with v[i] > x in a sorted slice.
pub fn upper_bound(v: &[f64], x: f64) -> usize {
    v.partition_point(|&a| a <= x)
}
