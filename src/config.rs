//! Search configuration. Times are days unless the field name says hours.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SearchConfig {
    pub period_min: f64,
    pub period_max: f64,
    pub durations_hours: Vec<f64>,
    pub threshold: f64,
    pub bin_minutes: f64,
    pub window_days: f64,
    pub window_factor: f64,
    pub window_max_days: f64,
    pub edge_trim_windows: f64,
    pub clip_sigma: f64,
    pub correlated_noise: bool,
    pub ingress_fractions: Vec<f64>,
    pub shape_layers: usize,
    pub phase_oversample: usize,
    pub shape_gate_fraction: f64,
    pub prescan_factor: usize,
    pub prescan_fraction: f64,
    pub period_oversample: f64,
    pub max_candidates: usize,
    pub period_batch: usize,
    pub max_fold_mib: f64,
    pub passes: usize,
    pub min_events: usize,
    pub single_events: bool,
    pub model_comparison: bool,
    pub evidence_candidates: usize,
    pub evidence_rtol: f64,
    pub planet_log_odds_min: f64,
    pub planet_prior: f64,
    pub binary_prior: f64,
    pub planet_depth_scale: f64,
    pub binary_depth_scale: f64,
    pub diluted_binary_depth_scale: f64,
    pub stellar_radius_solar: Option<f64>,
    pub planet_radius_scale_rj: f64,
    pub target_flux_fraction: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            period_min: 0.3,
            period_max: 10.0,
            durations_hours: (0..13).map(|i| (0.5 * 1.25f64.powi(i) * 1e6).round_ties_even() / 1e6).collect(),
            threshold: 7.1,
            bin_minutes: 10.0,
            window_days: 0.4,
            window_factor: 3.0,
            window_max_days: 3.5,
            edge_trim_windows: 0.5,
            clip_sigma: 4.0,
            correlated_noise: true,
            ingress_fractions: vec![0.0, 0.2, 0.45],
            shape_layers: 4,
            phase_oversample: 4,
            shape_gate_fraction: 0.6,
            prescan_factor: 2,
            prescan_fraction: 0.7,
            period_oversample: 3.0,
            max_candidates: 20,
            period_batch: 4096,
            max_fold_mib: 1024.0,
            passes: 3,
            min_events: 2,
            single_events: true,
            model_comparison: true,
            evidence_candidates: 32,
            evidence_rtol: 1e-12,
            planet_log_odds_min: 0.0,
            planet_prior: 0.01,
            binary_prior: 0.01,
            planet_depth_scale: 0.01,
            binary_depth_scale: 0.1,
            diluted_binary_depth_scale: 0.01,
            stellar_radius_solar: None,
            planet_radius_scale_rj: 1.0,
            target_flux_fraction: 1.0,
        }
    }
}

impl SearchConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0 < self.period_min && self.period_min < self.period_max) {
            return Err("require 0 < period_min < period_max".into());
        }
        if self.durations_hours.is_empty() || self.durations_hours.iter().any(|d| !(d.is_finite() && *d > 0.0)) {
            return Err("durations_hours must contain positive finite values".into());
        }
        let dmin = self.durations_hours.iter().cloned().fold(f64::INFINITY, f64::min);
        if dmin / 24.0 >= self.period_min / 4.0 {
            return Err("minimum duration must be below one quarter of period_min".into());
        }
        if self.ingress_fractions.is_empty() || self.ingress_fractions.iter().any(|x| !(0.0..0.5).contains(x)) {
            return Err("ingress fractions must be in [0, 0.5)".into());
        }
        if self.shape_layers < 1 || self.phase_oversample < 1 || self.period_batch < 1 || self.passes < 1 || self.prescan_factor < 1 {
            return Err("invalid integer search parameter".into());
        }
        if !(0.0 < self.prescan_fraction && self.prescan_fraction <= 1.0) {
            return Err("prescan_fraction must be in (0, 1]".into());
        }
        Ok(())
    }

    /// Sorted unique durations in days.
    pub fn durations_days(&self) -> Vec<f64> {
        let mut d: Vec<f64> = self.durations_hours.iter().map(|h| h / 24.0).collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        d.dedup();
        d
    }
}
