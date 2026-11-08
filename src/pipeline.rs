//! Multi-pass transit search: family preparation, per-period scan, peak
//! selection with masking, single-event scan and the JSON report.

use crate::config::SearchConfig;
use crate::prep::curve::{
    FamilyState, Observations, PreparedCurve, family_state, mask_curve, prepare_families,
};
use crate::search::events::single_event_search;
use crate::search::evidence::compare_search;
use crate::search::scan::{Gpu, Row, signal_search};
use crate::select::peaks::{Peak, select_events, select_peaks};
use serde_json::{Value, json};

const CANDIDATE_CAPACITY: usize = 131072;

fn merged_time(families: &[FamilyState]) -> Vec<f64> {
    let mut t: Vec<f64> = families
        .iter()
        .flat_map(|s| s.curve.time.iter().cloned())
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t.dedup();
    t
}

/// Raw mission flux: normalize, bin and detrend per duration family, then search.
pub fn search_transits(
    obs: &Observations,
    cfg: &SearchConfig,
    device: usize,
) -> Result<Value, Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    let durations = cfg.durations_days();
    let gpu = Gpu::shared(device)?;
    let families = prepare_families(&gpu, obs, &durations, &crate::prep_config(cfg))?;
    search_families(families, cfg, device, start)
}

/// Detrended fractional flux with baseline zero and per-observation errors,
/// searched as one family with every duration.
pub fn search_prepared(
    obs: &Observations,
    cfg: &SearchConfig,
    device: usize,
) -> Result<Value, Box<dyn std::error::Error>> {
    let err = obs.err.as_ref().ok_or("prepared input requires flux_err")?;
    let keep: Vec<usize> = (0..obs.time.len())
        .filter(|&i| {
            obs.time[i].is_finite() && obs.flux[i].is_finite() && err[i].is_finite() && err[i] > 0.0
        })
        .collect();
    let mut order = keep.clone();
    order.sort_by(|&a, &b| obs.time[a].partial_cmp(&obs.time[b]).unwrap());
    let curve = PreparedCurve {
        time: order.iter().map(|&i| obs.time[i]).collect(),
        flux: order.iter().map(|&i| obs.flux[i]).collect(),
        sigma: order.iter().map(|&i| err[i]).collect(),
        window_days: f64::NAN,
        clipped: 0,
    };
    let durations = cfg.durations_days();
    search_families(
        vec![(curve, (0..durations.len()).collect())],
        cfg,
        device,
        std::time::Instant::now(),
    )
}

fn search_families(
    families: Vec<(PreparedCurve, Vec<usize>)>,
    cfg: &SearchConfig,
    device: usize,
    start: std::time::Instant,
) -> Result<Value, Box<dyn std::error::Error>> {
    let durations = cfg.durations_days();
    let mut states: Vec<FamilyState> = families
        .into_iter()
        .map(|(curve, indices)| {
            let d: Vec<f64> = indices.iter().map(|&i| durations[i]).collect();
            family_state(curve, &d, &indices, cfg.correlated_noise)
        })
        .collect::<Result<_, _>>()?;
    let original: Vec<FamilyState> = states.clone();
    let preparation_seconds = start.elapsed().as_secs_f64();
    let time_all = merged_time(&states);
    if time_all.len() < 6 {
        return Err("at least six usable observations are required".into());
    }
    let baseline = time_all[time_all.len() - 1] - time_all[0];
    let gpu = Gpu::shared(device)?;
    let search_start = std::time::Instant::now();

    let mut peaks: Vec<Peak> = Vec::new();
    let mut masks: Vec<Vec<bool>> = Vec::new();
    let mut excluded = vec![false; time_all.len()];
    let mut pass_stats = Vec::new();
    let mut first_table: Vec<Row> = Vec::new();
    let mut periods_tested = 0usize;
    let mut seconds_scan = 0.0;
    let mut seconds_select = 0.0;
    for pass in 0..cfg.passes {
        let scan_start = std::time::Instant::now();
        let mut table: Vec<Row> = Vec::new();
        let mut family: Vec<usize> = Vec::new();
        let mut band_stats = Vec::new();
        for (k, state) in states.iter().enumerate() {
            let (rows, grid, stats) = signal_search(&gpu, state, cfg)?;
            if pass == 0 {
                periods_tested += grid.len();
            }
            band_stats.push(json!({
                "family": k, "window_days": state.curve.window_days, "observations": state.curve.time.len(),
                "clipped": state.curve.clipped, "error_inflation": state.noise.inflation, "segments": state.noise.segments,
                "durations_hours": state.durations.iter().map(|d| d * 24.0).collect::<Vec<_>>(),
                "periods": grid.len(), "rho": state.rho, "bands": stats,
            }));
            family.extend(std::iter::repeat_n(k, rows.len()));
            table.extend(rows);
        }
        if pass == 0 {
            first_table = table.clone();
        }
        seconds_scan += scan_start.elapsed().as_secs_f64();
        let select_start = std::time::Instant::now();
        let selection = select_peaks(
            &gpu,
            &table,
            &family,
            &states,
            &time_all,
            cfg.threshold,
            cfg.max_candidates - peaks.len(),
            cfg.min_events,
            &masks,
            &excluded,
            pass + 1,
        )?;
        masks = selection.masks;
        excluded = selection.excluded;
        seconds_select += select_start.elapsed().as_secs_f64();
        pass_stats.push(json!({"pass": pass + 1, "rows": table.len(), "accepted": selection.accepted.len(), "families": band_stats}));
        if selection.accepted.is_empty() {
            break;
        }
        let lead = selection.accepted[0].clone();
        peaks.extend(selection.accepted);
        if pass + 1 < cfg.passes {
            states = states
                .into_iter()
                .map(|s| {
                    let curve = mask_curve(
                        &s.curve,
                        lead.period_days,
                        lead.epoch_bjd,
                        lead.duration_hours / 24.0,
                        0.5,
                    );
                    family_state(curve, &s.durations, &s.indices, cfg.correlated_noise)
                })
                .collect::<Result<_, _>>()?;
            if states.iter().any(|s| s.curve.time.len() < 6) {
                break;
            }
        }
    }

    let mut evidence_seconds = 0.0;
    if cfg.model_comparison && !peaks.is_empty() {
        let evidence_start = std::time::Instant::now();
        let mut order: Vec<usize> = (0..peaks.len()).collect();
        order.sort_by(|&a, &b| peaks[b].snr.partial_cmp(&peaks[a].snr).unwrap());
        for &index in order.iter().take(cfg.evidence_candidates) {
            let peak = peaks[index].clone();
            let base = &original[peak.family];
            let mut curve = base.curve.clone();
            for other in &peaks {
                if other.row != peak.row || other.pass != peak.pass || other.family != peak.family {
                    curve = mask_curve(
                        &curve,
                        other.period_days,
                        other.epoch_bjd,
                        other.duration_hours / 24.0,
                        0.5,
                    );
                }
            }
            if curve.time.len() < 6 {
                continue;
            }
            let state = family_state(curve, &base.durations, &base.indices, cfg.correlated_noise)?;
            let rows = compare_search(&gpu, &state, peak.period_days, cfg)?;
            let row = rows
                .iter()
                .find(|r| r.selection_flags & 1 != 0)
                .or(rows.first());
            let Some(row) = row else { continue };
            let finite = |x: f64| if x.is_finite() { Some(x) } else { None };
            let p = &mut peaks[index];
            p.planet_log_odds = finite(row.planet_log_odds);
            p.log_bf_planet_noise = finite(row.log_bf_planet_noise);
            p.log_bf_binary_noise = finite(row.log_bf_binary_noise);
            p.binary_secondary_ratio = finite(row.binary_secondary_ratio);
            p.binary_secondary_phase = finite(row.binary_secondary_phase);
            p.binary_odd_even_ratio = finite(row.binary_odd_even_ratio);
            p.binary_period_days = finite(row.binary_period_days);
            p.ingress_fraction = finite(row.ingress_fraction);
            p.evidence_epoch_bjd = Some(row.epoch);
            p.evidence_duration_hours = Some(24.0 * row.duration);
            p.model_preferred = row.planet_log_odds >= cfg.planet_log_odds_min;
            p.binary_diagnostics_complete = [
                row.binary_secondary_ratio,
                row.binary_secondary_phase,
                row.binary_odd_even_ratio,
                row.binary_period_days,
            ]
            .iter()
            .all(|x| x.is_finite());
        }
        evidence_seconds = evidence_start.elapsed().as_secs_f64();
    }

    let mut events = Vec::new();
    let mut event_stats = serde_json::Map::new();
    let events_start = std::time::Instant::now();
    if cfg.single_events {
        let mut all: Vec<Row> = Vec::new();
        for (k, state) in states.iter().enumerate() {
            let (table, stats) = single_event_search(&gpu, state, cfg, CANDIDATE_CAPACITY)?;
            event_stats.insert(k.to_string(), json!(stats));
            let selected =
                select_events(&table, &time_all, cfg.threshold, cfg.max_candidates, &masks);
            all.extend(selected.iter().map(|e| Row {
                period: f64::INFINITY,
                epoch: e.epoch_bjd,
                duration: e.duration_hours / 24.0,
                score: e.snr,
                depth: e.depth,
            }));
        }
        all.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        events = select_events(&all, &time_all, cfg.threshold, cfg.max_candidates, &masks);
    }

    let seconds_events = events_start.elapsed().as_secs_f64();
    let mut by_signal = peaks.clone();
    by_signal.sort_by(|a, b| b.snr.partial_cmp(&a.snr).unwrap());
    let mut by_model = peaks.clone();
    by_model.sort_by(|a, b| {
        let key = |p: &Peak| {
            let evidence = if p.model_preferred {
                p.log_bf_planet_noise
            } else {
                None
            };
            (
                if p.model_preferred { 0 } else { 1 },
                -(evidence.or(p.planet_log_odds).unwrap_or(f64::NEG_INFINITY)),
            )
        };
        let (ka, kb) = (key(a), key(b));
        ka.0.cmp(&kb.0)
            .then(ka.1.partial_cmp(&kb.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    if !cfg.model_comparison {
        by_model = by_signal.clone();
    }
    let search_seconds = search_start.elapsed().as_secs_f64();
    Ok(json!({
        "schema_version": 3,
        "software_version": env!("CARGO_PKG_VERSION"),
        "config": cfg,
        "statistics": {
            "device": format!("cuda:{device}"), "observations": time_all.len(), "baseline_days": baseline,
            "periodic_period_max_days": cfg.period_max.min(baseline),
            "periods_tested": periods_tested,
            "qualifying_periods": first_table.len(),
            "resolved_peaks": peaks.len(), "single_events": events.len(),
            "seconds_preparation": preparation_seconds, "seconds_search": search_seconds,
            "evidence": {"seconds_search": evidence_seconds},
            "seconds_scan": seconds_scan, "seconds_select": seconds_select, "seconds_single_events": seconds_events,
            "seconds_total": preparation_seconds + search_seconds,
            "passes": pass_stats, "single_event_scan": event_stats,
        },
        "candidates": by_model.iter().take(cfg.max_candidates).collect::<Vec<_>>(),
        "signal_candidates": by_signal.iter().take(cfg.max_candidates).collect::<Vec<_>>(),
        "single_events": events.iter().take(cfg.max_candidates).collect::<Vec<_>>(),
        "period_table": first_table,
    }))
}
