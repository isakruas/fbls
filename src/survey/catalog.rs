//! Catalogued signals per star and the rules that match a detected signal to
//! them. Sources: `catalogs/toi.json` (TOI table) and `catalogs/known.json`
//! (CTOIs, published planets, eclipsing binaries, SPOC TCEs).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Known {
    pub source: String,
    pub name: String,
    pub toi: Option<String>,
    pub tic: i64,
    pub period: Option<f64>,
    pub epoch: Option<f64>,
    pub period_err: Option<f64>,
    pub epoch_err: Option<f64>,
    pub duration_hours: Option<f64>,
    pub disposition: Option<String>,
    pub transiting: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Match {
    pub source: String,
    pub name: String,
    pub toi: Option<String>,
    pub disposition: Option<String>,
    pub period_ratio: Option<f64>,
    pub phase_checked: bool,
}

pub fn wrap(phase: f64, period: f64) -> f64 {
    (phase + period / 2.0).rem_euclid(period) - period / 2.0
}

fn positive(v: &Value) -> Option<f64> {
    let x = v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))?;
    if x.is_finite() && x > 0.0 { Some(x) } else { None }
}

pub struct Catalog {
    pub by_star: HashMap<i64, Vec<Known>>,
}

impl Catalog {
    pub fn load(catalogs: &Path) -> Result<Self, String> {
        let mut by_star: HashMap<i64, Vec<Known>> = HashMap::new();
        if let Ok(text) = std::fs::read_to_string(catalogs.join("toi.json")) {
            let rows: Vec<Value> = serde_json::from_str(&text).map_err(|e| e.to_string())?;
            for r in rows {
                let Some(tic) = r["tid"].as_i64() else { continue };
                let toi = r["toi"].as_str().map(|s| s.to_string()).or(r["toi"].as_f64().map(|f| format!("{f}")));
                by_star.entry(tic).or_default().push(Known {
                    source: "TOI".into(),
                    name: format!("TOI {}", toi.clone().unwrap_or_default()),
                    toi,
                    tic,
                    period: positive(&r["pl_orbper"]),
                    epoch: positive(&r["pl_tranmid"]),
                    period_err: positive(&r["pl_orbpererr1"]),
                    epoch_err: positive(&r["pl_tranmiderr1"]),
                    duration_hours: positive(&r["pl_trandurh"]),
                    disposition: r["tfopwg_disp"].as_str().map(|s| s.to_string()),
                    transiting: true,
                });
            }
        }
        if let Ok(text) = std::fs::read_to_string(catalogs.join("known.json")) {
            let rows: Vec<Value> = serde_json::from_str(&text).map_err(|e| e.to_string())?;
            for r in rows {
                let Some(tic) = r["tic"].as_i64() else { continue };
                by_star.entry(tic).or_default().push(Known {
                    source: r["source"].as_str().unwrap_or("").into(),
                    name: r["name"].as_str().unwrap_or("").into(),
                    toi: None,
                    tic,
                    period: positive(&r["period"]),
                    epoch: positive(&r["epoch"]),
                    period_err: positive(&r["period_err"]),
                    epoch_err: positive(&r["epoch_err"]),
                    duration_hours: positive(&r["duration_hours"]),
                    disposition: r["disposition"].as_str().map(|s| s.to_string()),
                    transiting: r["transiting"].as_bool().unwrap_or(false),
                });
            }
        }
        Ok(Self { by_star })
    }

    pub fn of(&self, tic: i64) -> &[Known] {
        self.by_star.get(&tic).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

const RATIOS: [f64; 5] = [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0];

/// A periodic signal matches a catalogued one at the same period or a
/// harmonic (1, 2, 1/2, 3, 1/3) and the same phase while the catalog
/// ephemeris is still predictive at our epoch; past a quarter period of
/// propagated 3-sigma drift the period alone decides. Single-transit entries
/// match when one of our transits falls on their event; non-transiting
/// entries use the period alone.
pub fn match_known(period: f64, epoch: f64, duration_hours: f64, known: &[Known]) -> Option<Match> {
    let duration = duration_hours / 24.0;
    for k in known {
        let d = (k.duration_hours.unwrap_or(0.0) / 24.0).max(duration);
        let mut found: Option<(Option<f64>, bool)> = None;
        match (k.period, k.epoch) {
            (None, Some(e)) => {
                if wrap(e - epoch, period).abs() < d + 3.0 * k.epoch_err.unwrap_or(0.0) {
                    found = Some((None, true));
                }
            }
            (Some(p), e) => {
                for ratio in RATIOS {
                    if (period / (p * ratio) - 1.0).abs() < 0.01 {
                        let cycles = e.map(|e| (epoch - e).abs() / p).unwrap_or(0.0);
                        let drift = 3.0 * (cycles * k.period_err.unwrap_or(0.0)).hypot(k.epoch_err.unwrap_or(0.0));
                        if e.is_none() || !k.transiting || drift > p.min(p * ratio) / 4.0 {
                            found = Some((Some(ratio), false));
                        } else if wrap(epoch - e.unwrap(), p.min(p * ratio)).abs() < d + drift {
                            found = Some((Some(ratio), true));
                        }
                        if found.is_some() {
                            break;
                        }
                    }
                }
            }
            _ => {}
        }
        if let Some((ratio, checked)) = found {
            return Some(Match { source: k.source.clone(), name: k.name.clone(), toi: k.toi.clone(), disposition: k.disposition.clone(), period_ratio: ratio, phase_checked: checked });
        }
    }
    None
}

/// An isolated event matches a predicted transit of a periodic entry (within
/// the propagated 3-sigma drift) or the event time of a single-transit entry.
pub fn match_single(epoch: f64, duration_hours: f64, known: &[Known]) -> Option<Match> {
    let duration = duration_hours / 24.0;
    for k in known {
        let Some(e) = k.epoch else { continue };
        if !k.transiting {
            continue;
        }
        let d = (k.duration_hours.unwrap_or(0.0) / 24.0).max(duration);
        let hit = match k.period {
            None => (epoch - e).abs() < d + 3.0 * k.epoch_err.unwrap_or(0.0),
            Some(p) => {
                let drift = 3.0 * ((epoch - e).abs() / p * k.period_err.unwrap_or(0.0)).hypot(k.epoch_err.unwrap_or(0.0));
                drift < p / 4.0 && wrap(epoch - e, p).abs() < d + drift
            }
        };
        if hit {
            return Some(Match { source: k.source.clone(), name: k.name.clone(), toi: k.toi.clone(), disposition: k.disposition.clone(), period_ratio: None, phase_checked: true });
        }
    }
    None
}

pub fn same_period(period: f64, other: f64) -> bool {
    RATIOS.iter().any(|r| (period / (other * r) - 1.0).abs() < 0.01)
}

/// (period, epoch, duration_hours) of every catalogued transiting signal with
/// a full ephemeris, leaving out the entries at the period of the event or a
/// harmonic of it.
pub fn known_masks(known: &[Known], event_period: Option<f64>) -> Vec<(f64, f64, f64)> {
    known
        .iter()
        .filter(|k| k.transiting)
        .filter_map(|k| Some((k.period?, k.epoch?, k.duration_hours?)))
        .filter(|(p, _, _)| event_period.is_none_or(|ep| !same_period(ep, *p)))
        .collect()
}
