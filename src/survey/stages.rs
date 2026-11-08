//! Survey stages, each consuming the previous stage's files under
//! `<root>/survey/` and skipping work already done:
//!
//!   search   detector on the search window of sectors: peaks, single events
//!            and the detrended families the peaks use      -> survey/search/<tic>.json, <tic>.residuals.json
//!   vet      catalog cross-match, duplicates, vetting        -> survey/vetting/<tic>.json
//!   pixel    difference-image test of passing signals        -> survey/pixel/<tic>.json
//!   publish  tiers, candidate table and CTOI rows            -> public/candidates.csv, public/ctoi.csv,
//!                                                              public/<day>/ with the stars first published that day
//!   refine   every sector of each CTOI star: fetch, search,
//!            vet, pixel, panel data                          -> public/<day>/refine/<tic>/
//!
//! A stage never overwrites an existing output, except the vetting when
//! VETTING_VERSION changed. Catalog information is used only after the search.

use super::catalog::{Catalog, known_masks, match_known, match_single, same_period};
use super::fetch::{fetch_star, today_utc};
use super::pixel::{apply_pixel_test, pixel_test, sector_images};
use super::publish::{
    CANDIDATE_COLUMNS, CTOI_COLUMNS, candidate_row, ctoi_row, rank, tier, write_csv,
};
use super::star::{SEARCH_SECTORS, Star, load_star};
use super::vet::{Folded, VETTING_VERSION, fold, measure};
use crate::config::SearchConfig;
use crate::prep::curve::{Observations, PrepConfig, prepare_families};
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};

pub const PERIOD_MAX: f64 = 100.0;

pub struct Survey {
    pub root: PathBuf,
    pub public: PathBuf,
    pub catalog: Catalog,
    pub author: String,
    pub device: usize,
    pub pmax: f64,
    /// Stars processed concurrently: GPU searches share the device in `search`,
    /// one star per CPU thread in `vet`.
    pub workers: usize,
}

/// Runs `job` over `items` with `workers` threads pulling from a shared index;
/// `job` receives (position, item) and returns the progress line.
fn parallel<T: Sync, F: Fn(usize, &T) -> String + Sync>(items: &[T], workers: usize, job: F) {
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..workers.max(1).min(items.len().max(1)) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(item) = items.get(i) else { break };
                    eprintln!("[{}/{}] {}", i + 1, items.len(), job(i, item));
                }
            });
        }
    });
}

fn read_json(path: &Path) -> Option<Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
}

fn write_json(path: &Path, mut value: Value) -> Result<(), String> {
    std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
    if let Some(o) = value.as_object_mut() {
        o.insert("processed_utc".into(), json!(super::fetch::now_utc()));
    }
    std::fs::write(path, serde_json::to_string_pretty(&value).unwrap() + "\n")
        .map_err(|e| e.to_string())
}

/// Survey configuration: 19 durations up to 27.7 h for periods up to 100 days.
pub fn survey_config(pmax: f64) -> SearchConfig {
    SearchConfig {
        period_max: pmax,
        durations_hours: (0..19)
            .map(|i| (0.5 * 1.25f64.powi(i) * 1e6).round_ties_even() / 1e6)
            .collect(),
        ..SearchConfig::default()
    }
}

fn observations(star: &Star) -> Observations {
    Observations {
        time: star.time_bjd.clone(),
        flux: star.flux.clone(),
        err: Some(star.flux_err.clone()),
        group: star.sector.clone(),
    }
}

/// Detrended residuals per family on the original cadences (no binning):
/// (shortest duration, longest duration, residual per observation, NaN where clipped).
pub fn detrended(star: &Star, cfg: &SearchConfig) -> Result<Vec<(f64, f64, Vec<f64>)>, String> {
    let durations = cfg.durations_days();
    let prep = PrepConfig {
        bin_minutes: 0.0,
        window_min: cfg.window_days,
        window_factor: cfg.window_factor,
        window_max: cfg.window_max_days,
        clip_sigma: cfg.clip_sigma,
        edge_trim: cfg.edge_trim_windows,
    };
    let families = prepare_families(&observations(star), &durations, &prep)?;
    let mut out = Vec::new();
    for (curve, indices) in families {
        let mut corrected = vec![f64::NAN; star.time_bjd.len()];
        for (t, f) in curve.time.iter().zip(&curve.flux) {
            let pos = star
                .time_bjd
                .partition_point(|&x| x < *t)
                .min(star.time_bjd.len() - 1);
            corrected[pos] = *f;
        }
        let d: Vec<f64> = indices.iter().map(|&i| durations[i]).collect();
        out.push((
            d.iter().cloned().fold(f64::INFINITY, f64::min),
            d.iter().cloned().fold(0.0, f64::max),
            corrected,
        ));
    }
    Ok(out)
}

/// Family whose duration range is closest to the peak duration.
pub fn family_index(residuals: &[(f64, f64, Vec<f64>)], duration_hours: f64) -> usize {
    let d = duration_hours / 24.0;
    (0..residuals.len())
        .min_by(|&a, &b| {
            let fa = ((residuals[a].0 * residuals[a].1).sqrt() / d).ln().abs();
            let fb = ((residuals[b].0 * residuals[b].1).sqrt() / d).ln().abs();
            fa.partial_cmp(&fb).unwrap()
        })
        .unwrap()
}

/// Index of a stronger peak whose period is this one's period or a harmonic
/// within 1%: the same system seen again, not a new candidate.
fn related_peak(period: f64, stronger: &[Value]) -> Option<usize> {
    stronger
        .iter()
        .position(|o| same_period(period, o["period_days"].as_f64().unwrap_or(0.0)))
}

fn normalised_flux(star: &Star) -> Vec<f64> {
    let mut flux = star.flux.clone();
    let mut sectors = star.sector.clone();
    sectors.sort();
    sectors.dedup();
    for s in sectors {
        let idx: Vec<usize> = (0..flux.len()).filter(|&i| star.sector[i] == s).collect();
        let med = super::numeric::median(&idx.iter().map(|&i| flux[i]).collect::<Vec<_>>());
        for i in idx {
            flux[i] /= med;
        }
    }
    flux
}

impl Survey {
    pub fn search_path(&self, tic: i64) -> PathBuf {
        self.root.join("survey/search").join(format!("{tic}.json"))
    }
    pub fn residuals_path(&self, tic: i64) -> PathBuf {
        self.root
            .join("survey/search")
            .join(format!("{tic}.residuals.json"))
    }
    pub fn vet_path(&self, tic: i64) -> PathBuf {
        self.root.join("survey/vetting").join(format!("{tic}.json"))
    }
    pub fn pixel_path(&self, tic: i64) -> PathBuf {
        self.root.join("survey/pixel").join(format!("{tic}.json"))
    }

    /// Search of one star: the report plus the detrended residuals of the
    /// families its peaks use.
    pub fn search_star(
        &self,
        star: &Star,
        cfg: &SearchConfig,
        residuals_file: &Path,
    ) -> Result<Value, String> {
        let report = crate::pipeline::search_transits(&observations(star), cfg, self.device)
            .map_err(|e| e.to_string())?;
        let peaks = report["signal_candidates"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if !peaks.is_empty() {
            let residuals = detrended(star, cfg)?;
            let mut used: Vec<usize> = peaks
                .iter()
                .map(|p| family_index(&residuals, p["duration_hours"].as_f64().unwrap()))
                .collect();
            used.sort();
            used.dedup();
            let families: Vec<Value> = used.iter().map(|&i| json!({"shortest_duration_days": residuals[i].0, "longest_duration_days": residuals[i].1, "corrected": residuals[i].2.iter().map(|v| if v.is_finite() { json!((v * 1e9).round() / 1e9) } else { Value::Null }).collect::<Vec<_>>()})).collect();
            write_json(
                residuals_file,
                json!({"tic": star.tic, "families": families}),
            )?;
        }
        let mut sectors = star.sector.clone();
        sectors.sort();
        sectors.dedup();
        Ok(json!({
            "tic": star.tic, "config": report["config"], "baseline_days": star.time_bjd.last().unwrap() - star.time_bjd[0],
            "sectors": sectors, "statistics": report["statistics"], "peaks": peaks, "single_events": report["single_events"],
        }))
    }

    fn load_residuals(path: &Path) -> Result<Vec<(f64, f64, Vec<f64>)>, String> {
        let v = read_json(path).ok_or(format!("missing residual file {}", path.display()))?;
        Ok(v["families"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|f| {
                (
                    f["shortest_duration_days"].as_f64().unwrap(),
                    f["longest_duration_days"].as_f64().unwrap(),
                    f["corrected"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap_or(f64::NAN))
                        .collect(),
                )
            })
            .collect())
    }

    fn folded(
        star: &Star,
        residuals: &[(f64, f64, Vec<f64>)],
        peak: &Value,
        masks: &[(f64, f64, f64)],
    ) -> Result<Folded, String> {
        let corrected =
            &residuals[family_index(residuals, peak["duration_hours"].as_f64().unwrap())].2;
        fold(
            &star.time_bjd,
            corrected,
            &star.sector,
            peak["period_days"].as_f64().unwrap(),
            peak["epoch_bjd"].as_f64().unwrap(),
            peak["duration_hours"].as_f64().unwrap(),
            masks,
        )
    }

    pub fn vet_star(
        &self,
        star: &Star,
        search: &Value,
        residuals_file: &Path,
    ) -> Result<Value, String> {
        let tic = star.tic;
        let peaks = search["peaks"].as_array().cloned().unwrap_or_default();
        let singles = search["single_events"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let mut events = Vec::new();
        if peaks.is_empty() && singles.is_empty() {
            return Ok(
                json!({"tic": tic, "baseline_days": search["baseline_days"], "vetting_version": VETTING_VERSION, "events": events}),
            );
        }
        let residuals = if peaks.is_empty() {
            Vec::new()
        } else {
            Self::load_residuals(residuals_file)?
        };
        let known = self.catalog.of(tic);
        let parents: Vec<Option<usize>> = (0..peaks.len())
            .map(|i| related_peak(peaks[i]["period_days"].as_f64().unwrap(), &peaks[..i]))
            .collect();
        let raw = normalised_flux(star);
        for (i, peak) in peaks.iter().enumerate() {
            let (p, e, d) = (
                peak["period_days"].as_f64().unwrap(),
                peak["epoch_bjd"].as_f64().unwrap(),
                peak["duration_hours"].as_f64().unwrap(),
            );
            let mut event = json!({"id": i, "tic": tic, "search": "periodic", "peak": peak, "known": match_known(p, e, d, known)});
            if let Some(j) = parents[i] {
                event["related"] =
                    json!({"period_days": peaks[j]["period_days"], "snr": peaks[j]["snr"]});
                event["tier"] = json!("duplicate: harmonic or secondary of a stronger signal");
                events.push(event);
                continue;
            }
            let mut masks: Vec<(f64, f64, f64)> = peaks
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i && parents[*j] != Some(i))
                .map(|(_, q)| {
                    (
                        q["period_days"].as_f64().unwrap(),
                        q["epoch_bjd"].as_f64().unwrap(),
                        q["duration_hours"].as_f64().unwrap(),
                    )
                })
                .collect();
            masks.extend(known_masks(known, Some(p)));
            event["masks"] = json!(masks);
            match Self::folded(star, &residuals, peak, &masks) {
                Ok(prep) => {
                    let m = measure(
                        &prep,
                        &star.star,
                        Some(&star.centr1),
                        Some(&star.centr2),
                        star.crowdsap,
                        &star.meta,
                        Some(&raw),
                    );
                    event["metrics"] = serde_json::to_value(m).unwrap();
                }
                Err(err) => event["error"] = json!(err),
            }
            events.push(event);
        }
        for (k, single) in singles.iter().enumerate() {
            let (e, d) = (
                single["epoch_bjd"].as_f64().unwrap(),
                single["duration_hours"].as_f64().unwrap(),
            );
            events.push(json!({"id": peaks.len() + k, "tic": tic, "search": "single", "peak": single, "known": match_single(e, d, known),
                "metrics": {"period_days": null, "epoch_bjd": e, "duration_hours": d, "depth_ppm": single["depth_ppm"], "depth_err_ppm": null,
                            "snr": single["snr"], "transits": 1, "flags": {}, "verdict": "pass", "star": star.star, "sectors": [], "geometry": null,
                            "odd_depth_ppm": null, "even_depth_ppm": null, "odd_even_sigma": null, "secondary_depth_ppm": null, "secondary_sigma": null,
                            "secondary_phase": null, "secondary_albedo": null, "secondary_threshold_sigma": null, "brightening_sigma": null,
                            "ingress_fraction": null, "fitted_duration_hours": null, "fitted_duration_err_hours": null, "radius_earth": null,
                            "a_over_rstar": null, "density_ratio": null, "centroid_sigma": null, "source_offset_arcsec": null, "crowdsap": star.crowdsap, "robovetter": null}}));
        }
        Ok(
            json!({"tic": tic, "baseline_days": search["baseline_days"], "vetting_version": VETTING_VERSION, "events": events}),
        )
    }

    /// A periodic, non-duplicate signal that passed vetting (and the pixel test when given).
    fn passing(event: &Value, pixel: Option<&Value>) -> bool {
        if event["search"] != "periodic"
            || event.get("related").is_some()
            || event.get("metrics").is_none()
        {
            return false;
        }
        if event["metrics"]["verdict"] != "pass" {
            return false;
        }
        if let Some(result) = pixel.and_then(|p| p["events"].get(event["id"].to_string()))
            && let Ok(r) = serde_json::from_value::<super::pixel::PixelResult>(result.clone())
        {
            let mut m = event["metrics"].clone();
            apply_pixel_test(&mut m, &r);
            return m["verdict"] == "pass";
        }
        true
    }

    pub fn pixel_star(&self, star: &Star, vetting: &Value) -> Value {
        let mut out = Map::new();
        for event in vetting["events"].as_array().cloned().unwrap_or_default() {
            if Self::passing(&event, None) {
                let p = &event["peak"];
                let result = pixel_test(
                    star,
                    p["period_days"].as_f64(),
                    p["epoch_bjd"].as_f64().unwrap(),
                    p["duration_hours"].as_f64().unwrap(),
                    &self.root,
                    None,
                );
                out.insert(
                    event["id"].to_string(),
                    serde_json::to_value(result).unwrap(),
                );
            }
        }
        json!({"tic": star.tic, "events": out})
    }

    /// Final events of one star: pixel result applied, tier set.
    pub fn assemble(vetting: Option<&Value>, pixel: Option<&Value>) -> Vec<Value> {
        let Some(vetting) = vetting else {
            return Vec::new();
        };
        let mut events = Vec::new();
        for mut event in vetting["events"].as_array().cloned().unwrap_or_default() {
            if event.get("metrics").is_none() {
                if event.get("tier").is_some() {
                    events.push(event);
                }
                continue;
            }
            if let Some(result) = pixel.and_then(|p| p["events"].get(event["id"].to_string()))
                && let Ok(r) = serde_json::from_value::<super::pixel::PixelResult>(result.clone())
            {
                let mut m = event["metrics"].clone();
                apply_pixel_test(&mut m, &r);
                event["metrics"] = m;
            }
            event["tier"] = json!(tier(&event));
            events.push(event);
        }
        events
    }

    fn stage_files(&self, tic: i64) -> (Option<Value>, Option<Value>, Option<Value>) {
        (
            read_json(&self.search_path(tic)),
            read_json(&self.vet_path(tic)),
            read_json(&self.pixel_path(tic)),
        )
    }

    pub fn run_search(&self, tics: &[i64]) -> Result<(), String> {
        let cfg = survey_config(self.pmax);
        let pending: Vec<i64> = tics
            .iter()
            .cloned()
            .filter(|&t| read_json(&self.search_path(t)).is_none())
            .collect();
        eprintln!(
            "search: {} stars, {} pending, {} workers",
            tics.len(),
            pending.len(),
            self.workers
        );
        parallel(&pending, self.workers, |_, &tic| {
            let started = std::time::Instant::now();
            let out = load_star(&self.root, tic, Some(SEARCH_SECTORS))
                .and_then(|star| self.search_star(&star, &cfg, &self.residuals_path(tic)));
            let mut value = out.unwrap_or_else(|e| json!({"tic": tic, "error": e}));
            value["seconds"] = json!(started.elapsed().as_secs_f64());
            let summary = format!(
                "{} peaks, {} single events",
                value["peaks"].as_array().map_or(0, |a| a.len()),
                value["single_events"].as_array().map_or(0, |a| a.len())
            );
            if let Err(e) = write_json(&self.search_path(tic), value.clone()) {
                return format!("TIC {tic} write failed: {e}");
            }
            format!(
                "TIC {tic} {:.0}s {}",
                started.elapsed().as_secs_f64(),
                value["error"].as_str().unwrap_or(&summary)
            )
        });
        Ok(())
    }

    pub fn run_vet(&self, tics: &[i64]) -> Result<(), String> {
        let pending: Vec<i64> = tics
            .iter()
            .cloned()
            .filter(|&t| {
                read_json(&self.search_path(t)).is_some_and(|s| s.get("peaks").is_some())
                    && !read_json(&self.vet_path(t))
                        .is_some_and(|v| v["vetting_version"] == json!(VETTING_VERSION))
            })
            .collect();
        eprintln!(
            "vet: {} stars, {} pending, {} workers",
            tics.len(),
            pending.len(),
            self.workers
        );
        parallel(&pending, self.workers, |_, &tic| {
            let started = std::time::Instant::now();
            let search = read_json(&self.search_path(tic)).unwrap_or_default();
            let mut value = load_star(&self.root, tic, Some(SEARCH_SECTORS))
                .and_then(|star| self.vet_star(&star, &search, &self.residuals_path(tic)))
                .unwrap_or_else(|e| json!({"tic": tic, "error": e, "vetting_version": VETTING_VERSION, "events": []}));
            value["seconds"] = json!(started.elapsed().as_secs_f64());
            if let Err(e) = write_json(&self.vet_path(tic), value.clone()) {
                return format!("TIC {tic} write failed: {e}");
            }
            let summary: Vec<String> = value["events"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .iter()
                .map(|e| {
                    format!(
                        "P={} {} {}",
                        e["peak"]["period_days"],
                        e["known"]["name"].as_str().unwrap_or("not catalogued"),
                        e["tier"]
                            .as_str()
                            .or(e["metrics"]["verdict"].as_str())
                            .or(e["error"].as_str())
                            .unwrap_or("")
                    )
                })
                .collect();
            format!(
                "TIC {tic} {:.0}s {}",
                started.elapsed().as_secs_f64(),
                if summary.is_empty() {
                    "no events".to_string()
                } else {
                    summary.join("; ")
                }
            )
        });
        Ok(())
    }

    pub fn run_pixel(&self, tics: &[i64]) -> Result<(), String> {
        let pending: Vec<i64> = tics
            .iter()
            .cloned()
            .filter(|&t| {
                read_json(&self.vet_path(t)).is_some() && read_json(&self.pixel_path(t)).is_none()
            })
            .collect();
        eprintln!("pixel: {} stars, {} pending", tics.len(), pending.len());
        for (n, tic) in pending.iter().enumerate() {
            let started = std::time::Instant::now();
            let vetting = read_json(&self.vet_path(*tic)).unwrap();
            let mut value = match load_star(&self.root, *tic, Some(SEARCH_SECTORS)) {
                Ok(star) => self.pixel_star(&star, &vetting),
                Err(e) => json!({"tic": tic, "error": e, "events": {}}),
            };
            value["seconds"] = json!(started.elapsed().as_secs_f64());
            write_json(&self.pixel_path(*tic), value.clone())?;
            eprintln!(
                "[{}/{}] TIC {tic} {:.0}s {} signals",
                n + 1,
                pending.len(),
                started.elapsed().as_secs_f64(),
                value["events"].as_object().map_or(0, |o| o.len())
            );
        }
        Ok(())
    }

    fn searched_stars(&self) -> Vec<i64> {
        let mut out: Vec<i64> = std::fs::read_dir(self.root.join("survey/search"))
            .map(|d| {
                d.flatten()
                    .filter_map(|e| {
                        e.file_name()
                            .to_string_lossy()
                            .strip_suffix(".json")
                            .and_then(|s| s.parse().ok())
                    })
                    .collect()
            })
            .unwrap_or_default();
        out.sort();
        out
    }

    /// Table rows of every vetted signal and CTOI rows of every new periodic
    /// candidate, per star.
    fn candidate_rows(
        &self,
        tics: &[i64],
    ) -> (
        Vec<(i64, Map<String, Value>)>,
        Vec<(i64, Value, Map<String, Value>)>,
    ) {
        let mut rows = Vec::new();
        let mut ctoi = Vec::new();
        for &tic in tics {
            let (_, vetting, pixel) = self.stage_files(tic);
            for e in Self::assemble(vetting.as_ref(), pixel.as_ref()) {
                if e.get("metrics").is_none() {
                    continue;
                }
                rows.push((tic, candidate_row(&e)));
                let t = e["tier"].as_str().unwrap_or("");
                if e["known"].is_null()
                    && e["search"] == "periodic"
                    && (t.starts_with("strong") || t.starts_with("candidate"))
                {
                    ctoi.push((tic, e.clone(), ctoi_row(&e, &self.author)));
                }
            }
        }
        (rows, ctoi)
    }

    fn published_before(&self, today: &str) -> Vec<i64> {
        let mut seen = Vec::new();
        if let Ok(days) = std::fs::read_dir(&self.public) {
            for day in days.flatten() {
                let name = day.file_name().to_string_lossy().into_owned();
                if name == today || name.len() != 10 {
                    continue;
                }
                if let Ok(text) = std::fs::read_to_string(day.path().join("candidates.csv")) {
                    for line in text.lines().skip(1) {
                        if let Some(t) = line.split(',').next().and_then(|s| s.parse().ok()) {
                            seen.push(t);
                        }
                    }
                }
            }
        }
        seen
    }

    /// Cumulative `public/candidates.csv` and `public/ctoi.csv` over every
    /// searched star, plus the folder of the day with the stars first
    /// published today and the panel data of their CTOI rows.
    pub fn run_publish(&self, tics: &[i64]) -> Result<(usize, usize, usize), String> {
        std::fs::create_dir_all(&self.public).map_err(|e| e.to_string())?;
        let today = today_utc();
        let mut all: Vec<i64> = tics.to_vec();
        all.extend(self.searched_stars());
        all.sort();
        all.dedup();
        let (mut rows, ctoi) = self.candidate_rows(&all);
        let order = |r: &Map<String, Value>| {
            (
                rank(r["tier"].as_str().unwrap_or("")),
                !r["known"].is_null(),
                -(r["detector_snr"].as_f64().unwrap_or(0.0) * 1e6) as i64,
            )
        };
        rows.sort_by_key(|(_, r)| order(r));
        let table: Vec<Map<String, Value>> = rows.iter().map(|(_, r)| r.clone()).collect();
        write_csv(
            &self.public.join("candidates.csv"),
            &CANDIDATE_COLUMNS,
            &table,
        )?;
        write_csv(
            &self.public.join("ctoi.csv"),
            &CTOI_COLUMNS,
            &ctoi.iter().map(|(_, _, r)| r.clone()).collect::<Vec<_>>(),
        )?;
        let seen = self.published_before(&today);
        let new: Vec<i64> = rows
            .iter()
            .map(|(t, _)| *t)
            .filter(|t| !seen.contains(t))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        if !new.is_empty() {
            let day = self.public.join(&today);
            std::fs::create_dir_all(&day).map_err(|e| e.to_string())?;
            write_csv(
                &day.join("candidates.csv"),
                &CANDIDATE_COLUMNS,
                &rows
                    .iter()
                    .filter(|(t, _)| new.contains(t))
                    .map(|(_, r)| r.clone())
                    .collect::<Vec<_>>(),
            )?;
            let new_ctoi: Vec<&(i64, Value, Map<String, Value>)> =
                ctoi.iter().filter(|(t, _, _)| new.contains(t)).collect();
            if !new_ctoi.is_empty() {
                write_csv(
                    &day.join("ctoi.csv"),
                    &CTOI_COLUMNS,
                    &new_ctoi
                        .iter()
                        .map(|(_, _, r)| r.clone())
                        .collect::<Vec<_>>(),
                )?;
                for (tic, e, _) in new_ctoi {
                    let folder = day.join("panels").join(format!("{tic}_{}", e["id"]));
                    if folder.exists() {
                        continue;
                    }
                    if let Ok(star) = load_star(&self.root, *tic, Some(SEARCH_SECTORS))
                        && let Ok(residuals) = Self::load_residuals(&self.residuals_path(*tic))
                    {
                        let masks: Vec<(f64, f64, f64)> = e["masks"]
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .map(|m| {
                                        (
                                            m[0].as_f64().unwrap(),
                                            m[1].as_f64().unwrap(),
                                            m[2].as_f64().unwrap(),
                                        )
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        if let Ok(prep) = Self::folded(&star, &residuals, &e["peak"], &masks) {
                            panel_data(&prep, e, &folder)?;
                        }
                        if e["metrics"]["pixel_test"]["sectors"]
                            .as_array()
                            .is_some_and(|a| a.iter().any(|d| d["used"] == true))
                        {
                            let p = &e["peak"];
                            pixel_panel_data(
                                &star,
                                p["period_days"].as_f64(),
                                p["epoch_bjd"].as_f64().unwrap(),
                                p["duration_hours"].as_f64().unwrap(),
                                &self.root,
                                &folder,
                            )
                            .ok();
                        }
                    }
                }
            }
        }
        Ok((rows.len(), ctoi.len(), new.len()))
    }

    /// Every sector of each CTOI star: fetch, search, vetting, pixel test and
    /// panel data on the full baseline, each step skipped when its file
    /// exists. Ends with `ctoi.json`: the refined CTOI row of each survey
    /// candidate that survives, or the reason it did not.
    pub fn run_refine(&self, tics: &[i64]) -> Result<(), String> {
        let (_, ctoi) = self.candidate_rows(tics);
        let mut stars: std::collections::BTreeMap<i64, Vec<Value>> =
            std::collections::BTreeMap::new();
        for (tic, e, _) in ctoi {
            stars.entry(tic).or_default().push(e["peak"].clone());
        }
        let today = today_utc();
        let cfg = survey_config(self.pmax);
        for (tic, peaks) in stars {
            let folder = self.refine_folder(tic, &today);
            if folder.join("ctoi.json").exists() {
                continue;
            }
            std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
            let started = std::time::Instant::now();
            if !folder.join("download.json").exists() {
                let mut log = Vec::new();
                let result = fetch_star(&self.root, tic, None, &mut |line| {
                    log.push(line.to_string())
                });
                write_json(
                    &folder.join("download.json"),
                    json!({"tic": tic, "ok": result.is_ok(), "error": result.as_ref().err(), "log": log}),
                )?;
            }
            let star = load_star(&self.root, tic, None)?;
            let residuals_file = folder.join("search.residuals.json");
            let search = match read_json(&folder.join("search.json")) {
                Some(s) => s,
                None => {
                    let s = self.search_star(&star, &cfg, &residuals_file)?;
                    write_json(&folder.join("search.json"), s.clone())?;
                    s
                }
            };
            let vetting = match read_json(&folder.join("vetting.json"))
                .filter(|v| v["vetting_version"] == json!(VETTING_VERSION))
            {
                Some(v) => v,
                None => {
                    let v = self.vet_star(&star, &search, &residuals_file)?;
                    write_json(&folder.join("vetting.json"), v.clone())?;
                    v
                }
            };
            let pixel = match read_json(&folder.join("pixel.json")) {
                Some(p) => p,
                None => {
                    let p = self.pixel_star(&star, &vetting);
                    write_json(&folder.join("pixel.json"), p.clone())?;
                    p
                }
            };
            let events = Self::assemble(Some(&vetting), Some(&pixel));
            let residuals = if search["peaks"].as_array().is_some_and(|a| !a.is_empty()) {
                Self::load_residuals(&residuals_file)?
            } else {
                Vec::new()
            };
            let mut candidates = Vec::new();
            for peak in &peaks {
                let found = events.iter().find(|e| {
                    e["search"] == "periodic" && e.get("metrics").is_some() && same_signal(e, peak)
                });
                let mut entry =
                    json!({"survey_period_days": peak["period_days"], "survey_snr": peak["snr"]});
                match found {
                    None => entry["outcome"] = json!("not recovered on the full baseline"),
                    Some(e) => {
                        entry["outcome"] = json!("recovered");
                        entry["tier"] = e["tier"].clone();
                        entry["signal"] = e["id"].clone();
                        entry["snr"] = e["metrics"]["snr"].clone();
                        entry["period_days"] = e["metrics"]["period_days"].clone();
                        entry["transits"] = e["metrics"]["transits"].clone();
                        let t = e["tier"].as_str().unwrap_or("");
                        if (t.starts_with("strong") || t.starts_with("candidate"))
                            && e["known"].is_null()
                        {
                            entry["ctoi"] = Value::Object(ctoi_row(e, &self.author));
                            let masks: Vec<(f64, f64, f64)> = e["masks"]
                                .as_array()
                                .map(|a| {
                                    a.iter()
                                        .map(|m| {
                                            (
                                                m[0].as_f64().unwrap(),
                                                m[1].as_f64().unwrap(),
                                                m[2].as_f64().unwrap(),
                                            )
                                        })
                                        .collect()
                                })
                                .unwrap_or_default();
                            let panels = folder.join("panels").join(e["id"].to_string());
                            if let Ok(prep) = Self::folded(&star, &residuals, &e["peak"], &masks) {
                                panel_data(&prep, e, &panels)?;
                            }
                            if e["metrics"]["pixel_test"]["sectors"]
                                .as_array()
                                .is_some_and(|a| a.iter().any(|d| d["used"] == true))
                            {
                                let p = &e["peak"];
                                pixel_panel_data(
                                    &star,
                                    p["period_days"].as_f64(),
                                    p["epoch_bjd"].as_f64().unwrap(),
                                    p["duration_hours"].as_f64().unwrap(),
                                    &self.root,
                                    &panels,
                                )
                                .ok();
                            }
                        }
                    }
                }
                candidates.push(entry);
            }
            write_json(
                &folder.join("ctoi.json"),
                json!({"tic": tic, "sectors": search["sectors"], "baseline_days": search["baseline_days"], "candidates": candidates}),
            )?;
            eprintln!("refine TIC {tic} {:.0}s", started.elapsed().as_secs_f64());
        }
        Ok(())
    }

    fn refine_folder(&self, tic: i64, today: &str) -> PathBuf {
        if let Ok(days) = std::fs::read_dir(&self.public) {
            for day in days.flatten() {
                let candidate = day.path().join("refine").join(tic.to_string());
                if candidate.join("ctoi.json").exists() {
                    return candidate;
                }
            }
        }
        self.public.join(today).join("refine").join(tic.to_string())
    }
}

/// The refined event that is the survey peak: same period (or a harmonic) and the same transit phase.
fn same_signal(event: &Value, peak: &Value) -> bool {
    let (p0, p1) = (
        peak["period_days"].as_f64().unwrap_or(0.0),
        event["peak"]["period_days"].as_f64().unwrap_or(0.0),
    );
    let d = peak["duration_hours"]
        .as_f64()
        .unwrap_or(0.0)
        .max(event["peak"]["duration_hours"].as_f64().unwrap_or(0.0))
        / 24.0;
    let (e0, e1) = (
        peak["epoch_bjd"].as_f64().unwrap_or(0.0),
        event["peak"]["epoch_bjd"].as_f64().unwrap_or(0.0),
    );
    [1.0, 2.0, 0.5].iter().any(|r| {
        (p1 / (p0 * r) - 1.0).abs() < 0.01 && super::catalog::wrap(e1 - e0, p0.min(p1)).abs() < d
    })
}

/// Per-panel data of a candidate as CSV files: binned detrended curve with
/// transit times, folded curve with the fitted trapezoid, odd and even
/// folds, and the secondary window.
pub fn panel_data(prep: &Folded, event: &Value, folder: &Path) -> Result<(), String> {
    std::fs::create_dir_all(folder).map_err(|e| e.to_string())?;
    let m = &event["metrics"];
    let depth = m["depth_ppm"].as_f64().unwrap_or(0.0) / 1e6;
    let t14 = m["fitted_duration_hours"]
        .as_f64()
        .unwrap_or(m["duration_hours"].as_f64().unwrap_or(0.0))
        / 24.0;
    let ingress = m["ingress_fraction"].as_f64().unwrap_or(0.2);
    let valid: Vec<usize> = (0..prep.time.len()).filter(|&i| prep.valid[i]).collect();
    let mut light = String::from("time_bjd,flux_ppt\n");
    let mut folded = String::from("phase_hours,flux_ppt,model_ppt,parity\n");
    for &i in &valid {
        light += &format!("{},{}\n", prep.time[i], prep.corrected[i] * 1e3);
        let k = ((prep.time[i] - prep.epoch) / prep.period).round() as i64;
        folded += &format!(
            "{},{},{},{}\n",
            prep.phase[i] * 24.0,
            prep.corrected[i] * 1e3,
            super::vet::trapezoid(prep.phase[i], depth, t14, ingress) * 1e3,
            if k.rem_euclid(2) == 0 { "even" } else { "odd" }
        );
    }
    let mut transits = String::from("transit_bjd\n");
    let k0 = ((prep.time[0] - prep.epoch) / prep.period).ceil() as i64;
    let k1 = ((prep.time[prep.time.len() - 1] - prep.epoch) / prep.period).floor() as i64;
    for k in k0..=k1 {
        transits += &format!("{}\n", prep.epoch + k as f64 * prep.period);
    }
    std::fs::write(folder.join("light_curve.csv"), light).map_err(|e| e.to_string())?;
    std::fs::write(folder.join("folded.csv"), folded).map_err(|e| e.to_string())?;
    std::fs::write(folder.join("transits.csv"), transits).map_err(|e| e.to_string())?;
    std::fs::write(
        folder.join("event.json"),
        serde_json::to_string_pretty(event).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Pixel panel data: difference and out-of-transit images per sector as CSV grids.
pub fn pixel_panel_data(
    star: &Star,
    period: Option<f64>,
    epoch: f64,
    duration_hours: f64,
    root: &Path,
    folder: &Path,
) -> Result<(), String> {
    for image in sector_images(star, period, epoch, duration_hours, root, None)? {
        std::fs::create_dir_all(folder).map_err(|e| e.to_string())?;
        for (name, data) in [
            ("difference", &image.difference),
            ("difference_error", &image.error),
            ("oot", &image.oot),
        ] {
            let mut text = String::new();
            for y in 0..image.ny {
                text += &(0..image.nx)
                    .map(|x| format!("{}", data[y * image.nx + x]))
                    .collect::<Vec<_>>()
                    .join(",");
                text += "\n";
            }
            std::fs::write(
                folder.join(format!("sector-{}-{name}.csv", image.sector)),
                text,
            )
            .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
