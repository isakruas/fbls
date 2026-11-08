//! Recovery of catalog ephemerides over the reference star set.

use crate::config::SearchConfig;
use crate::read_inputs;
use serde_json::Value;

#[derive(Debug, Clone, serde::Serialize)]
struct Recovery {
    tic: String,
    toi: String,
    catalog_period: f64,
    catalog_depth_ppm: f64,
    catalog_duration_h: f64,
    found_period: Option<f64>,
    found_snr: Option<f64>,
    found_depth_ppm: Option<f64>,
    found_duration_h: Option<f64>,
    alias: Option<f64>,
    recovered: bool,
    seconds: f64,
}

/// Smallest relative distance of `period` to m x catalog period for m in {1/3, 1/2, 1, 2, 3}.
fn alias_error(period: f64, catalog: f64) -> (f64, f64) {
    [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0]
        .iter()
        .map(|&m| ((period / catalog - m).abs() / m, m))
        .fold(
            (f64::INFINITY, 1.0),
            |acc, x| if x.0 < acc.0 { x } else { acc },
        )
}

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let stars_path = args
        .iter()
        .position(|a| a == "--stars")
        .map(|i| args[i + 1].clone())
        .ok_or("--stars FILE is required")?;
    let output = args
        .iter()
        .position(|a| a == "--output")
        .map(|i| args[i + 1].clone());
    let only: Option<String> = args
        .iter()
        .position(|a| a == "--tic")
        .map(|i| args[i + 1].clone());
    let device: usize = args
        .iter()
        .position(|a| a == "--device")
        .map(|i| args[i + 1].parse().unwrap_or(0))
        .unwrap_or(0);
    let data = std::path::PathBuf::from(
        args.iter()
            .position(|a| a == "--data")
            .map(|i| args[i + 1].clone())
            .unwrap_or("data".into()),
    );
    let stars: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(&stars_path)?)?;
    let mut results = Vec::new();
    for star in &stars {
        let tic = star["tic"].as_str().unwrap().to_string();
        if only.as_ref().is_some_and(|t| *t != tic) {
            continue;
        }
        let mut paths: Vec<String> = std::fs::read_dir(data.join("curves").join(&tic))?
            .filter_map(|e| e.ok())
            .map(|e| e.path().to_string_lossy().into_owned())
            .filter(|p| p.ends_with(".npz") || p.ends_with(".fits"))
            .collect();
        paths.sort();
        let cfg = SearchConfig {
            period_max: star["pmax"].as_f64().unwrap_or(20.0),
            ..SearchConfig::default()
        };
        let start = std::time::Instant::now();
        let obs = read_inputs(&paths)?;
        let report = crate::pipeline::search_transits(&obs, &cfg, device)?;
        let seconds = start.elapsed().as_secs_f64();
        let catalog_period = star["period"].as_f64().unwrap();
        let catalog_depth = star["depth_ppm"].as_f64().unwrap_or(f64::NAN);
        let catalog_duration = star["duration_h"].as_f64().unwrap_or(f64::NAN);
        let candidates = report["signal_candidates"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let best = candidates
            .iter()
            .map(|c| {
                (
                    c,
                    alias_error(c["period_days"].as_f64().unwrap(), catalog_period),
                )
            })
            .filter(|(_, (err, _))| *err < 1e-3)
            .max_by(|a, b| {
                a.0["snr"]
                    .as_f64()
                    .partial_cmp(&b.0["snr"].as_f64())
                    .unwrap()
            });
        let (found, alias) = match best {
            Some((c, (_, m))) => (Some(c.clone()), Some(m)),
            None => (candidates.first().cloned(), None),
        };
        let depth_ok = |d: f64| {
            !catalog_depth.is_finite() || (d / catalog_depth > 0.5 && d / catalog_depth < 2.0)
        };
        let duration_ok = |h: f64| {
            !(catalog_duration > 0.0) || (h / catalog_duration > 0.5 && h / catalog_duration < 2.0)
        };
        let recovered = alias.is_some()
            && found.as_ref().is_some_and(|c| {
                depth_ok(c["depth_ppm"].as_f64().unwrap_or(0.0))
                    && duration_ok(c["duration_hours"].as_f64().unwrap_or(0.0))
            });
        let row = Recovery {
            tic: tic.clone(),
            toi: star["toi"].as_str().unwrap_or("").to_string(),
            catalog_period,
            catalog_depth_ppm: catalog_depth,
            catalog_duration_h: catalog_duration,
            found_period: found.as_ref().and_then(|c| c["period_days"].as_f64()),
            found_snr: found.as_ref().and_then(|c| c["snr"].as_f64()),
            found_depth_ppm: found.as_ref().and_then(|c| c["depth_ppm"].as_f64()),
            found_duration_h: found.as_ref().and_then(|c| c["duration_hours"].as_f64()),
            alias,
            recovered,
            seconds,
        };
        println!(
            "{:>10} TOI {:>8} P={:9.4} found={:<10} alias={:<5} snr={:<7} depth={:>7.0}/{:<7.0} dur={:5.2}/{:<5.2}h {} {:.1}s",
            row.tic,
            row.toi,
            row.catalog_period,
            row.found_period
                .map(|p| format!("{p:.4}"))
                .unwrap_or("-".into()),
            row.alias.map(|m| format!("{m:.2}")).unwrap_or("-".into()),
            row.found_snr
                .map(|s| format!("{s:.1}"))
                .unwrap_or("-".into()),
            row.found_depth_ppm.unwrap_or(0.0),
            row.catalog_depth_ppm,
            row.found_duration_h.unwrap_or(0.0),
            row.catalog_duration_h,
            if row.recovered { "RECOVERED" } else { "MISSED" },
            row.seconds
        );
        results.push(row);
    }
    let recovered = results.iter().filter(|r| r.recovered).count();
    let total_seconds: f64 = results.iter().map(|r| r.seconds).sum();
    println!(
        "recovered {recovered} of {} stars in {total_seconds:.0} s",
        results.len()
    );
    if let Some(path) = output {
        std::fs::write(path, serde_json::to_string_pretty(&results)?)?;
    }
    Ok(())
}
