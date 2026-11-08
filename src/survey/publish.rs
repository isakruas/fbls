//! Tiers, candidate table and ExoFOP CTOI rows from the vetting files.

use serde_json::{Map, Value, json};

pub const TIER_ORDER: [&str; 6] = [
    "strong",
    "candidate",
    "unconfirmed",
    "inconclusive",
    "rejected",
    "duplicate",
];
/// Transits a periodic signal must show to be called periodic.
pub const MIN_TRANSITS: usize = 3;

pub const CTOI_COLUMNS: [&str; 27] = [
    "TIC ID",
    "Transit Epoch (BJD)",
    "Transit Epoch (BJD) err",
    "Period (days)",
    "Period (days) Error",
    "Depth ppm",
    "Depth ppm Error",
    "Duration (hrs)",
    "Duration (hrs) Error",
    "Impact Param",
    "Impact Param Error",
    "Rad_p/Rad_s",
    "Rad_p/Rad_s Error",
    "a/Rad_s",
    "a/Rad_s Error",
    "Inclination (deg)",
    "Inclination (deg) Error",
    "Planet Radius (R_Earth)",
    "Planet Radius (R_Earth) Error",
    "Insolation (Earth Flux)",
    "Insolation (Earth Flux) Error",
    "Equilibrium Temp (K)",
    "Equilibrium Temp (K) Error",
    "Fitted Stellar Density (g/cm^3)",
    "Fitted Stellar Density (g/cm^3) Error",
    "User Disposition",
    "Notes",
];

/// Tier from vetting, model evidence and FPP; no learned classifier. A
/// periodic signal with fewer than MIN_TRANSITS transits is unconfirmed.
/// Without stellar radius and temperature the radius, density and FPP tests
/// are meaningless, so the signal stays a candidate.
pub fn tier(event: &Value) -> String {
    let m = &event["metrics"];
    let fpp = &event["fpp"];
    if m["verdict"] == "fail" {
        return "rejected: vetting".into();
    }
    if fpp["nfpp"].as_f64().is_some_and(|v| v > 0.1) {
        return "rejected: NFPP (likely nearby star)".into();
    }
    let preferred = event["peak"]["model_preferred"].as_bool().unwrap_or(false);
    if event["search"] == "single" {
        return "candidate: single event".into();
    }
    if m["transits"].as_u64().unwrap_or(0) < MIN_TRANSITS as u64 {
        return format!("unconfirmed: fewer than {MIN_TRANSITS} transits");
    }
    let star = &m["star"];
    if !(star["radius"].as_f64().is_some_and(|r| r > 0.0)
        && star["teff"].as_f64().is_some_and(|t| t > 0.0))
    {
        return "candidate: no stellar parameters".into();
    }
    if let (Some(f), Some(nf)) = (fpp["fpp"].as_f64(), fpp["nfpp"].as_f64()) {
        if f < 0.01 && nf < 0.001 && preferred {
            return "strong: FPP < 1% and NFPP < 0.1%".into();
        }
        if f < 0.5 && nf < 0.001 {
            return "candidate: likely planet".into();
        }
        return "inconclusive".into();
    }
    if preferred {
        "candidate: no FPP".into()
    } else {
        "inconclusive: binary-like evidence".into()
    }
}

pub fn rank(label: &str) -> usize {
    TIER_ORDER
        .iter()
        .position(|o| label.starts_with(o))
        .unwrap_or(TIER_ORDER.len())
}

fn finite(v: &Value) -> Option<f64> {
    v.as_f64().filter(|x| x.is_finite())
}

/// CTOI values for one vetted candidate; fields without a measurement stay
/// empty. Uncertainties come from the search grid and the measured noise.
pub fn ctoi_row(event: &Value, author: &str) -> Map<String, Value> {
    let m = &event["metrics"];
    let peak = &event["peak"];
    let period = finite(&m["period_days"]);
    let epoch = finite(&m["epoch_bjd"]).unwrap_or(f64::NAN);
    let duration_h = finite(&m["duration_hours"]).unwrap_or(f64::NAN);
    let snr = finite(&m["snr"]).or(finite(&peak["snr"])).unwrap_or(7.1);
    let periodic = event["search"] == "periodic";
    let baseline = finite(&event["baseline_days"]).or(period).unwrap_or(1.0);
    let period_err = period
        .filter(|_| periodic)
        .map(|p| p * (duration_h / 24.0) / (2.0 * baseline));
    let epoch_err = (duration_h / 24.0) / snr.max(1.0);
    let depth = finite(&m["depth_ppm"]);
    let depth_err = finite(&m["depth_err_ppm"]);
    let k = depth.filter(|d| *d > 0.0).map(|d| (d / 1e6).sqrt());
    let k_err = match (k, depth_err) {
        (Some(k), Some(de)) => Some((de / 1e6) / (2.0 * k)),
        _ => None,
    };
    let radius = finite(&m["radius_earth"]);
    let radius_err = match (radius, k, k_err) {
        (Some(r), Some(k), Some(ke)) => Some(r * ke / k),
        _ => None,
    };
    let g = &m["geometry"];
    let reliable = g["reliable"].as_bool().unwrap_or(false);
    let (a_over_r, a_over_r_err) = if reliable {
        (
            finite(&g["a_over_rstar_fit"]),
            finite(&g["a_over_rstar_fit_err"]),
        )
    } else {
        (
            finite(&m["a_over_rstar"]),
            finite(&m["a_over_rstar"]).map(|a| 0.1 * a),
        )
    };
    let flags: Vec<String> = m["flags"]
        .as_object()
        .map(|o| {
            o.iter()
                .map(|(k, v)| format!("{k}={}", v.as_str().unwrap_or("")))
                .collect()
        })
        .unwrap_or_default();
    let fpp = &event["fpp"];
    let notes = format!(
        "FBLS blind search ({}); detector SNR {:.1}, {} transits; tier {}; flags {}; FPP {}; NFPP {}; {}{author}",
        event["search"].as_str().unwrap_or("periodic"),
        finite(&peak["snr"]).unwrap_or(0.0),
        m["transits"].as_u64().unwrap_or(0),
        event["tier"].as_str().unwrap_or(""),
        if flags.is_empty() {
            "none".to_string()
        } else {
            flags.join(";")
        },
        fpp["fpp"]
            .as_f64()
            .map(|v| v.to_string())
            .unwrap_or("None".into()),
        fpp["nfpp"]
            .as_f64()
            .map(|v| v.to_string())
            .unwrap_or("None".into()),
        if reliable {
            "geometry from trapezoid fit, circular orbit; "
        } else {
            "a/Rs from TIC density; "
        }
    );
    let duration_err = finite(&m["fitted_duration_err_hours"]).unwrap_or(duration_h * 0.25);
    let mut row = Map::new();
    let mut put = |key: &str, value: Value| {
        row.insert(key.to_string(), value);
    };
    put("TIC ID", event["tic"].clone());
    put("Transit Epoch (BJD)", json!(epoch));
    put("Transit Epoch (BJD) err", json!(epoch_err));
    put("Period (days)", json!(period.filter(|_| periodic)));
    put("Period (days) Error", json!(period_err));
    put("Depth ppm", json!(depth));
    put("Depth ppm Error", json!(depth_err));
    put("Duration (hrs)", json!(duration_h));
    put("Duration (hrs) Error", json!(duration_err));
    put("Impact Param", g["impact"].clone());
    put("Impact Param Error", g["impact_err"].clone());
    put("Rad_p/Rad_s", json!(k));
    put("Rad_p/Rad_s Error", json!(k_err));
    put("a/Rad_s", json!(a_over_r));
    put("a/Rad_s Error", json!(a_over_r_err));
    put("Inclination (deg)", g["inclination_deg"].clone());
    put("Inclination (deg) Error", g["inclination_err"].clone());
    put("Planet Radius (R_Earth)", json!(radius));
    put("Planet Radius (R_Earth) Error", json!(radius_err));
    put("Insolation (Earth Flux)", g["insolation_earth"].clone());
    put("Insolation (Earth Flux) Error", g["insolation_err"].clone());
    put("Equilibrium Temp (K)", g["teq_k"].clone());
    put("Equilibrium Temp (K) Error", g["teq_err"].clone());
    put(
        "Fitted Stellar Density (g/cm^3)",
        g["fitted_density_gcc"].clone(),
    );
    put(
        "Fitted Stellar Density (g/cm^3) Error",
        g["fitted_density_err"].clone(),
    );
    put("User Disposition", json!("PC"));
    put("Notes", json!(notes));
    row
}

pub fn csv_cell(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => {
            if s.contains(',') || s.contains('"') || s.contains('\n') {
                format!("\"{}\"", s.replace('"', "\"\""))
            } else {
                s.clone()
            }
        }
        Value::Bool(b) => {
            if *b {
                "True".into()
            } else {
                "False".into()
            }
        }
        other => other.to_string(),
    }
}

pub fn write_csv(
    path: &std::path::Path,
    columns: &[&str],
    rows: &[Map<String, Value>],
) -> Result<(), String> {
    let mut text = columns.join(",") + "\n";
    for r in rows {
        text += &columns
            .iter()
            .map(|c| csv_cell(r.get(*c).unwrap_or(&Value::Null)))
            .collect::<Vec<_>>()
            .join(",");
        text += "\n";
    }
    std::fs::write(path, text).map_err(|e| e.to_string())
}

/// Candidate table row of one vetted event.
pub fn candidate_row(event: &Value) -> Map<String, Value> {
    let m = &event["metrics"];
    let peak = &event["peak"];
    let known = &event["known"];
    let flags: Vec<String> = m["flags"]
        .as_object()
        .map(|o| {
            o.iter()
                .map(|(k, v)| format!("{k}={}", v.as_str().unwrap_or("")))
                .collect()
        })
        .unwrap_or_default();
    let mut row = Map::new();
    for (k, v) in [
        ("tic", event["tic"].clone()),
        ("signal", event["id"].clone()),
        ("search", event["search"].clone()),
        ("tier", event["tier"].clone()),
        ("known", known["name"].clone()),
        ("known_source", known["source"].clone()),
        ("known_disposition", known["disposition"].clone()),
        ("period_days", m["period_days"].clone()),
        ("epoch_bjd", m["epoch_bjd"].clone()),
        ("duration_hours", m["duration_hours"].clone()),
        ("depth_ppm", m["depth_ppm"].clone()),
        ("radius_earth", m["radius_earth"].clone()),
        ("detector_snr", peak["snr"].clone()),
        ("vetting_snr", m["snr"].clone()),
        ("transits", m["transits"].clone()),
        ("planet_log_odds", peak["planet_log_odds"].clone()),
        ("model_preferred", peak["model_preferred"].clone()),
        ("fpp", event["fpp"]["fpp"].clone()),
        ("nfpp", event["fpp"]["nfpp"].clone()),
        (
            "pixel_offset_arcsec",
            m["pixel_test"]["offset_arcsec"].clone(),
        ),
        ("flags", json!(flags.join(";"))),
    ] {
        row.insert(k.to_string(), v);
    }
    row
}

pub const CANDIDATE_COLUMNS: [&str; 21] = [
    "tic",
    "signal",
    "search",
    "tier",
    "known",
    "known_source",
    "known_disposition",
    "period_days",
    "epoch_bjd",
    "duration_hours",
    "depth_ppm",
    "radius_earth",
    "detector_snr",
    "vetting_snr",
    "transits",
    "planet_log_odds",
    "model_preferred",
    "fpp",
    "nfpp",
    "pixel_offset_arcsec",
    "flags",
];
