//! FBLS: box least squares transit search for TESS light curves on CUDA.
//!
//! Usage:
//!   fbls search --input a.npz [b.npz ...] [--pmin D] [--pmax D] [--output out.json]
//!               [--save-table table.csv] [--prepared] [--device N] [--prescan FACTOR] [--threshold SNR]
//!   fbls validate --stars reference/stars.json [--data DIR] [--tic TIC] [--output recovery.json] [--device N]
//!   fbls batch --list stars.txt --output-dir DIR [--workers K] [--devices 0,1] [--pmax D]
//!   fbls fetch --tic ID... [--sectors N,M] [--root data]
//!   fbls survey <search|vet|pixel|refine|publish> (--targets FILE | --tic ID... | --all-downloaded) [--root data] [--public public]

mod config;
mod io {
    pub mod fits;
    pub mod npz;
}
mod prep {
    pub mod curve;
    pub mod stats;
}
mod search {
    pub mod events;
    pub mod evidence;
    pub mod grid;
    pub mod kernels;
    pub mod math;
    pub mod scan;
}
mod select {
    pub mod peaks;
}
mod batch;
mod pipeline;
mod survey;
mod validate;

use config::SearchConfig;
use prep::curve::{Observations, PrepConfig};
use std::collections::HashMap;

pub fn read_inputs(paths: &[String]) -> Result<Observations, String> {
    let mut time = Vec::new();
    let mut flux = Vec::new();
    let mut err: Option<Vec<f64>> = None;
    let mut group = Vec::new();
    let mut groups: HashMap<(usize, i64), i64> = HashMap::new();
    for (index, path) in paths.iter().enumerate() {
        if path.ends_with(".fits") || path.ends_with(".fits.gz") {
            let lc = io::fits::read_light_curve(path)?;
            let n = lc.time_bjd.len();
            if index == 0 {
                err = Some(Vec::new());
            }
            match err.as_mut() {
                Some(v) => v.extend_from_slice(&lc.flux_err),
                None => return Err("all input files must either provide flux_err or omit it".into()),
            }
            time.extend_from_slice(&lc.time_bjd);
            flux.extend_from_slice(&lc.flux);
            let next = groups.len() as i64;
            let g = *groups.entry((index, lc.sector)).or_insert(next);
            group.extend(std::iter::repeat_n(g, n));
            continue;
        }
        let source = io::npz::read_npz(path)?;
        let t = source.get("time_bjd").ok_or(format!("{path}: required columns are time_bjd and flux"))?;
        let f = source.get("flux").ok_or(format!("{path}: required columns are time_bjd and flux"))?;
        if t.shape.len() != 1 || f.shape != t.shape {
            return Err(format!("{path}: time_bjd and flux must be matching vectors"));
        }
        let n = t.len();
        match source.get("flux_err") {
            Some(e) if e.shape == t.shape => {
                if index == 0 {
                    err = Some(Vec::new());
                }
                match err.as_mut() {
                    Some(v) => v.extend_from_slice(&e.data),
                    None => return Err("all input files must either provide flux_err or omit it".into()),
                }
            }
            Some(_) => return Err(format!("{path}: flux_err must match time_bjd")),
            None if err.is_some() => return Err("all input files must either provide flux_err or omit it".into()),
            None => {}
        }
        let sector: Vec<i64> = match source.get("sector") {
            Some(s) if s.len() == 1 => vec![s.data[0] as i64; n],
            Some(s) if s.len() == n => s.data.iter().map(|x| *x as i64).collect(),
            Some(_) => return Err(format!("{path}: sector must be a scalar or an observation vector")),
            None => vec![0; n],
        };
        time.extend_from_slice(&t.data);
        flux.extend_from_slice(&f.data);
        for s in sector {
            let next = groups.len() as i64;
            group.push(*groups.entry((index, s)).or_insert(next));
        }
    }
    Ok(Observations { time, flux, err, group })
}

pub fn prep_config(cfg: &SearchConfig) -> PrepConfig {
    PrepConfig {
        bin_minutes: cfg.bin_minutes,
        window_min: cfg.window_days,
        window_factor: cfg.window_factor,
        window_max: cfg.window_max_days,
        clip_sigma: cfg.clip_sigma,
        edge_trim: cfg.edge_trim_windows,
    }
}

fn run_search(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut inputs = Vec::new();
    let mut output = "-".to_string();
    let mut table_path: Option<String> = None;
    let mut prepared = false;
    let mut device = 0usize;
    let mut cfg = SearchConfig::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                i += 1;
                while i < args.len() && !args[i].starts_with("--") {
                    inputs.push(args[i].clone());
                    i += 1;
                }
                continue;
            }
            "--output" => {
                output = args[i + 1].clone();
                i += 1;
            }
            "--save-table" => {
                table_path = Some(args[i + 1].clone());
                i += 1;
            }
            "--prepared" => prepared = true,
            "--device" => {
                device = args[i + 1].parse()?;
                i += 1;
            }
            "--pmin" => {
                cfg.period_min = args[i + 1].parse()?;
                i += 1;
            }
            "--pmax" => {
                cfg.period_max = args[i + 1].parse()?;
                i += 1;
            }
            "--prescan" => {
                cfg.prescan_factor = args[i + 1].parse()?;
                i += 1;
            }
            "--threshold" => {
                cfg.threshold = args[i + 1].parse()?;
                i += 1;
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
        i += 1;
    }
    if inputs.is_empty() {
        return Err("--input is required".into());
    }
    cfg.validate()?;
    let obs = read_inputs(&inputs)?;
    let mut report = if prepared { pipeline::search_prepared(&obs, &cfg, device)? } else { pipeline::search_transits(&obs, &cfg, device)? };
    report["input_files"] = serde_json::json!(inputs);
    report["input_mode"] = serde_json::json!(if prepared { "prepared" } else { "raw" });
    if let Some(path) = table_path {
        let rows = report["period_table"].as_array().cloned().unwrap_or_default();
        let mut text = String::from("period_days,epoch_bjd,duration_days,snr,depth\n");
        for r in rows {
            text += &format!("{},{},{},{},{}\n", r["period"], r["epoch"], r["duration"], r["score"], r["depth"]);
        }
        std::fs::write(path, text)?;
    }
    report.as_object_mut().unwrap().remove("period_table");
    let text = serde_json::to_string_pretty(&report)?;
    if output == "-" {
        println!("{text}");
    } else {
        std::fs::write(&output, text)?;
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("search") => run_search(&args[1..]),
        Some("validate") => validate::run(&args[1..]),
        Some("batch") => batch::run(&args[1..]),
        Some("fetch") => survey::run_fetch(&args[1..]),
        Some("survey") => survey::run_survey(&args[1..]),
        Some("catalog") => survey::release::run(&args[1..]),
        _ => Err("usage: fbls search | validate | batch | fetch | survey <stage>".into()),
    };
    if let Err(e) = result {
        eprintln!("fbls: {e}");
        std::process::exit(1);
    }
}
