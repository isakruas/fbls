//! Survey layer: product retrieval, catalog matching, vetting, pixel-level
//! tests and publication tables built on the search reports.

pub mod catalog;
pub mod fetch;
pub mod numeric;
pub mod pixel;
pub mod publish;
pub mod release;
pub mod stages;
pub mod star;
pub mod vet;

use std::path::PathBuf;

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).map(|i| args[i + 1].clone())
}

fn tics_from(args: &[String]) -> Result<Vec<i64>, String> {
    let mut tics: Vec<i64> = Vec::new();
    if let Some(i) = args.iter().position(|a| a == "--tic") {
        for a in &args[i + 1..] {
            if a.starts_with("--") {
                break;
            }
            tics.push(a.parse().map_err(|_| format!("invalid TIC {a}"))?);
        }
    }
    if let Some(path) = arg(args, "--targets") {
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
        let col = header.iter().position(|h| *h == "tic_id" || *h == "tic").ok_or("targets CSV needs a tic_id column")?;
        for line in lines {
            if let Some(t) = line.split(',').nth(col).and_then(|s| s.trim().parse().ok()) {
                tics.push(t);
            }
        }
    }
    if args.iter().any(|a| a == "--all-downloaded") {
        let root = PathBuf::from(arg(args, "--root").unwrap_or("data".into()));
        tics.extend(star::downloaded_stars(&root));
    }
    if tics.is_empty() {
        return Err("give --targets FILE, --tic ID... or --all-downloaded".into());
    }
    tics.sort();
    tics.dedup();
    Ok(tics)
}

/// `fbls fetch --tic ID... [--sectors N,M] [--root DIR]`
pub fn run_fetch(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(arg(args, "--root").unwrap_or("data".into()));
    let sectors: Option<Vec<i64>> = arg(args, "--sectors").map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect());
    for tic in tics_from(args)? {
        let written = fetch::fetch_star(&root, tic, sectors.as_deref(), &mut |line| eprintln!("{line}"))?;
        eprintln!("TIC {tic}: {} new sector files", written.len());
    }
    Ok(())
}

/// `fbls survey <search|vet|pixel|refine|publish> (--targets FILE | --tic ID... | --all-downloaded) [--root DIR] [--public DIR] [--author NAME] [--device N] [--pmax D]`
pub fn run_survey(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let stage = args.first().ok_or("survey stage: search, vet, pixel, refine or publish")?.clone();
    let root = PathBuf::from(arg(args, "--root").unwrap_or("data".into()));
    let public = PathBuf::from(arg(args, "--public").unwrap_or("public".into()));
    let catalog = catalog::Catalog::load(&root.join("catalogs"))?;
    let survey = stages::Survey {
        root,
        public,
        catalog,
        author: arg(args, "--author").unwrap_or_default(),
        device: arg(args, "--device").and_then(|d| d.parse().ok()).unwrap_or(0),
        pmax: arg(args, "--pmax").and_then(|d| d.parse().ok()).unwrap_or(stages::PERIOD_MAX),
    };
    let tics = tics_from(args)?;
    match stage.as_str() {
        "search" => survey.run_search(&tics)?,
        "vet" => survey.run_vet(&tics)?,
        "pixel" => survey.run_pixel(&tics)?,
        "refine" => survey.run_refine(&tics)?,
        "publish" => {
            let (rows, ctoi, new) = survey.run_publish(&tics)?;
            eprintln!("public: {rows} vetted signals, {ctoi} CTOI rows, {new} stars first published today");
        }
        other => return Err(format!("unknown survey stage {other}").into()),
    }
    Ok(())
}
