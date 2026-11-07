//! Batch mode: many stars through one or more GPUs with several worker
//! threads per device, so kernels of one star overlap the host stages
//! (preparation, selection, report) of another.

use crate::config::SearchConfig;
use crate::read_inputs;
use std::sync::{Arc, Mutex};

struct Job {
    name: String,
    inputs: Vec<String>,
    pmax: Option<f64>,
}

/// List format: one star per line, `name pmax file1 file2 ...`; `pmax` may be `-`.
fn parse_list(path: &str) -> Result<Vec<Job>, Box<dyn std::error::Error>> {
    let mut jobs = Vec::new();
    for line in std::fs::read_to_string(path)?.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 || fields[0].starts_with('#') {
            continue;
        }
        jobs.push(Job {
            name: fields[0].to_string(),
            pmax: if fields[1] == "-" { None } else { Some(fields[1].parse()?) },
            inputs: fields[2..].iter().map(|s| s.to_string()).collect(),
        });
    }
    Ok(jobs)
}

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let get = |flag: &str| args.iter().position(|a| a == flag).map(|i| args[i + 1].clone());
    let list = get("--list").ok_or("--list FILE is required")?;
    let output_dir = get("--output-dir").ok_or("--output-dir DIR is required")?;
    let workers: usize = get("--workers").map(|w| w.parse()).transpose()?.unwrap_or(2); // measured: 2 workers per GPU gain ~11%, 3 lose to contention
    let devices: Vec<usize> = get("--devices").map(|d| d.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![0]);
    let pmax_default: f64 = get("--pmax").map(|p| p.parse()).transpose()?.unwrap_or(20.0);
    std::fs::create_dir_all(&output_dir)?;
    let jobs = Arc::new(Mutex::new(parse_list(&list)?));
    let total = jobs.lock().unwrap().len();
    let start = std::time::Instant::now();
    let done = Arc::new(Mutex::new(0usize));
    let mut handles = Vec::new();
    for device in devices {
        for _ in 0..workers {
            let jobs = Arc::clone(&jobs);
            let done = Arc::clone(&done);
            let output_dir = output_dir.clone();
            handles.push(std::thread::spawn(move || -> Result<(), String> {
                loop {
                    let job = jobs.lock().unwrap().pop();
                    let Some(job) = job else { return Ok(()) };
                    let cfg = SearchConfig { period_max: job.pmax.unwrap_or(pmax_default), ..SearchConfig::default() };
                    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
                        let obs = read_inputs(&job.inputs)?;
                        let mut report = crate::pipeline::search_transits(&obs, &cfg, device)?;
                        report["input_files"] = serde_json::json!(job.inputs);
                        report["input_mode"] = serde_json::json!("raw");
                        std::fs::write(format!("{output_dir}/{}.json", job.name), serde_json::to_string_pretty(&report)?)?;
                        let seconds = report["statistics"]["seconds_total"].as_f64().unwrap_or(0.0);
                        let mut d = done.lock().unwrap();
                        *d += 1;
                        eprintln!("[{}/{}] {} cuda:{} {:.1}s", *d, total, job.name, device, seconds);
                        Ok(())
                    })();
                    if let Err(e) = result {
                        eprintln!("{}: {e}", job.name);
                        std::fs::write(format!("{output_dir}/{}.error", job.name), e.to_string()).ok();
                    }
                }
            }));
        }
    }
    for h in handles {
        h.join().map_err(|_| "worker thread panicked")??;
    }
    eprintln!("{total} stars in {:.0} s", start.elapsed().as_secs_f64());
    Ok(())
}
