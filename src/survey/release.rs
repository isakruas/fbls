//! Catalog release: the deposit folder assembled from the published tables
//! and its publication as a new version of the Zenodo record.
//!
//! `fbls catalog deposit [--public DIR] [--day YYYY-MM-DD] [--out DIR]`
//!   writes `<out>/<day>/` with `ctoi.csv`, `candidates.csv`, `CITATION.cff`
//!   (version and release date set to the day), `ReadMe.txt` (version line
//!   set), `.zenodo.json` and `figures.zip` holding one panel folder per CTOI
//!   signal (from the refine folders and the day figure folders).
//! `fbls catalog zenodo --deposit DIR`
//!   publishes the folder as a new version of the concept record
//!   (`ZENODO_TOKEN`, `ZENODO_RECORD`), skipping versions already published.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

const CONCEPT_RECORD: &str = "24026413";
const ZENODO_API: &str = "https://zenodo.org/api";

/// UTC civil date of today as YYYY-MM-DD (days-from-epoch to civil conversion).
fn today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let z = secs.div_euclid(86400) + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}-{m:02}-{d:02}")
}

fn replace_line(text: &str, prefix: &str, replacement: &str) -> String {
    text.lines()
        .map(|l| {
            if l.starts_with(prefix) {
                replacement.to_string()
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

/// CTOI signal -> panel folder, from `public/<day>/refine/<tic>/figures/<signal>`
/// and `public/<day>/figures/<tic>_<signal>`; refine folders take precedence.
fn figure_folders(public: &Path) -> BTreeMap<String, PathBuf> {
    let mut found = BTreeMap::new();
    let mut days: Vec<PathBuf> = std::fs::read_dir(public)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.is_dir()
                        && p.file_name()
                            .unwrap()
                            .to_string_lossy()
                            .matches('-')
                            .count()
                            == 2
                })
                .collect()
        })
        .unwrap_or_default();
    days.sort();
    for day in &days {
        if let Ok(rd) = std::fs::read_dir(day.join("refine")) {
            for star in rd.filter_map(|e| e.ok()) {
                let tic = star.file_name().to_string_lossy().into_owned();
                if let Ok(figs) = std::fs::read_dir(star.path().join("figures")) {
                    for f in figs.filter_map(|e| e.ok()) {
                        found.insert(
                            format!("{tic}_{}", f.file_name().to_string_lossy()),
                            f.path(),
                        );
                    }
                }
            }
        }
        if let Ok(rd) = std::fs::read_dir(day.join("figures")) {
            for f in rd.filter_map(|e| e.ok()) {
                let name = f.file_name().to_string_lossy().into_owned();
                if name.contains('_') {
                    found.entry(name).or_insert(f.path());
                }
            }
        }
    }
    found
}

fn zenodo_metadata(readme: &str, day: &str) -> serde_json::Value {
    let title = readme.lines().next().unwrap_or("").to_string();
    let description = readme
        .split("\n\n")
        .nth(1)
        .unwrap_or("")
        .trim()
        .replace('\n', " ");
    serde_json::json!({
        "upload_type": "dataset", "title": title, "version": format!("v{day}"), "publication_date": day,
        "creators": [{"name": "Ruas, Isak Paulo de Andrade", "affiliation": "NROLABS"}],
        "license": "cc-by-4.0", "keywords": ["TESS", "exoplanets", "transit search", "planet candidates", "CTOI"],
        "description": description,
        "related_identifiers": [{"identifier": "https://github.com/isakruas/fbls", "relation": "isSupplementTo", "resource_type": "software"}],
    })
}

fn add_tree(
    zip: &mut zip::ZipWriter<std::fs::File>,
    dir: &Path,
    base: &Path,
    options: zip::write::SimpleFileOptions,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    entries.sort();
    let mut count = 0;
    for p in entries {
        let rel = p.strip_prefix(base)?.to_string_lossy().replace('\\', "/");
        if p.is_dir() {
            zip.add_directory(format!("{rel}/"), options)?;
            count += add_tree(zip, &p, base, options)?;
        } else {
            zip.start_file(rel, options)?;
            zip.write_all(&std::fs::read(&p)?)?;
            count += 1;
        }
    }
    Ok(count)
}

fn zip_folder(
    root: &Path,
    folder_name: &str,
    out: &Path,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut zip = zip::ZipWriter::new(std::fs::File::create(out)?);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    zip.add_directory(format!("{folder_name}/"), options)?;
    let count = add_tree(&mut zip, &root.join(folder_name), root, options)?;
    zip.finish()?;
    Ok(count)
}

fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let target = dst.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

pub fn deposit(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let get = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .map(|i| args[i + 1].clone())
    };
    let public = PathBuf::from(get("--public").unwrap_or("public".into()));
    let day = get("--day")
        .or_else(|| std::env::var("FBLS_DAY").ok())
        .unwrap_or_else(today);
    let out = PathBuf::from(get("--out").unwrap_or("paper/deposit".into())).join(&day);
    if !public.join("ctoi.csv").exists() {
        return Err("nothing published yet".into());
    }
    if out.exists() {
        std::fs::remove_dir_all(&out)?;
    }
    std::fs::create_dir_all(out.join("figures"))?;
    for name in ["ctoi.csv", "candidates.csv"] {
        std::fs::copy(public.join(name), out.join(name))?;
    }
    let citation = std::fs::read_to_string("CITATION.cff")?;
    let citation = replace_line(
        &replace_line(&citation, "version:", &format!("version: v{day}")),
        "date-released:",
        &format!("date-released: {day}"),
    );
    std::fs::write(out.join("CITATION.cff"), citation)?;
    let readme = std::fs::read_to_string("paper/ReadMe.txt")?
        .replace("Version: <date>", &format!("Version: {day}"));
    std::fs::write(out.join("ReadMe.txt"), &readme)?;
    std::fs::write(
        out.join(".zenodo.json"),
        serde_json::to_string_pretty(&zenodo_metadata(&readme, &day))? + "\n",
    )?;
    let text = std::fs::read_to_string(public.join("ctoi.csv"))?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
    let col = header
        .iter()
        .position(|h| *h == "TIC ID")
        .ok_or("ctoi.csv without a TIC ID column")?;
    let tics: std::collections::HashSet<String> = lines
        .filter_map(|l| {
            l.split(',')
                .nth(col)
                .map(|s| s.trim_matches('"').to_string())
        })
        .collect();
    let mut copied = 0;
    for (key, folder) in figure_folders(&public) {
        if tics.contains(key.split('_').next().unwrap_or("")) {
            copy_dir(&folder, &out.join("figures").join(&key))?;
            copied += 1;
        }
    }
    zip_folder(&out, "figures", &out.join("figures.zip"))?;
    std::fs::remove_dir_all(out.join("figures"))?;
    println!(
        "{}: {} CTOI stars, {copied} figure folders",
        out.display(),
        tics.len()
    );
    Ok(())
}

fn call(
    method: &str,
    url: &str,
    token: &str,
    body: Option<(&[u8], &str)>,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let agent = ureq::agent();
    let auth = format!("Bearer {token}");
    let response = match method {
        "GET" => agent.get(url).set("Authorization", &auth).call()?,
        "DELETE" => agent.delete(url).set("Authorization", &auth).call()?,
        _ => {
            let (bytes, content_type) = body.unwrap_or((b"{}", "application/json"));
            let req = if method == "POST" {
                agent.post(url)
            } else {
                agent.put(url)
            };
            req.set("Authorization", &auth)
                .set("Content-Type", content_type)
                .send_bytes(bytes)?
        }
    };
    let text = response.into_string()?;
    Ok(if text.is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(&text)?
    })
}

pub fn zenodo(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let folder = PathBuf::from(
        args.iter()
            .position(|a| a == "--deposit")
            .map(|i| args[i + 1].clone())
            .ok_or("--deposit DIR is required")?,
    );
    let token = std::env::var("ZENODO_TOKEN").map_err(|_| "ZENODO_TOKEN is not set")?;
    let concept = std::env::var("ZENODO_RECORD").unwrap_or(CONCEPT_RECORD.to_string());
    let metadata: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(folder.join(".zenodo.json"))?)?;
    let version = metadata["version"].as_str().unwrap_or("").to_string();
    let published = call(
        "GET",
        &format!("{ZENODO_API}/records?q=conceptrecid:{concept}&all_versions=true&size=100"),
        &token,
        None,
    )?;
    let hits = published["hits"]["hits"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if let Some(same) = hits
        .iter()
        .find(|r| r["metadata"]["version"].as_str() == Some(version.as_str()))
    {
        println!(
            "version {version} is already published as {}; nothing to do",
            same["doi"]
        );
        return Ok(());
    }
    let draft = if let Some(latest) = hits
        .iter()
        .max_by_key(|r| r["created"].as_str().unwrap_or("").to_string())
    {
        let created = call(
            "POST",
            &format!(
                "{ZENODO_API}/deposit/depositions/{}/actions/newversion",
                latest["id"]
            ),
            &token,
            None,
        )?;
        call(
            "GET",
            created["links"]["latest_draft"]
                .as_str()
                .ok_or("no latest_draft link")?,
            &token,
            None,
        )?
    } else {
        let d = call(
            "POST",
            &format!("{ZENODO_API}/deposit/depositions"),
            &token,
            Some((b"{}", "application/json")),
        )?;
        println!(
            "new record: concept {} (set ZENODO_RECORD to it)",
            d["conceptrecid"]
        );
        d
    };
    let id = draft["id"].as_i64().ok_or("draft without id")?;
    for old in draft["files"].as_array().cloned().unwrap_or_default() {
        call(
            "DELETE",
            &format!(
                "{ZENODO_API}/deposit/depositions/{id}/files/{}",
                old["id"].as_str().unwrap_or("")
            ),
            &token,
            None,
        )?;
    }
    let bucket = draft["links"]["bucket"]
        .as_str()
        .ok_or("draft without bucket")?
        .to_string();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&folder)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().unwrap() != ".zenodo.json")
        .collect();
    files.sort();
    for path in files {
        let bytes = std::fs::read(&path)?;
        call(
            "PUT",
            &format!("{bucket}/{}", path.file_name().unwrap().to_string_lossy()),
            &token,
            Some((&bytes, "application/octet-stream")),
        )?;
    }
    let body = serde_json::to_vec(&serde_json::json!({"metadata": metadata}))?;
    call(
        "PUT",
        &format!("{ZENODO_API}/deposit/depositions/{id}"),
        &token,
        Some((&body, "application/json")),
    )?;
    let record = call(
        "POST",
        &format!("{ZENODO_API}/deposit/depositions/{id}/actions/publish"),
        &token,
        None,
    )?;
    println!("published {}", record["doi"]);
    Ok(())
}

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    match args.first().map(String::as_str) {
        Some("deposit") => deposit(&args[1..]),
        Some("zenodo") => zenodo(&args[1..]),
        _ => Err("usage: fbls catalog deposit [--public DIR] [--day D] [--out DIR] | fbls catalog zenodo --deposit DIR".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_date_is_well_formed() {
        let d = today();
        assert_eq!(d.len(), 10);
        assert!(d.starts_with("20"));
    }

    #[test]
    fn deposit_assembles_tables_metadata_and_archive() {
        let root = std::env::temp_dir().join(format!("fbls-release-{}", std::process::id()));
        let public = root.join("public");
        std::fs::create_dir_all(public.join("2026-01-02/figures/123_1")).unwrap();
        std::fs::write(public.join("ctoi.csv"), "TIC ID,Period (days)\n123,1.5\n").unwrap();
        std::fs::write(public.join("candidates.csv"), "tic,signal\n123,1\n").unwrap();
        std::fs::write(
            public.join("2026-01-02/figures/123_1/folded.csv"),
            "phase,flux\n0,1\n",
        )
        .unwrap();
        let cwd = std::env::current_dir().unwrap();
        std::fs::write(root.join("CITATION.cff"), "version: v0\ndate-released: 0\n").unwrap();
        std::fs::create_dir_all(root.join("paper")).unwrap();
        std::fs::write(
            root.join("paper/ReadMe.txt"),
            "Title\nAuthor\nVersion: <date>\n\nDescription here.\n",
        )
        .unwrap();
        std::env::set_current_dir(&root).unwrap();
        let result = deposit(&[
            "--public".into(),
            public.to_string_lossy().into(),
            "--day".into(),
            "2026-01-02".into(),
            "--out".into(),
            root.join("deposit").to_string_lossy().into(),
        ]);
        std::env::set_current_dir(cwd).unwrap();
        result.unwrap();
        let out = root.join("deposit/2026-01-02");
        for name in [
            "ctoi.csv",
            "candidates.csv",
            "CITATION.cff",
            "ReadMe.txt",
            ".zenodo.json",
            "figures.zip",
        ] {
            assert!(out.join(name).exists(), "{name}");
        }
        assert!(
            std::fs::read_to_string(out.join("CITATION.cff"))
                .unwrap()
                .contains("version: v2026-01-02")
        );
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(out.join(".zenodo.json")).unwrap())
                .unwrap();
        assert_eq!(meta["title"], "Title");
        assert_eq!(meta["description"], "Description here.");
        std::fs::remove_dir_all(root).unwrap();
    }
}
