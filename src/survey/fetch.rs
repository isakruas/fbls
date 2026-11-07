//! SPOC 2-minute light-curve products of a star from MAST: observation lookup
//! through the CAOM filtered service, download from the public mirror with the
//! MAST download endpoint as fallback, verification, and the provenance
//! sidecar next to each file.

use crate::io::fits::read_light_curve;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

const MAST_INVOKE: &str = "https://mast.stsci.edu/api/v0/invoke";
const MAST_DOWNLOAD: &str = "https://mast.stsci.edu/api/v0.1/Download/file?uri=";

#[derive(Debug, Clone)]
pub struct Product {
    pub sector: i64,
    pub filename: String,
    pub uri: String,
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().timeout_connect(std::time::Duration::from_secs(30)).timeout_read(std::time::Duration::from_secs(600)).build()
}

/// SPOC 120 s light-curve products of a TIC, one per sector.
pub fn products(tic: i64) -> Result<Vec<Product>, String> {
    let request = json!({
        "service": "Mast.Caom.Filtered", "format": "json",
        "params": {"columns": "obsid,obs_id,sequence_number,t_exptime,provenance_name,dataURL",
                   "filters": [{"paramName": "target_name", "values": [tic.to_string()]},
                               {"paramName": "obs_collection", "values": ["TESS"]},
                               {"paramName": "dataproduct_type", "values": ["timeseries"]}]}
    });
    let response: Value = agent()
        .post(MAST_INVOKE)
        .send_form(&[("request", &request.to_string())])
        .map_err(|e| format!("MAST query failed: {e}"))?
        .into_json()
        .map_err(|e| format!("MAST response: {e}"))?;
    if response["status"].as_str() != Some("COMPLETE") {
        return Err(format!("MAST status {}", response["status"]));
    }
    let mut out = Vec::new();
    for row in response["data"].as_array().cloned().unwrap_or_default() {
        if row["provenance_name"].as_str() != Some("SPOC") || row["t_exptime"].as_f64() != Some(120.0) {
            continue;
        }
        let Some(uri) = row["dataURL"].as_str() else { continue };
        if !uri.ends_with("_lc.fits") {
            continue;
        }
        out.push(Product { sector: row["sequence_number"].as_i64().unwrap_or(0), filename: uri.rsplit('/').next().unwrap().to_string(), uri: uri.to_string() });
    }
    out.sort_by_key(|p| p.sector);
    out.dedup_by_key(|p| p.sector);
    Ok(out)
}

pub fn mirror_url(tic: i64, sector: i64, filename: &str) -> String {
    let t = format!("{tic:016}");
    format!("https://stpubdata.s3.amazonaws.com/tess/public/tid/s{sector:04}/{}/{}/{}/{}/{filename}", &t[0..4], &t[4..8], &t[8..12], &t[12..16])
}

/// Downloads `url` into `output` through a temporary file; false on 404.
pub fn download(url: &str, output: &Path) -> Result<bool, String> {
    std::fs::create_dir_all(output.parent().unwrap()).map_err(|e| e.to_string())?;
    let part = output.with_extension("part");
    for attempt in 0..3 {
        match agent().get(url).call() {
            Ok(response) => {
                let mut file = std::fs::File::create(&part).map_err(|e| e.to_string())?;
                std::io::copy(&mut response.into_reader(), &mut file).map_err(|e| e.to_string())?;
                std::fs::rename(&part, output).map_err(|e| e.to_string())?;
                return Ok(true);
            }
            Err(ureq::Error::Status(404, _)) => return Ok(false),
            Err(e) => {
                if attempt == 2 {
                    return Err(format!("download failed {url}: {e}"));
                }
                std::thread::sleep(std::time::Duration::from_secs(3 * (attempt + 1)));
            }
        }
    }
    Ok(false)
}

fn sha256_hex(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

/// Fetches every SPOC light curve of a star into `root/mast/lc` and writes
/// `root/curves/<tic>/sector-N.fits` (a hard link or copy of the product) with
/// its provenance sidecar. Existing valid files are kept.
pub fn fetch_star(root: &Path, tic: i64, only_sectors: Option<&[i64]>, log: &mut dyn FnMut(&str)) -> Result<Vec<PathBuf>, String> {
    let mut written = Vec::new();
    for product in products(tic)? {
        if only_sectors.is_some_and(|s| !s.contains(&product.sector)) {
            continue;
        }
        let folder = root.join("curves").join(tic.to_string());
        let fits_path = root.join("mast").join("lc").join(&product.filename);
        let target = folder.join(format!("sector-{}.fits", product.sector));
        let sidecar = folder.join(format!("sector-{}.json", product.sector));
        if (target.exists() || folder.join(format!("sector-{}.npz", product.sector)).exists()) && sidecar.exists() {
            log(&format!("TIC {tic} sector {} present", product.sector));
            continue;
        }
        if !fits_path.exists() || read_light_curve(fits_path.to_str().unwrap()).is_err() {
            let mut source = "mirror";
            let ok = download(&mirror_url(tic, product.sector, &product.filename), &fits_path).unwrap_or(false) || {
                source = "mast";
                download(&format!("{MAST_DOWNLOAD}{}", product.uri), &fits_path)?
            };
            if !ok {
                log(&format!("TIC {tic} sector {} missing (404)", product.sector));
                continue;
            }
            if let Err(e) = read_light_curve(fits_path.to_str().unwrap()) {
                std::fs::remove_file(&fits_path).ok();
                log(&format!("TIC {tic} sector {} invalid FITS after transfer: {e}", product.sector));
                continue;
            }
            log(&format!("TIC {tic} sector {} downloaded from {source}", product.sector));
        }
        let lc = read_light_curve(fits_path.to_str().unwrap())?;
        if lc.tic != tic {
            return Err(format!("product {} belongs to TIC {}, not {tic}", product.filename, lc.tic));
        }
        std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
        if !target.exists() && std::fs::hard_link(&fits_path, &target).is_err() {
            std::fs::copy(&fits_path, &target).map_err(|e| e.to_string())?;
        }
        let meta = json!({
            "version": 3, "tic": tic, "sector": product.sector, "author": "SPOC", "exptime": 120,
            "quality_bitmask": "default", "flux": "PDCSAP_FLUX", "centroid": "MOM_CENTR", "time": "BJD TDB",
            "crowdsap": lc.crowdsap, "star": lc.star, "target_pixel": lc.target_pixel,
            "product": product.filename, "fits": format!("mast/lc/{}", product.filename),
            "fits_sha256": sha256_hex(&fits_path)?, "fits_size_bytes": std::fs::metadata(&fits_path).map(|m| m.len()).unwrap_or(0),
            "points": lc.time_bjd.len(), "points_total": lc.points_total,
            "time_first_bjd": lc.time_bjd.first(), "time_last_bjd": lc.time_bjd.last(),
            "retrieved_at_utc": now_utc(),
        });
        std::fs::write(&sidecar, serde_json::to_string_pretty(&meta).unwrap() + "\n").map_err(|e| e.to_string())?;
        written.push(target);
    }
    Ok(written)
}

pub fn now_utc() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let days = secs / 86400;
    let (h, m, s) = ((secs % 86400) / 3600, (secs % 3600) / 60, secs % 60);
    let (y, mo, d) = civil_from_days(days as i64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Gregorian date of a day count since 1970-01-01.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

pub fn today_utc() -> String {
    now_utc()[..10].to_string()
}
