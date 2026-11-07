//! Difference images from SPOC target pixel files: which pixel the transit
//! comes from. The out-of-transit image minus the in-transit image shows
//! where the light was lost; a Gaussian fit locates the source and its offset
//! from the target position (WCS of the same stamp) is the pixel-level source
//! test. Target pixel files are fetched on demand for up to three sectors
//! that contain transits.

use super::fetch::{download, mirror_url};
use super::numeric::{chi2_sigma_2, fit_errors_unit, least_squares, median, std};
use super::star::Star;
use crate::io::fits::{DEFAULT_QUALITY_BITMASK, read_fits, tan_world_to_pixel};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const MAX_SECTORS: usize = 3;
pub const MIN_DIFFERENCE_SNR: f64 = 5.0;
pub const OFFSET_SIGMA: f64 = 4.0;
pub const OFFSET_PIXELS: f64 = 0.5;
pub const SYSTEMATIC_PIXELS: f64 = 0.1;
pub const PIXEL_ARCSEC: f64 = 21.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectorDetail {
    pub sector: i64,
    pub transits: usize,
    pub difference_snr: Option<f64>,
    pub used: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dx_pixels: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dy_pixels: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dx_err: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dy_err: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PixelResult {
    pub sigma: Option<f64>,
    pub offset_pixels: Option<f64>,
    pub offset_arcsec: Option<f64>,
    pub sectors: Vec<SectorDetail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Local target pixel file of a sector, downloaded on first use.
pub fn tpf_file(star: &Star, sector: i64, root: &Path) -> Result<PathBuf, String> {
    let product = star.meta.get(&sector).and_then(|m| m.product.clone()).ok_or(format!("no product name for sector {sector}"))?;
    let name = Path::new(&product).file_name().unwrap().to_string_lossy().replace("_lc.fits", "_tp.fits");
    let path = root.join("mast").join("tp").join(&name);
    if !path.exists() {
        let ok = download(&mirror_url(star.tic, sector, &name), &path)? || download(&format!("https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:TESS/product/{name}"), &path)?;
        if !ok {
            return Err(format!("target pixel file not found for sector {sector}"));
        }
    }
    Ok(path)
}

pub struct Tpf {
    pub time: Vec<f64>,
    /// Cadence-major images, each ny x nx.
    pub flux: Vec<Vec<f64>>,
    pub ny: usize,
    pub nx: usize,
    pub target: (f64, f64),
}

pub fn read_tpf(path: &Path) -> Result<Tpf, String> {
    let hdus = read_fits(path.to_str().unwrap())?;
    let primary = &hdus[0];
    let table = hdus.iter().find(|h| h.name == "PIXELS").or(hdus.get(1)).ok_or("no PIXELS extension")?;
    let aperture = hdus.iter().find(|h| h.name == "APERTURE").ok_or("no APERTURE extension")?;
    let bjd_ref = table.header.f64("BJDREFI").unwrap_or(0.0) + table.header.f64("BJDREFF").unwrap_or(0.0);
    let time = table.column_f64("TIME")?;
    let quality = table.column_f64("QUALITY")?;
    let (flux, per_row) = table.column_array("FLUX")?;
    let (nx, ny) = (aperture.naxes.first().copied().unwrap_or(0), aperture.naxes.get(1).copied().unwrap_or(0));
    if nx * ny != per_row {
        return Err("pixel stamp shape mismatch".into());
    }
    let (ra, dec) = (primary.header.f64("RA_OBJ").ok_or("RA_OBJ")?, primary.header.f64("DEC_OBJ").ok_or("DEC_OBJ")?);
    let target = tan_world_to_pixel(&aperture.header, ra, dec).ok_or("target outside the stamp projection")?;
    let mut out = Tpf { time: Vec::new(), flux: Vec::new(), ny, nx, target };
    for r in 0..table.rows {
        let image = &flux[r * per_row..(r + 1) * per_row];
        if (quality[r] as i64 & DEFAULT_QUALITY_BITMASK) == 0 && time[r].is_finite() && image.iter().any(|v| v.is_finite()) {
            out.time.push(time[r] + bjd_ref);
            out.flux.push(image.to_vec());
        }
    }
    Ok(out)
}

fn nanmean_images(images: &[&Vec<f64>], n: usize) -> Vec<f64> {
    (0..n)
        .map(|p| {
            let vals: Vec<f64> = images.iter().map(|img| img[p]).filter(|v| v.is_finite()).collect();
            if vals.is_empty() { f64::NAN } else { vals.iter().sum::<f64>() / vals.len() as f64 }
        })
        .collect()
}

fn nanstd_images(images: &[&Vec<f64>], n: usize, ddof: usize) -> Vec<f64> {
    (0..n)
        .map(|p| {
            let vals: Vec<f64> = images.iter().map(|img| img[p]).filter(|v| v.is_finite()).collect();
            std(&vals, ddof)
        })
        .collect()
}

pub struct DifferenceImage {
    pub difference: Vec<f64>,
    pub error: Vec<f64>,
    pub oot: Vec<f64>,
    pub transits: usize,
}

/// Mean (out of transit - in transit) image over the covered transits with its per-pixel error.
pub fn difference_image(tpf: &Tpf, centers: &[f64], duration: f64) -> Option<DifferenceImage> {
    let n = tpf.ny * tpf.nx;
    let mut diffs: Vec<Vec<f64>> = Vec::new();
    let mut errors: Vec<Vec<f64>> = Vec::new();
    let mut oot: Vec<Vec<f64>> = Vec::new();
    for &c in centers {
        let inside: Vec<&Vec<f64>> = tpf.time.iter().zip(&tpf.flux).filter(|(t, _)| (**t - c).abs() < duration / 4.0).map(|(_, f)| f).collect();
        let outside_idx: Vec<usize> = (0..tpf.time.len()).filter(|&i| { let d = (tpf.time[i] - c).abs(); d > 0.75 * duration && d < 1.25 * duration }).collect();
        if inside.len() < 3 || outside_idx.len() < 6 || !outside_idx.iter().any(|&i| tpf.time[i] < c) || !outside_idx.iter().any(|&i| tpf.time[i] > c) {
            continue;
        }
        let outside: Vec<&Vec<f64>> = outside_idx.iter().map(|&i| &tpf.flux[i]).collect();
        let out = nanmean_images(&outside, n);
        let inn = nanmean_images(&inside, n);
        let sd = nanstd_images(&outside, n, 0);
        diffs.push(out.iter().zip(&inn).map(|(a, b)| a - b).collect());
        let factor = (1.0 / inside.len() as f64 + 1.0 / outside.len() as f64).sqrt();
        errors.push(sd.iter().map(|s| s * factor).collect());
        oot.push(out);
    }
    if diffs.is_empty() {
        return None;
    }
    let refs: Vec<&Vec<f64>> = diffs.iter().collect();
    let difference = nanmean_images(&refs, n);
    let error = if diffs.len() >= 3 {
        let sd = nanstd_images(&refs, n, 1);
        (0..n)
            .map(|p| {
                let a = sd[p] / (diffs.len() as f64).sqrt();
                let b = (errors.iter().map(|e| e[p] * e[p]).filter(|v| v.is_finite()).sum::<f64>() / errors.len() as f64 / diffs.len() as f64).sqrt();
                a.max(b)
            })
            .collect()
    } else {
        (0..n).map(|p| errors.iter().map(|e| if e[p].is_finite() { e[p] * e[p] } else { 0.0 }).sum::<f64>().sqrt() / diffs.len() as f64).collect()
    };
    let oot_refs: Vec<&Vec<f64>> = oot.iter().collect();
    Some(DifferenceImage { difference, error, oot: nanmean_images(&oot_refs, n), transits: diffs.len() })
}

pub struct Source {
    pub x: f64,
    pub y: f64,
    pub x_err: f64,
    pub y_err: f64,
    pub snr: f64,
}

/// Weighted Gaussian plus constant fit of the difference image: source
/// position, errors and amplitude significance in pixels.
pub fn fit_source(image: &[f64], error: &[f64], ny: usize, nx: usize) -> Option<Source> {
    let good: Vec<usize> = (0..ny * nx).filter(|&p| image[p].is_finite() && error[p].is_finite() && error[p] > 0.0).collect();
    if good.len() < 9 {
        return None;
    }
    let z: Vec<f64> = good.iter().map(|&p| image[p]).collect();
    let w: Vec<f64> = good.iter().map(|&p| 1.0 / error[p]).collect();
    let xg: Vec<f64> = good.iter().map(|&p| (p % nx) as f64).collect();
    let yg: Vec<f64> = good.iter().map(|&p| (p / nx) as f64).collect();
    let j = *good.iter().max_by(|&&a, &&b| image[a].partial_cmp(&image[b]).unwrap()).unwrap();
    let start = [image[j].max(1e-12), (j % nx) as f64, (j / nx) as f64, 0.9, median(&z)];
    let lower = [0.0, -0.5, -0.5, 0.3, f64::NEG_INFINITY];
    let upper = [f64::INFINITY, nx as f64 - 0.5, ny as f64 - 0.5, 3.0, f64::INFINITY];
    let (p, r, jac) = least_squares(
        |p| (0..z.len()).map(|i| (p[0] * (-((xg[i] - p[1]).powi(2) + (yg[i] - p[2]).powi(2)) / (2.0 * p[3] * p[3])).exp() + p[4] - z[i]) * w[i]).collect(),
        &start,
        &lower,
        &upper,
    );
    let dof = good.len().saturating_sub(5).max(1) as f64;
    let chi2 = r.iter().map(|v| v * v).sum::<f64>() / dof;
    // Weighted fit: the covariance is inv(JtJ) scaled by max(reduced chi-square, 1).
    let errors: Vec<f64> = fit_errors_unit(&jac).iter().map(|e| e * chi2.max(1.0).sqrt()).collect();
    if !errors[0].is_finite() {
        return None;
    }
    Some(Source { x: p[1], y: p[2], x_err: errors[1], y_err: errors[2], snr: if errors[0] > 0.0 { p[0] / errors[0] } else { 0.0 } })
}

pub fn transit_centers(time: &[f64], period: Option<f64>, epoch: f64) -> Vec<f64> {
    match period {
        Some(p) if p.is_finite() => {
            let k0 = ((time[0] - epoch) / p - 0.5).ceil() as i64;
            let k1 = ((time[time.len() - 1] - epoch) / p + 0.5).floor() as i64;
            (k0..=k1).map(|k| epoch + k as f64 * p).collect()
        }
        _ => vec![epoch],
    }
}

pub struct SectorImage {
    pub sector: i64,
    pub transits: usize,
    pub difference: Vec<f64>,
    pub error: Vec<f64>,
    pub oot: Vec<f64>,
    pub ny: usize,
    pub nx: usize,
    pub target: (f64, f64),
    pub source: Option<Source>,
}

/// Difference image, target position and fitted source per sector with
/// transits (up to three).
pub fn sector_images(star: &Star, period: Option<f64>, epoch: f64, duration_hours: f64, root: &Path, epochs: Option<&[f64]>) -> Result<Vec<SectorImage>, String> {
    let duration = duration_hours / 24.0;
    let times: Vec<f64> = epochs.map(|e| e.to_vec()).unwrap_or_else(|| transit_centers(&star.time_bjd, period, epoch));
    let mut sectors: Vec<i64> = star.sector.clone();
    sectors.sort();
    sectors.dedup();
    let with_transits: Vec<i64> = sectors
        .into_iter()
        .filter(|&s| {
            let t: Vec<f64> = (0..star.time_bjd.len()).filter(|&i| star.sector[i] == s).map(|i| star.time_bjd[i]).collect();
            times.iter().any(|&c| c > t[0] - duration && c < t[t.len() - 1] + duration)
        })
        .collect();
    let chosen = &with_transits[with_transits.len().saturating_sub(MAX_SECTORS)..];
    let mut out = Vec::new();
    for &s in chosen {
        let tpf = read_tpf(&tpf_file(star, s, root)?)?;
        if tpf.time.is_empty() {
            continue;
        }
        let centers: Vec<f64> = times.iter().cloned().filter(|&c| c > tpf.time[0] && c < tpf.time[tpf.time.len() - 1]).collect();
        let Some(image) = difference_image(&tpf, &centers, duration) else { continue };
        let source = fit_source(&image.difference, &image.error, tpf.ny, tpf.nx);
        out.push(SectorImage { sector: s, transits: image.transits, difference: image.difference, error: image.error, oot: image.oot, ny: tpf.ny, nx: tpf.nx, target: tpf.target, source });
    }
    Ok(out)
}

/// Difference-image source offset from the target, combined over up to three sectors.
pub fn pixel_test(star: &Star, period: Option<f64>, epoch: f64, duration_hours: f64, root: &Path, epochs: Option<&[f64]>) -> PixelResult {
    let images = match sector_images(star, period, epoch, duration_hours, root, epochs) {
        Ok(i) => i,
        Err(e) => return PixelResult { sigma: None, offset_pixels: None, offset_arcsec: None, sectors: Vec::new(), error: Some(e) },
    };
    let mut details = Vec::new();
    for image in &images {
        match &image.source {
            Some(src) if src.snr >= MIN_DIFFERENCE_SNR => details.push(SectorDetail {
                sector: image.sector,
                transits: image.transits,
                difference_snr: Some(src.snr),
                used: true,
                dx_pixels: Some(src.x - image.target.0),
                dy_pixels: Some(src.y - image.target.1),
                dx_err: Some(src.x_err.hypot(SYSTEMATIC_PIXELS)),
                dy_err: Some(src.y_err.hypot(SYSTEMATIC_PIXELS)),
            }),
            other => details.push(SectorDetail { sector: image.sector, transits: image.transits, difference_snr: other.as_ref().map(|s| s.snr), used: false, dx_pixels: None, dy_pixels: None, dx_err: None, dy_err: None }),
        }
    }
    let used: Vec<&SectorDetail> = details.iter().filter(|d| d.used).collect();
    if used.is_empty() {
        return PixelResult { sigma: None, offset_pixels: None, offset_arcsec: None, sectors: details, error: None };
    }
    let combine = |v: &dyn Fn(&SectorDetail) -> f64, e: &dyn Fn(&SectorDetail) -> f64| {
        let w: f64 = used.iter().map(|d| 1.0 / e(d).powi(2)).sum();
        let s: f64 = used.iter().map(|d| v(d) / e(d).powi(2)).sum();
        (s / w, 1.0 / w.sqrt())
    };
    let (dx, ex) = combine(&|d| d.dx_pixels.unwrap(), &|d| d.dx_err.unwrap());
    let (dy, ey) = combine(&|d| d.dy_pixels.unwrap(), &|d| d.dy_err.unwrap());
    let chi2 = (dx / ex).powi(2) + (dy / ey).powi(2);
    let offset = dx.hypot(dy);
    PixelResult { sigma: Some(chi2_sigma_2(chi2)), offset_pixels: Some(offset), offset_arcsec: Some(offset * PIXEL_ARCSEC), sectors: details, error: None }
}

/// Pixel-level flag on the metrics JSON: a significant offset of half a
/// pixel or more fails; the verdict is recomputed from the flags.
pub fn apply_pixel_test(metrics: &mut serde_json::Value, result: &PixelResult) {
    metrics["pixel_test"] = serde_json::to_value(result).unwrap_or(serde_json::Value::Null);
    if let (Some(sigma), Some(offset)) = (result.sigma, result.offset_pixels) {
        let flag = if sigma > OFFSET_SIGMA && offset > OFFSET_PIXELS {
            Some(super::vet::FAIL)
        } else if sigma > 3.0 && offset > 0.6 * OFFSET_PIXELS {
            Some(super::vet::WARN)
        } else {
            None
        };
        if let Some(f) = flag {
            metrics["flags"]["pixel_offset"] = serde_json::json!(f);
        }
    }
    let failed = metrics["flags"].as_object().is_some_and(|o| o.values().any(|v| v == super::vet::FAIL));
    metrics["verdict"] = serde_json::json!(if failed { super::vet::FAIL } else { "pass" });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_source_is_located() {
        let (ny, nx) = (11, 11);
        let (x0, y0) = (6.3, 4.7);
        let image: Vec<f64> = (0..ny * nx).map(|p| { let (x, y) = ((p % nx) as f64, (p / nx) as f64); 100.0 * (-((x - x0).powi(2) + (y - y0).powi(2)) / (2.0 * 0.8f64.powi(2))).exp() + 2.0 }).collect();
        let error = vec![1.0; ny * nx];
        let s = fit_source(&image, &error, ny, nx).unwrap();
        assert!((s.x - x0).abs() < 0.02 && (s.y - y0).abs() < 0.02, "{} {}", s.x, s.y);
        assert!(s.snr > 50.0);
    }
}
