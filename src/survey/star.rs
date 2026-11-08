//! Light curves of one star from the data directory: sector files
//! (`sector-N.npz` with a provenance sidecar, or `sector-N.fits`), the sector
//! window a survey searches, and the per-sector metadata the vetting needs.

use crate::io::fits::{StarParameters, read_light_curve};
use crate::io::npz::read_npz;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Sectors searched per star: eight sectors hold two transits of any period
/// up to 100 days.
pub const SEARCH_SECTORS: usize = 8;
/// Sectors farther apart than this (about a year) are never combined.
pub const MAX_GAP_SECTORS: i64 = 13;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Sidecar {
    pub version: Option<i64>,
    pub tic: Option<i64>,
    pub sector: Option<i64>,
    pub crowdsap: Option<f64>,
    #[serde(default)]
    pub star: StarParameters,
    pub target_pixel: Option<(f64, f64)>,
    pub product: Option<String>,
    pub fits: Option<String>,
    pub fits_sha256: Option<String>,
    pub points: Option<usize>,
    pub time_first_bjd: Option<f64>,
    pub time_last_bjd: Option<f64>,
    pub retrieved_at_utc: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Star {
    pub tic: i64,
    pub time_bjd: Vec<f64>,
    pub flux: Vec<f64>,
    pub flux_err: Vec<f64>,
    pub centr1: Vec<f64>,
    pub centr2: Vec<f64>,
    pub sector: Vec<i64>,
    pub meta: BTreeMap<i64, Sidecar>,
    pub star: StarParameters,
    pub crowdsap: Option<f64>,
}

/// The sectors to search: the largest group whose consecutive gaps are at
/// most MAX_GAP_SECTORS (shortest span, then most recent, on ties), reduced
/// to the SEARCH_SECTORS spanning the shortest time.
pub fn search_window(mut sectors: Vec<i64>, count: usize, max_gap: i64) -> Vec<i64> {
    sectors.sort();
    sectors.dedup();
    if sectors.is_empty() {
        return Vec::new();
    }
    let mut groups: Vec<Vec<i64>> = vec![vec![sectors[0]]];
    for w in sectors.windows(2) {
        if w[1] - w[0] <= max_gap {
            groups.last_mut().unwrap().push(w[1]);
        } else {
            groups.push(vec![w[1]]);
        }
    }
    let best = groups
        .into_iter()
        .max_by_key(|g| (g.len(), -(g[g.len() - 1] - g[0]), g[g.len() - 1]))
        .unwrap();
    if best.len() <= count {
        return best;
    }
    (0..=best.len() - count)
        .map(|i| best[i..i + count].to_vec())
        .min_by_key(|w| (w[count - 1] - w[0], -w[count - 1]))
        .unwrap()
}

pub fn sector_files(folder: &Path) -> BTreeMap<i64, PathBuf> {
    let mut out = BTreeMap::new();
    if let Ok(entries) = std::fs::read_dir(folder) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(rest) = name.strip_prefix("sector-")
                && let Some(num) = rest.strip_suffix(".npz").or(rest.strip_suffix(".fits"))
                && let Ok(s) = num.parse::<i64>()
            {
                let prefer_npz = name.ends_with(".npz");
                out.entry(s)
                    .and_modify(|p: &mut PathBuf| {
                        if prefer_npz {
                            *p = e.path()
                        }
                    })
                    .or_insert(e.path());
            }
        }
    }
    out
}

pub fn read_sidecar(path: &Path) -> Sidecar {
    let json = path.with_extension("json");
    std::fs::read_to_string(&json)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

/// Loads the given sectors (all when None) of a star under `root/curves/<tic>`.
pub fn load_star(root: &Path, tic: i64, sectors: Option<usize>) -> Result<Star, String> {
    let folder = root.join("curves").join(tic.to_string());
    let files = sector_files(&folder);
    if files.is_empty() {
        return Err(format!("no light curves for TIC {tic}"));
    }
    let chosen = match sectors {
        None => files.keys().cloned().collect(),
        Some(n) => search_window(files.keys().cloned().collect(), n, MAX_GAP_SECTORS),
    };
    let mut star = Star {
        tic,
        time_bjd: Vec::new(),
        flux: Vec::new(),
        flux_err: Vec::new(),
        centr1: Vec::new(),
        centr2: Vec::new(),
        sector: Vec::new(),
        meta: BTreeMap::new(),
        star: StarParameters::default(),
        crowdsap: None,
    };
    for s in chosen {
        let path = &files[&s];
        let mut sidecar = read_sidecar(path);
        if path.extension().and_then(|e| e.to_str()) == Some("fits") {
            let lc = read_light_curve(path.to_str().unwrap())?;
            let n = lc.time_bjd.len();
            star.time_bjd.extend(lc.time_bjd);
            star.flux.extend(lc.flux);
            star.flux_err.extend(lc.flux_err);
            star.centr1.extend(lc.centr1);
            star.centr2.extend(lc.centr2);
            star.sector.extend(std::iter::repeat_n(s, n));
            sidecar.crowdsap = sidecar.crowdsap.or(lc.crowdsap);
            sidecar.target_pixel = sidecar.target_pixel.or(lc.target_pixel);
            if sidecar.star.radius.is_none() {
                sidecar.star = lc.star;
            }
        } else {
            let data = read_npz(path.to_str().unwrap())?;
            let t = data.get("time_bjd").ok_or("npz without time_bjd")?;
            let n = t.len();
            star.time_bjd.extend_from_slice(&t.data);
            star.flux
                .extend_from_slice(&data.get("flux").ok_or("npz without flux")?.data);
            star.flux_err.extend(
                data.get("flux_err")
                    .map(|a| a.data.clone())
                    .unwrap_or(vec![f64::NAN; n]),
            );
            star.centr1.extend(
                data.get("centr1")
                    .map(|a| a.data.clone())
                    .unwrap_or(vec![f64::NAN; n]),
            );
            star.centr2.extend(
                data.get("centr2")
                    .map(|a| a.data.clone())
                    .unwrap_or(vec![f64::NAN; n]),
            );
            star.sector.extend(std::iter::repeat_n(s, n));
        }
        star.meta.insert(s, sidecar);
    }
    let mut order: Vec<usize> = (0..star.time_bjd.len()).collect();
    order.sort_by(|&a, &b| star.time_bjd[a].partial_cmp(&star.time_bjd[b]).unwrap());
    let reorder = |v: &Vec<f64>| order.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    star.flux = reorder(&star.flux);
    star.flux_err = reorder(&star.flux_err);
    star.centr1 = reorder(&star.centr1);
    star.centr2 = reorder(&star.centr2);
    star.sector = order.iter().map(|&i| star.sector[i]).collect();
    star.time_bjd = reorder(&star.time_bjd);
    star.star = star
        .meta
        .values()
        .find(|m| m.star.radius.is_some())
        .or(star.meta.values().next())
        .map(|m| m.star.clone())
        .unwrap_or_default();
    let crowd: Vec<f64> = star.meta.values().filter_map(|m| m.crowdsap).collect();
    star.crowdsap = if crowd.is_empty() {
        None
    } else {
        Some(super::numeric::median(&crowd))
    };
    Ok(star)
}

pub fn downloaded_stars(root: &Path) -> Vec<i64> {
    let mut out: Vec<i64> = std::fs::read_dir(root.join("curves"))
        .map(|d| {
            d.flatten()
                .filter_map(|e| e.file_name().to_string_lossy().parse::<i64>().ok())
                .filter(|t| !sector_files(&root.join("curves").join(t.to_string())).is_empty())
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_prefers_the_largest_close_group() {
        assert_eq!(
            search_window(vec![1, 2, 3, 30, 31, 32, 33], 8, 13),
            vec![30, 31, 32, 33]
        );
        assert_eq!(
            search_window((1..=12).collect(), 8, 13),
            (5..=12).collect::<Vec<_>>()
        );
        assert_eq!(search_window(vec![5, 40], 8, 13), vec![40]);
    }
}
