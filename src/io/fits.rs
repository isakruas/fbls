//! FITS reader for SPOC light-curve and target-pixel products: 2880-byte
//! blocks, 80-character header cards, big-endian binary tables and images.
//! Only the column types these products use are decoded (D, E, J, I, B, K,
//! and fixed-size array cells).

#[derive(Debug, Clone)]
pub struct Header {
    pub cards: Vec<(String, String)>,
}

impl Header {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.cards
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
    pub fn f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.trim().parse::<f64>().ok())
    }
    pub fn i64(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(|v| v.trim().parse::<i64>().ok())
    }
    pub fn text(&self, key: &str) -> Option<String> {
        self.get(key)
            .map(|v| v.trim().trim_matches('\'').trim().to_string())
    }
}

#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub code: char,
    pub repeat: usize,
    pub offset: usize,
    pub width: usize,
}

/// One header-data unit. `data` is the raw payload (padding removed).
#[derive(Debug, Clone)]
pub struct Hdu {
    pub name: String,
    pub header: Header,
    pub data: Vec<u8>,
    pub columns: Vec<Column>,
    pub rows: usize,
    pub row_bytes: usize,
    pub naxes: Vec<usize>,
}

const BLOCK: usize = 2880;

fn parse_header(bytes: &[u8], at: usize) -> Result<(Header, usize), String> {
    let mut cards = Vec::new();
    let mut pos = at;
    loop {
        if pos + 80 > bytes.len() {
            return Err("truncated FITS header".into());
        }
        let card =
            std::str::from_utf8(&bytes[pos..pos + 80]).map_err(|_| "non-ASCII header card")?;
        pos += 80;
        let key = card[..8].trim().to_string();
        if key == "END" {
            break;
        }
        if key.is_empty() || key == "COMMENT" || key == "HISTORY" {
            continue;
        }
        let value = if &card[8..10] == "= " {
            let rest = &card[10..];
            if rest.trim_start().starts_with('\'') {
                let start = rest.find('\'').unwrap();
                let mut end = start + 1;
                let b = rest.as_bytes();
                while end < b.len() {
                    if b[end] == b'\'' {
                        if end + 1 < b.len() && b[end + 1] == b'\'' {
                            end += 2;
                            continue;
                        }
                        break;
                    }
                    end += 1;
                }
                rest[start..=end.min(rest.len() - 1)].to_string()
            } else {
                rest.split('/').next().unwrap_or("").trim().to_string()
            }
        } else {
            String::new()
        };
        cards.push((key, value));
    }
    let end = (pos - at).div_ceil(BLOCK) * BLOCK + at;
    Ok((Header { cards }, end))
}

fn code_width(code: char) -> Result<usize, String> {
    Ok(match code {
        'D' | 'K' => 8,
        'E' | 'J' => 4,
        'I' => 2,
        'B' | 'L' | 'A' => 1,
        other => return Err(format!("unsupported FITS column type {other}")),
    })
}

pub fn read_fits(path: &str) -> Result<Vec<Hdu>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let mut hdus = Vec::new();
    let mut pos = 0;
    while pos + BLOCK <= bytes.len() {
        let (header, data_start) = parse_header(&bytes, pos)?;
        let bitpix = header.i64("BITPIX").unwrap_or(8);
        let naxis = header.i64("NAXIS").unwrap_or(0) as usize;
        let naxes: Vec<usize> = (1..=naxis)
            .map(|i| header.i64(&format!("NAXIS{i}")).unwrap_or(0) as usize)
            .collect();
        let mut size: usize = if naxis == 0 {
            0
        } else {
            naxes.iter().product::<usize>() * (bitpix.unsigned_abs() as usize / 8)
        };
        let pcount = header.i64("PCOUNT").unwrap_or(0) as usize;
        size += pcount;
        let name = header.text("EXTNAME").unwrap_or_else(|| {
            if hdus.is_empty() {
                "PRIMARY".into()
            } else {
                String::new()
            }
        });
        let mut columns = Vec::new();
        let (mut rows, mut row_bytes) = (0, 0);
        if header.text("XTENSION").as_deref() == Some("BINTABLE") {
            row_bytes = naxes.first().copied().unwrap_or(0);
            rows = naxes.get(1).copied().unwrap_or(0);
            let fields = header.i64("TFIELDS").unwrap_or(0) as usize;
            let mut offset = 0;
            for i in 1..=fields {
                let form = header
                    .text(&format!("TFORM{i}"))
                    .ok_or(format!("missing TFORM{i}"))?;
                let code = form.chars().last().ok_or("empty TFORM")?;
                let repeat: usize = form[..form.len() - 1].parse().unwrap_or(1);
                let width = code_width(code)?;
                columns.push(Column {
                    name: header.text(&format!("TTYPE{i}")).unwrap_or_default(),
                    code,
                    repeat,
                    offset,
                    width,
                });
                offset += repeat * width;
            }
        }
        let end = data_start + size;
        if end > bytes.len() {
            return Err(format!("{path}: truncated data unit {name}"));
        }
        hdus.push(Hdu {
            name,
            header,
            data: bytes[data_start..end].to_vec(),
            columns,
            rows,
            row_bytes,
            naxes,
        });
        pos = data_start + size.div_ceil(BLOCK) * BLOCK;
    }
    Ok(hdus)
}

impl Hdu {
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }

    fn scalar(&self, col: &Column, row: usize, index: usize) -> f64 {
        let at = row * self.row_bytes + col.offset + index * col.width;
        let b = &self.data[at..at + col.width];
        match col.code {
            'D' => f64::from_be_bytes(b.try_into().unwrap()),
            'E' => f32::from_be_bytes(b.try_into().unwrap()) as f64,
            'J' => i32::from_be_bytes(b.try_into().unwrap()) as f64,
            'K' => i64::from_be_bytes(b.try_into().unwrap()) as f64,
            'I' => i16::from_be_bytes(b.try_into().unwrap()) as f64,
            _ => b[0] as f64,
        }
    }

    /// Scalar column as f64 (NaN stays NaN).
    pub fn column_f64(&self, name: &str) -> Result<Vec<f64>, String> {
        let col = self.column(name).ok_or(format!("missing column {name}"))?;
        Ok((0..self.rows).map(|r| self.scalar(col, r, 0)).collect())
    }

    /// Array column: `repeat` values per row, row-major.
    pub fn column_array(&self, name: &str) -> Result<(Vec<f64>, usize), String> {
        let col = self.column(name).ok_or(format!("missing column {name}"))?;
        let mut out = Vec::with_capacity(self.rows * col.repeat);
        for r in 0..self.rows {
            for i in 0..col.repeat {
                out.push(self.scalar(col, r, i));
            }
        }
        Ok((out, col.repeat))
    }
}

/// Quality flags excluded by default (attitude tweak, safe mode, coarse point,
/// Earth point, desaturation, manual exclude, impulsive outlier, bad
/// calibration exclude): bitmask 17087.
pub const DEFAULT_QUALITY_BITMASK: i64 = 17087;

/// Pixel coordinates of (ra, dec) under the gnomonic (TAN) projection of a
/// FITS header with CRPIX/CRVAL/CDELT/PC keywords; zero-based.
pub fn tan_world_to_pixel(h: &Header, ra: f64, dec: f64) -> Option<(f64, f64)> {
    let (crpix1, crpix2) = (h.f64("CRPIX1")?, h.f64("CRPIX2")?);
    let (crval1, crval2) = (h.f64("CRVAL1")?, h.f64("CRVAL2")?);
    let (cdelt1, cdelt2) = (
        h.f64("CDELT1").unwrap_or(1.0),
        h.f64("CDELT2").unwrap_or(1.0),
    );
    let (pc11, pc12, pc21, pc22) = match (
        h.f64("CD1_1"),
        h.f64("CD1_2"),
        h.f64("CD2_1"),
        h.f64("CD2_2"),
    ) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => (
            h.f64("PC1_1").unwrap_or(1.0) * cdelt1,
            h.f64("PC1_2").unwrap_or(0.0) * cdelt1,
            h.f64("PC2_1").unwrap_or(0.0) * cdelt2,
            h.f64("PC2_2").unwrap_or(1.0) * cdelt2,
        ),
    };
    let d2r = std::f64::consts::PI / 180.0;
    let (a0, d0) = (crval1 * d2r, crval2 * d2r);
    let (a, d) = (ra * d2r, dec * d2r);
    let cos_c = d0.sin() * d.sin() + d0.cos() * d.cos() * (a - a0).cos();
    if cos_c <= 0.0 {
        return None;
    }
    let x = d.cos() * (a - a0).sin() / cos_c / d2r;
    let y = (d0.cos() * d.sin() - d0.sin() * d.cos() * (a - a0).cos()) / cos_c / d2r;
    let det = pc11 * pc22 - pc12 * pc21;
    if det == 0.0 {
        return None;
    }
    let p1 = (pc22 * x - pc12 * y) / det;
    let p2 = (-pc21 * x + pc11 * y) / det;
    Some((p1 + crpix1 - 1.0, p2 + crpix2 - 1.0))
}

/// Light curve of one SPOC product after the default quality mask.
#[derive(Debug, Clone)]
pub struct LightCurve {
    pub tic: i64,
    pub sector: i64,
    pub time_bjd: Vec<f64>,
    pub flux: Vec<f64>,
    pub flux_err: Vec<f64>,
    pub centr1: Vec<f64>,
    pub centr2: Vec<f64>,
    pub crowdsap: Option<f64>,
    pub star: StarParameters,
    /// Target position on the CCD (column, row), from the aperture WCS.
    pub target_pixel: Option<(f64, f64)>,
    pub points_total: usize,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct StarParameters {
    pub radius: Option<f64>,
    pub teff: Option<f64>,
    pub logg: Option<f64>,
    pub tmag: Option<f64>,
}

pub fn read_light_curve(path: &str) -> Result<LightCurve, String> {
    let hdus = read_fits(path)?;
    let primary = hdus.first().ok_or("empty FITS")?;
    let table = hdus
        .iter()
        .find(|h| h.name == "LIGHTCURVE")
        .ok_or("no LIGHTCURVE extension")?;
    let aperture = hdus.iter().find(|h| h.name == "APERTURE");
    let bjd_ref =
        table.header.f64("BJDREFI").unwrap_or(0.0) + table.header.f64("BJDREFF").unwrap_or(0.0);
    let time = table.column_f64("TIME")?;
    let flux = table.column_f64("PDCSAP_FLUX")?;
    let flux_err = table.column_f64("PDCSAP_FLUX_ERR")?;
    let c1 = table.column_f64("MOM_CENTR1")?;
    let c2 = table.column_f64("MOM_CENTR2")?;
    let quality = table.column_f64("QUALITY")?;
    let keep: Vec<usize> = (0..table.rows)
        .filter(|&i| {
            (quality[i] as i64 & DEFAULT_QUALITY_BITMASK) == 0
                && time[i].is_finite()
                && flux[i].is_finite()
        })
        .collect();
    let pick = |v: &Vec<f64>| keep.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let target_pixel = aperture.and_then(|ap| {
        let (ra, dec) = (
            primary.header.f64("RA_OBJ")?,
            primary.header.f64("DEC_OBJ")?,
        );
        let (x, y) = tan_world_to_pixel(&ap.header, ra, dec)?;
        Some((x + ap.header.f64("CRVAL1P")?, y + ap.header.f64("CRVAL2P")?))
    });
    Ok(LightCurve {
        tic: primary.header.i64("TICID").unwrap_or(0),
        sector: primary.header.i64("SECTOR").unwrap_or(0),
        time_bjd: keep.iter().map(|&i| time[i] + bjd_ref).collect(),
        flux: pick(&flux),
        flux_err: pick(&flux_err),
        centr1: pick(&c1),
        centr2: pick(&c2),
        crowdsap: table.header.f64("CROWDSAP"),
        star: StarParameters {
            radius: primary.header.f64("RADIUS"),
            teff: primary.header.f64("TEFF"),
            logg: primary.header.f64("LOGG"),
            tmag: primary.header.f64("TESSMAG"),
        },
        target_pixel,
        points_total: table.rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(key: &str, value: &str) -> Vec<u8> {
        let mut s = format!("{key:<8}= {value:>20}");
        s.truncate(80);
        let mut b = s.into_bytes();
        b.resize(80, b' ');
        b
    }

    fn pad(mut v: Vec<u8>) -> Vec<u8> {
        let n = v.len().div_ceil(BLOCK) * BLOCK;
        v.resize(n, 0);
        v
    }

    #[test]
    fn binary_table_round_trip() {
        let mut header = Vec::new();
        for (k, v) in [("SIMPLE", "T"), ("BITPIX", "8"), ("NAXIS", "0")] {
            header.extend(card(k, v));
        }
        header.extend(b"END".iter().chain(std::iter::repeat(&b' ').take(77)));
        let mut primary = header.clone();
        primary.resize(primary.len().div_ceil(BLOCK) * BLOCK, b' ');
        let rows = 3usize;
        let mut ext = Vec::new();
        for (k, v) in [
            ("XTENSION", "'BINTABLE'"),
            ("BITPIX", "8"),
            ("NAXIS", "2"),
            ("NAXIS1", "16"),
            ("NAXIS2", "3"),
            ("PCOUNT", "0"),
            ("GCOUNT", "1"),
            ("TFIELDS", "3"),
            ("TTYPE1", "'TIME'"),
            ("TFORM1", "'D'"),
            ("TTYPE2", "'FLUX'"),
            ("TFORM2", "'E'"),
            ("TTYPE3", "'QUALITY'"),
            ("TFORM3", "'J'"),
            ("EXTNAME", "'LIGHTCURVE'"),
        ] {
            ext.extend(card(k, v));
        }
        ext.extend(b"END".iter().chain(std::iter::repeat(&b' ').take(77)));
        ext.resize(ext.len().div_ceil(BLOCK) * BLOCK, b' ');
        let mut data = Vec::new();
        for r in 0..rows {
            data.extend((1.5 + r as f64).to_be_bytes());
            data.extend((2.0f32 * r as f32).to_be_bytes());
            data.extend((r as i32 * 7).to_be_bytes());
        }
        let mut file = primary;
        file.extend(ext);
        file.extend(pad(data));
        let path = std::env::temp_dir().join("fbls_fits_test.fits");
        std::fs::write(&path, &file).unwrap();
        let hdus = read_fits(path.to_str().unwrap()).unwrap();
        assert_eq!(hdus.len(), 2);
        let t = &hdus[1];
        assert_eq!(t.name, "LIGHTCURVE");
        assert_eq!(t.column_f64("TIME").unwrap(), vec![1.5, 2.5, 3.5]);
        assert_eq!(t.column_f64("FLUX").unwrap(), vec![0.0, 2.0, 4.0]);
        assert_eq!(t.column_f64("QUALITY").unwrap(), vec![0.0, 7.0, 14.0]);
    }

    #[test]
    fn tan_projection_inverse_at_reference_point() {
        let h = Header {
            cards: vec![
                ("CRPIX1".into(), "6.635".into()),
                ("CRPIX2".into(), "6.366".into()),
                ("CRVAL1".into(), "26.2247".into()),
                ("CRVAL2".into(), "-67.9748".into()),
                ("CDELT1".into(), "-0.00584".into()),
                ("CDELT2".into(), "0.00584".into()),
                ("PC1_1".into(), "-0.2788".into()),
                ("PC1_2".into(), "0.9622".into()),
                ("PC2_1".into(), "0.9573".into()),
                ("PC2_2".into(), "0.2828".into()),
            ],
        };
        let (x, y) = tan_world_to_pixel(&h, 26.2247, -67.9748).unwrap();
        assert!((x - 5.635).abs() < 1e-9 && (y - 5.366).abs() < 1e-9);
    }
}
