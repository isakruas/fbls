//! Minimal NPZ reader: zip archive of NPY members, decoded to f64 on the host.
use std::collections::HashMap;
use std::io::Read;

#[derive(Debug, Clone)]
pub struct Array {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Array {
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

fn parse_npy(bytes: &[u8]) -> Result<Array, String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not an NPY file".into());
    }
    let (header_len, offset) = if bytes[6] == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&bytes[offset..offset + header_len]).map_err(|e| e.to_string())?;
    let descr = extract(header, "'descr':").ok_or("NPY header without descr")?;
    if extract(header, "'fortran_order':").map(|s| s.starts_with("True")).unwrap_or(false) {
        return Err("fortran_order arrays are not supported".into());
    }
    let shape_text = header.split("'shape':").nth(1).ok_or("NPY header without shape")?;
    let shape_text = shape_text.split('(').nth(1).and_then(|s| s.split(')').next()).ok_or("invalid shape")?;
    let shape: Vec<usize> = shape_text
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let count: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
    let body = &bytes[offset + header_len..];
    let descr = descr.trim_matches(|c| c == '\'' || c == '"');
    let little = !descr.starts_with('>');
    let kind = &descr[1..];
    let mut data = Vec::with_capacity(count);
    macro_rules! read_all {
        ($t:ty, $w:expr) => {{
            if body.len() < count * $w {
                return Err("truncated NPY body".into());
            }
            for i in 0..count {
                let mut b = [0u8; $w];
                b.copy_from_slice(&body[i * $w..(i + 1) * $w]);
                let v = if little { <$t>::from_le_bytes(b) } else { <$t>::from_be_bytes(b) };
                data.push(v as f64);
            }
        }};
    }
    match kind {
        "f8" => read_all!(f64, 8),
        "f4" => read_all!(f32, 4),
        "i8" => read_all!(i64, 8),
        "i4" => read_all!(i32, 4),
        "i2" => read_all!(i16, 2),
        "u1" | "b1" => data.extend(body[..count].iter().map(|&b| b as f64)),
        other => return Err(format!("unsupported NPY dtype: {other}")),
    }
    Ok(Array { shape, data })
}

fn extract<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let rest = header.split(key).nth(1)?.trim_start();
    let end = rest.find(',')?;
    Some(rest[..end].trim())
}

pub fn read_npz(path: &str) -> Result<HashMap<String, Array>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| format!("{path}: {e}"))?;
    let mut out = HashMap::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
        let name = entry.name().trim_end_matches(".npy").to_string();
        let mut bytes = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        out.insert(name.clone(), parse_npy(&bytes).map_err(|e| format!("{path}:{name}: {e}"))?);
    }
    Ok(out)
}
