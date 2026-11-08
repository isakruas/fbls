//! Single-event scan: every template above threshold at one trial period
//! longer than the observing baseline, so no folding occurs.

use super::kernels::{CUM_BLOCK, EVENT_STRIDE, FOLD_BLOCK, NF, TILE};
use super::scan::{DeviceReady, Gpu, Row};
use crate::config::SearchConfig;
use crate::prep::curve::FamilyState;
use cuda_core::{DeviceBuffer, LaunchConfig};
use cuda_device::atomic::{DeviceAtomicF64, DeviceAtomicU32};

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct EventStats {
    pub templates_scanned: u64,
    pub threshold_survivors: usize,
    pub overflow: bool,
}

pub fn single_event_search(
    gpu: &Gpu,
    state: &FamilyState,
    cfg: &SearchConfig,
    capacity: usize,
) -> Result<(Vec<Row>, EventStats), Box<dyn std::error::Error>> {
    let mut ds: Vec<(f64, f64)> = state
        .durations
        .iter()
        .cloned()
        .zip(state.rho.iter().cloned())
        .collect();
    ds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let durations: Vec<f64> = ds.iter().map(|d| d.0).collect();
    let rho: Vec<f64> = ds.iter().map(|d| d.1).collect();
    let baseline = *state.ready.time.last().unwrap();
    let trial = baseline + 2.0 * durations[durations.len() - 1] + 1e-6;
    let nb = ((cfg.phase_oversample as f64 * trial / durations[0]).ceil() as usize).max(8);
    let dev = DeviceReady::upload(gpu, &state.ready)?;
    let stream = &gpu.stream;
    let layers = cfg.shape_layers;
    let factors: Vec<f64> = cfg
        .ingress_fractions
        .iter()
        .flat_map(|&ing| {
            (0..layers).map(move |l| {
                1.0 - 2.0
                    * ing
                    * if layers > 1 {
                        l as f64 / (layers - 1) as f64
                    } else {
                        0.0
                    }
            })
        })
        .collect();
    let d_dur = DeviceBuffer::from_host(stream, &durations)?;
    let d_rho = DeviceBuffer::from_host(stream, &rho)?;
    let d_ing = DeviceBuffer::from_host(stream, &cfg.ingress_fractions)?;
    let d_fac = DeviceBuffer::from_host(stream, &factors)?;
    let scale = nb as f64 / trial;
    let d_scales = DeviceBuffer::from_host(stream, &[scale])?;
    let folded = DeviceBuffer::<f64>::zeroed(stream, nb * NF)?.wrap_elem::<DeviceAtomicF64>();
    let mut cum = DeviceBuffer::<f64>::zeroed(stream, (2 * nb + 1) * NF)?;
    let mut cum32 = DeviceBuffer::<f32>::zeroed(stream, 2 * (2 * nb + 1) * NF)?;
    let mut records = DeviceBuffer::<f64>::zeroed(stream, capacity * EVENT_STRIDE)?;
    let counter = DeviceBuffer::<u32>::zeroed(stream, 1)?.wrap_elem::<DeviceAtomicU32>();
    let mut count_out = DeviceBuffer::<u32>::zeroed(stream, 1)?;
    let tiles = nb.div_ceil(TILE);
    // SAFETY: grid shapes match each kernel's indexing and every buffer covers the kernel's accesses.
    unsafe {
        gpu.module.fold(
            stream,
            LaunchConfig {
                grid_dim: ((dev.n as u32).div_ceil(FOLD_BLOCK), 1, 1),
                block_dim: (FOLD_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            &dev.time,
            &dev.fields,
            &d_scales,
            &folded,
            dev.n as u32,
            nb as u32,
            1,
        )?;
        gpu.module.cumulative(
            stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (CUM_BLOCK as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            &folded,
            &mut cum,
            &mut cum32,
            nb as u32,
        )?;
        gpu.module.scan_all(
            stream,
            LaunchConfig {
                grid_dim: (tiles as u32, 1, 1),
                block_dim: (TILE as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            &cum,
            &d_dur,
            &d_rho,
            &d_ing,
            &d_fac,
            scale,
            dev.baseline_norm,
            cfg.threshold,
            dev.n as u32,
            nb as u32,
            durations.len() as u32,
            cfg.ingress_fractions.len() as u32,
            layers as u32,
            capacity as u32,
            &counter,
            &mut records,
        )?;
    }
    unsafe {
        gpu.module.read_counter(
            stream,
            LaunchConfig::for_num_elems(1),
            &counter,
            &mut count_out,
        )?;
    }
    let count = count_out.to_host_vec(stream)?[0] as usize;
    let mut stats = EventStats {
        templates_scanned: (nb * durations.len() * cfg.ingress_fractions.len()) as u64,
        threshold_survivors: count,
        overflow: false,
    };
    if count > capacity {
        stats.overflow = true;
        return Ok((Vec::new(), stats));
    }
    let rec = records.to_host_vec(stream)?;
    let rows = (0..count)
        .map(|i| {
            let r = &rec[i * EVENT_STRIDE..(i + 1) * EVENT_STRIDE];
            Row {
                period: f64::INFINITY,
                epoch: (r[0] + r[1] / 2.0) / scale + dev.origin,
                duration: r[1] / scale,
                score: r[2],
                depth: r[3],
            }
        })
        .collect();
    Ok((rows, stats))
}
