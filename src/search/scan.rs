//! Host driver of the per-period signal scan: period bank, batching under a
//! memory budget, kernel launches and threshold selection.

use super::grid::period_grid;
use super::kernels::{CUM_BLOCK, FOLD_BLOCK, FOLD_PERIODS, FOLD_SHARED_BINS, NF, PART_STRIDE, RECORD_STRIDE, TILE, gpu};
use crate::config::SearchConfig;
use crate::prep::curve::{FamilyState, Ready};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig, Stream};
use cuda_device::atomic::DeviceAtomicF64;
use std::sync::Arc;

/// One qualifying period: best template above threshold.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Row {
    pub period: f64,
    pub epoch: f64,
    pub duration: f64,
    pub score: f64,
    pub depth: f64,
}

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct ScanStats {
    pub periods: usize,
    pub prescan_survivors: usize,
    pub phase_bins: usize,
    pub batches: usize,
    pub templates_scanned: u64,
    pub exact_tile_retries: u64,
    pub seconds: f64,
    pub seconds_fold: f64,
    pub seconds_cumulative: f64,
    pub seconds_scan: f64,
}

pub struct Gpu {
    /// Kept alive for the lifetime of the streams and modules.
    _ctx: Arc<CudaContext>,
    pub stream: Arc<Stream>,
    pub module: gpu::LoadedModule,
    pub evidence: super::evidence::evidence_gpu::LoadedModule,
}

impl Gpu {
    pub fn new(device: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(device)?;
        let stream = ctx.default_stream();
        let module = gpu::load(&ctx)?;
        let evidence = super::evidence::evidence_gpu::load(&ctx)?;
        Ok(Self { _ctx: ctx, stream, module, evidence })
    }

    /// (free, total) bytes of device memory.
    pub fn memory(&self) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: the context created by `new` is current on this thread; the pointers are valid.
        let status = unsafe { cuda_core::sys::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize) };
        if status != 0 {
            return Err(format!("cuMemGetInfo failed: {status:?}").into());
        }
        Ok((free, total))
    }

    /// Fold budget in MiB: the configured value, capped at half of the free device memory.
    pub fn fold_budget_mib(&self, configured: f64) -> f64 {
        match self.memory() {
            Ok((free, _)) => configured.min(free as f64 / 1048576.0 / 2.0),
            Err(_) => configured,
        }
    }
}

/// Device copy of a family's precision fields.
pub struct DeviceReady {
    pub time: DeviceBuffer<f64>,
    pub fields: DeviceBuffer<f64>,
    pub n: usize,
    pub baseline_norm: f64,
    pub origin: f64,
}

impl DeviceReady {
    pub fn upload(gpu: &Gpu, ready: &Ready) -> Result<Self, Box<dyn std::error::Error>> {
        let n = ready.time.len();
        let mut fields = Vec::with_capacity(NF * n);
        fields.extend_from_slice(&ready.v);
        fields.extend_from_slice(&ready.w);
        fields.extend(std::iter::repeat_n(1.0, n));
        Ok(Self {
            time: DeviceBuffer::from_host(&gpu.stream, &ready.time)?,
            fields: DeviceBuffer::from_host(&gpu.stream, &fields)?,
            n,
            baseline_norm: ready.baseline_norm,
            origin: ready.origin,
        })
    }
}

pub struct Bank<'a> {
    pub durations: &'a [f64],
    pub rho: &'a [f64],
    pub ingress: &'a [f64],
    pub layers: usize,
    pub threshold: f64,
    pub shape_gate: f64,
    pub phase_oversample: usize,
    pub period_batch: usize,
    pub max_fold_mib: f64,
    pub prescan_factor: usize,
    pub prescan_fraction: f64,
}

/// Best template per period over the finite staircase bank with `nb` phase
/// bins; returns (score, row) for every period, score -inf when no template
/// is usable.
fn scan_bank(gpu: &Gpu, dev: &DeviceReady, periods: &[f64], bank: &Bank, nb: usize, stats: &mut ScanStats) -> Result<Vec<(f64, Row)>, Box<dyn std::error::Error>> {
    let profile = std::env::var_os("FBLS_PROFILE").is_some();
    stats.phase_bins = nb;
    let nd = bank.durations.len();
    let ns = bank.ingress.len();
    let layers = bank.layers;
    let factors: Vec<f64> = bank
        .ingress
        .iter()
        .flat_map(|&ing| (0..layers).map(move |l| 1.0 - 2.0 * ing * if layers > 1 { l as f64 / (layers - 1) as f64 } else { 0.0 }))
        .collect();
    let stream = &gpu.stream;
    let d_dur = DeviceBuffer::from_host(stream, bank.durations)?;
    let d_rho = DeviceBuffer::from_host(stream, bank.rho)?;
    let d_ing = DeviceBuffer::from_host(stream, bank.ingress)?;
    let d_fac = DeviceBuffer::from_host(stream, &factors)?;
    let tiles = nb.div_ceil(TILE);
    let per_period_bytes = NF * (4 * nb + 2) * 8 + NF * (2 * nb + 1) * 8;
    let budget = ((bank.max_fold_mib * 1048576.0) / per_period_bytes as f64).floor() as usize;
    if budget == 0 {
        return Err("one folded period exceeds max_fold_mib".into());
    }
    let batch = bank.period_batch.min(budget).min(periods.len());
    let shared_fold = nb <= FOLD_SHARED_BINS;
    let mut folded_plain = DeviceBuffer::<f64>::zeroed(stream, if shared_fold { batch * nb * NF } else { 1 })?;
    let folded = DeviceBuffer::<f64>::zeroed(stream, if shared_fold { 1 } else { batch * nb * NF })?.wrap_elem::<DeviceAtomicF64>();
    let mut cum = DeviceBuffer::<f64>::zeroed(stream, batch * (2 * nb + 1) * NF)?;
    let mut cum32 = DeviceBuffer::<f32>::zeroed(stream, 2 * batch * (2 * nb + 1) * NF)?;
    let mut parts = DeviceBuffer::<f64>::zeroed(stream, batch * tiles * PART_STRIDE)?;
    let mut keys = DeviceBuffer::<u64>::zeroed(stream, batch * tiles)?;
    let mut flags = DeviceBuffer::<u32>::zeroed(stream, batch * tiles)?;
    let mut records = DeviceBuffer::<f64>::zeroed(stream, batch * RECORD_STRIDE)?;
    let mut score = DeviceBuffer::<f64>::zeroed(stream, batch)?;
    let mut depth = DeviceBuffer::<f64>::zeroed(stream, batch)?;
    let mut rows = Vec::new();
    let mut cursor = 0;
    while cursor < periods.len() {
        let stop = (cursor + batch).min(periods.len());
        let pp = &periods[cursor..stop];
        let p = pp.len();
        let scales: Vec<f64> = pp.iter().map(|x| nb as f64 / x).collect();
        let d_scales = DeviceBuffer::from_host(stream, &scales)?;
        let d_periods = DeviceBuffer::from_host(stream, pp)?;
        let fold_cfg = LaunchConfig {
            grid_dim: ((dev.n as u32).div_ceil(FOLD_BLOCK), (p as u32).div_ceil(FOLD_PERIODS as u32), 1),
            block_dim: (FOLD_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let scan_cfg = LaunchConfig { grid_dim: (tiles as u32, p as u32, 1), block_dim: (TILE as u32, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: grid shapes match each kernel's indexing and every buffer covers the kernel's accesses.
        let mut tick = std::time::Instant::now();
        let mut lap = |acc: &mut f64| {
            if profile {
                stream.synchronize().unwrap();
                *acc += tick.elapsed().as_secs_f64();
                tick = std::time::Instant::now();
            }
        };
        unsafe {
            if shared_fold {
                let per_block = (FOLD_SHARED_BINS / nb).clamp(1, 8) as u32;
                gpu.module.fold_shared(
                    stream,
                    LaunchConfig { grid_dim: ((p as u32).div_ceil(per_block), 1, 1), block_dim: (FOLD_BLOCK, 1, 1), shared_mem_bytes: 0 },
                    &dev.time,
                    &dev.fields,
                    &d_scales,
                    &mut folded_plain,
                    dev.n as u32,
                    nb as u32,
                    p as u32,
                    per_block,
                )?;
                lap(&mut stats.seconds_fold);
                gpu.module.cumulative_plain(
                    stream,
                    LaunchConfig { grid_dim: (p as u32, 1, 1), block_dim: (CUM_BLOCK as u32, 1, 1), shared_mem_bytes: 0 },
                    &folded_plain,
                    &mut cum,
                    &mut cum32,
                    nb as u32,
                )?;
            } else {
                gpu.module.zero_folded(stream, LaunchConfig::for_num_elems((p * nb * NF) as u32), &folded)?;
                gpu.module.fold(stream, fold_cfg, &dev.time, &dev.fields, &d_scales, &folded, dev.n as u32, nb as u32, p as u32)?;
                lap(&mut stats.seconds_fold);
                gpu.module.cumulative(
                    stream,
                    LaunchConfig { grid_dim: (p as u32, 1, 1), block_dim: (CUM_BLOCK as u32, 1, 1), shared_mem_bytes: 0 },
                    &folded,
                    &mut cum,
                    &mut cum32,
                    nb as u32,
                )?;
            }
            lap(&mut stats.seconds_cumulative);
            gpu.module.scan_tiles(
                stream,
                scan_cfg,
                &cum,
                &cum32,
                &d_scales,
                &d_dur,
                &d_rho,
                &d_ing,
                &d_fac,
                dev.baseline_norm,
                bank.shape_gate,
                dev.n as u32,
                nb as u32,
                nd as u32,
                ns as u32,
                layers as u32,
                &mut parts,
                &mut keys,
                &mut flags,
            )?;
            lap(&mut stats.seconds_scan);
            gpu.module.merge_tiles(
                stream,
                LaunchConfig::for_num_elems(p as u32),
                &parts,
                &keys,
                &d_periods,
                &d_scales,
                p as u32,
                tiles as u32,
                &mut records,
                &mut score,
                &mut depth,
            )?;
        }
        let rec = records.to_host_vec(stream)?;
        let sc = score.to_host_vec(stream)?;
        let dp = depth.to_host_vec(stream)?;
        let fl = flags.to_host_vec(stream)?;
        let fl = &fl[..p * tiles];
        if fl.iter().any(|f| f & 2 != 0) {
            return Err("nonpositive template precision; inspect noise model".into());
        }
        stats.exact_tile_retries += fl.iter().filter(|f| *f & 1 != 0).count() as u64;
        for i in 0..p {
            let r = &rec[i * RECORD_STRIDE..(i + 1) * RECORD_STRIDE];
            rows.push((
                sc[i],
                Row { period: r[0], epoch: (r[2] + r[3] / 2.0) / r[1] + dev.origin, duration: r[3] / r[1], score: sc[i], depth: dp[i] },
            ));
        }
        stats.templates_scanned += (p * nd * ns * nb) as u64;
        stats.batches += 1;
        cursor = stop;
    }
    Ok(rows)
}

/// Best template per period; returns the periods whose best score reaches
/// the threshold. With `prescan_factor` > 1 every period is first scanned on
/// a phase grid coarser by that factor and only periods whose coarse maximum
/// reaches `prescan_fraction` x threshold are scanned on the full grid.
pub fn scan_periods(gpu: &Gpu, dev: &DeviceReady, periods: &[f64], bank: &Bank) -> Result<(Vec<Row>, ScanStats), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    let mut stats = ScanStats { periods: periods.len(), ..Default::default() };
    if periods.is_empty() || bank.durations.is_empty() {
        return Ok((Vec::new(), stats));
    }
    let dmin = bank.durations.iter().cloned().fold(f64::INFINITY, f64::min);
    let nb = ((bank.phase_oversample as f64 * periods[periods.len() - 1] / dmin).ceil() as usize).max(8);
    let mut survivors: Vec<f64> = periods.to_vec();
    if bank.prescan_factor > 1 && nb / bank.prescan_factor >= 8 {
        let coarse = scan_bank(gpu, dev, periods, bank, nb / bank.prescan_factor, &mut stats)?;
        survivors = coarse.iter().filter(|(s, _)| s.is_finite() && *s >= bank.prescan_fraction * bank.threshold).map(|(_, r)| r.period).collect();
        stats.prescan_survivors = survivors.len();
    }
    let mut rows = Vec::new();
    if !survivors.is_empty() {
        let full = scan_bank(gpu, dev, &survivors, bank, nb, &mut stats)?;
        rows.extend(full.into_iter().filter(|(s, _)| s.is_finite() && *s >= bank.threshold).map(|(_, r)| r));
    }
    stats.seconds = start.elapsed().as_secs_f64();
    Ok((rows, stats))
}



/// Best template per period for one family over its own period grid, in
/// octave bands so that each band only uses durations below a quarter of its
/// shortest period.
pub fn signal_search(gpu: &Gpu, state: &FamilyState, cfg: &SearchConfig) -> Result<(Vec<Row>, Vec<f64>, Vec<ScanStats>), Box<dyn std::error::Error>> {
    let mut ds: Vec<(f64, f64)> = state.durations.iter().cloned().zip(state.rho.iter().cloned()).collect();
    ds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let baseline = *state.ready.time.last().unwrap();
    let maximum = cfg.period_max.min(baseline);
    let minimum = cfg.period_min.max(4.0 * ds[0].0 * (1.0 + 1e-9));
    if maximum <= minimum {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let grid = period_grid(minimum, maximum, baseline, ds[0].0, cfg.period_oversample);
    let dev = DeviceReady::upload(gpu, &state.ready)?;
    let shape_gate = if cfg.ingress_fractions[0] == 0.0 { cfg.shape_gate_fraction * cfg.threshold } else { 0.0 };
    let mut rows = Vec::new();
    let mut stats = Vec::new();
    let mut lo = grid[0];
    let last = grid[grid.len() - 1];
    while lo <= last {
        let hi = lo * 2.0;
        let selected: Vec<f64> = grid.iter().cloned().filter(|&p| p >= lo && p < hi).collect();
        let chosen: Vec<(f64, f64)> = ds.iter().cloned().filter(|d| d.0 < lo / 4.0).collect();
        if !selected.is_empty() && !chosen.is_empty() {
            let durations: Vec<f64> = chosen.iter().map(|d| d.0).collect();
            let rho: Vec<f64> = chosen.iter().map(|d| d.1).collect();
            let bank = Bank {
                durations: &durations,
                rho: &rho,
                ingress: &cfg.ingress_fractions,
                layers: cfg.shape_layers,
                threshold: cfg.threshold,
                shape_gate,
                phase_oversample: cfg.phase_oversample,
                period_batch: cfg.period_batch,
                max_fold_mib: gpu.fold_budget_mib(cfg.max_fold_mib),
                prescan_factor: cfg.prescan_factor,
                prescan_fraction: cfg.prescan_fraction,
            };
            let (band_rows, band_stats) = scan_periods(gpu, &dev, &selected, &bank)?;
            rows.extend(band_rows);
            stats.push(band_stats);
        }
        lo = hi;
    }
    Ok((rows, grid, stats))
}
