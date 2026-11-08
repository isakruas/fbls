//! Greedy peak selection with event-overlap rescoring.
//!
//! A row is accepted when at least `min_events` transit epochs hold three or
//! more observations, fewer than half of its in-transit observations are
//! shared with accepted ephemerides, its amplitude statistic without the
//! observations of accepted ephemerides reaches the threshold scaled by the
//! square root of its unshared fraction (floored at half the threshold), and
//! the statistic without its strongest single event keeps at least half of the
//! threshold. After each acceptance the strongest remaining rows are rescored.

use crate::prep::curve::{FamilyState, transit_mask};
use crate::prep::stats::lower_bound;
use crate::search::kernels::RESCORE_BLOCK;
use crate::search::scan::{Gpu, Row};
use cuda_core::{DeviceBuffer, LaunchConfig};
use std::collections::BTreeMap;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Peak {
    pub period_days: f64,
    pub epoch_bjd: f64,
    pub duration_hours: f64,
    pub snr: f64,
    pub snr_folded: f64,
    pub snr_without_strongest_event: f64,
    pub depth: f64,
    pub depth_ppm: f64,
    pub events: usize,
    pub family: usize,
    pub pass: usize,
    #[serde(skip)]
    pub row: usize,
    pub planet_log_odds: Option<f64>,
    pub log_bf_planet_noise: Option<f64>,
    pub log_bf_binary_noise: Option<f64>,
    pub binary_secondary_ratio: Option<f64>,
    pub binary_secondary_phase: Option<f64>,
    pub binary_odd_even_ratio: Option<f64>,
    pub binary_period_days: Option<f64>,
    pub ingress_fraction: Option<f64>,
    pub evidence_epoch_bjd: Option<f64>,
    pub evidence_duration_hours: Option<f64>,
    pub model_preferred: bool,
    pub binary_diagnostics_complete: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SingleEvent {
    pub epoch_bjd: f64,
    pub duration_hours: f64,
    pub snr: f64,
    pub depth: f64,
    pub depth_ppm: f64,
    pub points: usize,
}

fn overlap(a: &[bool], b: &[bool]) -> f64 {
    let total = a.iter().filter(|&&x| x).count();
    if total == 0 {
        return 0.0;
    }
    a.iter().zip(b).filter(|(x, y)| **x && **y).count() as f64 / total as f64
}

/// Number of distinct transit epochs with at least `minimum_points` in-transit observations.
pub fn event_count(
    time: &[f64],
    period: f64,
    epoch: f64,
    duration: f64,
    minimum_points: usize,
) -> usize {
    let inside = transit_mask(time, period, epoch, duration, 0.0);
    let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
    for (i, &t) in time.iter().enumerate() {
        if inside[i] {
            *counts
                .entry(((t - epoch) / period).round() as i64)
                .or_insert(0) += 1;
        }
    }
    counts.values().filter(|&&c| c >= minimum_points).count()
}

fn nearest_rho(state: &FamilyState, duration: f64) -> f64 {
    let mut best = 0;
    for i in 1..state.durations.len() {
        if (state.durations[i] - duration).abs() < (state.durations[best] - duration).abs() {
            best = i;
        }
    }
    state.rho[best]
}

/// Positions of the family's observations in the merged time vector.
fn positions(state: &FamilyState, time_all: &[f64]) -> Vec<usize> {
    state
        .ready
        .time
        .iter()
        .map(|&t| lower_bound(time_all, t + state.ready.origin).min(time_all.len() - 1))
        .collect()
}

struct FamilyDevice {
    time: DeviceBuffer<f64>,
    fields: DeviceBuffer<f64>,
    origin: f64,
    n: usize,
}

/// Batched rescoring on the GPU: box sums per row over the family
/// observations outside the excluded set, and in-transit counts over the
/// merged time vector.
struct Rescorer<'a> {
    gpu: &'a Gpu,
    families: Vec<FamilyDevice>,
    time_all: DeviceBuffer<f64>,
}

impl<'a> Rescorer<'a> {
    fn new(
        gpu: &'a Gpu,
        states: &[FamilyState],
        time_all: &[f64],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = &gpu.stream;
        let mut families = Vec::new();
        for s in states {
            let r = &s.ready;
            let mut fields = Vec::with_capacity(2 * r.time.len());
            fields.extend_from_slice(&r.v);
            fields.extend_from_slice(&r.w);
            families.push(FamilyDevice {
                time: DeviceBuffer::from_host(stream, &r.time)?,
                fields: DeviceBuffer::from_host(stream, &fields)?,
                origin: r.origin,
                n: r.time.len(),
            });
        }
        Ok(Self {
            gpu,
            families,
            time_all: DeviceBuffer::from_host(stream, time_all)?,
        })
    }

    /// (score, required threshold) of every row in `indices`.
    fn evaluate(
        &self,
        indices: &[usize],
        table: &[Row],
        family: &[usize],
        states: &[FamilyState],
        pos: &[Vec<usize>],
        excluded: &[bool],
        threshold: f64,
    ) -> Result<Vec<(f64, f64)>, Box<dyn std::error::Error>> {
        let stream = &self.gpu.stream;
        let mut out = vec![(f64::NEG_INFINITY, threshold); indices.len()];
        let excluded_all: Vec<u8> = excluded.iter().map(|&e| e as u8).collect();
        let d_excluded_all = DeviceBuffer::from_host(stream, &excluded_all)?;
        let rows_all: Vec<f64> = indices
            .iter()
            .flat_map(|&i| [table[i].period, table[i].epoch, table[i].duration])
            .collect();
        let d_rows_all = DeviceBuffer::from_host(stream, &rows_all)?;
        let mut d_counts = DeviceBuffer::<u32>::zeroed(stream, 2 * indices.len())?;
        // SAFETY: one block per row; every buffer covers the kernel's accesses.
        unsafe {
            self.gpu.module.count_rows(
                stream,
                LaunchConfig {
                    grid_dim: (indices.len() as u32, 1, 1),
                    block_dim: (RESCORE_BLOCK as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.time_all,
                &d_excluded_all,
                &d_rows_all,
                self.time_all.len() as u32,
                &mut d_counts,
            )?;
        }
        let counts = d_counts.to_host_vec(stream)?;
        for (slot, _) in indices.iter().enumerate() {
            let total = counts[slot * 2].max(1) as f64;
            let unshared = counts[slot * 2 + 1] as f64 / total;
            out[slot].1 = (threshold * unshared.sqrt()).max(0.5 * threshold);
        }
        for (k, dev) in self.families.iter().enumerate() {
            let slots: Vec<usize> = (0..indices.len())
                .filter(|&s| family[indices[s]] == k)
                .collect();
            if slots.is_empty() {
                continue;
            }
            let excluded_family: Vec<u8> = pos[k].iter().map(|&p| excluded[p] as u8).collect();
            let d_excluded = DeviceBuffer::from_host(stream, &excluded_family)?;
            let rows: Vec<f64> = slots
                .iter()
                .flat_map(|&s| {
                    let r = &table[indices[s]];
                    [r.period, r.epoch, r.duration]
                })
                .collect();
            let d_rows = DeviceBuffer::from_host(stream, &rows)?;
            let mut d_sums = DeviceBuffer::<f64>::zeroed(stream, 3 * slots.len())?;
            unsafe {
                self.gpu.module.rescore_rows(
                    stream,
                    LaunchConfig {
                        grid_dim: (slots.len() as u32, 1, 1),
                        block_dim: (RESCORE_BLOCK as u32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &dev.time,
                    &dev.fields,
                    &d_excluded,
                    &d_rows,
                    dev.origin,
                    dev.n as u32,
                    &mut d_sums,
                )?;
            }
            let sums = d_sums.to_host_vec(stream)?;
            for (m, &s) in slots.iter().enumerate() {
                let (a, hp, norm) = (sums[m * 3], sums[m * 3 + 1], sums[m * 3 + 2]);
                let q = norm - hp * hp / states[k].ready.baseline_norm;
                out[s].0 = if q > 0.0 && a > 0.0 {
                    (a / nearest_rho(&states[k], table[indices[s]].duration)) / q.max(1e-300).sqrt()
                } else {
                    f64::NEG_INFINITY
                };
            }
        }
        Ok(out)
    }
}

/// Box statistic after removing the strongest individual event; returns
/// (statistic, number of events with data).
fn without_strongest_event(
    state: &FamilyState,
    pos: &[usize],
    row: &Row,
    excluded: &[bool],
) -> (f64, usize) {
    let r = &state.ready;
    let mut sums: BTreeMap<i64, (f64, f64, f64)> = BTreeMap::new();
    let mut members: Vec<(usize, i64)> = Vec::new();
    for (i, &t) in r.time.iter().enumerate() {
        if excluded[pos[i]] {
            continue;
        }
        let ta = t + r.origin;
        let phase = (ta - row.epoch + row.period / 2.0).rem_euclid(row.period) - row.period / 2.0;
        if phase.abs() <= row.duration / 2.0 {
            let cycle = ((ta - row.epoch) / row.period).round() as i64;
            let e = sums.entry(cycle).or_insert((0.0, 0.0, 0.0));
            e.0 += r.v[i];
            e.1 += r.p[i];
            e.2 += r.w[i];
            members.push((i, cycle));
        }
    }
    if sums.is_empty() {
        return (f64::NEG_INFINITY, 0);
    }
    let rho = nearest_rho(state, row.duration);
    let strongest = sums
        .iter()
        .map(|(&c, &(a, hp, w))| {
            (
                c,
                a / rho / (w - hp * hp / r.baseline_norm).max(1e-300).sqrt(),
            )
        })
        .fold(
            (0i64, f64::NEG_INFINITY),
            |acc, x| if x.1 > acc.1 { x } else { acc },
        )
        .0;
    let (mut a, mut hp, mut norm) = (0.0, 0.0, 0.0);
    for &(i, cycle) in &members {
        if cycle != strongest {
            a += r.v[i];
            hp += r.p[i];
            norm += r.w[i];
        }
    }
    let q = norm - hp * hp / r.baseline_norm;
    let value = if q > 0.0 && a > 0.0 {
        a / rho / q.max(1e-300).sqrt()
    } else {
        f64::NEG_INFINITY
    };
    (value, sums.len())
}

pub struct Selection {
    pub accepted: Vec<Peak>,
    pub masks: Vec<Vec<bool>>,
    pub excluded: Vec<bool>,
}

#[allow(clippy::too_many_arguments)]
pub fn select_peaks(
    gpu: &Gpu,
    table: &[Row],
    family: &[usize],
    states: &[FamilyState],
    time_all: &[f64],
    threshold: f64,
    limit: usize,
    min_events: usize,
    taken: &[Vec<bool>],
    excluded_in: &[bool],
    pass: usize,
) -> Result<Selection, Box<dyn std::error::Error>> {
    let margin = 0.5;
    let max_iterations = 20000;
    let mut accepted = Vec::new();
    let mut masks: Vec<Vec<bool>> = taken.to_vec();
    let mut excluded = excluded_in.to_vec();
    if table.is_empty() {
        return Ok(Selection {
            accepted,
            masks,
            excluded,
        });
    }
    let pos: Vec<Vec<usize>> = states.iter().map(|s| positions(s, time_all)).collect();
    let rescorer = Rescorer::new(gpu, states, time_all)?;
    let mut remaining: Vec<f64> = table
        .iter()
        .map(|r| {
            if r.score.is_finite() && r.score >= threshold {
                r.score
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let mut fresh = vec![!excluded.iter().any(|&x| x); table.len()];

    // Candidates ordered by score with lazy invalidation: an entry whose
    // stored score no longer matches `remaining` is stale and dropped.
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;
    #[derive(PartialEq)]
    struct Entry(f64, usize);
    impl Eq for Entry {}
    impl PartialOrd for Entry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Entry {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0
                .partial_cmp(&other.0)
                .unwrap_or(Ordering::Equal)
                .then(other.1.cmp(&self.1))
        }
    }
    let mut heap: BinaryHeap<Entry> = (0..table.len())
        .filter(|&i| remaining[i].is_finite())
        .map(|i| Entry(remaining[i], i))
        .collect();
    // Lazy rescoring from the top of the heap: stale entries carry their
    // previous score, which bounds the rescored value, so rows below the best
    // fresh entry are never rescored.
    let block = 8192;
    let refresh = |remaining: &mut Vec<f64>,
                   fresh: &mut Vec<bool>,
                   heap: &mut BinaryHeap<Entry>,
                   excluded: &[bool]|
     -> Result<(), Box<dyn std::error::Error>> {
        let mut stale = Vec::new();
        let mut kept = Vec::new();
        while let Some(Entry(v, i)) = heap.pop() {
            if v != remaining[i] {
                continue;
            }
            if fresh[i] {
                kept.push(Entry(v, i));
            } else {
                stale.push(i);
                if stale.len() == block {
                    break;
                }
            }
        }
        heap.extend(kept);
        if stale.is_empty() {
            return Ok(());
        }
        let values = rescorer.evaluate(&stale, table, family, states, &pos, excluded, threshold)?;
        for (m, &i) in stale.iter().enumerate() {
            let (rescored, required) = values[m];
            remaining[i] = if rescored >= required {
                rescored
            } else {
                f64::NEG_INFINITY
            };
            fresh[i] = true;
            if remaining[i].is_finite() {
                heap.push(Entry(remaining[i], i));
            }
        }
        Ok(())
    };

    let mut iterations = 0;
    while accepted.len() < limit && iterations < max_iterations {
        iterations += 1;
        let top = loop {
            match heap.peek() {
                Some(Entry(v, i)) if *v != remaining[*i] => {
                    heap.pop();
                }
                other => break other.map(|e| e.1),
            }
        };
        let Some(i) = top else { break };
        if !fresh[i] {
            refresh(&mut remaining, &mut fresh, &mut heap, &excluded)?;
            continue;
        }
        heap.pop();
        let isolated = remaining[i];
        remaining[i] = f64::NEG_INFINITY;
        let row = table[i];
        let mask = transit_mask(time_all, row.period, row.epoch, row.duration, 0.0);
        if masks.iter().any(|other| overlap(&mask, other) >= 0.5) {
            continue;
        }
        let events = event_count(time_all, row.period, row.epoch, row.duration, 3);
        if events < min_events {
            continue;
        }
        let k = family[i];
        let (leftover, _) = without_strongest_event(&states[k], &pos[k], &row, &excluded);
        if min_events > 1 && leftover < 0.5 * threshold {
            continue;
        }
        let wide = transit_mask(time_all, row.period, row.epoch, row.duration, margin);
        for (e, &w) in excluded.iter_mut().zip(&wide) {
            *e |= w;
        }
        masks.push(wide);
        accepted.push(Peak {
            period_days: row.period,
            epoch_bjd: row.epoch,
            duration_hours: 24.0 * row.duration,
            snr: isolated,
            snr_folded: row.score,
            snr_without_strongest_event: leftover,
            depth: row.depth,
            depth_ppm: 1e6 * row.depth,
            events,
            family: k,
            pass,
            row: i,
            planet_log_odds: None,
            log_bf_planet_noise: None,
            log_bf_binary_noise: None,
            binary_secondary_ratio: None,
            binary_secondary_phase: None,
            binary_odd_even_ratio: None,
            binary_period_days: None,
            ingress_fraction: None,
            evidence_epoch_bjd: None,
            evidence_duration_hours: None,
            model_preferred: false,
            binary_diagnostics_complete: false,
        });
        fresh.iter_mut().for_each(|f| *f = false);
        refresh(&mut remaining, &mut fresh, &mut heap, &excluded)?;
    }
    Ok(Selection {
        accepted,
        masks,
        excluded,
    })
}

/// Greedy suppression of single-event templates overlapping accepted ephemerides.
pub fn select_events(
    table: &[Row],
    time: &[f64],
    threshold: f64,
    limit: usize,
    taken: &[Vec<bool>],
) -> Vec<SingleEvent> {
    let mut output = Vec::new();
    let mut masks: Vec<Vec<bool>> = taken.to_vec();
    if table.is_empty() {
        return output;
    }
    let mut remaining: Vec<f64> = table
        .iter()
        .map(|r| {
            if r.score.is_finite() && r.score >= threshold {
                r.score
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let mut iterations = 0;
    while output.len() < limit && iterations < 5000 {
        iterations += 1;
        let i = (0..table.len())
            .max_by(|&a, &b| remaining[a].partial_cmp(&remaining[b]).unwrap())
            .unwrap();
        if !remaining[i].is_finite() {
            break;
        }
        let row = table[i];
        let inside: Vec<bool> = time
            .iter()
            .map(|&t| (t - row.epoch).abs() <= row.duration / 2.0)
            .collect();
        for j in 0..table.len() {
            if (table[j].epoch - row.epoch).abs() <= (table[j].duration + row.duration) / 2.0 {
                remaining[j] = f64::NEG_INFINITY;
            }
        }
        let points = inside.iter().filter(|&&x| x).count();
        if points < 3 || masks.iter().any(|other| overlap(&inside, other) >= 0.5) {
            continue;
        }
        masks.push(
            time.iter()
                .map(|&t| (t - row.epoch).abs() <= 0.75 * row.duration)
                .collect(),
        );
        output.push(SingleEvent {
            epoch_bjd: row.epoch,
            duration_hours: 24.0 * row.duration,
            snr: row.score,
            depth: row.depth,
            depth_ppm: 1e6 * row.depth,
            points,
        });
    }
    output
}
