"""Pipeline-facing input preparation, search configuration and GPU results."""
from dataclasses import asdict, dataclass, field
import math
import time

import torch

from .core import (PreparedCurve, gpu64, prepare_families, prepare_astrometry, family_state, signal_search,
                   single_event_search, evidence_at, transit_mask, event_count, mask_curve)


DEFAULT_DURATIONS_HOURS = tuple(round(.5 * 1.25 ** i, 6) for i in range(13))
TABLE_COLUMNS = ("period_days", "epoch_bjd", "duration_days", "snr", "depth",
                 "planet_log_odds", "log_bf_planet_noise", "log_bf_binary_noise",
                 "binary_secondary_ratio", "binary_secondary_phase", "binary_odd_even_ratio",
                 "binary_period_days", "ingress_fraction", "selection_flags")


@dataclass(frozen=True)
class SearchConfig:
    """Period-bank, preparation and reporting parameters; all times are days or hours."""
    period_min: float = .5
    period_max: float = 10.
    durations_hours: tuple[float, ...] = DEFAULT_DURATIONS_HOURS
    threshold: float = 7.1
    bin_minutes: float = 10.
    window_days: float = .4
    window_factor: float = 3.
    window_max_days: float = 3.5
    clip_sigma: float = 4.
    noise_degree: int = 0
    correlated_noise: bool = True
    ingress_fractions: tuple[float, ...] = (0., .2, .45)
    shape_layers: int = 4
    phase_oversample: int = 4
    period_oversample: float = 3.
    max_candidates: int = 20
    period_batch: int | None = None
    candidate_capacity: int = 131072
    max_fold_mib: float = 128.
    model_comparison: bool = True
    planet_prior: float = .01
    binary_prior: float = .01
    planet_depth_scale: float = .01
    binary_depth_scale: float = .1
    diluted_binary_depth_scale: float = .01
    stellar_radius_solar: float | None = None
    planet_radius_scale_rj: float = 1.
    target_flux_fraction: float = 1.
    passes: int = 3
    min_events: int = 2
    evidence_candidates: int = 32
    single_events: bool = True
    evidence_rtol: float = 1e-12
    planet_log_odds_min: float = 0.
    extended_physics: bool = False
    limb_darkening: tuple[float, float] | None = (.3, .2)
    eccentricity_nodes: int = 2
    periastron_nodes: int = 4
    eccentricity_max: float = .8
    stellar_mass_solar: float | None = None
    stellar_logg: float | None = None
    stellar_teff: float | None = None
    stellar_tmag: float | None = None
    stellar_duration_scatter: float = .5
    ellipsoidal: bool = True
    ellipsoidal_planet_scale: float = 1e-5
    ellipsoidal_binary_scale: float = 1e-3
    ellipsoidal_alpha: float = 1.5
    planet_mass_scale_mj: float = 1.
    centroid_planet_scale: float = .001
    centroid_binary_scale: float = .05

    def __post_init__(self):
        durations = tuple(sorted(set(float(d) for d in self.durations_hours)))
        ingress = tuple(float(x) for x in self.ingress_fractions)
        object.__setattr__(self, "durations_hours", durations)
        object.__setattr__(self, "ingress_fractions", ingress)
        for name in ("period_min", "period_max", "threshold", "bin_minutes", "window_days",
                     "window_factor", "window_max_days", "clip_sigma", "period_oversample", "max_fold_mib", "planet_prior", "binary_prior",
                     "planet_depth_scale", "binary_depth_scale", "diluted_binary_depth_scale",
                     "planet_radius_scale_rj", "target_flux_fraction",
                     "evidence_rtol", "planet_log_odds_min", "eccentricity_max", "stellar_duration_scatter",
                     "ellipsoidal_planet_scale", "ellipsoidal_binary_scale", "ellipsoidal_alpha", "planet_mass_scale_mj",
                     "centroid_planet_scale", "centroid_binary_scale"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError("Numeric search parameters cannot be boolean")
            try:
                value = float(value)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{name} must be numeric") from error
            if not math.isfinite(value):
                raise ValueError("Search parameters must be finite")
            object.__setattr__(self, name, value)
        if not isinstance(self.correlated_noise, bool) or not isinstance(self.model_comparison, bool) \
                or not isinstance(self.single_events, bool):
            raise ValueError("correlated_noise, model_comparison and single_events must be boolean")
        if not isinstance(self.extended_physics,bool) or not isinstance(self.ellipsoidal,bool):
            raise ValueError("extended_physics and ellipsoidal must be boolean")
        for name in ("stellar_mass_solar","stellar_logg","stellar_teff","stellar_tmag"):
            value=getattr(self,name)
            if value is not None:
                if isinstance(value,bool) or not math.isfinite(float(value)):
                    raise ValueError(f"{name} must be finite")
                value=float(value)
                if name in ("stellar_mass_solar","stellar_teff") and value<=0:
                    raise ValueError(f"{name} must be positive")
                object.__setattr__(self,name,value)
        if self.limb_darkening is not None:
            limb=tuple(float(x) for x in self.limb_darkening)
            if len(limb)!=2 or any(not math.isfinite(x) for x in limb) or limb[0]<0 or limb[0]+2*limb[1]<0 or sum(limb)>=1:
                raise ValueError("Quadratic limb coefficients must define positive monotonic intensity")
            object.__setattr__(self,"limb_darkening",limb)
        if not 0<self.eccentricity_max<1 or self.stellar_duration_scatter<=0:
            raise ValueError("Invalid eccentricity support or stellar duration uncertainty")
        if min(self.ellipsoidal_planet_scale,self.ellipsoidal_binary_scale,self.ellipsoidal_alpha,
               self.planet_mass_scale_mj,self.centroid_planet_scale,self.centroid_binary_scale)<=0:
            raise ValueError("Physical amplitude prior scales must be positive")
        if self.stellar_radius_solar is not None:
            if isinstance(self.stellar_radius_solar, bool):
                raise ValueError("stellar_radius_solar must be positive")
            radius = float(self.stellar_radius_solar)
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError("stellar_radius_solar must be positive and finite")
            object.__setattr__(self, "stellar_radius_solar", radius)
        if not (0 < self.planet_prior < 1 and 0 < self.binary_prior < 1
                and self.planet_prior + self.binary_prior < 1):
            raise ValueError("Positive planet/binary prior masses must sum to less than one")
        if not 0 <= self.evidence_rtol < 1:
            raise ValueError("evidence_rtol must be in [0, 1)")
        if not (0 < self.target_flux_fraction <= 1 and self.planet_radius_scale_rj > 0
                and self.planet_depth_scale > 0 and self.binary_depth_scale > 0
                and self.diluted_binary_depth_scale > 0):
            raise ValueError("Depth scales and planet radius must be positive; target_flux_fraction must be in (0, 1]")
        if not 0 < self.period_min < self.period_max:
            raise ValueError("Require 0 < period_min < period_max")
        if not durations or any(not math.isfinite(d) or d <= 0 for d in durations):
            raise ValueError("durations_hours must contain positive finite values")
        if durations[0] / 24 >= self.period_min / 4:
            raise ValueError("Minimum duration must be below one quarter of period_min")
        if not ingress or any(not math.isfinite(x) or not 0 <= x < .5 for x in ingress):
            raise ValueError("Ingress fractions must be in [0, 0.5)")
        if (self.threshold < 0 or self.bin_minutes < 0 or self.window_days <= 0 or self.window_factor <= 0
                or self.window_max_days < self.window_days or
                self.clip_sigma <= 0 or self.period_oversample < 1 or self.max_fold_mib <= 0):
            raise ValueError("Invalid threshold, preparation window or resource budget")
        integers = ((self.noise_degree, 0, 4), (self.shape_layers, 1, None),
                    (self.eccentricity_nodes,0,16),(self.periastron_nodes,4,64),
                    (self.phase_oversample, 1, None), (self.max_candidates, 0, None),
                    (self.candidate_capacity, 1, None), (self.passes, 1, None), (self.min_events, 1, None),
                    (self.evidence_candidates, 0, None))
        for value, minimum, maximum in integers:
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
                raise ValueError("Invalid integer search parameter")
        if self.period_batch is not None and (isinstance(self.period_batch, bool)
                or not isinstance(self.period_batch, int) or self.period_batch < 1):
            raise ValueError("period_batch must be a positive integer or None")


@dataclass
class TransitSearchResult:
    """GPU float64 period maxima plus resolved periodic peaks and single events."""
    table: torch.Tensor
    statistics: dict
    config: SearchConfig
    peaks: list = field(default_factory=list)
    single_events: list = field(default_factory=list)

    def candidates(self, limit=None, *, ranking="planet"):
        """Resolved periodic peaks: model-preferred evidence order, or amplitude order."""
        limit = self.config.max_candidates if limit is None else limit
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("Candidate limit must be a nonnegative integer")
        if ranking not in ("planet", "signal"):
            raise ValueError("ranking must be 'planet' or 'signal'")
        rows = [dict(peak) for peak in self.peaks]
        if ranking == "signal" or not self.config.model_comparison:
            rows.sort(key=lambda r: -r["snr"])
        else:
            def key(r):
                odds = r.get("planet_log_odds")
                evidence = r.get("log_bf_planet_noise")
                preferred = r.get("model_preferred", False)
                return (0 if preferred else 1, -(evidence if preferred and evidence is not None else
                                                 (odds if odds is not None else -math.inf)))
            rows.sort(key=key)
        return rows[:limit]

    def to_dict(self):
        """Convert resolved peaks, single events and scalar metadata into a JSON mapping."""
        from . import __version__
        return {"schema_version": 3, "software_version": __version__, "config": asdict(self.config),
                "statistics": self.statistics, "table_columns": TABLE_COLUMNS,
                "candidates": self.candidates(), "signal_candidates": self.candidates(ranking="signal"),
                "single_events": self.single_events[:self.config.max_candidates]}


def _input_vectors(time_values, flux_values, errors=None, sectors=None):
    t, flux = gpu64(time_values), gpu64(flux_values)
    if t.ndim != 1 or flux.shape != t.shape:
        raise ValueError("Time and flux must be matching one-dimensional arrays")
    err = gpu64(errors) if errors is not None else None
    if err is not None and err.shape != t.shape:
        raise ValueError("Flux errors must match time and flux")
    labels = torch.as_tensor(sectors, device="cuda") if sectors is not None else None
    if labels is not None:
        if labels.ndim == 0:
            labels = labels.expand(len(t))
        if labels.shape != t.shape or not bool(torch.isfinite(labels).all() & (labels == labels.long()).all()):
            raise ValueError("Sector labels must be finite integers matching the observation count")
        labels = labels.long()
    return t, flux, err, labels


def _comparison(config):
    return {key: getattr(config, key) for key in ("planet_prior", "binary_prior",
            "planet_depth_scale", "binary_depth_scale", "diluted_binary_depth_scale",
            "stellar_radius_solar", "planet_radius_scale_rj", "target_flux_fraction", "evidence_rtol",
            "extended_physics", "limb_darkening", "eccentricity_nodes", "periastron_nodes", "eccentricity_max",
            "stellar_mass_solar", "stellar_logg", "stellar_teff", "stellar_tmag", "stellar_duration_scatter",
            "ellipsoidal", "ellipsoidal_planet_scale", "ellipsoidal_binary_scale", "ellipsoidal_alpha",
            "planet_mass_scale_mj", "centroid_planet_scale", "centroid_binary_scale")}


def _overlap(mask_a, mask_b):
    """Fraction of mask_a observations shared with mask_b."""
    total = int(mask_a.sum())
    return float((mask_a & mask_b).sum()) / total if total else 0.


def _rescore(states, table, family, rows, time_all, excluded):
    """Box-template amplitude statistics of selected rows with excluded observations removed.

    Uses each family's precision-weighted fields and duration noise scale, so
    a row explained by already accepted events loses its significance.
    """
    scores = torch.full((len(rows),), -torch.inf, device="cuda", dtype=torch.float64)
    for k, state in enumerate(states):
        local = torch.nonzero(family[rows] == k).flatten()
        if not len(local):
            continue
        t = state.ready.time + state.ready.origin
        position = torch.searchsorted(time_all, t).clamp_max(len(time_all) - 1)
        keep = ~excluded[position]
        v, p, w = state.ready.fields[0], state.ready.fields[1], state.ready.fields[2]
        chunk = max(1, 2**24 // max(1, len(t)))
        for begin in range(0, len(local), chunk):
            part = local[begin:begin + chunk]
            values = table[rows[part]]
            phase = (t[None, :] - values[:, 1:2] + values[:, 0:1] / 2).remainder(values[:, 0:1]) - values[:, 0:1] / 2
            inside = (phase.abs() <= values[:, 2:3] / 2) & keep[None, :]
            weight = inside.double()
            a, hp, norm = weight @ v, weight @ p, weight @ w
            index = (values[:, 2:3] - state.durations[None, :]).abs().argmin(1)
            rho = state.rho[index]
            q = norm - hp.square() / state.ready.baseline_norm
            scores[part] = torch.where((q > 0) & (a > 0), (a / rho) / q.clamp_min(1e-300).sqrt(), -torch.inf)
    return scores


def _without_strongest_event(state, row, time_all, excluded):
    """Box statistic of one ephemeris after removing its strongest individual event.

    An isolated event folded with an arbitrary long period reaches the amplitude
    threshold on its own; a periodic signal keeps most of its significance when
    one event is removed. Returns (statistic, event_count_with_data).
    """
    period, epoch, duration = [float(x) for x in row[:3]]
    t = state.ready.time + state.ready.origin
    position = torch.searchsorted(time_all, t).clamp_max(len(time_all) - 1)
    keep = ~excluded[position]
    inside = transit_mask(t, period, epoch, duration, margin=0.) & keep
    if not bool(inside.any()):
        return -math.inf, 0
    cycles = ((t - epoch) / period).round().long()
    labels, inverse = torch.unique(cycles[inside], return_inverse=True)
    v, p, w = state.ready.fields[0], state.ready.fields[1], state.ready.fields[2]
    sums = torch.zeros((3, len(labels)), device="cuda", dtype=torch.float64)
    for k, values in enumerate((v, p, w)):
        sums[k].scatter_add_(0, inverse, values[inside])
    rho = state.rho[(state.durations - duration).abs().argmin()]
    q = (sums[2] - sums[1].square() / state.ready.baseline_norm).clamp_min(1e-300)
    individual = sums[0] / rho / q.sqrt()
    strongest = labels[individual.argmax()]
    rest = inside & (cycles != strongest)
    a, hp, norm = v[rest].sum(), p[rest].sum(), w[rest].sum()
    q = norm - hp.square() / state.ready.baseline_norm
    value = float(a / rho / q.clamp_min(1e-300).sqrt()) if float(q) > 0 and float(a) > 0 else -math.inf
    return value, len(labels)


def _select_peaks(table, time_all, family, states, threshold, limit, min_events, taken, excluded, margin=.5,
                  block=1024, max_iterations=20000):
    """Greedy peak selection with event-overlap rescoring over the five-column table.

    A row is accepted when at least `min_events` transit epochs hold three or
    more observations, fewer than half of its in-transit observations are
    shared with accepted ephemerides, its amplitude statistic without the
    observations of accepted ephemerides still reaches the threshold, and the
    statistic without its strongest single event keeps at least half of the
    threshold (with `min_events` above one). After each acceptance the
    strongest remaining rows are rescored the same way.
    """
    accepted, masks = [], list(taken)
    excluded = excluded.clone()
    if not len(table):
        return accepted, masks, excluded
    remaining = table[:, 3].clone()
    remaining[~torch.isfinite(remaining) | (remaining < threshold)] = -torch.inf
    fresh = torch.full((len(table),), not bool(excluded.any()), device="cuda", dtype=torch.bool)

    def refresh():
        order = torch.argsort(remaining, descending=True)[:block]
        order = order[torch.isfinite(remaining[order]) & ~fresh[order]]
        if len(order):
            rescored = _rescore(states, table, family, order, time_all, excluded)
            remaining[order] = torch.where(rescored >= threshold, rescored, -torch.inf)
            fresh[order] = True

    iterations = 0
    while len(accepted) < limit and iterations < max_iterations:
        iterations += 1
        i = int(remaining.argmax())
        if not math.isfinite(float(remaining[i])):
            break
        if not bool(fresh[i]):
            refresh()
            continue
        isolated = float(remaining[i])
        remaining[i] = -torch.inf
        p, epoch, duration, score, depth = [float(x) for x in table[i, :5]]
        mask = transit_mask(time_all, p, epoch, duration, margin=0.)
        if any(_overlap(mask, other) >= .5 for other in masks):
            continue
        events = event_count(time_all, p, epoch, duration)
        if events < min_events:
            continue
        leftover, _ = _without_strongest_event(states[int(family[i])], table[i], time_all, excluded)
        if min_events > 1 and leftover < .5 * threshold:
            continue
        wide = transit_mask(time_all, p, epoch, duration, margin)
        masks.append(wide)
        excluded |= wide
        accepted.append({"period_days": p, "epoch_bjd": epoch, "duration_hours": 24 * duration,
                         "snr": isolated, "snr_folded": score, "snr_without_strongest_event": leftover,
                         "depth": depth, "depth_ppm": 1e6 * depth,
                         "events": events, "family": int(family[i]), "row": i})
        fresh[:] = False
        refresh()
    return accepted, masks, excluded


def _select_events(table, time, threshold, limit, taken, max_iterations=5000):
    """Greedy suppression of single-event templates overlapping accepted ephemerides."""
    output, masks = [], list(taken)
    if not len(table):
        return output
    remaining = table[:, 3].clone()
    remaining[~torch.isfinite(remaining) | (remaining < threshold)] = -torch.inf
    iterations = 0
    while len(output) < limit and iterations < max_iterations:
        iterations += 1
        i = int(remaining.argmax())
        if not math.isfinite(float(remaining[i])):
            break
        epoch, duration, score, depth = [float(x) for x in table[i, 1:5]]
        inside = (time - epoch).abs() <= duration / 2
        remaining[(table[:, 1] - epoch).abs() <= (table[:, 2] + duration) / 2] = -torch.inf
        if int(inside.sum()) < 3 or any(_overlap(inside, other) >= .5 for other in masks):
            continue
        masks.append((time - epoch).abs() <= .75 * duration)
        output.append({"epoch_bjd": epoch, "duration_hours": 24 * duration, "snr": score,
                       "depth": depth, "depth_ppm": 1e6 * depth, "points": int(inside.sum())})
    return output


def _search(families, config, preparation_seconds, time_all):
    """Multi-pass family search: signal scan, event-overlap peaks, masking, evidence."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    if len(time_all) < 6:
        raise ValueError("At least six usable observations are required")
    baseline = float(time_all[-1] - time_all[0])
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError("Observations must span a positive finite time interval")
    durations = gpu64(config.durations_hours) / 24
    options = dict(ingress_fractions=config.ingress_fractions, shape_layers=config.shape_layers,
                   phase_oversample=config.phase_oversample, period_batch=config.period_batch,
                   candidate_capacity=config.candidate_capacity, max_fold_mib=config.max_fold_mib)
    states = [family_state(curve, durations[indices], indices, config.noise_degree, config.correlated_noise)
              for curve, indices in families]
    peaks, masks, pass_stats, first_table, periods_tested = [], [], [], None, 0
    excluded = torch.zeros(len(time_all), device="cuda", dtype=torch.bool)
    for pass_index in range(config.passes):
        tables, labels, band_stats = [], [], []
        for k, state in enumerate(states):
            table, grid, stats = signal_search(state, config.period_min, config.period_max, config.threshold,
                                               period_oversample=config.period_oversample, **options)
            tables.append(table)
            labels.append(torch.full((len(table),), k, device="cuda", dtype=torch.int64))
            band_stats.append({"family": k, "window_days": state.curve.provenance.get("window_days"),
                               "durations_hours": (24 * state.durations).tolist(), "periods": len(grid),
                               "rho": state.rho.tolist(), "bands": stats})
            if pass_index == 0:
                periods_tested += len(grid)
        table = torch.cat(tables) if tables else torch.empty((0, 5), device="cuda", dtype=torch.float64)
        family = torch.cat(labels) if labels else torch.empty(0, device="cuda", dtype=torch.int64)
        if pass_index == 0:
            first_table = table
        found, masks, excluded = _select_peaks(table, time_all, family, states, config.threshold,
                                               config.max_candidates - len(peaks), config.min_events, masks, excluded)
        for peak in found:
            peak["pass"] = pass_index + 1
        pass_stats.append({"pass": pass_index + 1, "rows": len(table), "accepted": len(found),
                           "families": band_stats})
        if not found:
            break
        peaks.extend(found)
        if pass_index + 1 < config.passes:
            lead = found[0]
            states = [family_state(mask_curve(s.curve, lead["period_days"], lead["epoch_bjd"],
                                              lead["duration_hours"] / 24),
                                   s.durations, s.duration_indices, config.noise_degree, config.correlated_noise)
                      for s in states]
            if any(len(s.curve.time) < 6 for s in states):
                break
    if first_table is None:
        first_table = torch.empty((0, 5), device="cuda", dtype=torch.float64)
    full = torch.full((len(first_table), 14), torch.nan, device="cuda", dtype=torch.float64)
    full[:, :5] = first_table
    full[:, 13] = 2.
    evidence_stats = {}
    if config.model_comparison and peaks:
        comparison = _comparison(config)
        chosen = sorted(peaks, key=lambda r: -r["snr"])[:config.evidence_candidates]
        evidence_seconds = 0.
        for peak in chosen:
            # Other accepted ephemerides are removed so the evidence describes this signal alone.
            curve, indices = families[peak["family"]]
            for other in peaks:
                if other is not peak:
                    curve = mask_curve(curve, other["period_days"], other["epoch_bjd"], other["duration_hours"] / 24)
            if len(curve.time) < 6:
                continue
            state = family_state(curve, durations[indices], indices, config.noise_degree, config.correlated_noise)
            rows, stats = evidence_at(state, [peak["period_days"]], comparison, threshold=config.threshold, **options)
            evidence_seconds += stats.get("seconds_search", 0.)
            evidence_stats = {key: value for key, value in stats.items()
                              if key not in ("seconds_search", "periods", "batches")}
            hit = (rows[:, 0] == peak["period_days"]) & ((rows[:, 13].long() & 1) != 0)
            if not bool(hit.any()):
                hit = rows[:, 0] == peak["period_days"]
            if not bool(hit.any()):
                continue
            row = [float(x) for x in rows[torch.nonzero(hit).flatten()[0]]]
            peak.update({name: value if math.isfinite(value) else None
                         for name, value in zip(TABLE_COLUMNS[5:13], row[5:13])})
            peak["evidence_epoch_bjd"], peak["evidence_duration_hours"] = row[1], 24 * row[2]
            peak["model_preferred"] = row[5] >= config.planet_log_odds_min
            peak["binary_diagnostics_complete"] = all(math.isfinite(v) for v in row[8:12])
            if peak["pass"] == 1 and peak["row"] < len(full):
                full[peak["row"], 5:13] = torch.tensor(row[5:13], device="cuda", dtype=torch.float64)
                full[peak["row"], 13] = 3.
        evidence_stats["seconds_search"] = evidence_seconds
    for peak in peaks:
        if config.model_comparison and "planet_log_odds" not in peak:
            peak.update({name: None for name in TABLE_COLUMNS[5:13]})
            peak["model_preferred"] = False
            peak["binary_diagnostics_complete"] = False
        peak.pop("row", None)
    events, event_stats = [], {}
    if config.single_events:
        for k, state in enumerate(states):
            table, stats = single_event_search(state, config.threshold, config.candidate_capacity,
                                               **{key: value for key, value in options.items()
                                                  if key not in ("period_batch", "candidate_capacity")})
            event_stats[k] = {key: value for key, value in stats.items() if key in ("templates_scanned", "threshold_survivors", "overflow")}
            events.extend(_select_events(table, time_all, config.threshold, config.max_candidates, masks))
        events.sort(key=lambda r: -r["snr"])
        events = _select_events(gpu64([[math.inf, e["epoch_bjd"], e["duration_hours"] / 24, e["snr"], e["depth"]]
                                       for e in events]) if events else torch.empty((0, 5), device="cuda", dtype=torch.float64),
                                time_all, config.threshold, config.max_candidates, masks)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    stats = {"device": torch.cuda.get_device_name(time_all.device), "dtype": "float64",
             "observations": len(time_all), "baseline_days": baseline,
             "periodic_period_max_days": min(config.period_max, baseline),
             "periods_tested": periods_tested, "qualifying_periods": int(torch.isfinite(full[:, 3]).sum()),
             "retained_ephemerides": len(full), "resolved_peaks": len(peaks), "single_events": len(events),
             "seconds_preparation": preparation_seconds, "seconds_search": elapsed,
             "seconds_total": preparation_seconds + elapsed,
             "families": [{"family": k, "window_days": curve.provenance.get("window_days"),
                           "durations_hours": [config.durations_hours[i] for i in indices],
                           "preparation": curve.provenance} for k, (curve, indices) in enumerate(families)],
             "passes": pass_stats, "evidence": evidence_stats, "single_event_scan": event_stats}
    return TransitSearchResult(full, stats, config, peaks, events)


def search_transits(time_bjd, flux, flux_err=None, sector=None, *, config=None, centroid=None, centroid_err=None):
    """Normalize, bin and detrend positive mission flux per duration family, then search on CUDA."""
    config = SearchConfig() if config is None else config
    if not isinstance(config, SearchConfig):
        raise TypeError("config must be a SearchConfig")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU is required")
    torch.cuda.synchronize()
    start = time.perf_counter()
    t, f, error, labels = _input_vectors(time_bjd, flux, flux_err, sector)
    centers, center_error = _centroid_vectors(centroid, centroid_err, len(t))
    families = prepare_families(t, f, error, labels, gpu64(config.durations_hours) / 24, bin_minutes=config.bin_minutes,
                                window_min=config.window_days, window_factor=config.window_factor,
                                window_max=config.window_max_days, clip_sigma=config.clip_sigma,
                                centroid=centers, centroid_err=center_error)
    torch.cuda.synchronize()
    time_all = torch.unique(torch.cat([curve.time for curve, _ in families]))
    return _search(families, config, time.perf_counter() - start, time_all)


def search_prepared(time_bjd, relative_flux, flux_err, *, config=None, centroid=None, centroid_err=None, sector=None):
    """Search detrended fractional flux with baseline zero and absolute-BJD times as one family."""
    config = SearchConfig() if config is None else config
    if not isinstance(config, SearchConfig):
        raise TypeError("config must be a SearchConfig")
    if flux_err is None:
        raise ValueError("Prepared input requires per-observation flux errors")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU is required")
    torch.cuda.synchronize()
    start = time.perf_counter()
    t, f, error, labels = _input_vectors(time_bjd, relative_flux, flux_err, sector)
    centers, center_error = _centroid_vectors(centroid, centroid_err, len(t))
    keep = torch.isfinite(t) & torch.isfinite(f) & torch.isfinite(error) & (error > 0)
    t, f, error = t[keep], f[keep], error[keep]
    order = t.argsort(stable=True)
    astrometry = prepare_astrometry(t[order], centers[keep][order], center_error[keep][order] if center_error is not None else None,
                                    labels[keep][order] if labels is not None else None) if centers is not None else None
    curve = PreparedCurve(t[order], f[order], error[order],
                          {"input_mode": "prepared", "observations": len(t), "window_days": None}, astrometry)
    torch.cuda.synchronize()
    return _search([(curve, list(range(len(config.durations_hours))))], config, time.perf_counter() - start, curve.time)


def _centroid_vectors(values, errors, count):
    if values is None:
        if errors is not None:
            raise ValueError("Centroid errors require centroid coordinates")
        return None,None
    c=gpu64(values)
    e=gpu64(errors) if errors is not None else None
    if c.shape!=(count,2) or (e is not None and e.shape!=c.shape):
        raise ValueError("Centroid coordinates and errors must have shape (observations, 2)")
    return c,e
