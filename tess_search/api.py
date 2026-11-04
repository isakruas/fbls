"""Pipeline-facing input preparation, search configuration and GPU results."""
from dataclasses import asdict, dataclass
import math
import time

import torch
import triton

from .core import PreparedCurve, gpu64, period_grid, prepare_curve, prepare_astrometry, search_curve
from .period_kernel import evidence_peaks


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
    window_days: float = .75
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
    peak_drift_durations: float = 2.
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
                     "clip_sigma", "period_oversample", "max_fold_mib", "planet_prior", "binary_prior",
                     "planet_depth_scale", "binary_depth_scale", "diluted_binary_depth_scale",
                     "planet_radius_scale_rj", "target_flux_fraction", "peak_drift_durations",
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
        if not isinstance(self.correlated_noise, bool) or not isinstance(self.model_comparison, bool):
            raise ValueError("correlated_noise and model_comparison must be boolean")
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
        if self.peak_drift_durations < 0:
            raise ValueError("peak_drift_durations must be nonnegative")
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
        if (self.threshold < 0 or self.bin_minutes < 0 or self.window_days <= 0 or
                self.clip_sigma <= 0 or self.period_oversample < 1 or self.max_fold_mib <= 0):
            raise ValueError("Invalid threshold, preparation window or resource budget")
        integers = ((self.noise_degree, 0, 4), (self.shape_layers, 1, None),
                    (self.eccentricity_nodes,0,16),(self.periastron_nodes,4,64),
                    (self.phase_oversample, 1, None), (self.max_candidates, 0, None),
                    (self.candidate_capacity, 1, None))
        for value, minimum, maximum in integers:
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
                raise ValueError("Invalid integer search parameter")
        if self.period_batch is not None and (isinstance(self.period_batch, bool)
                or not isinstance(self.period_batch, int) or self.period_batch < 1):
            raise ValueError("period_batch must be a positive integer or None")


@dataclass
class TransitSearchResult:
    """GPU float64 period maxima for independent model and amplitude rankings."""
    table: torch.Tensor
    statistics: dict
    config: SearchConfig

    def candidates(self, limit=None, *, ranking="planet"):
        """Prioritize model-preferred evidence peaks or independently select amplitude peaks."""
        limit = self.config.max_candidates if limit is None else limit
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("Candidate limit must be a nonnegative integer")
        if ranking not in ("planet", "signal"):
            raise ValueError("ranking must be 'planet' or 'signal'")
        joint = ranking == "planet" and self.config.model_comparison
        flag = 1 if joint else 2
        eligible = (self.table[:, 13].long() & flag) != 0
        if joint and self.config.peak_drift_durations and self.statistics.get("baseline_days") and len(self.table):
            rows = torch.nonzero(eligible).flatten()
            bank = self.table[rows].contiguous()
            if len(bank):
                baseline = self.statistics["baseline_days"]
                maximum_duration = min(max(self.config.durations_hours) / 24, self.config.period_max / 4)
                radius = self.config.peak_drift_durations * maximum_duration / baseline
                count = self.statistics["periods_tested"]
                step = math.log(self.config.period_max / self.config.period_min) / max(1, count - 1)
                window = len(bank) - 1 if radius >= 1 else min(len(bank) - 1, math.ceil(-math.log1p(-radius) / step) + 2)
                peaks = torch.empty(len(bank), device=self.table.device, dtype=torch.bool)
                evidence_peaks[(len(bank),)](bank, peaks, len(bank), radius, window,
                                             min(128, triton.next_power_of_2(2 * window + 1)),
                                             num_warps=4, enable_fp_fusion=False)
                eligible[rows] = peaks
        preferred = eligible & (self.table[:, 5] >= self.config.planet_log_odds_min) if joint else eligible
        remaining = torch.where(preferred, self.table[:, 6 if joint else 3], -torch.inf)
        fallback = torch.where(eligible & ~preferred, self.table[:, 5], -torch.inf)
        output = []
        for _ in range(min(limit, len(self.table))):
            i = int(remaining.argmax())
            if not math.isfinite(float(remaining[i])):
                if not joint or not bool(torch.isfinite(fallback).any()):
                    break
                remaining, fallback = fallback, torch.full_like(fallback, -torch.inf)
                i = int(remaining.argmax())
            row = [float(x) for x in self.table[i]]
            p, epoch, duration, score, depth = row[:5]
            candidate = {"period_days": p, "epoch_bjd": epoch, "duration_hours": 24 * duration,
                         "snr": score, "depth": depth, "depth_ppm": 1e6 * depth}
            if self.config.model_comparison:
                candidate.update({name: value if math.isfinite(value) else None
                                  for name, value in zip(TABLE_COLUMNS[5:], row[5:])})
                candidate["model_preferred"] = row[5] >= self.config.planet_log_odds_min
                candidate["binary_diagnostics_complete"] = all(math.isfinite(v) for v in row[8:12])
            output.append(candidate)
            phase = (self.table[:, 1] - epoch + p / 2).remainder(p) - p / 2
            duplicate = ((self.table[:, 0] / p - 1).abs() <= .002) & (phase.abs() <= duration)
            remaining[duplicate] = -torch.inf
            fallback[duplicate] = -torch.inf
            remaining[i] = -torch.inf
        return output

    def to_dict(self):
        """Convert selected candidates and scalar metadata into a JSON-compatible mapping."""
        from . import __version__
        return {"schema_version": 2, "software_version": __version__, "config": asdict(self.config),
                "statistics": self.statistics, "table_columns": TABLE_COLUMNS,
                "candidates": self.candidates(), "signal_candidates": self.candidates(ranking="signal")}


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


def _search(curve, config, preparation_seconds):
    torch.cuda.synchronize()
    start = time.perf_counter()
    if len(curve.time) < 6:
        raise ValueError("At least six usable observations are required")
    baseline = float(curve.time[-1] - curve.time[0])
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError("Observations must span a positive finite time interval")
    durations = gpu64(config.durations_hours) / 24
    periods = period_grid(config.period_min, config.period_max, baseline,
                          float(durations.min()), config.period_oversample)
    table, details = search_curve(curve, periods, durations, threshold=config.threshold,
        noise_degree=config.noise_degree, correlated_noise=config.correlated_noise,
        ingress_fractions=config.ingress_fractions, shape_layers=config.shape_layers,
        phase_oversample=config.phase_oversample, period_batch=config.period_batch,
        candidate_capacity=config.candidate_capacity, max_fold_mib=config.max_fold_mib,
        comparison={key: getattr(config, key) for key in ("planet_prior", "binary_prior",
            "planet_depth_scale", "binary_depth_scale", "diluted_binary_depth_scale",
            "stellar_radius_solar", "planet_radius_scale_rj", "target_flux_fraction", "evidence_rtol",
            "extended_physics","limb_darkening","eccentricity_nodes","periastron_nodes","eccentricity_max",
            "stellar_mass_solar","stellar_logg","stellar_teff","stellar_tmag","stellar_duration_scatter",
            "ellipsoidal","ellipsoidal_planet_scale","ellipsoidal_binary_scale","ellipsoidal_alpha",
            "planet_mass_scale_mj","centroid_planet_scale","centroid_binary_scale")} if config.model_comparison else None)
    if not config.model_comparison:
        extra = torch.zeros((len(table), 9), device="cuda", dtype=torch.float64)
        extra[:, -1] = 2.
        table = torch.cat((table, extra), dim=1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    stats = {"device": torch.cuda.get_device_name(curve.time.device), "dtype": "float64",
             "observations": len(curve.time), "baseline_days": baseline,
             "periods_tested": len(periods), "qualifying_periods": len(torch.unique(table[:, 0])),
             "retained_ephemerides": len(table),
             "seconds_preparation": preparation_seconds, "seconds_search": elapsed,
             "seconds_total": preparation_seconds + elapsed,
             "preparation": curve.provenance, **details}
    return TransitSearchResult(table, stats, config)


def search_transits(time_bjd, flux, flux_err=None, sector=None, *, config=None, centroid=None, centroid_err=None):
    """Normalize and detrend positive mission flux, then search on CUDA in float64."""
    config = SearchConfig() if config is None else config
    if not isinstance(config, SearchConfig):
        raise TypeError("config must be a SearchConfig")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU is required")
    torch.cuda.synchronize()
    start = time.perf_counter()
    t, f, error, labels = _input_vectors(time_bjd, flux, flux_err, sector)
    centers, center_error = _centroid_vectors(centroid,centroid_err,len(t))
    curve = prepare_curve(t, f, error, labels, bin_minutes=config.bin_minutes,
                          window=config.window_days, clip_sigma=config.clip_sigma,centroid=centers,centroid_err=center_error)
    torch.cuda.synchronize()
    return _search(curve, config, time.perf_counter() - start)


def search_prepared(time_bjd, relative_flux, flux_err, *, config=None, centroid=None, centroid_err=None, sector=None):
    """Search detrended fractional flux with baseline zero and absolute-BJD times."""
    config = SearchConfig() if config is None else config
    if not isinstance(config, SearchConfig):
        raise TypeError("config must be a SearchConfig")
    if flux_err is None:
        raise ValueError("Prepared input requires per-observation flux errors")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU is required")
    torch.cuda.synchronize()
    start = time.perf_counter()
    t, f, error, labels = _input_vectors(time_bjd, relative_flux, flux_err,sector)
    centers, center_error = _centroid_vectors(centroid,centroid_err,len(t))
    keep = torch.isfinite(t) & torch.isfinite(f) & torch.isfinite(error) & (error > 0)
    t, f, error = t[keep], f[keep], error[keep]
    order = t.argsort(stable=True)
    astrometry=prepare_astrometry(t[order],centers[keep][order],center_error[keep][order] if center_error is not None else None,
        labels[keep][order] if labels is not None else None) if centers is not None else None
    curve = PreparedCurve(t[order], f[order], error[order],
                          {"input_mode": "prepared", "observations": len(t)},astrometry)
    torch.cuda.synchronize()
    return _search(curve, config, time.perf_counter() - start)


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
