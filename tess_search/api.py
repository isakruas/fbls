"""Pipeline-facing input preparation, search configuration and GPU results."""
from dataclasses import asdict, dataclass
import math
import time

import torch

from .core import PreparedCurve, gpu64, period_grid, prepare_curve, search_curve


DEFAULT_DURATIONS_HOURS = tuple(round(.5 * 1.25 ** i, 6) for i in range(13))
TABLE_COLUMNS = ("period_days", "epoch_bjd", "duration_days", "snr", "depth")


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
    ingress_fractions: tuple[float, ...] = (0., .2)
    shape_layers: int = 4
    phase_oversample: int = 4
    period_oversample: float = 3.
    max_candidates: int = 20
    period_batch: int | None = None
    candidate_capacity: int = 131072
    max_fold_mib: float = 128.

    def __post_init__(self):
        durations = tuple(sorted(set(float(d) for d in self.durations_hours)))
        ingress = tuple(float(x) for x in self.ingress_fractions)
        object.__setattr__(self, "durations_hours", durations)
        object.__setattr__(self, "ingress_fractions", ingress)
        for name in ("period_min", "period_max", "threshold", "bin_minutes", "window_days",
                     "clip_sigma", "period_oversample", "max_fold_mib"):
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
        if not isinstance(self.correlated_noise, bool):
            raise ValueError("correlated_noise must be boolean")
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
    """GPU float64 table with one row per qualifying period and scalar run metadata."""
    table: torch.Tensor
    statistics: dict
    config: SearchConfig

    def candidates(self, limit=None):
        """Select descending-score peaks with period/epoch suppression on the GPU."""
        limit = self.config.max_candidates if limit is None else limit
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("Candidate limit must be a nonnegative integer")
        remaining = self.table[:, 3].clone()
        output = []
        for _ in range(min(limit, len(self.table))):
            i = int(remaining.argmax())
            if not math.isfinite(float(remaining[i])):
                break
            p, epoch, duration, score, depth = [float(x) for x in self.table[i]]
            output.append({"period_days": p, "epoch_bjd": epoch, "duration_hours": 24 * duration,
                           "snr": score, "depth": depth, "depth_ppm": 1e6 * depth})
            phase = (self.table[:, 1] - epoch + p / 2).remainder(p) - p / 2
            duplicate = ((self.table[:, 0] / p - 1).abs() <= .002) & (phase.abs() <= duration)
            remaining[duplicate] = -torch.inf
            remaining[i] = -torch.inf
        return output

    def to_dict(self):
        """Convert selected candidates and scalar metadata into a JSON-compatible mapping."""
        from . import __version__
        return {"schema_version": 1, "software_version": __version__, "config": asdict(self.config),
                "statistics": self.statistics, "candidates": self.candidates()}


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
        candidate_capacity=config.candidate_capacity, max_fold_mib=config.max_fold_mib)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    stats = {"device": torch.cuda.get_device_name(curve.time.device), "dtype": "float64",
             "observations": len(curve.time), "baseline_days": baseline,
             "periods_tested": len(periods), "qualifying_periods": len(table),
             "seconds_preparation": preparation_seconds, "seconds_search": elapsed,
             "seconds_total": preparation_seconds + elapsed,
             "preparation": curve.provenance, **details}
    return TransitSearchResult(table, stats, config)


def search_transits(time_bjd, flux, flux_err=None, sector=None, *, config=None):
    """Normalize and detrend positive mission flux, then search on CUDA in float64."""
    config = SearchConfig() if config is None else config
    if not isinstance(config, SearchConfig):
        raise TypeError("config must be a SearchConfig")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU is required")
    torch.cuda.synchronize()
    start = time.perf_counter()
    t, f, error, labels = _input_vectors(time_bjd, flux, flux_err, sector)
    curve = prepare_curve(t, f, error, labels, bin_minutes=config.bin_minutes,
                          window=config.window_days, clip_sigma=config.clip_sigma)
    torch.cuda.synchronize()
    return _search(curve, config, time.perf_counter() - start)


def search_prepared(time_bjd, relative_flux, flux_err, *, config=None):
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
    t, f, error, _ = _input_vectors(time_bjd, relative_flux, flux_err)
    keep = torch.isfinite(t) & torch.isfinite(f) & torch.isfinite(error) & (error > 0)
    t, f, error = t[keep], f[keep], error[keep]
    order = t.argsort()
    curve = PreparedCurve(t[order], f[order], error[order],
                          {"input_mode": "prepared", "observations": len(t)})
    torch.cuda.synchronize()
    return _search(curve, config, time.perf_counter() - start)
