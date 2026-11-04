"""Complete GPU search on an explicit finite ephemeris/shape bank.

Heavy computation is CUDA float64 only. CPU work is limited to orchestration,
input loading and scalar metadata.
"""
from dataclasses import dataclass, field
import math
import time

import torch
import triton

from .kernels import fold_fields, scan_bounds, scan_floor, refine_modes
from .local_kernel import refine_local_modes
from .period_kernel import diagonal_tiles, diagonal_merge, fold_comparison, comparison_screen, joint_tiles, joint_merge


def gpu64(value):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this implementation has no CPU fallback")
    return torch.as_tensor(value, device="cuda", dtype=torch.float64).contiguous()


@dataclass
class NoiseModel:
    time: torch.Tensor
    deficit: torch.Tensor
    variance: torch.Tensor
    modes: torch.Tensor
    eigenvalues: torch.Tensor
    origin: float = 0.
    provenance: dict = field(default_factory=dict)

    def prepare(self):
        """Precision applications once per star; mode sums once per model."""
        t, y, d, u, lam = [gpu64(x) for x in
                           (self.time, self.deficit, self.variance, self.modes, self.eigenvalues)]
        n = len(t)
        if n < 6 or t.ndim != 1 or y.shape != t.shape or d.shape != t.shape:
            raise ValueError("At least six observations with matching time/deficit/variance are required")
        if u.shape != (n, len(lam)) or lam.ndim != 1:
            raise ValueError("Modes must be (observations, rank); eigenvalues must be (rank,)")
        if not bool(torch.isfinite(t).all() & torch.isfinite(y).all() & torch.isfinite(d).all() & (d > 0).all()):
            raise ValueError("Time, deficit and positive variances must be finite")
        if not bool((t[1:] >= t[:-1]).all()) or float(t[0]) < 0:
            raise ValueError("Relative timestamps must be sorted and nonnegative")
        if not bool(torch.isfinite(u).all() & torch.isfinite(lam).all() & (lam >= 0).all()):
            raise ValueError("Finite modes and nonnegative eigenvalues are required")
        order = torch.argsort(lam, descending=True)
        lam, u = lam[order], u[:, order].contiguous()
        if len(lam):
            orthogonality = float((u.T @ u - torch.eye(len(lam), device="cuda", dtype=t.dtype)).abs().max())
            if orthogonality > 1e-10:
                raise ValueError(f"Whitened modes are not orthonormal: error={orthogonality}")
        else:
            orthogonality = 0.
        inv_sqrt = d.rsqrt()
        beta = lam / (1 + lam)
        if len(beta) and not bool((beta < 1).all()):
            raise ArithmeticError("Covariance conditioning exceeds float64 modal precision")
        def precision(x):
            z = inv_sqrt * x
            return inv_sqrt * (z - u @ (beta * (u.T @ z)))
        p = precision(torch.ones_like(t))
        c = p.sum()
        if not bool(torch.isfinite(c) & (c > 0)):
            raise ArithmeticError("Invalid baseline precision; inspect covariance conditioning")
        py = precision(y)
        v = py - p * (py.sum() / c)
        fields = torch.stack((v, p, d.reciprocal(), torch.ones_like(t))).contiguous()
        # Only R x (N+1) is retained, never trial x N or trial x R.
        mode_prefix = torch.empty((len(lam), n + 1), device="cuda", dtype=t.dtype)
        mode_prefix[:, 0] = 0.
        if len(lam):
            torch.cumsum((u * inv_sqrt[:, None]).T.contiguous(), dim=1, out=mode_prefix[:, 1:])
            support_mask = (u != 0).T.to(torch.int8)
            first = support_mask.argmax(1)
            last = n - 1 - support_mask.flip(1).argmax(1)
            support = torch.stack((t[first], t[last]), dim=1).contiguous()
            max_support_span = float((support[:, 1] - support[:, 0]).max())
        else:
            support = torch.empty((0, 2), device="cuda", dtype=t.dtype)
            max_support_span = 0.
        group_constant = group_span = None
        if len(lam):
            groups, inverse = torch.unique(torch.stack((first, last), 1), dim=0, return_inverse=True)
            ordering = groups[:, 0].argsort()
            sorted_groups = groups[ordering]
            disjoint = len(groups) == 1 or bool((sorted_groups[1:, 0] > sorted_groups[:-1, 1]).all())
            if disjoint:
                # Sum beta_j max_i |u_ij|^2 within each disjoint support group.
                group_constant = torch.zeros(len(groups), device="cuda", dtype=t.dtype)
                group_constant.scatter_add_(0, inverse, beta * u.abs().amax(0).square())
                group_span = t[groups[:, 1]] - t[groups[:, 0]]
        return Prepared(t, fields, mode_prefix.contiguous(), beta.contiguous(), c,
                        self.origin, self.provenance, orthogonality, support, max_support_span,
                        group_constant, group_span, (y * v).sum())


@dataclass
class Prepared:
    time: torch.Tensor
    fields: torch.Tensor
    prefix: torch.Tensor
    beta: torch.Tensor
    baseline_norm: torch.Tensor
    origin: float
    provenance: dict
    orthogonality: float
    support: torch.Tensor
    max_support_span: float
    group_constant: torch.Tensor | None
    group_span: torch.Tensor | None
    residual_energy: torch.Tensor | None = None


@dataclass
class SearchResult:
    records: torch.Tensor
    scores: torch.Tensor
    depths: torch.Tensor
    modes_used: torch.Tensor
    statistics: dict
    origin: float

    def candidates(self, limit=20, period_tolerance=0.002, epoch_tolerance=1.):
        """GPU nonmaximum suppression; raw survivors remain available in records."""
        if limit < 1 or not len(self.scores):
            return []
        remaining = self.scores.clone()
        output = []
        for _ in range(min(limit, len(remaining))):
            index = int(torch.argmax(remaining))
            if not math.isfinite(float(remaining[index])):
                break
            row = self.records[index]
            p, scale, j, width = [float(x) for x in row[:4]]
            epoch = (j + width / 2) / scale + self.origin
            duration = width / scale
            output.append({"period_days": p, "epoch_bjd": epoch,
                           "duration_hours": duration * 24,
                           "ingress_fraction": float(row[7]),
                           "snr": float(self.scores[index]),
                           "depth": float(self.depths[index]),
                           "noise_modes_evaluated": int(self.modes_used[index])})
            periods = self.records[:, 0]
            epochs = (self.records[:, 2] + self.records[:, 3] / 2) / self.records[:, 1] + self.origin
            phase = (epochs - epoch + p / 2).remainder(p) - p / 2
            duplicate = ((periods / p - 1).abs() <= period_tolerance) & (phase.abs() <= duration * epoch_tolerance)
            remaining[duplicate] = -torch.inf
            remaining[index] = -torch.inf
        return output


def period_grid(minimum, maximum, baseline, minimum_duration, oversample=3.):
    """GPU logarithmic period grid with a uniform accumulated-time drift bound.

    Adjacent periods obey baseline * (P_next/P - 1) <= duration/oversample.
    """
    if not (0 < minimum < maximum and baseline > 0 and minimum_duration > 0 and oversample >= 1):
        raise ValueError("Invalid period-grid parameters")
    log_step = math.log1p(minimum_duration / (oversample * baseline))
    count = math.ceil(math.log(maximum / minimum) / log_step) + 1
    grid = torch.linspace(math.log(minimum), math.log(maximum), count, device="cuda", dtype=torch.float64).exp()
    grid[0], grid[-1] = minimum, maximum
    return grid.contiguous()


def fit_segment_noise(time_values, flux_values, errors=None, sectors=None,
                      bin_minutes=10., degree=2, gap_days=0.75):
    """Mission-flux preparation and empirical low-rank noise model on GPU.

    Normalize each sector by its median; never clip negative dips. Smooth
    Legendre modes on gap-separated segments are orthogonalized in whitened
    coordinates. Eigenvalues use the empirical estimate max(projection^2-1,0).
    Mode powers are estimated independently within each observing segment.
    """
    if degree not in range(0, 5) or bin_minutes < 0 or gap_days <= 0:
        raise ValueError("degree must be 0..4, bin_minutes >= 0 and gap_days > 0")
    t, flux = gpu64(time_values), gpu64(flux_values)
    err = gpu64(errors) if errors is not None else torch.ones_like(flux)
    sec = torch.as_tensor(sectors, device="cuda", dtype=torch.int64) if sectors is not None else torch.zeros_like(t, dtype=torch.int64)
    keep = torch.isfinite(t) & torch.isfinite(flux) & torch.isfinite(err) & (err > 0)
    t, flux, err, sec = t[keep], flux[keep], err[keep], sec[keep]
    order = t.argsort()
    t, flux, err, sec = t[order], flux[order], err[order], sec[order]
    if len(t) < 6:
        raise ValueError("Insufficient usable observations")
    for label in torch.unique(sec):
        mask = sec == label
        median = flux[mask].median()
        if float(median) <= 0:
            raise ValueError("Mission flux must have a positive median")
        flux[mask] = flux[mask] / median - 1
        err[mask] /= median
    origin = float(t[0])
    t = t - origin
    if bin_minutes:
        # Never merge sectors; an inverse-variance mean also propagates errors.
        cell = (t / (bin_minutes / 1440)).floor().to(torch.int64)
        keys = torch.stack((cell, sec), dim=1)
        unique, inv = torch.unique(keys, dim=0, return_inverse=True)
        w = err.square().reciprocal()
        denominator = torch.zeros(len(unique), device="cuda", dtype=torch.float64).scatter_add_(0, inv, w)
        def mean(x):
            return torch.zeros_like(denominator).scatter_add_(0, inv, x * w) / denominator
        t, flux, err, sec = mean(t), mean(flux), denominator.rsqrt(), unique[:, 1]
        order = t.argsort()
        t, flux, err, sec = t[order], flux[order], err[order], sec[order]
    delta = flux[1:] - flux[:-1]
    adjacent = (t[1:] - t[:-1] < gap_days) & (sec[1:] == sec[:-1])
    if not bool(adjacent.any()):
        raise ValueError("No adjacent observations available for noise estimation")
    standardized = delta[adjacent] / (err[1:][adjacent].square() + err[:-1][adjacent].square()).sqrt()
    inflation = (1.4826 * (standardized - standardized.median()).abs().median()).clamp_min(1.)
    variance = (err * inflation).square()
    if errors is None:
        white = (1.4826 / math.sqrt(2) * (delta[adjacent] - delta[adjacent].median()).abs().median()).clamp_min(1e-12)
        variance = torch.ones_like(t) * white.square()
    breaks = torch.where(((t[1:] - t[:-1]) > gap_days) | (sec[1:] != sec[:-1]))[0] + 1
    bounds = [0] + breaks.cpu().tolist() + [len(t)]
    rank = sum(min(degree + 1, b - a) for a, b in zip(bounds[:-1], bounds[1:])) if degree else 0
    u = torch.zeros((len(t), rank), device="cuda", dtype=torch.float64)
    cursor = 0
    if degree:
        for a, b in zip(bounds[:-1], bounds[1:]):
            x = 2 * (t[a:b] - t[a]) / (t[b - 1] - t[a]).clamp_min(1e-12) - 1
            polynomials = [torch.ones_like(x), x]
            for k in range(2, degree + 1):
                polynomials.append(((2 * k - 1) * x * polynomials[-1] - (k - 1) * polynomials[-2]) / k)
            columns = min(degree + 1, b - a)
            local = torch.stack(polynomials[:columns], dim=1) / variance[a:b].sqrt()[:, None]
            q = torch.linalg.qr(local, mode="reduced").Q
            u[a:b, cursor:cursor + columns] = q
            cursor += columns
    deficit = -flux
    projections = u.T @ (deficit / variance.sqrt())
    lam = (projections.square() - 1).clamp_min(0.)
    active = lam > 0
    u, lam = u[:, active].contiguous(), lam[active].contiguous()
    return NoiseModel(t, deficit, variance, u, lam, origin,
                      {"normalization": "per-sector median, no dip clipping",
                       "bin_minutes": bin_minutes, "polynomial_degree": degree,
                       "segments": len(bounds) - 1, "rank": len(lam),
                       "error_inflation": float(inflation),
                       "covariance_estimator": "positive empirical modal power minus white variance"})


def biweight_flatten(time, relative_flux, window=.75, gap=.5, c=5., iterations=10):
    """Robust Tukey-biweight location in a sliding time window, on GPU.

    Returns relative_flux detrended as (1 + f) / trend - 1. Segments are split
    at gaps longer than `gap` days. The scale is c times the median absolute
    deviation; location updates run for the specified number of iterations.
    """
    t, flux = time, 1 + relative_flux
    boundaries = [0] + (torch.where(t[1:] - t[:-1] > gap)[0] + 1).cpu().tolist() + [len(t)]
    output = torch.empty_like(flux)
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        local_t, local_f = t[a:b], flux[a:b]
        lo = torch.searchsorted(local_t, local_t - window / 2)
        hi = torch.searchsorted(local_t, local_t + window / 2, right=True)
        width = int((hi - lo).max())
        offsets = torch.arange(width, device="cuda")
        rows = max(1, 1_000_000 // width)
        for start in range(0, b - a, rows):
            index = lo[start:start + rows, None] + offsets
            mask = index < hi[start:start + rows, None]
            values = local_f[index.clamp_max(b - a - 1)]
            location = values.masked_fill(~mask, torch.nan).nanmedian(1).values
            for _ in range(iterations):
                deviations = (values - location[:, None]).abs().masked_fill(~mask, torch.nan)
                scale = (c * deviations.nanmedian(1).values).clamp_min(1e-12)
                u = (values - location[:, None]) / scale[:, None]
                weights = torch.where(mask & (u.abs() < 1), (1 - u.square()).square(), 0.)
                total = weights.sum(1)
                updated = (weights * values).sum(1) / total.clamp_min(1e-300)
                location = torch.where(total > 0, updated, location)
            output[a + start:a + start + len(location)] = local_f[start:start + len(location)] / location - 1
    return output


def correlated_noise_factors(time, residual, sigma, durations, min_fill=.5, minimum_bins=20):
    """Per-duration factor rho_D >= 1 for correlated noise at the transit time scale.

    Residuals are averaged in contiguous bins of length D with inverse-variance
    weights; rho_D is the robust (MAD) scatter of the standardized bin means.
    For white noise with correct errors rho_D is approximately one. Dividing
    the amplitude statistic by rho_D applies an empirical duration-wise scale
    correction estimated from the standardized residual means.
    """
    weights = sigma.square().reciprocal()
    steps = time[1:] - time[:-1]
    cadence = float(steps[steps > 0].median())
    factors = []
    for duration in durations.tolist() if torch.is_tensor(durations) else durations:
        cell = ((time - time[0]) / duration).floor().to(torch.int64)
        _, inverse, counts = torch.unique(cell, return_inverse=True, return_counts=True)
        total = torch.zeros(len(counts), device="cuda", dtype=torch.float64).scatter_add_(0, inverse, weights)
        mean = torch.zeros_like(total).scatter_add_(0, inverse, weights * residual) / total
        z = (mean * total.sqrt())[counts >= min_fill * duration / cadence]
        if len(z) < minimum_bins:
            factors.append(1.)
            continue
        factors.append(max(1., float(1.4826 * (z - z.median()).abs().median())))
    return torch.tensor(factors, device="cuda", dtype=torch.float64)


@dataclass
class PreparedCurve:
    time: torch.Tensor  # absolute BJD
    flux: torch.Tensor  # detrended relative flux (transit = negative)
    sigma: torch.Tensor  # white-noise errors after robust inflation
    provenance: dict


def prepare_curve(time_values, flux_values, errors=None, sectors=None, bin_minutes=10.,
                  window=.75, clip_sigma=4.):
    """Common detector input: normalize, bin, detrend and clip positive outliers (GPU).

    Negative excursions (dips) are never clipped. Errors are the propagated
    formal errors inflated to the robust adjacent-difference scatter.
    """
    diagonal = fit_segment_noise(time_values, flux_values, errors, sectors, bin_minutes=bin_minutes, degree=0)
    flat = biweight_flatten(diagonal.time, -diagonal.deficit, window)
    sigma = diagonal.variance.sqrt()
    scale = 1.4826 * (flat - flat.median()).abs().median()
    keep = torch.isfinite(flat) & (flat <= clip_sigma * scale)
    return PreparedCurve(diagonal.time[keep] + diagonal.origin, flat[keep], sigma[keep],
                         {**diagonal.provenance, "detrending": f"biweight window {window} d",
                          "positive_clip_sigma": clip_sigma, "clipped": int((~keep).sum()),
                          "observations": int(keep.sum())})


def search_curve(curve, periods, durations, threshold=4., noise_degree=2, correlated_noise=True,
                 ingress_fractions=(0., 0.2), band_factor=2., comparison=None, **options):
    """Best template per period over a full period range for one prepared curve.

    Periods are searched in bands spanning at most `band_factor`; each band
    uses durations below a quarter of its shortest period. The first five table
    columns are period, epoch_bjd, duration_days, score and depth. Model
    comparison appends evidence, binary diagnostics and selection flags.
    """
    if not math.isfinite(band_factor) or band_factor <= 1:
        raise ValueError("band_factor must be finite and greater than one")
    ps, ds = gpu64(periods), gpu64(durations).sort().values
    if (ps.ndim != 1 or not len(ps) or ds.ndim != 1 or not len(ds)
            or not bool(torch.isfinite(ps).all() & (ps > 0).all() & (ps[1:] >= ps[:-1]).all()
                        & torch.isfinite(ds).all() & (ds > 0).all())):
        raise ValueError("Use positive finite periods and durations, with periods sorted")
    if float(ds[0]) >= float(ps[0]) / 4:
        raise ValueError("At least one duration must be below one quarter of the minimum period")
    model = fit_segment_noise(curve.time, 1 + curve.flux, curve.sigma, bin_minutes=0, degree=noise_degree)
    ready = model.prepare()
    rho = correlated_noise_factors(model.time, -model.deficit, model.variance.sqrt(), ds) if correlated_noise else torch.ones_like(ds)
    tables, stats = [], []
    lo = float(ps[0])
    while lo <= float(ps[-1]):
        hi = lo * band_factor
        selected = (ps >= lo) & (ps < hi)
        chosen = ds < lo / 4
        if bool(selected.any()) and bool(chosen.any()):
            if comparison is not None:
                table, details = compare_search(ready, ps[selected], ds[chosen], comparison,
                    threshold=threshold, ingress_fractions=ingress_fractions,
                    noise_scale=rho[chosen], **options)
                tables.append(table)
            else:
                result = search(ready, ps[selected], ds[chosen], threshold=threshold, reduce="period",
                                ingress_fractions=ingress_fractions, noise_scale=rho[chosen], **options)
                row = result.records
                tables.append(torch.stack((row[:, 0], (row[:, 2] + row[:, 3] / 2) / row[:, 1] + result.origin,
                                           row[:, 3] / row[:, 1], result.scores, result.depths), dim=1))
                details = result.statistics
            stats.append({"band": [lo, min(hi, float(ps[-1]))], **{k: v for k, v in details.items()
                                                                     if k != "noise_model"}})
        lo = hi
    table = torch.cat(tables) if tables else torch.empty((0, 14 if comparison is not None else 5), device="cuda", dtype=torch.float64)
    return table[table[:, 0].argsort()], {"noise_model": model.provenance,
                                         "correlated_noise_factors": rho.tolist(), "bands": stats}


def compare_search(ready, periods, durations, comparison, threshold=7.1,
                   phase_oversample=4, ingress_fractions=(0., .2, .45), shape_layers=4,
                   period_batch=None, candidate_capacity=131072, max_fold_mib=128, noise_scale=None):
    """Joint planet/binary/noise search with integrated evidence and amplitude maxima.

    Each local model integrates a positive eclipse depth with a half-normal
    prior and sums normalized discrete scenario masses. Both orbital families
    share the fitted constant baseline and the complete supplied covariance.
    The second family models alternating primary events at twice the trial
    period. Phase, duration and shape are marginalized independently under
    both models. Scores are conditional period evidence ratios, not calibrated
    astrophysical probabilities. The amplitude threshold gates reported periods.
    """
    start = time.perf_counter()
    ready = ready.prepare() if isinstance(ready, NoiseModel) else ready
    ps, ds = gpu64(periods), gpu64(durations)
    if ps.ndim != 1 or ds.ndim != 1 or not len(ps) or not len(ds):
        raise ValueError("Nonempty period and duration vectors required")
    if not bool(torch.isfinite(ps).all() & (ps > 0).all() & (ps[1:] >= ps[:-1]).all()
                & torch.isfinite(ds).all() & (ds > 0).all()):
        raise ValueError("Finite positive periods and durations, with periods sorted, are required")
    rho = torch.ones_like(ds) if noise_scale is None else gpu64(noise_scale)
    if rho.shape != ds.shape or not bool(torch.isfinite(rho).all() & (rho >= 1).all()):
        raise ValueError("Duration noise scales must be finite and at least one")
    if ready.residual_energy is None:
        if not bool((rho == 1).all()):
            raise ValueError("Duration-dependent noise evidence requires residual_energy from NoiseModel.prepare")
        duration_log_mass = torch.zeros_like(ds)
    else:
        duration_log_mass = -.5 * ((len(ready.time) - 1) * rho.square().log()
                                   + ready.residual_energy / rho.square())
        duration_log_mass -= duration_log_mass.max()
    if (threshold < 0 or not math.isfinite(threshold) or phase_oversample < 1 or
            shape_layers < 1 or max_fold_mib <= 0 or not ingress_fractions or
            any(not math.isfinite(x) or not 0 <= x < .5 for x in ingress_fractions)):
        raise ValueError("Invalid comparison bank configuration")
    batch = 128 if period_batch is None else period_batch
    if batch < 1:
        raise ValueError("period_batch must be positive")
    # Eight-cell alignment makes all quarter-phase offsets and parity exact.
    nb = max(8, 4 * math.ceil(phase_oversample * float(ps[-1]) / float(ds.min()) / 4))
    bins, tile, n, rank = 2 * nb, 64, len(ready.time), len(ready.beta)
    nf = 4 + rank
    fields = torch.cat((ready.fields, ready.prefix[:, 1:] - ready.prefix[:, :-1]), dim=0).contiguous()
    ingress = gpu64(ingress_fractions)
    shape_logits = torch.stack((-4. * ingress, 2. * ingress), dim=1)
    shape_priors = (shape_logits - torch.logsumexp(shape_logits, dim=0)).contiguous()
    p_prior, b_prior = comparison["planet_prior"], comparison["binary_prior"]
    planet_scale = comparison["planet_depth_scale"]
    if comparison.get("stellar_radius_solar") is not None:
        planet_scale = (.10045 * comparison["planet_radius_scale_rj"] / comparison["stellar_radius_solar"]) ** 2
    dilution = comparison["target_flux_fraction"]
    planet_scale *= dilution
    parameter_values = [planet_scale, comparison["binary_depth_scale"] * dilution,
                        comparison["diluted_binary_depth_scale"],
                        math.log(p_prior), math.log(b_prior), math.log(1. - p_prior - b_prior)]
    parameter_values.extend(value for scenario in zip(
        (0., .01, .1, .5, 1.), (.75, .15, .08, .015, .005), (.10, .10, .20, .25, .35)) for value in scenario)
    parameter_values.extend((.25, .5, .75, 1., 4. / 3., 2., 4.))
    parameters = gpu64(parameter_values)
    if not bool(torch.isfinite(parameters).all() & (parameters[:3] > 0).all()):
        raise ValueError("Depth priors must remain positive and finite after stellar conditioning")
    tiles = triton.cdiv(nb, tile)
    output, batches, integrated_periods, omitted_groups, total_groups = [], 0, 0, 0, 0
    cursor = 0
    while cursor < len(ps):
        stop = min(cursor + batch, len(ps))
        pp = ps[cursor:stop]
        expected_bytes = len(pp) * (nf * (2 * bins + 1) + tiles * 2 * 15 + 28) * 8
        if expected_bytes > max_fold_mib * 2**20:
            if len(pp) == 1:
                raise MemoryError("One comparison fold exceeds max_fold_mib; increase the explicit budget")
            batch = max(1, len(pp) // 2)
            continue
        scales = (nb / pp).contiguous()
        folded = torch.zeros((len(pp), nf, bins), device="cuda", dtype=torch.float64)
        fold_comparison[(len(pp), nf, triton.cdiv(n, 256))](
            ready.time, fields, scales, folded, n, bins, nf, 256, enable_fp_fusion=False)
        cumulative = torch.cat((torch.zeros((len(pp), nf, 1), device="cuda", dtype=torch.float64),
                                folded.cumsum(2)), dim=2).contiguous()
        del folded
        screen = torch.empty((len(pp), tiles, 4), device="cuda", dtype=torch.float64)
        comparison_screen[(tiles, len(pp))](cumulative, ready.beta, scales, ds, rho, duration_log_mass, ingress,
            shape_priors, parameters, ready.baseline_norm, screen, n, bins, nf, rank, len(ds), len(ingress_fractions),
            shape_layers, tiles, tile, num_warps=4, enable_fp_fusion=False)
        # Every phase/shape is integrated for qualifying periods; no local evidence is truncated.
        eligible = screen[:, :, 0].amax(1) >= threshold * (1. - 1e-12)
        rtol = comparison.get("evidence_rtol", 1e-12)
        margin = math.log(rtol) - math.log(4 * nb * len(ds) * len(ingress_fractions)) if rtol else -math.inf
        winner = screen[:, :, 1].argmax(1)
        pilot_key = screen[torch.arange(len(pp), device="cuda"), winner, 3][eligible]
        floors = torch.cat((screen[:, :, 1:3].amax(1)[eligible] + margin, pilot_key[:, None]), dim=1).contiguous()
        cumulative, pp, scales = cumulative[eligible].contiguous(), pp[eligible].contiguous(), scales[eligible].contiguous()
        integrated_periods += len(pp)
        cursor, batches = stop, batches + 1
        if not len(pp):
            continue
        durations_in_bank = torch.floor(ds[None, :] * scales[:, None] + .5).clamp_min(1) < nb // 4
        log_mass = (math.log(nb) + torch.logsumexp(torch.where(durations_in_bank,
                    duration_log_mass[None, :], -torch.inf), dim=1)).contiguous()
        # Full scenario integration at one incumbent tightens bounds without restricting the search bank.
        pilot = torch.empty((len(pp), 1, 2, 15), device="cuda", dtype=torch.float64)
        joint_tiles[(1, len(pp))](cumulative, ready.beta, pp, scales, ds, rho, duration_log_mass, ingress,
            shape_priors, ready.baseline_norm, parameters, floors, pilot, n, bins, nf, rank,
            len(ds), len(ingress_fractions), shape_layers, 1, 1, threshold, True,
            num_warps=1, enable_fp_fusion=False)
        floors[:, :2] = torch.maximum(floors[:, :2], pilot[:, 0, 0, 11:13] + margin)
        del pilot
        parts = torch.empty((len(pp), tiles, 2, 15), device="cuda", dtype=torch.float64)
        table = torch.empty((len(pp), 2, 14), device="cuda", dtype=torch.float64)
        joint_tiles[(tiles, len(pp))](cumulative, ready.beta, pp, scales, ds, rho, duration_log_mass, ingress,
            shape_priors, ready.baseline_norm, parameters, floors, parts, n, bins, nf, rank,
            len(ds), len(ingress_fractions), shape_layers, tiles, tile, threshold,
            num_warps=4, enable_fp_fusion=False)
        omitted_groups += int(parts[:, :, 0, 13].sum())
        total_groups += len(pp) * tiles * len(ds) * len(ingress_fractions) * 4
        joint_merge[(len(pp), 2)](parts, pp, scales, ds, ingress, parameters, log_mass, table, tiles,
            len(ds), bins, triton.next_power_of_2(tiles), num_warps=4, enable_fp_fusion=False)
        hit = torch.isfinite(table[:, :, 3])
        # Identical maxima are represented once with both selection bits set.
        same = (table[:, 0, :5] == table[:, 1, :5]).all(1) & (table[:, 0, 12] == table[:, 1, 12]) & hit.all(1)
        table[same, 0, 13] = 3.
        hit[same, 1] = False
        selected = table[hit]
        selected[:, 1] += ready.origin
        output.append(selected)
        del cumulative, parts, table
    result = torch.cat(output) if output else torch.empty((0, 14), device="cuda", dtype=torch.float64)
    torch.cuda.synchronize()
    stats = {"backend": "joint_eclipse_evidence", "rank": rank, "periods": len(ps),
             "phase_bins_per_period": nb, "batches": batches,
             "periods_evidence_integrated": integrated_periods,
             "evidence_rtol": comparison.get("evidence_rtol", 1e-12),
             "evidence_groups_omitted": omitted_groups, "evidence_groups_total": total_groups,
             "templates_scanned": len(ps) * nb * len(ds) * len(ingress_fractions),
             "threshold": threshold, "seconds_search": time.perf_counter() - start,
             "effective_planet_depth_scale": planet_scale,
             "score_definition": "log(pi_planet Z_planet(P) / (pi_noise Z_noise + pi_binary Z_binary(P)))",
             "ephemeris_integration": "uniform phase and duration masses; normalized shape masses",
             "duration_noise_null_log_mass": duration_log_mass.tolist(),
             "evidence_model": "positive-depth linear eclipse surrogate, conditional on fitted noise",
             "binary_period_multipliers": [1, 2], "secondary_phases": [.25, .5, .75],
             "secondary_ratios": [0., .01, .1, .5, 1.],
             "planet_secondary_weights": [.75, .15, .08, .015, .005],
             "binary_secondary_weights": [.10, .10, .20, .25, .35],
             "secondary_phase_weights": [.1, .8, .1],
             "alternating_depth_ratios": [.25, .5, .75, 1., 4. / 3., 2., 4.],
             "binary_family_weights": [.5, .5], "binary_scale_weights": [.5, .5],
             "shape_log_masses": shape_priors.tolist()}
    return result, stats


def search(model, periods, durations, threshold=7.1, phase_oversample=4,
           ingress_fractions=(0., 0.2), shape_layers=4, period_batch=None,
           candidate_capacity=131072, adaptive=True, numerical_guard=1e-10,
           max_fold_mib=128, max_cycles=32768, reduce="all", noise_scale=None):
    """Search the finite bank; peak extraction is separate.

    reduce="all" returns every bank template above threshold. reduce="period"
    returns, for every period whose best template reaches the threshold,
    exactly that best template. Diagonal models use fused phase-tile scans;
    covariance models use partial-mode score bounds and modal refinement.
    Ties keep the lowest (shape, width, phase) index.
    Numerical slack expands score bounds at floating-point comparison boundaries.

    noise_scale optionally gives one correlated-noise factor (> 0) per
    duration; scores of that duration are divided by it before thresholding
    and maximization. Depths are not affected.

    Period batches shrink and retry on candidate-buffer overflow. They never
    silently truncate. Phase bins define the exact finite staircase bank;
    covariance-mode refinement uses the identical integer-cell convention.
    """
    if reduce not in ("all", "period"):
        raise ValueError("reduce must be 'all' or 'period'")
    period_tile, period_warps = 128, 4
    per_period = reduce == "period"
    if period_batch is None:
        period_batch = 512 if per_period else 32
    if threshold < 0 or phase_oversample < 1 or shape_layers < 1 or period_batch < 1 or candidate_capacity < 1:
        raise ValueError("Invalid search configuration")
    if not ingress_fractions or any(not 0 <= x < 0.5 for x in ingress_fractions):
        raise ValueError("Ingress fractions must be in [0, 0.5)")
    if numerical_guard < 0 or max_fold_mib <= 0:
        raise ValueError("Invalid numerical or memory budget")
    torch.cuda.synchronize()
    start = time.perf_counter()
    ready = model.prepare() if isinstance(model, NoiseModel) else model
    torch.cuda.synchronize()
    preparation = time.perf_counter() - start
    ps, ds = gpu64(periods), gpu64(durations)
    if ps.ndim != 1 or ds.ndim != 1 or not len(ps) or not len(ds):
        raise ValueError("Nonempty period and duration vectors required")
    if not bool(torch.isfinite(ps).all() & (ps > 0).all() & torch.isfinite(ds).all() & (ds > 0).all()):
        raise ValueError("Periods and durations must be finite and positive")
    if not bool((ps[1:] >= ps[:-1]).all()):
        raise ValueError("Periods must be sorted")
    rho = torch.ones_like(ds) if noise_scale is None else gpu64(noise_scale)
    if rho.shape != ds.shape or not bool(torch.isfinite(rho).all() & (rho > 0).all()):
        raise ValueError("noise_scale needs one positive finite factor per duration")
    minimum_duration = float(ds.min())
    # The finite template bank cannot change on a memory-budget retry.
    nb = max(8, math.ceil(phase_oversample * float(ps[-1]) / minimum_duration))
    n, rank = len(ready.time), len(ready.beta)
    fused_diagonal = per_period and rank == 0
    ingress_tensor = gpu64(ingress_fractions) if fused_diagonal else None
    shape_factors = gpu64([[1 - 2 * ingress * (layer / (shape_layers - 1) if shape_layers > 1 else 0.)
                           for layer in range(shape_layers)] for ingress in ingress_fractions]) if fused_diagonal else None
    tmax = float(ready.time[-1])
    localized = rank > 0 and ready.max_support_span < tmax / 4
    steps = math.ceil(math.log2(n + 1)) + 1
    beta0 = ready.beta[:1] if rank else torch.zeros(1, device="cuda", dtype=torch.float64)
    max_window_counts = None
    if ready.group_constant is not None:
        arange = torch.arange(n, device="cuda")
        max_window_counts = torch.stack([
            (torch.searchsorted(ready.time, ready.time + duration + minimum_duration / phase_oversample
                                + max(1e-12, tmax * 4e-15), right=True) - arange).max()
            for duration in ds
        ]).double()
    outputs, scores, depths, modal_counts = [], [], [], []
    no_floor = torch.zeros(1, device="cuda", dtype=torch.float64)
    stats = {"periods": len(ps), "rank": rank, "adaptive": adaptive, "reduction": reduce,
             "backend": "fused_diagonal" if fused_diagonal else "covariance_modes",
             "phase_tile_winners": 0,
             "exact_tile_retries": 0,
             "templates_scanned": 0, "upper_bound_survivors": 0,
             "mode_evaluations": 0, "batches": 0, "overflow_retries": 0,
             "phase_oversample": phase_oversample, "shape_layers": shape_layers,
             "threshold": threshold, "numerical_guard": numerical_guard,
             "orthogonality_error": ready.orthogonality,
             "localized_mode_support": localized,
             "support_energy_screening": max_window_counts is not None,
             "noise_model": ready.provenance, "seconds_preparation": preparation,
             "score_definition": "one-sided generalized least-squares amplitude statistic"}
    fold_time = scan_time = refine_time = 0.
    cursor, batch = 0, period_batch
    scan_start = time.perf_counter()
    stats["seconds_bank_setup"] = scan_start - start - preparation
    while cursor < len(ps):
        stop = min(cursor + batch, len(ps))
        pp = ps[cursor:stop]
        expected_bytes = len(pp) * 4 * (2 * nb + 1) * 8
        if expected_bytes > max_fold_mib * 2**20:
            if len(pp) == 1:
                raise MemoryError("One folded period exceeds max_fold_mib; increase the explicit memory budget")
            batch = max(1, len(pp) // 2)
            continue
        cycles = math.ceil(tmax / float(pp[0])) + 2
        if localized:
            cycles = math.ceil(ready.max_support_span / float(pp[0])) + 6
        if cycles + 1 > max_cycles:
            raise ValueError("Cycle register budget exceeded; split the observing baseline or raise max_cycles explicitly")
        scales = (nb / pp).contiguous()
        if max_window_counts is not None:
            coefficient = (ready.group_constant[None, :] *
                           ((ready.group_span[None, :] / pp[:, None]).floor() + 2)).amax(1)
            beta_bound = torch.minimum(beta0, coefficient[:, None] * max_window_counts[None, :]).contiguous()
        else:
            beta_bound = beta0.expand(len(pp), len(ds)).contiguous()
        folded = torch.zeros((len(pp), 4, nb), device="cuda", dtype=torch.float64)
        torch.cuda.synchronize(); tic = time.perf_counter()
        fold_fields[(len(pp), triton.cdiv(n, 256))](ready.time, ready.fields, scales, folded, n, nb, 256, enable_fp_fusion=False)
        cumulative = torch.cat((torch.zeros((len(pp), 4, 1), device="cuda", dtype=torch.float64), folded.cumsum(2)), dim=2).contiguous()
        del folded
        torch.cuda.synchronize(); fold_time += time.perf_counter() - tic
        if fused_diagonal:
            torch.cuda.synchronize(); tic = time.perf_counter()
            tiles = triton.cdiv(nb, period_tile)
            partials = tiles
            parts = torch.empty((len(pp), partials, 9), device="cuda", dtype=torch.float64)
            keys = torch.empty((len(pp), partials), device="cuda", dtype=torch.int64)
            invalid = torch.zeros(len(pp), device="cuda", dtype=torch.int32)
            retries = torch.empty((len(pp), partials), device="cuda", dtype=torch.int32)
            records = torch.empty((len(pp), 9), device="cuda", dtype=torch.float64)
            score = torch.empty(len(pp), device="cuda", dtype=torch.float64)
            depth = torch.empty_like(score)
            compiled_scan = diagonal_tiles[(tiles, len(pp))](
                cumulative, pp, scales, ds, rho, ingress_tensor, shape_factors, ready.baseline_norm, parts, keys, invalid, retries,
                n, nb, len(ds), len(ingress_fractions), shape_layers, tiles, period_tile,
                num_warps=period_warps, enable_fp_fusion=False)
            stats["scan_registers"] = compiled_scan.n_regs
            stats["scan_spills"] = compiled_scan.n_spills
            diagonal_merge[(len(pp),)](parts, keys, pp, scales, records, score, depth,
                                       partials, triton.next_power_of_2(partials), num_warps=4,
                                       enable_fp_fusion=False)
            if bool(invalid.any()):
                raise ArithmeticError("Nonpositive full-model template precision; inspect covariance conditioning")
            hit = torch.isfinite(score) & (score >= threshold)
            outputs.append(records[hit]); scores.append(score[hit]); depths.append(depth[hit])
            modal_counts.append(torch.zeros(int(hit.sum()), device="cuda", dtype=torch.int32))
            torch.cuda.synchronize(); scan_time += time.perf_counter() - tic
            stats["phase_tile_winners"] += len(pp) * partials
            stats["exact_tile_retries"] += int(retries.sum())
            stats["templates_scanned"] += len(pp) * len(ds) * len(ingress_fractions) * nb
            stats["batches"] += 1
            del cumulative, parts, keys, invalid, records, score, depth, retries
            cursor = stop
            continue
        floor = no_floor
        if per_period:
            torch.cuda.synchronize(); tic = time.perf_counter()
            floor_bits = torch.zeros(len(pp), device="cuda", dtype=torch.int64)
            for ingress in ingress_fractions:
                scan_floor[(len(pp), len(ds), triton.cdiv(nb, 128))](
                    cumulative, scales, ds, rho, ready.baseline_norm, floor_bits, n, nb, float(ingress),
                    1 if ingress == 0 else shape_layers, numerical_guard, 128, enable_fp_fusion=False)
            floor = floor_bits.view(torch.float64)
            torch.cuda.synchronize(); scan_time += time.perf_counter() - tic
        batch_outputs = []
        overflow = False
        for shape_index, ingress in enumerate(ingress_fractions):
            records = torch.empty((candidate_capacity, 9), device="cuda", dtype=torch.float64)
            counter = torch.zeros(1, device="cuda", dtype=torch.int32)
            layers = 1 if ingress == 0 else shape_layers
            torch.cuda.synchronize(); tic = time.perf_counter()
            scan_bounds[(len(pp), len(ds), triton.cdiv(nb, 128))](
                cumulative, pp, scales, ds, rho, beta_bound, ready.baseline_norm, records, counter, floor,
                n, nb, len(ds), candidate_capacity, threshold, float(ingress), layers,
                numerical_guard, 128, per_period, enable_fp_fusion=False)
            count = int(counter[0])
            torch.cuda.synchronize(); scan_time += time.perf_counter() - tic
            if count > candidate_capacity:
                overflow = True
                break
            if not count:
                continue
            records = records[:count]
            period_index = torch.searchsorted(pp, records[:, 0].contiguous())
            torch.cuda.synchronize(); tic = time.perf_counter()
            if rank == 0:
                # The rank-zero enclosure is already the exact score.
                q = records[:, 5] - records[:, 6]
                status = q <= 0
                score = torch.where(status, -torch.inf, records[:, 4] / q.clamp_min(1e-300).sqrt())
                depth = torch.where(status, 0., records[:, 4] / q.clamp_min(1e-300))
                used = torch.zeros(count, device="cuda", dtype=torch.int32)
            else:
                score = torch.empty(count, device="cuda", dtype=torch.float64)
                depth = torch.empty_like(score)
                used = torch.empty(count, device="cuda", dtype=torch.int32)
                status = torch.empty_like(used)
                candidate_floor = floor[period_index].contiguous() if per_period else no_floor
                if localized:
                    refine_local_modes[(count,)](
                        ready.time, ready.prefix, ready.beta, ready.support, records,
                        score, depth, used, status, candidate_floor, n, nb, rank, threshold,
                        adaptive, shape_layers, numerical_guard, steps, cycles,
                        min(64, triton.next_power_of_2(cycles + 1)), 16, per_period,
                        num_warps=4, enable_fp_fusion=False)
                else:
                    refine_modes[(count,)](ready.time, ready.prefix, ready.beta, ready.support, records,
                                           score, depth, used, status, candidate_floor, n, nb, rank, threshold,
                                           adaptive, shape_layers, numerical_guard, steps, cycles,
                                           min(256, triton.next_power_of_2(cycles + 1)), False, per_period,
                                           num_warps=4, enable_fp_fusion=False)
            if bool(status.any()):
                raise ArithmeticError("Nonpositive full-model template precision; inspect covariance conditioning")
            # Column 4 holds a/rho, so the kernels return a scaled depth.
            depth = depth * records[:, 8]
            torch.cuda.synchronize(); refine_time += time.perf_counter() - tic
            hit = torch.isfinite(score) & (score >= threshold)
            key = (shape_index * (nb + 1) + records[hit, 3].long()) * (nb + 1) + records[hit, 2].long()
            batch_outputs.append((records[hit].clone(), score[hit], depth[hit], used[hit],
                                  period_index[hit], key, count, int(used.sum())))
        del cumulative
        if overflow:
            stats["overflow_retries"] += 1
            if len(pp) == 1:
                raise MemoryError("One period exceeds candidate_capacity; increase capacity; no hypotheses were returned or dropped")
            batch = max(1, len(pp) // 2)
            continue
        for *_, count, used_sum in batch_outputs:
            stats["upper_bound_survivors"] += count
            stats["mode_evaluations"] += used_sum
        if per_period and batch_outputs:
            rr, ss, dd, mm, index, key = [torch.cat(column) for column in zip(*[item[:6] for item in batch_outputs])]
            if len(ss):
                best = torch.full((len(pp),), -torch.inf, device="cuda", dtype=torch.float64)
                best.scatter_reduce_(0, index, ss, reduce="amax")
                top = ss == best[index]
                big = torch.iinfo(torch.int64).max
                first_key = torch.full((len(pp),), big, device="cuda", dtype=torch.int64)
                first_key.scatter_reduce_(0, index, torch.where(top, key, big), reduce="amin")
                position = torch.arange(len(ss), device="cuda")
                chosen = torch.full((len(pp),), len(ss), device="cuda", dtype=torch.int64)
                chosen.scatter_reduce_(0, index, torch.where(top & (key == first_key[index]), position, len(ss)),
                                       reduce="amin")
                chosen = chosen[chosen < len(ss)]
                outputs.append(rr[chosen]); scores.append(ss[chosen])
                depths.append(dd[chosen]); modal_counts.append(mm[chosen])
        elif batch_outputs:
            for rr, ss, dd, mm, *_ in batch_outputs:
                outputs.append(rr); scores.append(ss); depths.append(dd); modal_counts.append(mm)
        stats["templates_scanned"] += len(pp) * len(ds) * len(ingress_fractions) * nb
        stats["batches"] += 1
        cursor = stop
    torch.cuda.synchronize()
    stats.update({"seconds_search": time.perf_counter() - scan_start,
                  "seconds_fold": fold_time, "seconds_scan": scan_time,
                  "seconds_refine": refine_time,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "mode_prefix_mib": ready.prefix.numel() * 8 / 2**20})
    def cat_or_empty(items, columns=None, dtype=torch.float64):
        return torch.cat(items) if items else torch.empty((0, columns) if columns else (0,), device="cuda", dtype=dtype)
    rr, ss, dd, mm = cat_or_empty(outputs, 9), cat_or_empty(scores), cat_or_empty(depths), cat_or_empty(modal_counts, dtype=torch.int32)
    stats["threshold_survivors"] = len(ss)
    torch.cuda.synchronize()
    stats["seconds_total"] = time.perf_counter() - start
    return SearchResult(rr, ss, dd, mm, stats, ready.origin)
