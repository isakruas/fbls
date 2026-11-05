"""Local false-positive tests for a signal with given period, epoch and duration.

Inputs are the detrended relative flux of the detector family whose duration is
closest to the signal, with the other accepted signals masked, so the vetting
sees the same data the detection used. Tests: depth and SNR against the
out-of-transit noise at the transit time scale, odd/even depth difference,
secondary eclipse search at every phase, trapezoid shape fit, duration versus
stellar density, centroid shift versus target position, crowding. Thresholds
follow SPOC/TPS conventions: detection at 7.1 sigma, at least two transits.
"""
import numpy as np
from scipy import stats
from scipy.optimize import least_squares

from .catalog import wrap

G = 6.674e-11
DAY = 86400.
REARTH_PER_RSUN = 109.1
RSUN_AU = 0.00465047
PIXEL_ARCSEC = 21.
MIN_SNR = 7.1
MIN_TRANSITS = 2
ODD_EVEN_SIGMA = 4.
CENTROID_SIGMA = 4.
SECONDARY_SIGMA = 5.
MAX_RADIUS_EARTH = 25.
MAX_DURATION_FRACTION = .15  # t14 / P of a grazing transit at a/R* = 2; no stellar transit lasts longer
FAIL, WARN = "fail", "warn"


def fold(time, corrected, sector, period, epoch, duration_hours, masks=()):
    """Phase-fold a detrended light curve, masking the other signals of the star."""
    time, corrected = np.asarray(time, float), np.asarray(corrected, float)
    sector = np.zeros(len(time)) if sector is None else np.asarray(sector)
    duration = duration_hours / 24
    if len(time) < 200 or not (0 < duration < period / 2):
        raise ValueError("invalid coverage or duration")
    phase = wrap(time - epoch, period)
    inside = np.abs(phase) < duration / 2
    others = np.zeros(len(time), dtype=bool)
    for p, e, d in masks:
        others |= np.abs(wrap(time - e, p)) < d / 24
    valid = np.isfinite(corrected) & ~others
    outside = (np.abs(phase) > duration) & valid
    if (inside & valid).sum() < 5 or outside.sum() < 100:
        raise ValueError("not enough coverage of the predicted transit")
    steps = np.diff(time)
    cadence = float(np.median(steps[steps > 0]))
    out = corrected[outside]
    scatter = max(1.4826 * np.median(np.abs(out - np.median(out))), 1e-9)
    return {"time": time, "phase": phase, "corrected": corrected, "valid": valid, "outside": outside,
            "scatter": scatter, "cadence": cadence, "period": period, "epoch": epoch, "duration": duration,
            "sector": sector}


def window_noise(time, values, mask, width, cadence):
    """Noise of a mean over `width` days, measured out of transit (includes correlated noise)."""
    t, v = time[mask], values[mask]
    expected = width / cadence
    if len(t) < 10:
        return np.nan, expected
    _, inverse, counts = np.unique(np.floor((t - t[0]) / width), return_inverse=True, return_counts=True)
    means = np.bincount(inverse, v) / counts
    full = counts >= expected / 2
    white = 1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(expected)
    if full.sum() < 5:
        return white, expected
    red = 1.4826 * np.median(np.abs(means[full] - np.median(means[full])))
    return max(red, white), expected


def trapezoid(x, depth, total, ingress):
    half, ramp = total / 2, max(ingress * total, 1e-6)
    return -depth * np.clip((half - np.abs(x)) / ramp, 0, 1)


def fit_shape(prep):
    x = prep["phase"] / prep["duration"]
    ok = prep["valid"] & (np.abs(x) < 2)
    edges = np.linspace(-2, 2, 81)
    centers, profile = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        chosen = ok & (x >= lo) & (x < hi)
        if chosen.sum() >= 3:
            centers.append((lo + hi) / 2)
            profile.append(np.median(prep["corrected"][chosen]))
    if len(centers) < 20:
        return np.nan, np.nan, np.nan, np.nan
    centers, profile = np.array(centers), np.array(profile)
    start = [min(max(-profile.min(), 1e-6), .99), 1., .2]
    fit = least_squares(lambda p: trapezoid(centers, *p) - profile, start, bounds=([0, .3, .01], [1, 2.5, .5]))
    errors = np.full(3, np.nan)
    if len(profile) > 3:
        try:
            scale = np.sum(fit.fun ** 2) / (len(profile) - 3)
            variance = np.diag(np.linalg.inv(fit.jac.T @ fit.jac)) * scale
            errors = np.where(variance > 0, np.sqrt(np.abs(variance)), np.nan)
        except np.linalg.LinAlgError:
            pass
    return float(fit.x[2]), float(fit.x[1] * prep["duration"]), float(errors[2]), float(errors[1] * prep["duration"])


def geometry(period, depth, depth_err, t14, t14_err, ingress, ingress_err, cadence, star, a_over_r_tic, samples=4000):
    """Circular-orbit transit geometry from the trapezoid fit (Seager & Mallen-Ornelas 2003) with
    Monte Carlo propagation of the depth, duration and ingress uncertainties.

    Impact parameter, inclination, a/R* and the fitted stellar density are reported only when the
    ingress is resolved (longer than two cadences and measured to better than 50%), the total
    duration is measured to better than 20%, the depth is significant and most samples give a valid
    geometry. Insolation and equilibrium temperature (zero albedo, full redistribution) use the fitted
    a/R* when reliable and the a/R* implied by the TIC density otherwise, with TIC uncertainties of
    10% in that a/R* and 150 K in the effective temperature. Values are medians, errors half the 16-84% range.
    """
    def finite(x):
        return x is not None and np.isfinite(x)

    def summary(values):
        lo, mid, hi = np.nanpercentile(values, (16, 50, 84))
        return float(mid), float((hi - lo) / 2)

    out = {k: None for k in ("impact", "impact_err", "inclination_deg", "inclination_err", "a_over_rstar_fit", "a_over_rstar_fit_err",
                             "fitted_density_gcc", "fitted_density_err", "insolation_earth", "insolation_err", "teq_k", "teq_err")}
    reasons = []
    if not (finite(depth) and finite(depth_err) and depth_err > 0 and depth / depth_err >= MIN_SNR):
        reasons.append("depth")
    if not (finite(t14) and finite(t14_err) and t14_err < .2 * t14):
        reasons.append("duration")
    if not (finite(ingress) and finite(ingress_err) and ingress_err < .5 * ingress and ingress * t14 >= 2 * cadence):
        reasons.append("ingress")
    a_over_r = None
    if not reasons:
        rng = np.random.default_rng(0)
        d = rng.normal(depth, depth_err, samples)
        t = rng.normal(t14, t14_err, samples)
        f = rng.normal(ingress, ingress_err, samples)
        ok = (d > 0) & (t > 0) & (t < period / 2) & (f > 0) & (f < .5)
        k = np.sqrt(np.where(ok, d, np.nan))
        r = 1 - 2 * f
        b2 = ((1 - k) ** 2 - r ** 2 * (1 + k) ** 2) / (1 - r ** 2)
        s2 = np.sin(np.pi * t / period) ** 2
        ok &= (b2 >= 0) & (b2 < (1 + k) ** 2)
        b = np.sqrt(np.where(ok, b2, np.nan))
        ar = np.sqrt(((1 + k) ** 2 - b2 * (1 - s2)) / s2)
        ok &= np.isfinite(ar) & (ar > b)
        if ok.mean() >= .68:
            b, ar = np.where(ok, b, np.nan), np.where(ok, ar, np.nan)
            out["impact"], out["impact_err"] = summary(b)
            out["inclination_deg"], out["inclination_err"] = summary(np.degrees(np.arccos(b / ar)))
            out["a_over_rstar_fit"], out["a_over_rstar_fit_err"] = summary(ar)
            out["fitted_density_gcc"], out["fitted_density_err"] = summary(3 * np.pi * ar ** 3 / (G * (period * DAY) ** 2) / 1000)
            a_over_r = (out["a_over_rstar_fit"], out["a_over_rstar_fit_err"])
        else:
            reasons.append("geometry")
    if a_over_r is None and finite(a_over_r_tic):
        a_over_r = (a_over_r_tic, .1 * a_over_r_tic)
    teff = (star or {}).get("teff")
    if a_over_r is not None and teff and teff > 0:
        rng = np.random.default_rng(1)
        ar = rng.normal(*a_over_r, samples)
        te = rng.normal(teff, 150., samples)
        ok = (ar > 1) & (te > 0)
        ar, te = np.where(ok, ar, np.nan), np.where(ok, te, np.nan)
        out["teq_k"], out["teq_err"] = summary(te * np.sqrt(1 / (2 * ar)))
        out["insolation_earth"], out["insolation_err"] = summary((te / 5772.) ** 4 / (ar * RSUN_AU) ** 2)
    out["reliable"] = not reasons
    out["unreliable"] = reasons
    return out


def centroid_test(prep, centr1, centr2, depth, meta):
    """Source position from the flux-weighted centroid shift, compared with the target position.

    If a source at x_s dims a fraction delta of the aperture flux, the centroid moves by
    delta (c_out - x_s) / (1 - delta). Sectors are combined by weighted mean on each axis.
    """
    core = np.abs(prep["phase"]) < prep["duration"] / 4
    near = prep["outside"] & (np.abs(prep["phase"]) < 3 * prep["duration"])
    offsets = []
    for sector in np.unique(prep["sector"]):
        info = meta.get(int(sector), {})
        if not info.get("target_pixel") or not depth > 0:
            continue
        aperture_depth = depth * (info.get("crowdsap") or 1.)
        select = prep["sector"] == sector
        shifts = []
        for values, target in zip((centr1, centr2), info["target_pixel"]):
            good = select & np.isfinite(values)
            if (good & core).sum() < 3 or (good & near).sum() < 10:
                break
            reference = good & near
            baseline = np.median(values[reference])
            residual = values - baseline
            noise, expected = window_noise(prep["time"], residual, reference, prep["duration"] / 2, prep["cadence"])
            inside = good & core
            if not np.isfinite(noise) or inside.sum() < 3:
                break
            shift = np.mean(residual[inside]) - np.median(residual[reference])
            factor = (1 - aperture_depth) / aperture_depth
            source = baseline - shift * factor
            error = np.hypot(noise * np.sqrt(expected / inside.sum()) * factor, .1)
            shifts.append((source - target, error))
        if len(shifts) == 2:
            offsets.append(shifts)
    if not offsets:
        return np.nan, np.nan
    total = []
    for axis in range(2):
        weights = np.array([1 / o[axis][1] ** 2 for o in offsets])
        values = np.array([o[axis][0] for o in offsets])
        total.append((np.sum(weights * values) / weights.sum(), 1 / np.sqrt(weights.sum())))
    chi2 = sum((o / e) ** 2 for o, e in total)
    sigma = stats.norm.isf(max(stats.chi2.sf(chi2, 2), 1e-300) / 2)
    return float(min(sigma, 40.)), float(np.hypot(*[o for o, _ in total]) * PIXEL_ARCSEC)


def stellar_density(star):
    """Mean density (kg/m^3) from log g and radius."""
    if not star or star.get("logg") is None or star.get("radius") is None or not star["radius"] > 0:
        return np.nan
    gravity = 10 ** star["logg"] / 100
    return 3 * gravity / (4 * np.pi * G * star["radius"] * 6.957e8)


def measure(prep, star=None, centr1=None, centr2=None, crowding=None, meta=None):
    """Vetting metrics and flags of one folded signal."""
    star = star or {}
    period, duration = prep["period"], prep["duration"]
    corrected, valid, time = prep["corrected"], prep["valid"], prep["time"]
    noise, expected = window_noise(time, corrected, prep["outside"], duration / 2, prep["cadence"])
    box_noise, box_expected = window_noise(time, corrected, prep["outside"], .8 * duration, prep["cadence"])
    number = np.round((time - prep["epoch"]) / period)
    odd = number % 2 == 1

    def depth(offset=0., select=True, half=duration / 4):
        chosen = valid & select & (np.abs(wrap(prep["phase"] - offset * period, period)) < half)
        if chosen.sum() == 0:
            return np.nan, np.nan, 0
        sigma, points = (noise, expected) if half == duration / 4 else (box_noise, box_expected)
        error = sigma * np.sqrt(points / chosen.sum())
        cycle = np.round((time[chosen] - prep["epoch"] - offset * period) / period)
        _, inverse, counts = np.unique(cycle, return_inverse=True, return_counts=True)
        full = counts >= max(3, points / 4)
        events = -np.bincount(inverse, corrected[chosen])[full] / counts[full]
        if len(events) >= 5:
            error = max(error, 1.4826 * np.median(np.abs(events - np.median(events))) / np.sqrt(len(events)))
        return float(-np.mean(corrected[chosen])), float(error), int(full.sum())

    d, d_err, transits = depth()
    box, box_err, _ = depth(half=.4 * duration)
    d_odd, e_odd, n_odd = depth(select=odd)
    d_even, e_even, n_even = depth(select=~odd)
    odd_even = abs(d_odd - d_even) / np.hypot(e_odd, e_even) if min(n_odd, n_even) >= 2 else np.nan

    nbins = max(64, int(np.ceil(8 * period / duration)))
    cells = np.floor((prep["phase"] / period % 1) * nbins).astype(int) % nbins
    sums = np.bincount(cells[valid], corrected[valid], nbins)
    hits = np.bincount(cells[valid], minlength=nbins).astype(float)
    width = max(1, int(round(nbins * duration / (2 * period))))
    kernel = np.ones(width)
    window_sum = np.convolve(np.r_[sums, sums[:width]], kernel, "valid")[:nbins]
    window_hits = np.convolve(np.r_[hits, hits[:width]], kernel, "valid")[:nbins]
    center = ((np.arange(nbins) + width / 2) / nbins) % 1
    allowed = (np.abs(center - .5) < .5 - 1.5 * duration / period) & (window_hits >= max(3, expected / 4))
    sec_depth, sec_sigma, sec_phase, brightening, sec_threshold = np.nan, 0., np.nan, 0., np.inf
    if allowed.any():
        depths = -window_sum / np.maximum(window_hits, 1)
        sigmas = depths / (noise * np.sqrt(expected / np.maximum(window_hits, 1)))
        best = int(np.argmax(np.where(allowed, sigmas, -np.inf)))
        sec_phase = float(center[best])
        sec_depth, sec_err, _ = depth(offset=sec_phase)
        sec_sigma = float(max(sec_depth / sec_err, 0.)) if sec_err > 0 else 0.
        brightening = float(max(-sigmas[allowed].min(), 0.))
        sec_threshold = max(3., float(stats.norm.isf(.01 / max(1, allowed.sum() / width))))

    ingress, total, ingress_err, total_err = fit_shape(prep)
    density = stellar_density(star)
    k = np.sqrt(max(d, 0.))
    if np.isfinite(density):
        a_over_r = (G * density * (period * DAY) ** 2 / (3 * np.pi)) ** (1 / 3)
        albedo = sec_depth * a_over_r ** 2 / d if d > 0 and np.isfinite(sec_depth) else np.nan
    else:
        a_over_r = albedo = np.nan
    t14 = total if np.isfinite(total) else duration
    max_density = 3 * np.pi * (period / (np.pi * t14) * (1 + k)) ** 3 / (G * (period * DAY) ** 2)
    density_ratio = max_density / density if np.isfinite(density) else np.nan
    radius = k * star["radius"] * REARTH_PER_RSUN if star.get("radius") else np.nan
    if centr1 is not None and meta:
        centroid_sigma, centroid_arcsec = centroid_test(prep, centr1, centr2, d, meta)
    else:
        centroid_sigma = centroid_arcsec = np.nan
    snr = box / box_err if np.isfinite(box_err) and box_err > 0 else np.nan
    shape = geometry(period, d, d_err, total, total_err, ingress, ingress_err, prep["cadence"], star, a_over_r)

    flags = {}
    if not snr >= MIN_SNR:
        flags["low_snr"] = FAIL
    if transits < MIN_TRANSITS:
        flags["single_transit"] = FAIL
    if odd_even > ODD_EVEN_SIGMA:
        flags["odd_even"] = FAIL
    elif odd_even > 3:
        flags["odd_even"] = WARN
    if (sec_sigma > sec_threshold and sec_sigma - brightening > 1
            and (albedo > 1 if np.isfinite(albedo) else sec_depth > .1 * d)):
        flags["secondary_eclipse"] = FAIL if sec_sigma > SECONDARY_SIGMA else WARN
    if radius > MAX_RADIUS_EARTH:
        flags["too_large"] = FAIL
    if centroid_sigma > CENTROID_SIGMA and centroid_arcsec > PIXEL_ARCSEC:
        flags["centroid_offset"] = FAIL
    elif centroid_sigma > 3 and centroid_arcsec > PIXEL_ARCSEC / 2:
        flags["centroid_offset"] = WARN
    if crowding is not None and crowding < .8:
        flags["crowded"] = WARN
    if t14 / period > MAX_DURATION_FRACTION:
        flags["duration_too_long"] = FAIL
    elif density_ratio < .2:
        flags["duration_too_long"] = WARN
    if ingress > .4:
        flags["v_shape"] = WARN

    def num(v):
        return float(v) if v is not None and np.isfinite(v) else None
    return {
        "period_days": period, "epoch_bjd": prep["epoch"], "duration_hours": duration * 24,
        "depth_ppm": num(d * 1e6), "depth_err_ppm": num(d_err * 1e6), "snr": num(snr), "transits": transits,
        "odd_depth_ppm": num(d_odd * 1e6), "even_depth_ppm": num(d_even * 1e6), "odd_even_sigma": num(odd_even),
        "secondary_depth_ppm": num(sec_depth * 1e6), "secondary_sigma": num(sec_sigma), "secondary_phase": num(sec_phase),
        "secondary_albedo": num(albedo), "secondary_threshold_sigma": num(sec_threshold), "brightening_sigma": num(brightening),
        "ingress_fraction": num(ingress), "fitted_duration_hours": num(total * 24 if np.isfinite(total) else np.nan),
        "fitted_duration_err_hours": num(total_err * 24 if np.isfinite(total_err) else np.nan),
        "radius_earth": num(radius), "a_over_rstar": num(a_over_r), "density_ratio": num(density_ratio), "geometry": shape,
        "centroid_sigma": num(centroid_sigma), "source_offset_arcsec": num(centroid_arcsec),
        "crowdsap": num(crowding) if crowding is not None else None, "star": star,
        "sectors": sorted(int(s) for s in np.unique(prep["sector"])),
        "flags": flags, "verdict": FAIL if FAIL in flags.values() else "pass",
    }
