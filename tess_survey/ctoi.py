"""Candidate rows in the column layout of the ExoFOP CTOI upload table.

Values come from the detector (period, epoch, duration, depth) and the vetting
metrics; uncertainties are propagated from the search grid and the measured
noise, not from a posterior fit. Check the current upload template on ExoFOP
before submitting.
"""
import math

COLUMNS = ["TIC ID", "Transit Epoch (BJD)", "Transit Epoch (BJD) err", "Period (days)", "Period (days) Error",
           "Depth ppm", "Depth ppm Error", "Duration (hrs)", "Duration (hrs) Error", "Impact Param",
           "Impact Param Error", "Rad_p/Rad_s", "Rad_p/Rad_s Error", "a/Rad_s", "a/Rad_s Error",
           "Inclination (deg)", "Inclination (deg) Error", "Planet Radius (R_Earth)", "Planet Radius (R_Earth) Error",
           "Insolation (Earth Flux)", "Equilibrium Temp (K)", "Fitted Stellar Density (g/cm^3)",
           "Fitted Stellar Density (g/cm^3) Error", "User Disposition", "Notes"]


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def ctoi_row(event, author):
    """CTOI values for one vetted candidate; fields without a measurement stay empty."""
    m, peak = event["metrics"], event["peak"]
    period, epoch, duration_h = m["period_days"], m["epoch_bjd"], m["duration_hours"]
    snr = m.get("snr") or peak.get("snr") or 7.1
    baseline = event.get("baseline_days") or period
    # Period resolution of the search grid (one duration of drift over the baseline) and
    # epoch precision from the transit duration and the detection significance.
    period_err = period * (duration_h / 24) / (2 * baseline) if event.get("search") == "periodic" else None
    epoch_err = (duration_h / 24) / max(snr, 1.)
    depth, depth_err = m.get("depth_ppm"), m.get("depth_err_ppm")
    k = math.sqrt(depth / 1e6) if finite(depth) and depth > 0 else None
    k_err = (depth_err / 1e6) / (2 * k) if k and finite(depth_err) else None
    radius = m.get("radius_earth")
    radius_err = radius * k_err / k if finite(radius) and k and k_err else None
    fpp = event.get("fpp") or {}
    notes = (f"tess-gpu-search blind search ({event.get('search', 'periodic')}); detector SNR {peak.get('snr', 0):.1f}, "
             f"{m.get('transits')} transits; tier {event.get('tier')}; flags "
             f"{';'.join(f'{a}={b}' for a, b in m.get('flags', {}).items()) or 'none'}; "
             f"FPP {fpp.get('fpp')}; NFPP {fpp.get('nfpp')}; {author}")
    return {
        "TIC ID": event["tic"], "Transit Epoch (BJD)": epoch, "Transit Epoch (BJD) err": epoch_err,
        "Period (days)": period if event.get("search") == "periodic" else None, "Period (days) Error": period_err,
        "Depth ppm": depth, "Depth ppm Error": depth_err, "Duration (hrs)": duration_h,
        "Duration (hrs) Error": duration_h * .25, "Impact Param": None, "Impact Param Error": None,
        "Rad_p/Rad_s": k, "Rad_p/Rad_s Error": k_err, "a/Rad_s": m.get("a_over_rstar"), "a/Rad_s Error": None,
        "Inclination (deg)": None, "Inclination (deg) Error": None, "Planet Radius (R_Earth)": radius,
        "Planet Radius (R_Earth) Error": radius_err, "Insolation (Earth Flux)": None, "Equilibrium Temp (K)": None,
        "Fitted Stellar Density (g/cm^3)": None, "Fitted Stellar Density (g/cm^3) Error": None,
        "User Disposition": "PC", "Notes": notes,
    }
