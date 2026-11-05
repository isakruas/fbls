"""False positive probability with TRICERATOPS (optional dependency, same environment).

TRICERATOPS' TESScut and SPOC aperture downloads are routed to the local SPOC
light-curve files, whose APERTURE extension carries the same WCS and CCD corner,
so nothing else is downloaded for the pixel geometry.
"""
import time
from pathlib import Path

import numpy as np


class LocalCutout:
    """Stands in for a TESScut download, built from the local SPOC light-curve file."""

    def __init__(self, path):
        from types import SimpleNamespace
        from astropy.io import fits
        with fits.open(path) as hdul:
            header = hdul["APERTURE"].header.copy()
            image = hdul["APERTURE"].data.astype(float)
        corner = fits.Header({"1CRV4P": header["CRVAL1P"], "2CRV4P": header["CRVAL2P"]})
        self.hdu = [None, SimpleNamespace(header=corner, data={"FLUX": image[None]}), SimpleNamespace(header=header)]


def use_local_files(paths):
    import lightkurve
    import triceratops.funcs as funcs
    from types import SimpleNamespace

    def search_tesscut(target, sector):
        return SimpleNamespace(download_all=lambda **_: [LocalCutout(paths[str(int(sector))])])
    lightkurve.search_tesscut = search_tesscut
    funcs.find_url = lambda ID, sector: paths[str(int(sector))]


def fpp_job(star, prep, metrics, root, period=None, runs=3, attempts=3):
    """TRICERATOPS false positive probability of one vetted signal, retried on network errors
    (the catalog queries go to MAST and Gaia)."""
    for attempt in range(attempts):
        result = fpp_once(star, prep, metrics, root, period, runs)
        if "error" not in result or not any(w in result["error"] for w in ("Connection", "Timeout", "timed out", "RemoteDisconnected")):
            return result
        time.sleep(30 * (attempt + 1))
    return result


def fpp_once(star, prep, metrics, root, period=None, runs=3):
    """One TRICERATOPS run. The light curve is folded within 1.5 durations and binned to 5 minutes;
    the pixel geometry comes from the local SPOC files. Returns FPP, NFPP, their Monte Carlo
    spread and the leading scenarios, or an error message."""
    try:
        import triceratops.triceratops as tr
    except ImportError:
        return {"error": "triceratops is not installed in this environment"}
    duration = metrics["duration_hours"] / 24
    sectors = sorted(set(prep["sector"][prep["valid"] & (np.abs(prep["phase"]) < duration / 2)].astype(int)))[-3:]
    near = prep["valid"] & (np.abs(prep["phase"]) < 1.5 * duration) & np.isin(prep["sector"], sectors)
    if near.sum() < 10:
        return {"error": "not enough points around the transit"}
    cells = np.floor(prep["phase"][near] / (5 / 1440)).astype(int)
    _, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
    phase = np.bincount(inverse, prep["phase"][near]) / counts
    flux = 1 + np.bincount(inverse, prep["corrected"][near]) / counts
    outside = np.abs(phase) > duration / 2
    flux_err = float(np.std(flux[outside])) if outside.sum() > 5 else float(prep["scatter"])
    root = Path(root)
    paths = {str(s): str(root / star["meta"][s]["fits"]) for s in sectors if s in star["meta"] and star["meta"][s].get("fits")}
    if len(paths) < len(sectors):
        return {"error": "SPOC files missing for some sectors"}
    use_local_files(paths)
    work = root / "triceratops"
    work.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    try:
        import os
        os.chdir(work)
        target = tr.target(ID=int(star["tic"]), sectors=np.array(sectors))
        if len(target.pix_coords) < len(sectors):
            return {"error": f"pixel geometry failed for {len(sectors) - len(target.pix_coords)} of {len(sectors)} sectors"}
        apertures = target.get_spoc_apertures()
        keep = [i for i, a in enumerate(apertures) if a is not None]
        if not keep:
            return {"error": "no SPOC apertures for the sectors"}
        if len(keep) < len(sectors):
            target = tr.target(ID=int(star["tic"]), sectors=np.array(sectors)[keep])
            apertures = [apertures[i] for i in keep]
        target.calc_depths(tdepth=max(metrics.get("depth_ppm") or 0, 1) / 1e6, all_ap_pixels=apertures)
        results = []
        for _ in range(runs):
            target.calc_probs(time=phase, flux_0=flux, flux_err_0=flux_err, P_orb=period or metrics["period_days"],
                              parallel=True, verbose=0)
            results.append([float(target.FPP), float(target.NFPP)])
        results = np.array(results)
        probs = target.probs.sort_values("prob", ascending=False).head(5)
        return {"fpp": float(results[:, 0].mean()), "nfpp": float(results[:, 1].mean()),
                "fpp_std": float(results[:, 0].std()), "nfpp_std": float(results[:, 1].std()),
                "sectors_used": [int(np.array(sectors)[i]) for i in keep],
                "top_scenarios": [{"id": str(r["ID"]), "scenario": r["scenario"], "prob": float(r["prob"])} for _, r in probs.iterrows()]}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"[-500:]}
    finally:
        os.chdir(cwd)
