"""Difference images from SPOC target pixel files: which pixel the transit comes from.

The light-curve file holds only the aperture sum and its centroid; the target
pixel file (TPF) holds every pixel at every cadence. The out-of-transit image
minus the in-transit image shows where the light was lost; a Gaussian fit
locates the source and its offset from the target position (WCS of the same
stamp) is the pixel-level source test. TPFs (~50 MB per sector) are fetched on
demand, for up to three sectors that contain transits, in parallel byte ranges.
"""
from pathlib import Path

import numpy as np

from .fetch import download

MAX_SECTORS = 3
MIN_DIFFERENCE_SNR = 5.
OFFSET_SIGMA = 4.
OFFSET_PIXELS = .5
SYSTEMATIC_PIXELS = .1
PIXEL_ARCSEC = 21.
FAIL, WARN = "fail", "warn"


def mirror_url(tic, sector, filename):
    t = f"{int(tic):016d}"
    return (f"https://stpubdata.s3.amazonaws.com/tess/public/tid/s{int(sector):04d}/"
            f"{t[0:4]}/{t[4:8]}/{t[8:12]}/{t[12:16]}/{filename}")


def tpf_file(star, sector, root):
    """Local TPF for a sector, downloaded on first use (AWS mirror first, MAST as fallback)."""
    name = Path(star["meta"][int(sector)]["product"]).name.replace("_lc.fits", "_tp.fits")
    path = Path(root) / "mast" / "tp" / name
    if not path.exists():
        if not download(mirror_url(star["tic"], sector, name), path) and \
                not download(f"https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/{name}", path):
            raise RuntimeError(f"TPF download failed for sector {sector}")
    return path


def read_tpf(path):
    from astropy.io import fits
    from astropy.wcs import WCS
    from lightkurve.utils import TessQualityFlags
    with fits.open(path) as hdul:
        data, header = hdul[1].data, hdul[1].header
        time = data["TIME"] + header["BJDREFI"] + header["BJDREFF"]
        flux, error = np.asarray(data["FLUX"], float), np.asarray(data["FLUX_ERR"], float)
        good = (TessQualityFlags.create_quality_mask(data["QUALITY"], "default") & np.isfinite(time)
                & np.isfinite(flux).any(axis=(1, 2)))
        x, y = WCS(hdul[2].header).all_world2pix(hdul[0].header["RA_OBJ"], hdul[0].header["DEC_OBJ"], 0)
    return time[good], flux[good], error[good], (float(x), float(y))


def difference_image(time, flux, centers, duration):
    """Mean (out of transit - in transit) image over the covered transits with its per-pixel error."""
    diffs, errors, oot = [], [], []
    for c in centers:
        inside = np.abs(time - c) < duration / 4
        outside = (np.abs(time - c) > .75 * duration) & (np.abs(time - c) < 1.25 * duration)
        if inside.sum() < 3 or outside.sum() < 6:
            continue
        before, after = outside & (time < c), outside & (time > c)
        if not before.any() or not after.any():
            continue
        out = np.nanmean(flux[outside], axis=0)
        diffs.append(out - np.nanmean(flux[inside], axis=0))
        errors.append(np.nanstd(flux[outside], axis=0) * np.sqrt(1 / inside.sum() + 1 / outside.sum()))
        oot.append(out)
    if not diffs:
        return None
    diffs = np.array(diffs)
    if len(diffs) >= 3:
        error = np.nanstd(diffs, axis=0, ddof=1) / np.sqrt(len(diffs))
        error = np.maximum(error, np.sqrt(np.nanmean(np.square(errors), axis=0) / len(diffs)))
    else:
        error = np.sqrt(np.nansum(np.square(errors), axis=0)) / len(diffs)
    return {"difference": np.nanmean(diffs, axis=0), "error": error, "oot": np.nanmean(oot, axis=0), "transits": len(diffs)}


def fit_source(image, error):
    """Weighted Gaussian plus constant fit: source position, errors and amplitude SNR in pixels."""
    from scipy.optimize import least_squares
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    good = np.isfinite(image) & np.isfinite(error) & (error > 0)
    if good.sum() < 9:
        return None
    z, w, xg, yg = image[good], 1 / error[good], xx[good], yy[good]
    j = int(np.argmax(np.where(good, image, -np.inf)))
    start = [max(image.flat[j], 1e-12), j % nx, j // nx, .9, np.nanmedian(z)]

    def residual(p):
        amplitude, x0, y0, sigma, offset = p
        return (amplitude * np.exp(-((xg - x0) ** 2 + (yg - y0) ** 2) / (2 * sigma ** 2)) + offset - z) * w
    try:
        result = least_squares(residual, start, bounds=([0, -.5, -.5, .3, -np.inf], [np.inf, nx - .5, ny - .5, 3., np.inf]))
    except ValueError:
        return None
    dof = max(1, good.sum() - 5)
    chi2 = float(np.sum(result.fun ** 2)) / dof
    try:
        covariance = np.linalg.inv(result.jac.T @ result.jac) * max(chi2, 1.)
    except np.linalg.LinAlgError:
        return None
    errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    amplitude, x0, y0 = result.x[:3]
    return {"x": float(x0), "y": float(y0), "x_err": float(errors[1]), "y_err": float(errors[2]),
            "snr": float(amplitude / errors[0]) if errors[0] > 0 else 0.}


def transit_centers(time, period, epoch):
    if period is None or not np.isfinite(period):
        return np.array([epoch])
    k = np.arange(np.ceil((time[0] - epoch) / period - .5), np.floor((time[-1] - epoch) / period + .5) + 1)
    return epoch + k * period


def sector_images(star, period, epoch, duration_hours, root, epochs=None):
    """Per sector with transits (up to three): difference image, out-of-transit image, target
    position, fitted source (or None) and the number of transits combined."""
    duration = duration_hours / 24
    times = np.asarray(epochs if epochs is not None else transit_centers(star["time_bjd"], period, epoch))
    sectors = []
    for s in np.unique(star["sector"]):
        t = star["time_bjd"][star["sector"] == s]
        if np.any((times > t[0] - duration) & (times < t[-1] + duration)):
            sectors.append(int(s))
    out = []
    for s in sectors[-MAX_SECTORS:]:
        time, flux, _, target = read_tpf(tpf_file(star, s, root))
        centers = times[(times > time[0]) & (times < time[-1])]
        image = difference_image(time, flux, centers, duration)
        if image is None:
            continue
        out.append({"sector": s, "transits": image["transits"], "difference": image["difference"], "error": image["error"],
                    "oot": image["oot"], "target": target, "source": fit_source(image["difference"], image["error"])})
    return out


def pixel_test(star, period, epoch, duration_hours, root, epochs=None):
    """Difference-image source offset from the target, combined over up to three sectors."""
    from scipy import stats
    details = []
    for image in sector_images(star, period, epoch, duration_hours, root, epochs):
        s, source = image["sector"], image["source"]
        if source is None or source["snr"] < MIN_DIFFERENCE_SNR:
            details.append({"sector": s, "transits": image["transits"], "difference_snr": source["snr"] if source else None, "used": False})
            continue
        details.append({"sector": s, "transits": image["transits"], "difference_snr": source["snr"], "used": True,
                        "dx_pixels": float(source["x"] - image["target"][0]), "dy_pixels": float(source["y"] - image["target"][1]),
                        "dx_err": float(np.hypot(source["x_err"], SYSTEMATIC_PIXELS)),
                        "dy_err": float(np.hypot(source["y_err"], SYSTEMATIC_PIXELS))})
    used = [d for d in details if d["used"]]
    if not used:
        return {"sigma": None, "offset_pixels": None, "offset_arcsec": None, "sectors": details}
    combined = []
    for key in ("dx", "dy"):
        w = np.array([1 / d[f"{key}_err"] ** 2 for d in used])
        v = np.array([d[f"{key}_pixels"] for d in used])
        combined.append((float(np.sum(w * v) / w.sum()), float(1 / np.sqrt(w.sum()))))
    chi2 = sum((v / e) ** 2 for v, e in combined)
    sigma = float(min(stats.norm.isf(max(stats.chi2.sf(chi2, 2), 1e-300) / 2), 40.))
    offset = float(np.hypot(combined[0][0], combined[1][0]))
    return {"sigma": sigma, "offset_pixels": offset, "offset_arcsec": offset * PIXEL_ARCSEC, "sectors": details}


def apply_pixel_test(metrics, result):
    """Pixel-level flag: a significant offset of half a pixel or more fails."""
    metrics["pixel_test"] = result
    sigma, offset = result.get("sigma"), result.get("offset_pixels")
    if sigma is None:
        return
    if sigma > OFFSET_SIGMA and offset > OFFSET_PIXELS:
        metrics["flags"]["pixel_offset"] = FAIL
    elif sigma > 3 and offset > .6 * OFFSET_PIXELS:
        metrics["flags"]["pixel_offset"] = WARN
    metrics["verdict"] = FAIL if FAIL in metrics["flags"].values() else "pass"
