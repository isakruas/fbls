"""Download the SPOC light curves listed in download_manifest.csv and convert them to NPZ.

Each product is fetched from the public AWS mirror first and from the MAST URL of
the manifest as fallback, with resume (curl --continue-at), a configurable number
of parallel transfers, validation of the FITS file (last TIME row readable) and a
per-product log. Existing valid files are skipped. The NPZ layout matches the
labeled cache (time_bjd, flux, flux_err, centr1, centr2 with default quality
mask and PDCSAP flux), with a JSON sidecar recording provenance and the SHA-256
of the FITS file.

Usage:
  download.py --pilot            # the pilot stars flagged in the manifest
  download.py                    # the complete manifest
  download.py --workers 4 --root ../data/blind
"""
import argparse
import csv
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).resolve().parent / "download_manifest.csv"
CURVE_VERSION = 2


def curl(url, output, attempts=3):
    """Resume-capable transfer; returns False on HTTP 404, raises on other failures."""
    for attempt in range(attempts):
        result = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location", "--continue-at", "-",
                                 "--connect-timeout", "20", "--max-time", "900", "--speed-limit", "1000",
                                 "--speed-time", "120", "--output", str(output), url], capture_output=True, text=True)
        if result.returncode == 0:
            return True
        if result.returncode == 22 and "404" in result.stderr:
            return False
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"transfer failed {url}: {result.stderr.strip()[-200:]}")


def valid_fits(path):
    try:
        from astropy.io import fits
        with fits.open(path) as hdul:
            hdul[1].data["TIME"][-1]
        return True
    except Exception:
        return False


def convert(fits_path, tic, sector, filename, root):
    import numpy as np
    from astropy.io import fits
    from lightkurve.utils import TessQualityFlags
    from astropy.wcs import WCS
    with fits.open(fits_path) as hdul:
        head, data = hdul[0].header, hdul[1].data
        aperture = hdul[2].header
        x, y = WCS(aperture).all_world2pix(head["RA_OBJ"], head["DEC_OBJ"], 0)
        target_pixel = [float(x + aperture["CRVAL1P"]), float(y + aperture["CRVAL2P"])]
        time_bjd = data["TIME"] + hdul[1].header["BJDREFI"] + hdul[1].header["BJDREFF"]
        good = (TessQualityFlags.create_quality_mask(data["QUALITY"], "default")
                & np.isfinite(time_bjd) & np.isfinite(data["PDCSAP_FLUX"]))
        crowding = hdul[1].header.get("CROWDSAP")
        star = {k: head.get(key) for k, key in (("radius", "RADIUS"), ("teff", "TEFF"), ("logg", "LOGG"), ("tmag", "TESSMAG"))}
        arrays = {"time_bjd": time_bjd[good], "flux": data["PDCSAP_FLUX"][good], "flux_err": data["PDCSAP_FLUX_ERR"][good],
                  "centr1": data["MOM_CENTR1"][good], "centr2": data["MOM_CENTR2"][good]}
    out = root / "curves" / str(tic) / f"sector-{sector}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".tmp").open("wb") as handle:
        np.savez_compressed(handle, **{k: np.asarray(v, dtype=float) for k, v in arrays.items()})
    out.with_suffix(".tmp").replace(out)
    sidecar = {"version": CURVE_VERSION, "tic": tic, "sector": sector, "author": "SPOC", "exptime": 120,
               "quality_bitmask": "default", "flux": "PDCSAP_FLUX", "centroid": "MOM_CENTR", "time": "BJD TDB",
               "crowdsap": crowding, "star": {k: (float(v) if v is not None else None) for k, v in star.items()},
               "target_pixel": target_pixel, "product": filename, "fits": str(fits_path.relative_to(root)),
               "fits_sha256": hashlib.sha256(fits_path.read_bytes()).hexdigest(),
               "fits_size_bytes": fits_path.stat().st_size, "points": int(good.sum()),
               "time_first_bjd": float(arrays["time_bjd"].min()) if good.any() else None,
               "time_last_bjd": float(arrays["time_bjd"].max()) if good.any() else None,
               "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar


def fetch(row, root):
    tic, sector, filename = int(row["tic_id"]), int(row["sector"]), row["filename"]
    fits_path = root / "mast" / "lc" / filename
    fits_path.parent.mkdir(parents=True, exist_ok=True)
    npz = root / "curves" / str(tic) / f"sector-{sector}.npz"
    if npz.exists() and npz.with_suffix(".json").exists() and fits_path.exists():
        return {"tic": tic, "sector": sector, "status": "present"}
    part = fits_path.with_suffix(".part")
    if not (fits_path.exists() and valid_fits(fits_path)):
        source = "mirror"
        try:
            ok = curl(row["mirror_url"], part)
        except RuntimeError:
            ok = False
        if not ok:
            source = "mast"
            part.unlink(missing_ok=True)
            if not curl(row["download_url"], part):
                return {"tic": tic, "sector": sector, "status": "missing (404)"}
        part.replace(fits_path)
        if not valid_fits(fits_path):
            fits_path.unlink(missing_ok=True)
            return {"tic": tic, "sector": sector, "status": "invalid FITS after transfer"}
    else:
        source = "existing"
    sidecar = convert(fits_path, tic, sector, filename, root)
    return {"tic": tic, "sector": sector, "status": "ok", "source": source, "bytes": sidecar["fits_size_bytes"],
            "sha256": sidecar["fits_sha256"], "points": sidecar["points"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot", action="store_true", help="Only rows flagged pilot=True")
    parser.add_argument("--limit", type=int, help="Stop after this many products")
    args = parser.parse_args()
    with args.manifest.open() as f:
        rows = [r for r in csv.DictReader(f) if not args.pilot or r.get("pilot") == "True"]
    if args.limit:
        rows = rows[:args.limit]
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / ("download_log_pilot.json" if args.pilot else "download_log.json")
    log, counts = [], {}
    started = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch, row, root): row for row in rows}
        for done, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                entry = future.result()
            except Exception as error:
                entry = {"tic": int(row["tic_id"]), "sector": int(row["sector"]), "status": "error", "error": str(error)[-300:]}
            log.append(entry)
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
            if done % 25 == 0 or done == len(rows):
                log_path.write_text(json.dumps({"started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
                                                "products": len(rows), "counts": counts, "entries": log}, indent=1) + "\n")
                print(f"[{done}/{len(rows)}] {counts}", flush=True)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
