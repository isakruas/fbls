"""Build a control sample of confirmed-planet hosts from an existing light-curve cache.

The control group measures recovery of known planets with the frozen detector.
Hosts are drawn from the TOI table (dispositions CP and KP) and from published
transiting planets, stratified by the orbital period of their longest-period
planet (0.5-3, 3-10, 10-30, 30-100 d), excluding stars listed in the given
exclusion files (development samples) and requiring that the cached data span
at least two periods. Sector NPZ/JSON files and the original FITS files are
copied into the data directory with the same layout the search CLI reads.

Usage:
  control_sample.py --source /path/to/snapshots --per-bin 50 [--exclude tics.json ...] [--copy]
  control_sample.py --source /path/to/snapshots --tics dev_tics.json --label development --copy
With --tics, the listed stars are copied as the named group instead of drawing a sample.
"""
import argparse
import csv
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
BINS = ((.5, 3.), (3., 10.), (10., 30.), (30., 100.))
SECTOR_DAYS = 27.4


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def confirmed_planets(source):
    """{tic: [(name, period_days, duration_hours)]} for CP/KP TOIs and published transiting planets."""
    hosts = defaultdict(list)
    for r in json.loads((source / "catalogs/toi.json").read_text()):
        if r.get("tfopwg_disp") in ("CP", "KP") and finite(r.get("pl_orbper")) and r["pl_orbper"] > 0:
            hosts[int(r["tid"])].append(("TOI " + str(r["toi"]), float(r["pl_orbper"]), r.get("pl_trandurh")))
    for r in json.loads((source / "catalogs/known.json").read_text()):
        if r.get("source") == "planet" and r.get("transiting") is True and finite(r.get("period")) and r["period"] > 0:
            rows = hosts[int(r["tic"])]
            if not any(abs(p / r["period"] - 1) < 2e-3 for _, p, _ in rows):
                rows.append((r["name"], float(r["period"]), r.get("duration_hours")))
    return hosts


def copy_files(rows, source, root):
    """Copy NPZ/JSON sector files and their FITS originals; existing files are kept."""
    copied = 0
    for r in rows:
        for npz in r["files"]:
            destination = root / "curves" / str(r["tic_id"])
            destination.mkdir(parents=True, exist_ok=True)
            for path in (npz, npz.with_suffix(".json")):
                if path.exists() and not (destination / path.name).exists():
                    shutil.copy2(path, destination / path.name)
                    copied += 1
            sidecar = json.loads(npz.with_suffix(".json").read_text()) if npz.with_suffix(".json").exists() else {}
            fits = source.parent / sidecar["fits"] if sidecar.get("fits") else None
            if fits and fits.exists():
                (root / "mast" / "lc").mkdir(parents=True, exist_ok=True)
                if not (root / "mast" / "lc" / fits.name).exists():
                    shutil.copy2(fits, root / "mast" / "lc" / fits.name)
                    copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="Directory with catalogs/toi.json, catalogs/known.json and curves/<tic>/sector-N.npz of labeled stars")
    parser.add_argument("--exclude", type=Path, nargs="*", default=[], help="JSON lists of TIC IDs to leave out")
    parser.add_argument("--per-bin", type=int, default=50)
    parser.add_argument("--fp-hosts", type=int, default=40, help="Stars whose only catalogued signals are TOI false positives")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument("--copy", action="store_true", help="Copy NPZ/JSON and FITS files into --root")
    parser.add_argument("--tics", type=Path, nargs="*", help="JSON lists of TIC IDs to copy as a named group instead of sampling")
    parser.add_argument("--label", default="control", help="Group name used in the output file names")
    args = parser.parse_args()
    source = args.source.resolve()
    if args.tics:
        tics = sorted({int(t) for path in args.tics for t in json.loads(path.read_text())})
        hosts = confirmed_planets(source)
        rows = []
        for tic in tics:
            folder = source / "curves" / str(tic)
            files = sorted(folder.glob("sector-*.npz"), key=lambda p: int(p.stem.split("-")[1])) if folder.is_dir() else []
            if files:
                rows.append({"tic_id": tic, "n_sectors": len(files), "sectors": " ".join(p.stem.split("-")[1] for p in files),
                             "planets": "; ".join(f"{n} P={p:.4f} d" for n, p, _ in sorted(hosts.get(tic, []), key=lambda x: x[1])),
                             "files": files})
        with (OUT / f"{args.label}_targets.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tic_id", "planets", "n_sectors", "sectors"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in ("tic_id", "planets", "n_sectors", "sectors")})
        print(f"{args.label}: {len(rows)} stars, {sum(r['n_sectors'] for r in rows)} sector files")
        if args.copy:
            print(f"copied {copy_files(rows, source, args.root)} files into {args.root}")
        return
    excluded = set()
    for path in args.exclude:
        excluded.update(int(t) for t in json.loads(path.read_text()))
    hosts = confirmed_planets(source)
    pools = {b: [] for b in BINS}
    for tic, planets in hosts.items():
        if tic in excluded:
            continue
        folder = source / "curves" / str(tic)
        files = sorted(folder.glob("sector-*.npz"), key=lambda p: int(p.stem.split("-")[1])) if folder.is_dir() else []
        if not files:
            continue
        sectors = [int(p.stem.split("-")[1]) for p in files]
        span = (max(sectors) - min(sectors) + 1) * SECTOR_DAYS
        in_range = [p for p in planets if .5 <= p[1] <= 100.]
        if not in_range or any(span < 2 * p[1] for p in in_range):
            continue
        longest = max(p[1] for p in in_range)
        for lo, hi in BINS:
            if lo <= longest < hi or (hi == 100. and longest == hi):
                pools[(lo, hi)].append({"tic_id": tic, "n_sectors": len(files), "sectors": " ".join(map(str, sectors)),
                                       "span_days_estimate": round(span, 1), "period_bin": f"{lo:g}-{hi:g}",
                                       "planets": "; ".join(f"{n} P={p:.4f} d" for n, p, _ in sorted(in_range, key=lambda x: x[1])),
                                       "n_planets_in_range": len(in_range), "files": files})
    fp_only = []
    fp_hosts = defaultdict(list)
    for r in json.loads((source / "catalogs/toi.json").read_text()):
        if finite(r.get("pl_orbper")) and r["pl_orbper"] > 0:
            fp_hosts[int(r["tid"])].append((r.get("tfopwg_disp"), float(r["pl_orbper"])))
    for tic, rows in fp_hosts.items():
        if tic in excluded or tic in hosts or any(d != "FP" for d, _ in rows):
            continue
        folder = source / "curves" / str(tic)
        files = sorted(folder.glob("sector-*.npz"), key=lambda p: int(p.stem.split("-")[1])) if folder.is_dir() else []
        if not files:
            continue
        sectors = [int(p.stem.split("-")[1]) for p in files]
        span = (max(sectors) - min(sectors) + 1) * SECTOR_DAYS
        if any(span < 2 * p for _, p in rows):
            continue
        fp_only.append({"tic_id": tic, "n_sectors": len(files), "sectors": " ".join(map(str, sectors)),
                        "span_days_estimate": round(span, 1), "period_bin": "false positive",
                        "planets": "; ".join(f"FP P={p:.4f} d" for _, p in sorted(rows, key=lambda x: x[1])),
                        "n_planets_in_range": 0, "files": files})
    # Separate generators keep the planet-host draw independent of the false-positive draw.
    rng = random.Random(args.seed)
    selected = []
    fp_only.sort(key=lambda r: r["tic_id"])
    random.Random(args.seed + 1).shuffle(fp_only)
    selected.extend(fp_only[:args.fp_hosts])
    print(f"false-positive-only hosts: available {len(fp_only)}, selected {min(len(fp_only), args.fp_hosts)}")
    for b, pool in pools.items():
        pool.sort(key=lambda r: r["tic_id"])
        rng.shuffle(pool)
        selected.extend(pool[:args.per_bin])
        print(f"period bin {b[0]:g}-{b[1]:g} d: available {len(pool)}, selected {min(len(pool), args.per_bin)}")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{args.label}_targets.csv").open("w", newline="") as f:
        fields = ["tic_id", "period_bin", "n_planets_in_range", "planets", "n_sectors", "sectors", "span_days_estimate"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in selected:
            w.writerow({k: r[k] for k in fields})
    (OUT / f"{args.label}_sources.json").write_text(json.dumps({
        "source": str(source), "excluded_lists": [str(p) for p in args.exclude], "excluded_stars": len(excluded),
        "rule": "CP/KP TOIs and published transiting planets; cached span >= 2 x period of every planet in 0.5-100 d; "
                "stratified by the longest period; seeded shuffle", "seed": args.seed, "per_bin": args.per_bin,
        "selected": len(selected), "fp_hosts": min(len(fp_only), args.fp_hosts),
        "available": {f"{b[0]:g}-{b[1]:g}": len(p) for b, p in pools.items()}}, indent=2) + "\n")
    print(f"control sample: {len(selected)} stars, {sum(r['n_sectors'] for r in selected)} sector files")
    if args.copy:
        print(f"copied {copy_files(selected, source, args.root)} files into {args.root}")


if __name__ == "__main__":
    main()
