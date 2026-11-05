"""Select TESS 120 s targets without catalogued signals and write verifiable download manifests.

Sources (all local snapshots of official products, see catalog_sources.json):
- MAST sector manifests (tesscurl_sector_N_lc.sh): every SPOC 120 s light-curve
  product per sector with its real MAST URL. No URL is constructed by guess.
- TOI table (Exoplanet Archive), ExoFOP CTOIs, published planets, TESS EB
  catalog and every SPOC TCE DV run: TIC identifiers to exclude.
- TIC (MAST Mast.Catalogs.Filtered.Tic): Tmag, Teff, radius, mass, logg,
  distance, contamination ratio and object type, fetched by TIC ID in batches
  and cached in tic_cache.jsonl.

Absence from these catalogs is not evidence that a star was never searched;
the report records each catalog's coverage and limitations. Known planet
parameters are used only to exclude stars, never to rank them.

Usage:
  select_targets.py fetch-snapshots                       # MAST sector manifests and catalog snapshots
  select_targets.py fetch-tic   [--min-sectors 2]        # cache TIC rows for the candidate pool
  select_targets.py select      [--pilot 10]              # write targets_download.csv and manifests
"""
import argparse
import csv
import json
import math
import re
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
TIC_CACHE = OUT / "tic_cache.jsonl"
DEFAULT_SOURCE = ROOT / "data" / "snapshots"
SOURCE = DEFAULT_SOURCE
PRODUCT = re.compile(r"https://\S+/(tess(\d{13})-s(\d{4})-(\d{16})-\d{4}-s_lc\.fits)$")
MAST_API = "https://mast.stsci.edu/api/v0/invoke"
MANIFEST_URL = "https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_{}_lc.sh"
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
PLANETS_QUERY = ("select pl_name, tic_id, pl_orbper, pl_orbpererr1, pl_tranmid, pl_tranmiderr1, pl_trandur, tran_flag "
                 "from pscomppars where tic_id is not null")
CTOI_URL = "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv"
EXOFOP_TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
EB_URL = "https://archive.stsci.edu/hlsps/tess-ebs/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv"
TCE_INDEX = "https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html"
TCE_FILE = re.compile(r"/missions/tess/catalogs/tce/[^\"]+_dvr-tcestats\.csv")
TIC_COLUMNS = "ID,ra,dec,Tmag,Teff,rad,mass,logg,d,contratio,objType,lumclass,disposition,duplicate_id"
SECTOR_DAYS = 27.4
QUOTA = {"A": 500, "B": 350, "C": 150}


def read_json(path):
    return json.loads(Path(path).read_text())


def products():
    """{tic: {sector: (filename, url, start_utc)}} from the local MAST sector manifests."""
    table = defaultdict(dict)
    for path in sorted(SOURCE.glob("manifests/sector-*.sh"), key=lambda p: int(p.stem.split("-")[1])):
        for line in path.read_text().splitlines():
            m = PRODUCT.search(line)
            if not m:
                continue
            stamp = m[2]
            start = datetime(int(stamp[:4]), 1, 1, tzinfo=timezone.utc) + timedelta(days=int(stamp[4:7]) - 1)
            table[int(m[4])][int(m[3])] = (m[1], line.split()[-1], start.isoformat())
    return table


def exclusions():
    """TIC identifiers with any catalogued signal, keyed by catalog."""
    toi = read_json(SOURCE / "catalogs/toi.json")
    known = read_json(SOURCE / "catalogs/known.json")
    by_catalog = {"TOI": {int(r["tid"]) for r in toi}}
    for source in ("CTOI", "planet", "EB", "TCE"):
        by_catalog[source] = {int(r["tic"]) for r in known if r["source"] == source}
    with (SOURCE / "catalogs/exofop_toi.csv").open() as f:
        by_catalog["ExoFOP TOI table"] = {int(r["TIC ID"]) for r in csv.DictReader(f) if r.get("TIC ID", "").isdigit()}
    used = set()
    for name in ("development_targets.csv", "control_targets.csv"):
        path = OUT / name
        if path.exists():
            with path.open() as f:
                used.update(int(r["tic_id"]) for r in csv.DictReader(f))
    by_catalog["development and control groups"] = used
    return by_catalog


def longest_run(sectors):
    best, current = [], []
    for sector in sorted(sectors):
        if current and sector != current[-1] + 1:
            best = max(best, current, key=len)
            current = []
        current.append(sector)
    return max(best, current, key=len)


def fetch_tic(ids, attempts=4):
    request = {"service": "Mast.Catalogs.Filtered.Tic", "format": "json",
               "params": {"columns": TIC_COLUMNS, "filters": [{"paramName": "ID", "values": ids}]}}
    for attempt in range(attempts):
        result = subprocess.run(["curl", "--silent", "--show-error", "--max-time", "300", "-X", "POST", MAST_API,
                                 "--data-urlencode", "request=" + json.dumps(request)], capture_output=True, text=True)
        try:
            payload = json.loads(result.stdout)
            if payload.get("status") == "COMPLETE":
                return payload["data"]
        except json.JSONDecodeError:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"TIC query failed after {attempts} attempts: {result.stderr[-300:]}")


def fetch_text(url, query=None):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--retry", "3", "--retry-all-errors",
               "--connect-timeout", "20", "--max-time", "600", url]
    if query:
        command[-1:] = ["-G", url, "--data-urlencode", f"query={query}", "--data-urlencode", "format=json"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"download failed ({url}): {result.stderr.strip()[-200:]}")
    return result.stdout


def positive(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def command_fetch_snapshots(args):
    """Download the MAST sector manifests and the exclusion catalogs into the snapshot directory."""
    import io
    from concurrent.futures import ThreadPoolExecutor
    manifests = SOURCE / "manifests"
    catalogs = SOURCE / "catalogs"
    manifests.mkdir(parents=True, exist_ok=True)
    catalogs.mkdir(parents=True, exist_ok=True)

    def manifest(sector):
        path = manifests / f"sector-{sector}.sh"
        if path.exists():
            return True
        result = subprocess.run(["curl", "--fail", "--silent", "--location", "--retry", "2", "--connect-timeout", "20",
                                 "--output", str(path.with_suffix(".tmp")), MANIFEST_URL.format(sector)], capture_output=True, text=True)
        if result.returncode == 22 and "404" in result.stderr:
            return False
        if result.returncode:
            raise RuntimeError(f"manifest {sector}: {result.stderr.strip()[-200:]}")
        path.with_suffix(".tmp").replace(path)
        return True
    sector = 1
    with ThreadPoolExecutor(8) as pool:
        while True:
            found = list(pool.map(manifest, range(sector, sector + 8)))
            if not all(found):
                sector += found.index(False)
                break
            sector += 8
    print(f"manifests: sectors 1-{sector - 1}", flush=True)
    toi = json.loads(fetch_text(TAP, "select * from toi"))
    if not isinstance(toi, list) or not toi or "tid" not in toi[0]:
        raise ValueError("unexpected TOI table response")
    (catalogs / "toi.json").write_text(json.dumps(toi))
    (catalogs / "exofop_toi.csv").write_text(fetch_text(EXOFOP_TOI_URL))
    ctois = list(csv.DictReader(io.StringIO(fetch_text(CTOI_URL))))
    planets = json.loads(fetch_text(TAP, PLANETS_QUERY))
    if not ctois or "CTOI" not in ctois[0] or not planets or "tic_id" not in planets[0]:
        raise ValueError("unexpected CTOI or planet catalog response")
    known = [{"source": "CTOI", "name": f"CTOI {r['CTOI']}", "tic": int(r["TIC ID"]), "period": positive(r["Period (days)"]),
              "epoch": positive(r["Transit Epoch (BJD)"]), "duration_hours": positive(r["Duration (hrs)"]),
              "disposition": r["TFOPWG Disposition"] or r["User Disposition"] or None, "transiting": True} for r in ctois]
    known += [{"source": "planet", "name": r["pl_name"], "tic": int(r["tic_id"].split()[-1]), "period": positive(r["pl_orbper"]),
               "epoch": positive(r["pl_tranmid"]), "duration_hours": positive(r["pl_trandur"]), "disposition": "published",
               "transiting": bool(r["tran_flag"])} for r in planets]
    known += [{"source": "EB", "name": f"EB TIC {r['tess_id']}", "tic": int(r["tess_id"]), "period": positive(r["period"]),
               "epoch": positive(r["bjd0"]) and positive(r["bjd0"]) + 2457000, "duration_hours": None,
               "disposition": "eclipsing binary", "transiting": False}
              for r in csv.DictReader(io.StringIO(fetch_text(EB_URL)))]
    tce_dir = catalogs / "tce"
    tce_dir.mkdir(exist_ok=True)
    files = sorted(set(TCE_FILE.findall(fetch_text(TCE_INDEX))))
    if not files:
        raise ValueError("no TCE files found on the MAST bulk download page")

    def tce(path):
        run = re.search(r"-(s\d{4}-s\d{4})_", path).group(1)
        local = tce_dir / Path(path).name
        if not local.exists():
            local.write_text(fetch_text("https://archive.stsci.edu" + path))
        with local.open() as f:
            return [{"source": "TCE", "name": f"TCE {run} {r['tceid'].lstrip('0')}", "tic": int(r["ticid"]),
                     "period": positive(r["tce_period"]), "epoch": positive(r["tce_time0"]),
                     "duration_hours": positive(r["tce_duration"]), "disposition": None, "transiting": True}
                    for r in csv.DictReader(line for line in f if not line.startswith("#"))]
    with ThreadPoolExecutor(8) as pool:
        for part in pool.map(tce, files):
            known.extend(part)
    (catalogs / "known.json").write_text(json.dumps(known))
    summary = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "toi_objects": len(toi),
               "manifest_sectors": sector - 1, "tce_runs": len(files),
               "known_counts": dict(Counter(k["source"] for k in known)),
               "sources": {"toi": f"{TAP} (select * from toi)", "exofop_toi": EXOFOP_TOI_URL, "ctoi": CTOI_URL,
                           "planets": f"{TAP} ({PLANETS_QUERY})", "eb": EB_URL, "tce_index": TCE_INDEX, "manifests": MANIFEST_URL}}
    (catalogs / "snapshot.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def command_fetch_tic(args):
    table = products()
    excluded = set().union(*exclusions().values())
    pool = sorted(t for t, sectors in table.items() if t not in excluded and len(sectors) >= args.min_sectors)
    cached = set()
    if TIC_CACHE.exists():
        with TIC_CACHE.open() as f:
            cached = {json.loads(line)["ID"] for line in f}
    pending = [t for t in pool if t not in cached]
    print(f"pool {len(pool)} stars with >= {args.min_sectors} sectors and no catalogued signal; cached {len(cached)}; pending {len(pending)}", flush=True)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    batches = [pending[i:i + args.batch] for i in range(0, len(pending), args.batch)]
    done = 0
    with TIC_CACHE.open("a") as f, ThreadPoolExecutor(args.threads) as pool_executor:
        futures = {pool_executor.submit(fetch_tic, ids): ids for ids in batches}
        for future in as_completed(futures):
            ids = futures[future]
            rows = future.result()
            found = {r["ID"] for r in rows}
            for r in rows:
                f.write(json.dumps({k: r.get(k) for k in TIC_COLUMNS.split(",")}) + "\n")
            for missing in set(ids) - found:
                f.write(json.dumps({"ID": missing, "missing": True}) + "\n")
            f.flush()
            done += len(ids)
            print(f"  {done}/{len(pending)} ({len(rows)} rows in batch)", flush=True)


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def command_select(args):
    started = datetime.now(timezone.utc).isoformat()
    table = products()
    by_catalog = exclusions()
    tic = {}
    with TIC_CACHE.open() as f:
        for line in f:
            r = json.loads(line)
            tic[r["ID"]] = r
    counts = Counter()
    candidates, excluded_rows = [], []
    for t, sectors in table.items():
        hits = [name for name, ids in by_catalog.items() if t in ids]
        if hits:
            counts["excluded: " + hits[0]] += 1
            continue
        if len(sectors) < 2:
            counts["excluded: fewer than 2 sectors"] += 1
            continue
        r = tic.get(t)
        if r is None or r.get("missing"):
            counts["excluded: no TIC row"] += 1
            excluded_rows.append({"tic_id": t, "reason": "no TIC row", "n_sectors": len(sectors)})
            continue
        tmag = r.get("Tmag")
        if not finite(tmag) or not 8. <= tmag <= 12.:
            counts["excluded: Tmag outside 8-12"] += 1
            continue
        if r.get("objType") != "STAR":
            counts["excluded: objType != STAR"] += 1
            excluded_rows.append({"tic_id": t, "reason": f"objType {r.get('objType')}", "n_sectors": len(sectors)})
            continue
        if r.get("disposition") not in (None, "", "SPLIT"):
            counts["excluded: TIC disposition " + str(r.get("disposition"))] += 1
            excluded_rows.append({"tic_id": t, "reason": f"TIC disposition {r.get('disposition')}", "n_sectors": len(sectors)})
            continue
        contamination = r.get("contratio")
        if finite(contamination) and contamination > .2:
            counts["excluded: contamination ratio > 0.2"] += 1
            excluded_rows.append({"tic_id": t, "reason": f"contratio {contamination:.3f}", "n_sectors": len(sectors)})
            continue
        run = longest_run(sectors.keys())
        starts = sorted(datetime.fromisoformat(v[2]) for v in sectors.values())
        baseline = (starts[-1] - starts[0]).days + SECTOR_DAYS
        n = len(sectors)
        if 8. <= tmag < 10. and n >= 4:
            priority = "A"
        elif 10. <= tmag <= 12. and n >= 3:
            priority = "B"
        elif n >= 2:
            priority = "C"
        else:
            continue
        candidates.append({
            "tic_id": t, "ra": r.get("ra"), "dec": r.get("dec"), "tmag": tmag, "teff": r.get("Teff"),
            "stellar_radius": r.get("rad"), "stellar_mass": r.get("mass"), "stellar_logg": r.get("logg"),
            "distance_pc": r.get("d"), "contamination_ratio": contamination, "n_sectors": n,
            "sectors": " ".join(str(s) for s in sorted(sectors)), "longest_contiguous_sectors": len(run),
            "baseline_days_estimate": round(baseline, 1), "observed_days_estimate": round(n * SECTOR_DAYS, 1),
            "valid_points_estimate": int(n * 18000), "cadence_seconds": 120, "photometric_source": "SPOC PDCSAP",
            "priority": priority, "catalog_match_status": "no match in TOI, CTOI, planets, EB, TCE",
            "quality_status": "pending download", "estimated_download_mb": round(n * 2.0, 1),
        })
    counts["candidates after criteria"] = len(candidates)
    # Deterministic, catalog-agnostic sample: a seeded shuffle within each priority keeps the sector-count
    # distribution of the available pool instead of concentrating on continuous-viewing-zone stars.
    import random
    candidates.sort(key=lambda c: c["tic_id"])
    random.Random(args.seed).shuffle(candidates)
    selected, taken = [], Counter()
    for priority in ("A", "B", "C"):
        group = [c for c in candidates if c["priority"] == priority]
        selected.extend(group[:QUOTA[priority]])
        taken[priority] = min(len(group), QUOTA[priority])
        counts[f"available priority {priority}"] = len(group)
    pilot = selected[:args.pilot]
    for c in pilot:
        c["pilot"] = True
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(selected[0].keys()) if selected else []
    if "pilot" not in fields:
        fields.append("pilot")
    with (OUT / "targets_download.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in selected:
            w.writerow({**c, "pilot": c.get("pilot", False)})
    with (OUT / "download_manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tic_id", "sector", "product_type", "product_source", "cadence_seconds", "mast_obsid",
                    "mast_product_uri", "download_url", "mirror_url", "filename", "file_size_bytes", "checksum", "pilot"])
        for c in selected:
            for sector in sorted(table[c["tic_id"]]):
                filename, url, _ = table[c["tic_id"]][sector]
                t = f"{c['tic_id']:016d}"
                mirror = (f"https://stpubdata.s3.amazonaws.com/tess/public/tid/s{sector:04d}/"
                          f"{t[0:4]}/{t[4:8]}/{t[8:12]}/{t[12:16]}/{filename}")
                w.writerow([c["tic_id"], sector, "light curve", "SPOC", 120, None,
                            url.split("uri=")[-1] if "uri=" in url else None, url, mirror, filename, None, None,
                            c.get("pilot", False)])
    with (OUT / "excluded_targets.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tic_id", "reason", "n_sectors"])
        w.writeheader()
        w.writerows(excluded_rows)
    sources = {
        "generated_utc": started,
        "manifests": {"sectors": len(list(SOURCE.glob("manifests/sector-*.sh"))),
                      "url_pattern": "https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_N_lc.sh"},
        "catalog_snapshot": read_json(SOURCE / "catalogs/snapshot.json"),
        "tic": {"service": "Mast.Catalogs.Filtered.Tic", "api": MAST_API, "columns": TIC_COLUMNS,
                "rows_cached": len(tic)},
        "exclusion_counts": {name: len(ids) for name, ids in by_catalog.items()},
        "limitations": [
            "TOI/CTOI/planet/TCE/EB catalogs are the local snapshots listed in catalog_snapshot; a star absent from them may still have been searched by SPOC, QLP or published work.",
            "The TESS EB catalog snapshot covers sectors 1-26 (FFI based); later eclipsing binaries are not excluded by it.",
            "SPOC TCE files cover the DV runs listed on the MAST bulk page at snapshot time.",
            "Baseline, observed days and point counts are estimates from sector start dates; exact coverage is measured after download.",
            "Contamination uses the TIC contratio field; a missing value is not treated as low contamination but is not excluded either.",
        ],
        "selection_rules": {"tmag": "8 <= Tmag <= 12", "object": "TIC objType STAR, no non-empty disposition except SPLIT",
                            "contamination": "contratio <= 0.2 when present", "sectors": ">= 2 SPOC 120 s sectors",
                            "priority": {"A": "8 <= Tmag < 10 and >= 4 sectors", "B": "10 <= Tmag <= 12 and >= 3 sectors", "C": ">= 2 sectors"},
                            "ordering": f"seeded shuffle within priority (seed {args.seed})",
                            "quota": QUOTA},
    }
    (OUT / "catalog_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    lines = [f"# Target selection report ({started})", "",
             f"Stars with SPOC 120 s light curves in {sources['manifests']['sectors']} sector manifests: {len(table)}", "",
             "| Step | Stars |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    lines += ["", "| Priority | Available | Selected |", "|---|---:|---:|"]
    lines += [f"| {p} | {counts[f'available priority {p}']} | {taken[p]} |" for p in ("A", "B", "C")]
    lines += ["", f"Selected: {len(selected)} stars, {sum(c['n_sectors'] for c in selected)} sector products, "
              f"~{sum(c['estimated_download_mb'] for c in selected) / 1024:.1f} GB; pilot batch: {len(pilot)} stars.", "",
              "Limitations:", *[f"- {x}" for x in sources["limitations"]]]
    (OUT / "selection_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Snapshot directory: manifests/sector-*.sh and catalogs/ (see fetch-snapshots); optional curves/ cache of used stars")
    sub = parser.add_subparsers(dest="command", required=True)
    snapshots = sub.add_parser("fetch-snapshots")
    snapshots.set_defaults(run=command_fetch_snapshots)
    fetch = sub.add_parser("fetch-tic")
    fetch.add_argument("--min-sectors", type=int, default=2)
    fetch.add_argument("--batch", type=int, default=1000)
    fetch.add_argument("--threads", type=int, default=6)
    fetch.set_defaults(run=command_fetch_tic)
    select = sub.add_parser("select")
    select.add_argument("--pilot", type=int, default=10)
    select.add_argument("--seed", type=int, default=20261105, help="Seed of the within-priority shuffle")
    select.set_defaults(run=command_select)
    args = parser.parse_args()
    global SOURCE
    SOURCE = args.source.resolve()
    args.run(args)


if __name__ == "__main__":
    main()
