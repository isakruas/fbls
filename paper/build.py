"""Assemble the Zenodo deposit from the published survey tables: paper/deposit/<day>/ (not tracked; the
release workflow builds it from public/) with the two tables, one figure folder per CTOI signal (zipped), the ReadMe
and the citation file, which states the version date."""
import csv
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public"
TODAY = os.environ.get("FBLS_DAY") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
DEPOSIT = ROOT / "paper" / "deposit" / TODAY


def figure_folders():
    """CTOI signal -> folder with its panels, from the refine folders and the day figure folders."""
    found = {}
    for day in sorted(PUBLIC.glob("*-*-*")):
        for folder in day.glob("refine/*/figures/*"):
            found[f"{folder.parts[-3]}_{folder.name}"] = folder
        for folder in day.glob("figures/*_*"):
            found.setdefault(folder.name, folder)
    return found


def zenodo_metadata(readme):
    """Metadata Zenodo reads before CITATION.cff: the same content, with the record typed as a dataset."""
    lines = readme.splitlines()
    return {"upload_type": "dataset", "title": lines[0], "version": f"v{TODAY}", "publication_date": TODAY,
            "creators": [{"name": "Ruas, Isak Paulo de Andrade", "affiliation": "NROLABS"}],
            "license": "cc-by-4.0", "keywords": ["TESS", "exoplanets", "transit search", "planet candidates", "CTOI"],
            "description": readme.split("\n\n")[1].replace("\n", " "),
            "related_identifiers": [{"identifier": "https://github.com/isakruas/fbls", "relation": "isSupplementTo",
                                     "resource_type": "software"}]}


def main():
    if not (PUBLIC / "ctoi.csv").exists():
        sys.exit("nothing published yet")
    if DEPOSIT.exists():
        shutil.rmtree(DEPOSIT)
    (DEPOSIT / "figures").mkdir(parents=True)
    for name in ("ctoi.csv", "candidates.csv"):
        shutil.copy(PUBLIC / name, DEPOSIT / name)
    citation = re.sub(r"^version: .*$", f"version: v{TODAY}", (ROOT / "CITATION.cff").read_text(), flags=re.M)
    (DEPOSIT / "CITATION.cff").write_text(re.sub(r"^date-released: .*$", f"date-released: {TODAY}", citation, flags=re.M))
    readme = (ROOT / "paper" / "ReadMe.txt").read_text().replace("Version: <date>", f"Version: {TODAY}")
    (DEPOSIT / "ReadMe.txt").write_text(readme)
    (DEPOSIT / ".zenodo.json").write_text(json.dumps(zenodo_metadata(readme), indent=2) + "\n")
    with (PUBLIC / "ctoi.csv").open() as f:
        tics = {row["TIC ID"] for row in csv.DictReader(f)}
    folders = figure_folders()
    copied = 0
    for key, folder in folders.items():
        if key.split("_")[0] in tics:
            shutil.copytree(folder, DEPOSIT / "figures" / key)
            copied += 1
    shutil.make_archive(str(DEPOSIT / "figures"), "zip", DEPOSIT, "figures")
    shutil.rmtree(DEPOSIT / "figures")
    print(f"{DEPOSIT}: {len(tics)} CTOI stars, {copied} figure folders")


if __name__ == "__main__":
    main()
