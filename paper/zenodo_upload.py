"""Publish a deposit folder as a Zenodo record version: a new record when ZENODO_RECORD is empty,
otherwise a new version of that concept record, with only the folder's files. Standard library only."""
import json
import os
import sys
import urllib.request
from pathlib import Path

API = "https://zenodo.org/api"
TOKEN = os.environ["ZENODO_TOKEN"]


def call(method, url, data=None, content_type="application/json"):
    body = json.dumps(data).encode() if isinstance(data, dict) else data
    request = urllib.request.Request(url, data=body, method=method,
                                     headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": content_type})
    with urllib.request.urlopen(request) as response:
        text = response.read()
        return json.loads(text) if text else {}


def metadata(folder):
    day = folder.name
    readme = (folder / "ReadMe.txt").read_text()
    description = readme.split("\n\n")[1].replace("\n", " ") + \
        "<br><br>Files: ctoi.csv (ExoFOP CTOI rows), candidates.csv (every vetted signal), " \
        "figures.zip (one folder per CTOI signal, one PNG per panel), ReadMe.txt (column definitions)."
    return {"metadata": {
        "upload_type": "dataset",
        "title": readme.splitlines()[0],
        "creators": [{"name": "Ruas, Isak Paulo de Andrade", "affiliation": "NROLABS"}],
        "description": description,
        "version": day,
        "publication_date": day,
        "license": "cc-by-4.0",
        "keywords": ["TESS", "exoplanets", "transit search", "planet candidates", "CTOI"],
        "related_identifiers": [{"identifier": "https://github.com/isakruas/fbls", "relation": "isSupplementedBy",
                                 "resource_type": "software"}]}}


def main(folder):
    folder = Path(folder)
    concept = os.environ.get("ZENODO_RECORD", "").strip()
    if concept:
        latest = call("GET", f"{API}/records/{concept}")["id"]
        draft = call("POST", f"{API}/deposit/depositions/{latest}/actions/newversion")
        draft = call("GET", draft["links"]["latest_draft"])
        for old in draft.get("files", []):
            call("DELETE", f"{API}/deposit/depositions/{draft['id']}/files/{old['id']}")
    else:
        draft = call("POST", f"{API}/deposit/depositions", {})
    bucket = draft["links"]["bucket"]
    for path in sorted(folder.iterdir()):
        call("PUT", f"{bucket}/{path.name}", path.read_bytes(), "application/octet-stream")
    call("PUT", f"{API}/deposit/depositions/{draft['id']}", metadata(folder))
    record = call("POST", f"{API}/deposit/depositions/{draft['id']}/actions/publish")
    print(f"published {record['doi']} (concept record {record['conceptrecid']}, set ZENODO_RECORD to it)")


if __name__ == "__main__":
    main(sys.argv[1])
