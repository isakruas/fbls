"""Publish a deposit folder as a new version of the Zenodo record, with its files at the root of the
record and the metadata of its .zenodo.json. Standard library only; run by the catalog workflow.

  ZENODO_TOKEN   personal access token (deposit:write, deposit:actions)
  ZENODO_RECORD  concept record id (default: the catalog's)
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

API = "https://zenodo.org/api"
CONCEPT = "24026413"


def call(method, url, data=None, content_type="application/json"):
    body = json.dumps(data).encode() if isinstance(data, dict) else data
    request = urllib.request.Request(url, data=body, method=method, headers={
        "Authorization": f"Bearer {os.environ['ZENODO_TOKEN']}", "Content-Type": content_type})
    with urllib.request.urlopen(request) as response:
        text = response.read()
        return json.loads(text) if text else {}


def main(folder):
    folder = Path(folder)
    metadata = json.loads((folder / ".zenodo.json").read_text())
    concept = os.environ.get("ZENODO_RECORD") or CONCEPT
    published = call("GET", f"{API}/records?q=conceptrecid:{concept}&all_versions=true&size=100")["hits"]["hits"]
    same = [r for r in published if r["metadata"].get("version") == metadata["version"]]
    if same:
        print(f"version {metadata['version']} is already published as {same[0]['doi']}; nothing to do")
        return
    if published:
        latest = max(published, key=lambda r: r["created"])["id"]
        draft = call("POST", f"{API}/deposit/depositions/{latest}/actions/newversion")
        draft = call("GET", draft["links"]["latest_draft"])
    else:
        draft = call("POST", f"{API}/deposit/depositions", {})
        print(f"new record: concept {draft['conceptrecid']} (set ZENODO_RECORD and CONCEPT to it)")
    for old in draft.get("files", []):
        call("DELETE", f"{API}/deposit/depositions/{draft['id']}/files/{old['id']}")
    for path in sorted(p for p in folder.iterdir() if p.name != ".zenodo.json"):
        call("PUT", f"{draft['links']['bucket']}/{path.name}", path.read_bytes(), "application/octet-stream")
    call("PUT", f"{API}/deposit/depositions/{draft['id']}", {"metadata": metadata})
    record = call("POST", f"{API}/deposit/depositions/{draft['id']}/actions/publish")
    print(f"published {record['doi']}")


if __name__ == "__main__":
    main(sys.argv[1])
