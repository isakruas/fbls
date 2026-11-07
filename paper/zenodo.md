# Zenodo deposit

One Zenodo record for the catalog (concept DOI 10.5281/zenodo.24026050), one version per day,
with the five deposit files at the root of the record. Zenodo's GitHub integration stays off for
this repository: it would add a duplicate version holding only a source zip.

A push to `main` that adds a day under `paper/deposit/` runs `.github/workflows/catalog.yml`:
the day folder becomes the whole content of the orphan branch `catalog`, tagged `v<day>` and
released on GitHub with the files as assets; then `paper/zenodo_upload.py` publishes them as a
new version of the record with the metadata of the folder's `.zenodo.json`. A manual run of the
workflow republishes the newest day.

Setup, once: a Zenodo personal access token with `deposit:write` and `deposit:actions` as the
repository secret `ZENODO_TOKEN`.

After a publish:

```bash
.venv/bin/python paper/build.py
git add paper/deposit && git commit -m "Catalog <day>" && git push
```
