# Zenodo deposit

One Zenodo record for the catalog, versioned by date. A tag `v<day>` runs
`.github/workflows/catalog-release.yml`, which creates the GitHub release with the four files of
`paper/deposit/<day>/` as assets and publishes them as a new version of the record
(`paper/zenodo_upload.py`, Zenodo REST API). Nothing else goes to Zenodo, so Zenodo's own GitHub
integration stays off for this repository.

Setup, once: a Zenodo personal access token with `deposit:write` and `deposit:actions` as the
repository secret `ZENODO_TOKEN`. After the first release, the run prints the concept record id:
store it as the repository variable `ZENODO_RECORD` and add the DOI to `CITATION.cff`.

After a publish:

```bash
.venv/bin/python paper/build.py
git add paper/deposit && git commit -m "Catalog <day>"
git tag v<day> && git push origin master v<day>
```
