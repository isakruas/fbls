# Zenodo deposit

One Zenodo record for the catalog, versioned by date, through Zenodo's GitHub integration
(enabled once on zenodo.org for this repository).

A push to `main` that changes `paper/deposit/` runs `.github/workflows/catalog.yml`: the newest
day folder becomes the whole content of the orphan branch `catalog` (five files: `ctoi.csv`,
`candidates.csv`, `figures.zip`, `ReadMe.txt`, `CITATION.cff`), tagged `v<day>` and released with
those files as assets. The release's source archive, which Zenodo archives, holds only them.

After a publish:

```bash
.venv/bin/python paper/build.py
git add paper/deposit && git commit -m "Catalog <day>" && git push
```
