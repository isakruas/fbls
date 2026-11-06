# Zenodo deposit

One Zenodo record for the catalog, versioned by date, through Zenodo's GitHub integration
(enabled once on zenodo.org for this repository; metadata from `CITATION.cff`).

After a publish:

```bash
.venv/bin/python paper/build.py                         # paper/deposit/<day>/
git add paper/deposit && git commit -m "Catalog <day>" && git push
paper/release.sh                                        # GitHub release v<day> with the deposit files
```
