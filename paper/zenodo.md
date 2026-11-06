# Zenodo deposit

One Zenodo record for the catalog, versioned by date, through Zenodo's GitHub integration
(enabled once on zenodo.org for this repository; metadata from the `CITATION.cff` of the deposit folder). Pushing a tag
`v<day>` runs `.github/workflows/release.yml`, which creates the GitHub release with the files of
`paper/deposit/<day>/` as its assets.

After a publish:

```bash
.venv/bin/python paper/build.py
git add paper/deposit && git commit -m "Catalog <day>"
git tag v<day> && git push origin master v<day>
```
