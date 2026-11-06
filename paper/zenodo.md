# Zenodo deposit

One Zenodo record, fed by GitHub releases (Zenodo's GitHub integration, enabled once on
zenodo.org for this repository). Metadata come from `CITATION.cff`.

After a publish:

```bash
.venv/bin/python paper/build.py        # paper/deposit/<day>/ (tracked)
git add paper/deposit && git commit -m "Catalog <day>"
git tag v<day> && git push origin master --tags
gh release create v<day> --title "Catalog <day>" --notes-file paper/deposit/<day>/ReadMe.txt
```

Each release becomes a new version of the record, with its own DOI; the concept DOI always
points to the latest. Add the DOI to `CITATION.cff` (`doi:`) after the first release.
