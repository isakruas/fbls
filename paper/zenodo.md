# Zenodo deposit

Build the deposit folder after a publish, then upload the files of `paper/deposit/<day>/`
to the Zenodo record (new version when the record exists; one folder per version):

```bash
.venv/bin/python paper/build.py
```

Form fields:

- Resource type: Dataset
- Title: FBLS: Preliminary Catalog of Exoplanet Transit Candidates Identified in TESS Photometry
- Creators: Isak Paulo de Andrade Ruas (NROLABS)
- License: Creative Commons Attribution 4.0 International
- Version: date of the publish, `YYYY-MM-DD`
- Keywords: TESS, exoplanets, transit search, planet candidates, CTOI
- Related works: https://github.com/isakruas/fbls (is supplemented by this software)
- Description: paste the first paragraph and the file list of `ReadMe.txt`.
