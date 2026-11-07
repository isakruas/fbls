"""Build the reference star set: confirmed or known TOI hosts (TFOPWG disposition CP/KP)
with local light curves, sampled per period bin.

Usage: python select_reference.py EXOFOP_TOI_CSV CURVE_DIR [CURVE_DIR ...]
Writes reference/stars.json; the light curves stay under data/curves/<tic>/.
"""
import csv, glob, json, os, shutil, sys
import numpy as np

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reference')
catalog, dirs = sys.argv[1], sys.argv[2:]
rows = [r for r in csv.DictReader(open(catalog)) if r.get('TFOPWG Disposition') in ('CP', 'KP')]
edges = [0.3, 0.7, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 25, 40, 60, 100, 400]
per_bin = 6
candidates = []
for r in rows:
    tic = r['TIC ID']
    try:
        period, depth = float(r['Period (days)']), float(r['Depth (ppm)'])
    except (ValueError, KeyError):
        continue
    files = []
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, tic, 'sector-*.npz')))
        if files:
            break
    if len(files) < 2 or period <= 0:
        continue
    tmin, tmax = np.inf, -np.inf
    for f in files:
        t = np.load(f)['time_bjd']
        t = t[np.isfinite(t)]
        tmin, tmax = min(tmin, t.min()), max(tmax, t.max())
    baseline = tmax - tmin
    if period > baseline / 2.5:
        continue
    candidates.append({'tic': tic, 'toi': r['TOI'], 'period': period, 'depth_ppm': depth,
                       'duration_h': float(r.get('Duration (hours)') or 0), 'tmag': float(r.get('TESS Mag') or 0),
                       'epoch_bjd': float(r.get('Epoch (BJD)') or 0), 'sectors': len(files), 'baseline_days': baseline,
                       'pmax': float(min(max(20.0, round(1.6 * period)), baseline)), 'files': files})
selected = []
for a, b in zip(edges[:-1], edges[1:]):
    group = sorted([c for c in candidates if a <= c['period'] < b], key=lambda c: c['depth_ppm'])
    if not group:
        continue
    picks = [group[int(i * (len(group) - 1) / (min(per_bin, len(group)) - 1))] for i in range(min(per_bin, len(group)))] if len(group) > 1 else group
    for i, c in enumerate(picks):
        c['python_reference'] = i in (0, len(picks) - 1)
        selected.append(c)
for c in selected:
    c['files'] = [os.path.relpath(f, os.path.join(root, '..')) for f in c['files']]
json.dump(selected, open(os.path.join(root, 'stars.json'), 'w'), indent=1)
print(len(candidates), 'candidates,', len(selected), 'selected')
for c in selected:
    print(f"{c['tic']:>10} TOI {c['toi']:>8} P={c['period']:8.3f} depth={c['depth_ppm']:8.0f} ppm sectors={c['sectors']} baseline={c['baseline_days']:.0f} pmax={c['pmax']:.0f}")
