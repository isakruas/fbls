"""Regenerate data/manifest.txt and data/inventory.csv from data/curves and the catalogs."""
import os, glob, json, csv
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
curves = os.path.join(root, 'data', 'curves')
tics = sorted(os.listdir(curves), key=int)
ref = {s['tic'] for s in json.load(open(os.path.join(root, 'reference', 'stars.json')))}
toi = {}
for r in csv.DictReader(open(os.path.join(root, 'data', 'catalogs', 'exofop_toi.csv'))):
    toi.setdefault(r['TIC ID'], []).append((r['TOI'], r.get('TFOPWG Disposition', ''), r.get('Period (days)', '')))
with open(os.path.join(root, 'data', 'manifest.txt'), 'w') as m, open(os.path.join(root, 'data', 'inventory.csv'), 'w') as inv:
    inv.write('tic,sectors,files,in_reference_set,toi,disposition,catalog_period_days\n')
    for t in tics:
        files = sorted(glob.glob(f'{curves}/{t}/sector-*.npz'))
        if not files:
            continue
        rel = [os.path.relpath(f, root) for f in files]
        m.write(f"{t} - {' '.join(rel)}\n")
        entries = toi.get(t, [])
        inv.write('%s,%d,%s,%s,%s,%s,%s\n' % (t, len(files), ';'.join(os.path.basename(f) for f in files), 'yes' if t in ref else 'no',
                                            ';'.join(e[0] for e in entries), ';'.join(e[1] for e in entries), ';'.join(e[2] for e in entries)))
print(len(tics), 'stars')
