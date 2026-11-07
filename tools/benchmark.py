"""Sensitivity and run-time comparison with third-party transit search tools.

Usage: python benchmark.py FBLS_BINARY STARS_JSON [--tools astropy,tls,cuvarbase] [--limit N] [--workdir DIR]
Light curves are read from data/curves/<tic>/ next to the reference folder.
For each reference star the same light curves (median-normalized, wotan
biweight-flattened for the third-party tools, which have no preparation stage)
are searched over the same period range; each tool's best period is compared
with the ExoFOP ephemeris (aliases 1/3, 1/2, 1, 2, 3 accepted).
"""
import argparse, glob, json, os, subprocess, time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('binary'); parser.add_argument('stars')
parser.add_argument('--tools', default='astropy,tls')
parser.add_argument('--limit', type=int, default=0)
parser.add_argument('--output', default='benchmark.json')
parser.add_argument('--workdir', default='/tmp')
args = parser.parse_args()
tools = args.tools.split(',')
root = os.path.dirname(os.path.abspath(args.stars))
stars = json.load(open(args.stars))
if args.limit:
    stars = stars[:args.limit]

def alias(p, cat):
    return min(abs(p / cat - m) / m for m in (1/3, 0.5, 1, 2, 3))

def load(tic):
    files = sorted(glob.glob(os.path.join(root, '..', 'data', 'curves', tic, 'sector-*.npz')))
    t = np.concatenate([np.load(f)['time_bjd'] for f in files]); y = np.concatenate([np.load(f)['flux'] for f in files])
    e = np.concatenate([np.load(f)['flux_err'] for f in files])
    keep = np.isfinite(t) & np.isfinite(y) & np.isfinite(e)
    t, y, e = t[keep], y[keep], e[keep]
    order = np.argsort(t)
    return files, t[order], y[order], e[order]

rows = []
for s in stars:
    tic, cat = s['tic'], s['period']
    files, t, y, e = load(tic)
    row = {'tic': tic, 'toi': s['toi'], 'catalog_period': cat, 'depth_ppm': s['depth_ppm'], 'sectors': len(files), 'n': len(t)}
    t0 = time.time()
    out = os.path.join(args.workdir, f'{tic}.json')
    subprocess.run([args.binary, 'search', '--input', *files, '--pmax', str(s['pmax']), '--output', out], check=True, capture_output=True)
    row['fbls_seconds'] = time.time() - t0
    res = json.load(open(out))
    c = res['signal_candidates']
    row['fbls_period'] = c[0]['period_days'] if c else None
    row['fbls_alias'] = alias(c[0]['period_days'], cat) if c else None
    if any(tool in tools for tool in ('astropy', 'tls', 'cuvarbase')):
        from wotan import flatten
        med = np.nanmedian(y)
        flat = flatten(t, y / med, method='biweight', window_length=0.5)
        ok = np.isfinite(flat)
        tf, yf, ef = t[ok], flat[ok], e[ok] / med
        periods = np.exp(np.linspace(np.log(0.5), np.log(s['pmax']), 20000))
        durations = np.array([0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8]) / 24
    if 'astropy' in tools:
        from astropy.timeseries import BoxLeastSquares
        t0 = time.time()
        power = BoxLeastSquares(tf, yf, ef).power(periods, durations, objective='snr')
        row['astropy_seconds'] = time.time() - t0
        row['astropy_period'] = float(power.period[np.argmax(power.power)])
        row['astropy_alias'] = alias(row['astropy_period'], cat)
    if 'tls' in tools:
        from transitleastsquares import transitleastsquares
        t0 = time.time()
        r = transitleastsquares(tf, yf, ef).power(period_min=0.5, period_max=s['pmax'], use_threads=os.cpu_count(), show_progress_bar=False)
        row['tls_seconds'] = time.time() - t0
        row['tls_period'] = float(r.period); row['tls_alias'] = alias(float(r.period), cat); row['tls_sde'] = float(r.SDE)
    if 'cuvarbase' in tools:
        import cuvarbase.bls as bls
        t0 = time.time()
        freqs = 1 / periods[::-1]
        p = bls.eebls_gpu_fast(tf.astype(np.float32), yf.astype(np.float32), ef.astype(np.float32), freqs.astype(np.float32))
        row['cuvarbase_seconds'] = time.time() - t0
        row['cuvarbase_period'] = float(1 / freqs[np.argmax(p)]); row['cuvarbase_alias'] = alias(row['cuvarbase_period'], cat)
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump(rows, open(args.output, 'w'), indent=1)
for tool in ['fbls'] + tools:
    n = sum(1 for r in rows if r.get(f'{tool}_alias') is not None and r[f'{tool}_alias'] < 1e-3)
    secs = sum(r.get(f'{tool}_seconds', 0) for r in rows)
    print(f'{tool}: recovered {n}/{len(rows)}, {secs:.0f} s')
