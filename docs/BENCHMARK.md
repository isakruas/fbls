# Benchmark against other transit search tools

Sixteen confirmed or known TOI hosts (`reference/stars.json`, first sixteen rows), the same
light curves and the same period range (0.5 d to `pmax`) for every tool. The third-party tools
search a wotan-flattened curve on a 20000-period logarithmic grid with nine box durations;
FBLS runs its full pipeline (five duration families, 13 durations, three shapes, up to three
passes, single events, model comparison). A star counts as recovered when the tool's best
period is within 0.1% of the ExoFOP period or of its 1/3, 1/2, 2x or 3x alias. Times are wall
seconds on one RTX 3050 and a 12-thread CPU (TLS uses every thread); `tools/benchmark.py`
reproduces the table.

| TIC | sectors | P (d) | FBLS | s | astropy BLS | s | TLS | s | cuvarbase | s |
|---|---|---|---|---|---|---|---|---|---|---|
| 153976959 | 7 | 0.686 | yes | 56 | yes | 13 | yes | 125 | no | 0.8 |
| 31374837 | 5 | 0.490 | no | 90 | no | 11 | no | 158 | no | 0.6 |
| 377064495 | 5 | 0.447 | no | 36 | no | 11 | no | 89 | no | 0.5 |
| 180695581 | 4 | 0.549 | yes | 43 | no | 7.9 | yes | 50 | no | 0.4 |
| 370133522 | 8 | 0.518 | yes | 46 | yes | 14 | yes | 214 | no | 0.8 |
| 392476080 | 2 | 0.672 | yes | 16 | no | 5.0 | yes | 7.7 | yes | 0.2 |
| 55315929 | 2 | 0.720 | yes | 1.3 | yes | 6.1 | yes | 0.9 | no | 0.3 |
| 377780790 | 11 | 0.837 | yes | 42 | yes | 18 | yes | 219 | no | 1.0 |
| 239332587 | 7 | 0.926 | yes | 34 | yes | 14 | yes | 106 | no | 0.8 |
| 287139872 | 42 | 0.935 | yes | 143 | no | 58 | yes | 1703 | no | 3.6 |
| 139285832 | 6 | 0.777 | yes | 32 | yes | 11 | yes | 141 | no | 0.6 |
| 37348844 | 3 | 0.767 | yes | 11 | yes | 7.9 | yes | 23 | no | 0.4 |
| 91555165 | 6 | 1.053 | yes | 34 | yes | 12 | yes | 153 | no | 0.7 |
| 143143769 | 5 | 1.337 | yes | 30 | yes | 11 | yes | 139 | yes | 0.6 |
| 452006073 | 4 | 1.267 | yes | 21 | no | 11 | yes | 58 | no | 0.5 |
| 22529346 | 6 | 1.275 | yes | 88 | no | 12 | yes | 177 | no | 0.7 |

Recovered: FBLS 14/16, TLS 14/16, astropy BLS 9/16, cuvarbase 2/16.
Total time: FBLS 724 s, TLS 3365 s, astropy BLS 223 s, cuvarbase 12 s.

The two stars no tool recovers have catalog periods of 0.447 and 0.490 d, below the 0.5 d
lower bound used in this run. The astropy misses are 2x aliases or neighbouring grid points of
the true period; cuvarbase runs in float32 on a coarse grid and returns the wrong period on most
stars.

With the default lower bound now at 0.3 d, FBLS recovers both: TIC 31374837 at 0.4901 d
(SNR 40.4, 278 ppm) and TIC 377064495 at 0.4466 d (SNR 21.8, 255 ppm), 16 of 16.
