"""Parallel mode tiles for gap-localized covariance bases."""
import triton
import triton.language as tl

from .kernels import lower_cell


@triton.jit
def refine_local_modes(T, Prefix, Beta, Support, Records, Score, Depth, Used, Status, Floor,
                       N: tl.constexpr, NB: tl.constexpr, R: tl.constexpr,
                       THRESHOLD: tl.constexpr, ADAPTIVE: tl.constexpr,
                       LAYERS: tl.constexpr, GUARD: tl.constexpr,
                       STEPS: tl.constexpr, CYCLES: tl.constexpr,
                       B: tl.constexpr, GROUP: tl.constexpr, USE_FLOOR: tl.constexpr):
    candidate = tl.program_id(0)
    limit = tl.full([], THRESHOLD, tl.float64)
    if USE_FLOOR:
        limit = tl.maximum(limit, tl.load(Floor + candidate) * (1. - 1.e-9))
    rec = Records + candidate * 9
    scale = tl.load(rec + 1)
    j = tl.load(rec + 2).to(tl.int64)
    k = tl.load(rec + 3).to(tl.int64)
    a = tl.load(rec + 4)
    norm = tl.load(rec + 5)
    d = tl.load(rec + 6)
    ingress = tl.load(rec + 7)
    actual_layers = tl.where(ingress == 0., 1, LAYERS)
    mode_lane = tl.arange(0, GROUP)
    orbit_lane = tl.arange(0, B)
    q = norm - d
    remaining = norm
    r = 0
    active = tl.full([], True, tl.int1)
    rejected = tl.full([], False, tl.int1)
    margin = GUARD * (norm + tl.abs(d) + 1.)
    while (r < R) & active:
        next_beta = tl.load(Beta + r)
        qmin = q - next_beta * tl.maximum(remaining, 0.)
        upper = tl.where(qmin > margin, a / tl.sqrt(tl.maximum(qmin - margin, 1.e-300)), float("inf"))
        if ADAPTIVE & (upper < limit - GUARD):
            active = tl.full([], False, tl.int1)
            rejected = tl.full([], True, tl.int1)
        else:
            mode = r + mode_lane
            mode_valid = mode < R
            first_time = tl.load(Support + mode * 2, mode_valid, other=0.)
            first_cycle = tl.floor(first_time * scale / NB).to(tl.int64) - 2
            gamma = tl.zeros([GROUP], tl.float64)
            for chunk in range(triton.cdiv(CYCLES + 1, B)):
                orbit = chunk * B + orbit_lane
                cycle = first_cycle[:, None] + orbit[None, :]
                for layer in tl.static_range(LAYERS):
                    fraction = 0. if LAYERS == 1 else layer / (LAYERS - 1)
                    width = tl.maximum(1, tl.floor(k.to(tl.float64) * (1 - 2 * ingress * fraction) + 0.5).to(tl.int64))
                    start = cycle * NB + j + (k - width) // 2
                    lo = lower_cell(T, start.to(tl.float64), scale, N, STEPS)
                    hi = lower_cell(T, (start + width).to(tl.float64), scale, N, STEPS)
                    mask = mode_valid[:, None] & (orbit[None, :] < CYCLES) & (hi > lo) & ((ingress != 0.) | (layer == 0))
                    row = mode.to(tl.int64)[:, None] * (N + 1)
                    value = tl.load(Prefix + row + hi, mask, other=0.) - tl.load(Prefix + row + lo, mask, other=0.)
                    gamma += tl.sum(value, 1) / actual_layers
            beta = tl.load(Beta + mode, mode_valid, other=0.)
            q -= tl.sum(beta * gamma * gamma, 0)
            remaining -= tl.sum(gamma * gamma, 0)
            r = tl.minimum(r + GROUP, R)
    invalid = (~rejected) & (q <= 0.)
    score = tl.where(rejected | invalid, -float("inf"), a / tl.sqrt(tl.maximum(q, 1.e-300)))
    tl.store(Score + candidate, score)
    tl.store(Depth + candidate, tl.where(rejected | invalid, 0., a / tl.maximum(q, 1.e-300)))
    tl.store(Used + candidate, r)
    tl.store(Status + candidate, invalid.to(tl.int32))
