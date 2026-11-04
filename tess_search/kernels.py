"""Fused float64 kernels; no trial-by-observation or trial-by-mode tensors."""
import torch
import triton
import triton.language as tl


@triton.jit
def fold_fields(T, X, Scale, Out, N, NB, B: tl.constexpr):
    """Accumulate the four fields of each observation into interleaved phase bins (period, bin, field)."""
    f = tl.program_id(0)
    i = tl.program_id(1) * B + tl.arange(0, B)
    t = tl.load(T + i, i < N, other=0.)
    scale = tl.load(Scale + f)
    # Identical integer-cell convention is used by the interval refinement.
    cell = tl.floor(t * scale).to(tl.int64) % NB
    base = (f.to(tl.int64) * NB + cell) * 4
    for field in tl.static_range(4):
        x = tl.load(X + field * N + i, i < N, other=0.)
        tl.atomic_add(Out + base + field, x, i < N, sem="relaxed")


def doubled_cumulative(folded):
    """Cumulative sums over (period, bin, field) with the bins repeated once: (period, 2 nb + 1, field)."""
    first = folded.cumsum(1)
    zero = torch.zeros((folded.shape[0], 1, folded.shape[2]), device=folded.device, dtype=folded.dtype)
    return torch.cat((zero, first, first + first[:, -1:, :]), dim=1).contiguous()


@triton.jit
def circular_sum(C, row, start, width, NB):
    j = start % NB
    end = j + width
    left = tl.load(C + row + j)
    right = tl.load(C + row + tl.minimum(end, NB))
    wrap = tl.load(C + row + tl.maximum(end - NB, 0))
    return right - left + wrap


@triton.jit
def interval_sum(Row, field, start, width):
    """Interval sum of one of four interleaved fields on a doubled cumulative row.

    Row points at the period's (2 NB + 1) positions times 4 fields; start lies
    in [0, NB) and start + width <= 2 NB, so offsets fit int32 and never wrap.
    """
    return tl.load(Row + ((start + width) * 4 + field)) - tl.load(Row + (start * 4 + field))


@triton.jit
def template_sums(C, f, di, j, Scale, Dur, N, NB,
                  INGRESS: tl.constexpr, LAYERS: tl.constexpr, B: tl.constexpr):
    """Numerator, baseline projection and diagonal norm of one staircase template.

    C holds, per period, the four fields interleaved per position of a
    cumulative sum over the folded bins repeated twice (2 NB + 1 positions).
    """
    valid = j < NB
    jj = tl.minimum(j, NB - 1)
    scale = tl.load(Scale + f)
    raw_k = tl.maximum(1, tl.floor(tl.load(Dur + di) * scale + 0.5).to(tl.int32))
    # Invalid wide templates must not issue out-of-range prefix loads.
    k = tl.minimum(raw_k, NB // 2)
    a = tl.zeros([B], tl.float64)
    hp = tl.zeros([B], tl.float64)
    norm = tl.zeros([B], tl.float64)
    row = C + f.to(tl.int64) * 4 * (2 * NB + 1)
    count = interval_sum(row, 3, jj, k)
    for layer in tl.static_range(LAYERS):
        fraction = 0. if LAYERS == 1 else layer / (LAYERS - 1)
        factor = tl.full([], 1 - 2 * INGRESS * fraction, tl.float64)
        width = tl.maximum(1, tl.floor(k.to(tl.float64) * factor + 0.5).to(tl.int32))
        start = jj + (k - width) // 2
        a += interval_sum(row, 0, start, width) / LAYERS
        hp += interval_sum(row, 1, start, width) / LAYERS
        norm += interval_sum(row, 2, start, width) * ((2. * layer + 1.) / (LAYERS * LAYERS))
    usable = valid & (raw_k < NB // 2) & (count >= 3) & (count <= N - 3) & (a > 0.)
    return jj, k, a, hp, norm, usable


@triton.jit
def scan_floor(C, Scale, Dur, Rho, BaseNorm, Floor,
               N, NB,
               INGRESS: tl.constexpr, LAYERS: tl.constexpr,
               GUARD: tl.constexpr, B: tl.constexpr):
    """Per-period maximum of a lower score bound in exact arithmetic.

    Covariance modes only remove precision, so a/sqrt(norm - d) never exceeds
    a template's exact score. The largest value over a period therefore never
    exceeds that period's best exact score. Positive float64 values have
    order-preserving int64 bit patterns, which permits an integer atomic max.
    """
    f = tl.program_id(0)
    di = tl.program_id(1)
    j = tl.program_id(2) * B + tl.arange(0, B)
    jj, k, a, hp, norm, usable = template_sums(C, f, di, j, Scale, Dur, N, NB, INGRESS, LAYERS, B)
    a = a / tl.load(Rho + di)
    d = hp * hp / tl.load(BaseNorm)
    q = norm - d
    margin = GUARD * (norm + tl.abs(d) + 1.)
    lower = a / tl.sqrt(tl.maximum(q, 1.e-300))
    lower = tl.where(usable & (q > margin), lower, 0.)
    best = tl.max(lower.to(tl.int64, bitcast=True), 0)
    tl.atomic_max(Floor + f, best, sem="relaxed")


@triton.jit
def scan_bounds(C, Period, Scale, Dur, Rho, Beta0, BaseNorm, Records, Counter, Floor,
                N, NB, ND: tl.constexpr,
                CAP: tl.constexpr, THRESHOLD: tl.constexpr,
                INGRESS: tl.constexpr, LAYERS: tl.constexpr,
                GUARD: tl.constexpr, B: tl.constexpr, USE_FLOOR: tl.constexpr):
    f = tl.program_id(0)
    di = tl.program_id(1)
    j = tl.program_id(2) * B + tl.arange(0, B)
    p = tl.load(Period + f)
    scale = tl.load(Scale + f)
    jj, k, a, hp, norm, usable = template_sums(C, f, di, j, Scale, Dur, N, NB, INGRESS, LAYERS, B)
    # Correlated-noise factor per duration; rho = 1 is the plain GLS score.
    rho = tl.load(Rho + di)
    a = a / rho
    d = hp * hp / tl.load(BaseNorm)
    q = norm - d
    qmin = q - tl.load(Beta0 + f * ND + di) * norm
    margin = GUARD * (norm + tl.abs(d) + 1.)
    upper = tl.where(qmin > margin, tl.maximum(a, 0.) / tl.sqrt(tl.maximum(qmin - margin, 1.e-300)), float("inf"))
    limit = tl.full([], THRESHOLD, tl.float64)
    if USE_FLOOR:
        # The relative slack keeps the best template despite last-bit rounding.
        limit = tl.maximum(limit, tl.load(Floor + f) * (1. - 1.e-9))
    keep = usable & (upper >= limit - GUARD)
    num = tl.sum(keep.to(tl.int32), 0)
    offset = tl.atomic_add(Counter, num, sem="relaxed")
    dest = offset + tl.cumsum(keep.to(tl.int32), 0) - 1
    write = keep & (dest < CAP)
    ptr = Records + dest.to(tl.int64) * 9
    tl.store(ptr + 0, p, write)
    tl.store(ptr + 1, scale, write)
    tl.store(ptr + 2, jj.to(tl.float64), write)
    tl.store(ptr + 3, k.to(tl.float64), write)
    tl.store(ptr + 4, a, write)
    tl.store(ptr + 5, norm, write)
    tl.store(ptr + 6, d, write)
    tl.store(ptr + 7, tl.full([], INGRESS, tl.float64), write)
    tl.store(ptr + 8, tl.zeros([B], tl.float64) + rho, write)


@triton.jit
def lower_cell(T, edge, scale, N, STEPS: tl.constexpr):
    lo = tl.full(edge.shape, 0, tl.int32)
    hi = tl.full(edge.shape, N, tl.int32)
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        value = tl.load(T + mid, mid < N, other=float("inf"))
        before = value * scale < edge
        lo = tl.where(before, mid + 1, lo)
        hi = tl.where(before, hi, mid)
    return lo


@triton.jit
def refine_modes(T, Prefix, Beta, Support, Records, Score, Depth, Used, Status, Floor,
                 N, NB, R: tl.constexpr,
                 THRESHOLD: tl.constexpr, ADAPTIVE: tl.constexpr,
                 LAYERS: tl.constexpr, GUARD: tl.constexpr,
                 STEPS: tl.constexpr, CYCLES: tl.constexpr, B: tl.constexpr,
                 LOCALIZED: tl.constexpr, USE_FLOOR: tl.constexpr):
    c = tl.program_id(0)
    limit = tl.full([], THRESHOLD, tl.float64)
    if USE_FLOOR:
        limit = tl.maximum(limit, tl.load(Floor + c) * (1. - 1.e-9))
    rec = Records + c * 9
    scale = tl.load(rec + 1)
    j = tl.load(rec + 2).to(tl.int64)
    k = tl.load(rec + 3).to(tl.int64)
    a = tl.load(rec + 4)
    norm = tl.load(rec + 5)
    d = tl.load(rec + 6)
    ingress = tl.load(rec + 7)
    actual_layers = tl.where(ingress == 0., 1, LAYERS)
    # Store geometry in registers and reuse it across every covariance mode.
    cycle = tl.arange(0, B).to(tl.int64) - 1
    lowers = ()
    uppers = ()
    for layer in tl.static_range(LAYERS):
        fraction = 0. if LAYERS == 1 else layer / (LAYERS - 1)
        width = tl.maximum(1, tl.floor(k * (1 - 2 * ingress * fraction) + 0.5).to(tl.int64))
        start = cycle * NB + j + (k - width) // 2
        lo = lower_cell(T, start.to(tl.float64), scale, N, STEPS)
        hi = lower_cell(T, (start + width).to(tl.float64), scale, N, STEPS)
        lowers += (lo,)
        uppers += (hi,)
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
        reject = ADAPTIVE & (upper < limit - GUARD)
        if reject:
            active = tl.full([], False, tl.int1)
            rejected = tl.full([], True, tl.int1)
        else:
            gamma = tl.full([], 0., tl.float64)
            row = r.to(tl.int64) * (N + 1)
            for chunk in range(triton.cdiv(CYCLES + 1, B)):
                cc = chunk * B + cycle
                for layer in tl.static_range(LAYERS):
                    if LOCALIZED:
                        first_time = tl.load(Support + r * 2)
                        first_cycle = tl.floor(first_time * scale / NB).to(tl.int64) - 2
                        fraction = 0. if LAYERS == 1 else layer / (LAYERS - 1)
                        width = tl.maximum(1, tl.floor(k.to(tl.float64) * (1 - 2 * ingress * fraction) + 0.5).to(tl.int64))
                        start = (cc + 1 + first_cycle) * NB + j + (k - width) // 2
                        lo = lower_cell(T, start.to(tl.float64), scale, N, STEPS)
                        hi = lower_cell(T, (start + width).to(tl.float64), scale, N, STEPS)
                    elif chunk == 0:
                        lo = lowers[layer]
                        hi = uppers[layer]
                    else:
                        fraction = 0. if LAYERS == 1 else layer / (LAYERS - 1)
                        width = tl.maximum(1, tl.floor(k.to(tl.float64) * (1 - 2 * ingress * fraction) + 0.5).to(tl.int64))
                        start = cc * NB + j + (k - width) // 2
                        lo = lower_cell(T, start.to(tl.float64), scale, N, STEPS)
                        hi = lower_cell(T, (start + width).to(tl.float64), scale, N, STEPS)
                    include = (ingress != 0.) | (layer == 0)
                    valid = (cc < CYCLES) & (hi > lo) & include
                    value = tl.load(Prefix + row + hi, valid, other=0.) - tl.load(Prefix + row + lo, valid, other=0.)
                    gamma += tl.sum(value, 0) / actual_layers
            q -= next_beta * gamma * gamma
            remaining -= gamma * gamma
            r += 1
    invalid = (~rejected) & (q <= 0.)
    score = tl.where(rejected | invalid, -float("inf"), a / tl.sqrt(tl.maximum(q, 1.e-300)))
    tl.store(Score + c, score)
    tl.store(Depth + c, tl.where(rejected | invalid, 0., a / tl.maximum(q, 1.e-300)))
    tl.store(Used + c, r)
    tl.store(Status + c, invalid.to(tl.int32))
