"""Single-pass diagonal GLS period maxima on the existing finite shape bank."""
import triton
import triton.language as tl

from .kernels import circular_sum


@triton.jit
def shape_sums(C, row, j, k, outer_a, outer_p, outer_w, ingress, Factors, si,
               NB: tl.constexpr, LAYERS: tl.constexpr):
    a = tl.where(ingress == 0., outer_a, outer_a / LAYERS)
    hp = tl.where(ingress == 0., outer_p, outer_p / LAYERS)
    coefficient = tl.full([], 1. / (LAYERS * LAYERS), tl.float64)
    norm = tl.where(ingress == 0., outer_w, outer_w * coefficient)
    if ingress != 0.:
        for layer in tl.static_range(1, LAYERS):
            factor = tl.load(Factors + si * LAYERS + layer)
            width = tl.maximum(1, tl.floor(k.to(tl.float64) * factor + .5).to(tl.int32))
            start = j + (k - width) // 2
            a += circular_sum(C, row, start, width, NB) / LAYERS
            hp += circular_sum(C, row + NB + 1, start, width, NB) / LAYERS
            coefficient = tl.full([], (2. * layer + 1.) / (LAYERS * LAYERS), tl.float64)
            norm += circular_sum(C, row + 2 * (NB + 1), start, width, NB) * coefficient
    return a, hp, norm


@triton.jit
def tile_choice(C, Scale, Dur, Rho, Ingress, Factors, BaseNorm, f, tile,
                N, NB: tl.constexpr, ND: tl.constexpr, NS: tl.constexpr,
                LAYERS: tl.constexpr, B: tl.constexpr, FAST: tl.constexpr):
    j = tile * B + tl.arange(0, B)
    jj = tl.minimum(j, NB - 1)
    scale = tl.load(Scale + f)
    baseline = tl.load(BaseNorm)
    row = f.to(tl.int64) * 4 * (NB + 1)
    best = tl.full([B], -float("inf"), tl.float64)
    best_key = tl.full([B], 9223372036854775807, tl.int64)
    best_duration = tl.zeros([B], tl.int32)
    bad = tl.full([B], False, tl.int1)
    best_aa = tl.zeros([B], tl.float64)
    best_qq = tl.full([B], 1., tl.float64)
    sensitive = tl.full([B], False, tl.int1)
    unsafe = tl.full([B], False, tl.int1)
    tiny = tl.full([], 1.e-280, tl.float64)
    huge = tl.full([], 1.e280, tl.float64)
    guard = tl.full([], 1.e-10, tl.float64)
    for di in range(ND):
        raw_k = tl.maximum(1, tl.floor(tl.load(Dur + di) * scale + .5).to(tl.int32))
        k = tl.minimum(raw_k, NB // 2)
        rho = tl.load(Rho + di)
        count = circular_sum(C, row + 3 * (NB + 1), jj, k, NB)
        outer_a = circular_sum(C, row, jj, k, NB)
        outer_p = circular_sum(C, row + NB + 1, jj, k, NB)
        outer_w = circular_sum(C, row + 2 * (NB + 1), jj, k, NB)
        usable = (j < NB) & (raw_k < NB // 2) & (count >= 3) & (count <= N - 3)
        for si in tl.static_range(NS):
            ingress = tl.load(Ingress + si)
            a, hp, norm = shape_sums(C, row, jj, k, outer_a, outer_p, outer_w,
                                     ingress, Factors, si, NB, LAYERS)
            valid = usable & (a > 0.)
            key = (si * (NB + 1) + k.to(tl.int64)) * (NB + 1) + jj
            if FAST:
                # c is common to all templates. Compare a^2/[rho^2(c*n-hp^2)]
                # by cross products, without per-template divides or roots.
                # Ill-conditioned/abnormal ranges and near ties are rescanned.
                cn = baseline * norm
                qq0 = cn - hp * hp
                qq = rho * rho * qq0
                aa = a * a
                left = aa * best_qq
                right = best_aa * qq
                safe = ((cn > 0.) & (qq0 >= .001 * cn) &
                        (aa >= tiny) & (aa <= huge) & (qq >= tiny) & (qq <= huge) &
                        (left >= tiny) & (left <= huge) &
                        ((right == 0.) | ((right >= tiny) & (right <= huge))))
                unsafe |= valid & ~safe
                near = (best_aa > 0.) & (tl.abs(left - right) <= guard * tl.maximum(left, right))
                far_better = left > right * (1. + 4. * guard)
                sensitive = tl.where(valid & safe & far_better, False,
                                     sensitive | (valid & safe & near))
                better = valid & safe & ((left > right) | ((left == right) & (key < best_key)))
                best_aa = tl.where(better, aa, best_aa)
                best_qq = tl.where(better, qq, best_qq)
            else:
                d = hp * hp / baseline
                q = norm - d
                bad |= valid & (q <= 0.)
                candidate = (a / rho) / tl.sqrt(tl.maximum(q, 1.e-300))
                value = tl.where(valid & (q > 0.) & (candidate < float("inf")), candidate, -float("inf"))
                better = (value > best) | ((value == best) & (key < best_key))
                best = tl.where(better, value, best)
            best_key = tl.where(better, key, best_key)
            best_duration = tl.where(better, di, best_duration)
    if FAST:
        best = tl.where(best_aa > 0., best_aa / best_qq, -float("inf"))
        # Empty tiles still need a harmless decodable key; they are not returned.
        best_key = tl.where(best_aa > 0., best_key, (NB + 1) + jj)
    value = tl.max(best, 0)
    key = tl.min(tl.where(best == value, best_key, 9223372036854775807), 0)
    winner = (best == value) & (best_key == key)
    lane = tl.min(tl.where(winner, tl.arange(0, B), 2147483647), 0)
    chosen = tl.arange(0, B) == lane
    di = tl.sum(tl.where(chosen, best_duration, 0), 0)
    near_top = (best > 0.) & (best >= value * (1. - guard))
    retry = ((tl.sum(unsafe.to(tl.int32), 0) > 0) |
             (tl.sum(near_top.to(tl.int32), 0) > 1) |
             (tl.sum((near_top & sensitive).to(tl.int32), 0) > 0))
    return key, di, value, retry, tl.sum(bad.to(tl.int32), 0)


@triton.jit
def diagonal_tiles(C, Period, Scale, Dur, Rho, Ingress, Factors, BaseNorm, Parts, Keys, Invalid, Retries,
                   N, NB: tl.constexpr, ND: tl.constexpr,
                   NS: tl.constexpr, LAYERS: tl.constexpr, NT: tl.constexpr,
                   B: tl.constexpr):
    tile = tl.program_id(0)
    f = tl.program_id(1)
    key, di, value, retry, bad = tile_choice(C, Scale, Dur, Rho, Ingress, Factors, BaseNorm,
                                            f, tile, N, NB, ND, NS, LAYERS, B, True)
    if retry:
        key, di, value, _, bad = tile_choice(C, Scale, Dur, Rho, Ingress, Factors, BaseNorm,
                                            f, tile, N, NB, ND, NS, LAYERS, B, False)
    # Reconstruct only the winner instead of carrying its fields per phase.
    phase = key % (NB + 1)
    width = key // (NB + 1) % (NB + 1)
    si = key // ((NB + 1) * (NB + 1))
    ingress = tl.load(Ingress + si)
    rho = tl.load(Rho + di)
    row = f.to(tl.int64) * 4 * (NB + 1)
    baseline = tl.load(BaseNorm)
    outer_a = circular_sum(C, row, phase, width, NB)
    outer_p = circular_sum(C, row + NB + 1, phase, width, NB)
    outer_w = circular_sum(C, row + 2 * (NB + 1), phase, width, NB)
    a, hp, norm = shape_sums(C, row, phase, width, outer_a, outer_p, outer_w,
                             ingress, Factors, si, NB, LAYERS)
    a = a / rho
    d = hp * hp / baseline
    value = tl.where(value > -float("inf"), a / tl.sqrt(tl.maximum(norm - d, 1.e-300)), -float("inf"))
    depth = a / tl.maximum(norm - d, 1.e-300) * rho
    out = Parts + (f.to(tl.int64) * NT + tile) * 9
    tl.store(out + 0, phase.to(tl.float64))
    tl.store(out + 1, width.to(tl.float64))
    tl.store(out + 2, a)
    tl.store(out + 3, norm)
    tl.store(out + 4, d)
    tl.store(out + 5, ingress)
    tl.store(out + 6, rho)
    tl.store(out + 7, value)
    tl.store(out + 8, depth)
    tl.store(Keys + f * NT + tile, key)
    tl.store(Retries + f * NT + tile, retry.to(tl.int32))
    if bad > 0:
        tl.atomic_or(Invalid + f, 1, sem="relaxed")


@triton.jit
def diagonal_merge(Parts, Keys, Period, Scale, Records, Score, Depth,
                   NT: tl.constexpr, B: tl.constexpr):
    f = tl.program_id(0)
    i = tl.arange(0, B)
    index = f.to(tl.int64) * NT + i
    values = tl.load(Parts + index * 9 + 7, i < NT, other=-float("inf"))
    keys = tl.load(Keys + index, i < NT, other=9223372036854775807)
    best = tl.max(values, 0)
    key = tl.min(tl.where(values == best, keys, 9223372036854775807), 0)
    tile = tl.min(tl.where((i < NT) & (values == best) & (keys == key), i, 2147483647), 0)
    part = Parts + (f.to(tl.int64) * NT + tile) * 9
    out = Records + f * 9
    tl.store(out, tl.load(Period + f))
    tl.store(out + 1, tl.load(Scale + f))
    for field in tl.static_range(7):
        tl.store(out + 2 + field, tl.load(part + field))
    tl.store(Score + f, best)
    tl.store(Depth + f, tl.load(part + 8))
