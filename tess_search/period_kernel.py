"""Single-pass diagonal GLS period maxima on the existing finite shape bank."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

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


@triton.jit
def fold_comparison(T, X, Scale, Out, N: tl.constexpr, NB: tl.constexpr,
                    NF: tl.constexpr, B: tl.constexpr):
    """Retain cycle parity by accumulating sufficient statistics over two periods."""
    f = tl.program_id(0)
    field = tl.program_id(1)
    i = tl.program_id(2) * B + tl.arange(0, B)
    cell = tl.floor(tl.load(T + i, i < N, other=0.) * tl.load(Scale + f)).to(tl.int64) % NB
    value = tl.load(X + field * N + i, i < N, other=0.)
    tl.atomic_add(Out + (f.to(tl.int64) * NF + field) * NB + cell, value, i < N, sem="relaxed")


@triton.jit
def layered_sum(C, row, start, width, ingress, NB: tl.constexpr, LAYERS: tl.constexpr,
                SQUARED: tl.constexpr):
    """Integrate a staircase or its square using nested interval sums."""
    value = tl.full(start.shape, 0., tl.float64)
    for layer in tl.static_range(LAYERS):
        fraction = tl.full([], layer if LAYERS > 1 else 0, tl.float64) / max(LAYERS - 1, 1)
        inner = tl.maximum(1, tl.floor(width.to(tl.float64) * (1 - 2 * ingress * fraction) + .5).to(tl.int32))
        coefficient = tl.full([], 2 * layer + 1 if SQUARED else 1, tl.float64) / (LAYERS * LAYERS if SQUARED else LAYERS)
        value += coefficient * circular_sum(C, row, start + (width - inner) // 2, inner, NB)
    return value


@triton.jit
def log_add(x, y):
    high = tl.maximum(x, y)
    return tl.where(high == -float("inf"), high,
                    high + tl.log(1. + tl.exp(tl.minimum(x, y) - high)))


@triton.jit
def positive_amplitude_evidence(a, q, tau):
    """Log evidence against zero amplitude for a normalized half-normal depth prior."""
    q = tl.maximum(q, 0.)
    precision = 1. + tau * tau * q
    z = tau * a / tl.sqrt(precision)
    # The negative-tail expansion avoids cancellation between z^2/2 and log Phi(z).
    zz = tl.maximum(z * z, 1.)
    inverse = 1. / zz
    series = 1. + inverse * (-1. + inverse * (3. + inverse * (-15. + inverse *
             (105. + inverse * (-945. + inverse * (10395. - inverse * 135135.))))))
    negative_tail = -.5 * tl.log(precision) - tl.log(tl.maximum(-z, 1.)) \
                    - tl.full([], .22579135264472743236, tl.float64) + tl.log(series)
    regular = -.5 * tl.log(precision) + .5 * z * z \
              + tl.log(tl.maximum(libdevice.erfc(-z * tl.full([], .7071067811865475244, tl.float64)), 1.e-300))
    return tl.where(z < -26., negative_tail, regular)


@triton.jit
def comparison_screen(C, Beta, Scale, Dur, Rho, DurationLogMass, Ingress, ShapePrior, Parameters, BaseNorm, Scores, N,
                      NB: tl.constexpr, NF: tl.constexpr, RANK: tl.constexpr,
                      ND: tl.constexpr, NS: tl.constexpr, LAYERS: tl.constexpr,
                      NT: tl.constexpr, B: tl.constexpr):
    """Exact amplitude gate on the identical finite bank before full evidence integration."""
    tile, f = tl.program_id(0), tl.program_id(1)
    j = tile * B + tl.arange(0, B)
    jj = tl.minimum(j, NB // 2 - 1)
    row = f.to(tl.int64) * NF * (NB + 1)
    scale, c = tl.load(Scale + f), tl.load(BaseNorm)
    best = tl.full([B], -float("inf"), tl.float64)
    best_a = tl.full([B], 0., tl.float64)
    bound_merit = tl.full([B], -float("inf"), tl.float64)
    best_q = tl.full([B], 1., tl.float64)
    best_null = tl.full([B], 0., tl.float64)
    best_p_shape = tl.full([B], 0., tl.float64)
    best_b_shape = tl.full([B], 0., tl.float64)
    best_key = tl.full([B], 0, tl.int64)
    for di in tl.range(ND, loop_unroll_factor=1):
        raw_k = tl.maximum(1, tl.floor(tl.load(Dur + di) * scale + .5).to(tl.int32))
        k = tl.minimum(raw_k, NB // 8 - 1)
        rho = tl.load(Rho + di)
        count = circular_sum(C, row + 3 * (NB + 1), jj, k, NB) \
                + circular_sum(C, row + 3 * (NB + 1), jj + NB // 2, k, NB)
        usable = (j < NB // 2) & (raw_k < NB // 8) & (count >= 3) & (count <= N - 3)
        for si in tl.range(NS, loop_unroll_factor=1):
            ingress = tl.load(Ingress + si)
            a = layered_sum(C, row, jj, k, ingress, NB, LAYERS, False) \
                + layered_sum(C, row, jj + NB // 2, k, ingress, NB, LAYERS, False)
            pe = layered_sum(C, row + NB + 1, jj, k, ingress, NB, LAYERS, False)
            po = layered_sum(C, row + NB + 1, jj + NB // 2, k, ingress, NB, LAYERS, False)
            ne = layered_sum(C, row + 2 * (NB + 1), jj, k, ingress, NB, LAYERS, True)
            no = layered_sum(C, row + 2 * (NB + 1), jj + NB // 2, k, ingress, NB, LAYERS, True)
            qe, qo, qeo = ne - pe * pe / c, no - po * po / c, -pe * po / c
            for mode in range(RANK):
                mr = row + (4 + mode) * (NB + 1)
                ge = layered_sum(C, mr, jj, k, ingress, NB, LAYERS, False)
                go = layered_sum(C, mr, jj + NB // 2, k, ingress, NB, LAYERS, False)
                beta = tl.load(Beta + mode)
                qe -= beta * ge * ge
                qo -= beta * go * go
                qeo -= beta * ge * go
            q = qe + qo + 2. * qeo
            score = a / tl.sqrt(tl.maximum(q * (rho * rho), 1.e-300))
            score_valid = usable & (a > 0.) & (q > 0.)
            merit = .5 * score * score + tl.load(DurationLogMass + di) + tl.load(ShapePrior + si * 2)
            better = score_valid & (merit > bound_merit)
            best_a = tl.where(better, a / (rho * rho), best_a)
            best_q = tl.where(better, q / (rho * rho), best_q)
            best_null = tl.where(better, tl.load(DurationLogMass + di), best_null)
            best_p_shape = tl.where(better, tl.load(ShapePrior + si * 2), best_p_shape)
            best_b_shape = tl.where(better, tl.load(ShapePrior + si * 2 + 1), best_b_shape)
            best_key = tl.where(better, (si * ND + di).to(tl.int64) * (NB // 2) + jj, best_key)
            bound_merit = tl.where(better, merit, bound_merit)
            best = tl.maximum(best, tl.where(score_valid, score, -float("inf")))
    maximum = tl.max(best, 0)
    bound_maximum = tl.max(bound_merit, 0)
    lane = tl.min(tl.where(bound_merit == bound_maximum, tl.arange(0, B), 2147483647), 0)
    take = tl.arange(0, B) == lane
    aa = tl.sum(tl.where(take, best_a, 0.), 0)
    qq = tl.sum(tl.where(take, best_q, 0.), 0)
    null = tl.sum(tl.where(take, best_null, 0.), 0)
    lower_p = positive_amplitude_evidence(aa, qq, tl.load(Parameters)) \
              + tl.log(tl.full([], .75, tl.float64)) + null + tl.sum(tl.where(take, best_p_shape, 0.), 0)
    lower_b = log_add(positive_amplitude_evidence(aa, qq, tl.load(Parameters + 1)),
                      positive_amplitude_evidence(aa, qq, tl.load(Parameters + 2))) \
              + tl.log(tl.full([], .025, tl.float64)) + null + tl.sum(tl.where(take, best_b_shape, 0.), 0)
    out = Scores + (f * NT + tile) * 4
    tl.store(out, maximum)
    tl.store(out + 1, tl.where(bound_maximum > -float("inf"), lower_p, -float("inf")))
    tl.store(out + 2, tl.where(bound_maximum > -float("inf"), lower_b, -float("inf")))
    tl.store(out + 3, tl.sum(tl.where(take, best_key, 0), 0).to(tl.float64))


@triton.jit
def subspace_upper(a, b, q, r, cross, rho2):
    """A two-dimensional nonnegative cone bounds every positive-depth mixture."""
    product, det = q * r, q * r - cross * cross
    energy = (a * a * r + b * b * q - 2. * a * b * cross) / tl.maximum(det * rho2, 1.e-300)
    safe = (q > 0.) & (r > 0.) & (det > tl.full([], 1.e-8, tl.float64) * product)
    positive_a, positive_b = tl.maximum(a, 0.), tl.maximum(b, 0.)
    boundary = .5 * tl.maximum(positive_a * positive_a / tl.maximum(q * rho2, 1.e-300),
                               positive_b * positive_b / tl.maximum(r * rho2, 1.e-300))
    interior = (a * r - b * cross > 0.) & (b * q - a * cross > 0.)
    upper = tl.where(safe, tl.where(interior, .5 * tl.maximum(energy, 0.), boundary), float("inf"))
    only_a = (r == 0.) & (b == 0.) & (cross == 0.)
    only_b = (q == 0.) & (a == 0.) & (cross == 0.)
    upper = tl.where(only_a & (q > 0.), .5 * a * a / (q * rho2), upper)
    upper = tl.where(only_b & (r > 0.), .5 * b * b / (r * rho2), upper)
    upper = tl.where(only_a & only_b, 0., upper)
    upper = tl.where((a <= 0.) & (b <= 0.), 0., upper)
    # Expand the exact-arithmetic bound; near-singular cases bypass pruning.
    return upper * tl.full([], 1.000001, tl.float64) + tl.full([], 1.e-6, tl.float64)


@triton.jit
def log_sum(values):
    high = tl.max(values, 0)
    safe_high = tl.where(high > -float("inf"), high, 0.)
    return tl.where(high > -float("inf"), safe_high + tl.log(tl.sum(tl.exp(values - safe_high), 0)), high)


@triton.jit
def joint_tiles(C, Beta, Period, Scale, Dur, Rho, DurationLogMass, Ingress, ShapePrior, BaseNorm,
                Parameters, Floors, Parts, N, NB: tl.constexpr, NF: tl.constexpr,
                RANK: tl.constexpr, ND: tl.constexpr, NS: tl.constexpr,
                LAYERS: tl.constexpr, NT: tl.constexpr, B: tl.constexpr,
                THRESHOLD: tl.constexpr, PILOT: tl.constexpr = False):
    """Compare normalized eclipse hypotheses at every ephemeris before either maximization."""
    tile, f = tl.program_id(0), tl.program_id(1)
    pilot_key = tl.load(Floors + f * 3 + 2).to(tl.int64)
    j = tl.full([B], 0, tl.int32) + pilot_key % (NB // 2) if PILOT else tile * B + tl.arange(0, B)
    jj = tl.minimum(j, NB // 2 - 1)
    row = f.to(tl.int64) * NF * (NB + 1)
    c = tl.load(BaseNorm)
    scale = tl.load(Scale + f)
    tau_p, tau_b, tau_d = tl.load(Parameters), tl.load(Parameters + 1), tl.load(Parameters + 2)
    log_pi_p, log_pi_b, log_pi_0 = tl.load(Parameters + 3), tl.load(Parameters + 4), tl.load(Parameters + 5)
    best_joint = tl.full([B], -float("inf"), tl.float64)
    best_signal = tl.full([B], -float("inf"), tl.float64)
    joint_key = tl.full([B], 9223372036854775807, tl.int64)
    signal_key = tl.full([B], 9223372036854775807, tl.int64)
    j_snr = tl.full([B], 0., tl.float64)
    j_depth = tl.full([B], 0., tl.float64)
    j_lp = tl.full([B], 0., tl.float64)
    j_lb = tl.full([B], 0., tl.float64)
    j_sr = tl.full([B], 0., tl.float64)
    j_sp = tl.full([B], 0., tl.float64)
    j_odd = tl.full([B], 0., tl.float64)
    j_mult = tl.full([B], 0., tl.float64)
    s_joint = tl.full([B], 0., tl.float64)
    s_depth = tl.full([B], 0., tl.float64)
    s_lp = tl.full([B], 0., tl.float64)
    s_lb = tl.full([B], 0., tl.float64)
    s_sr = tl.full([B], 0., tl.float64)
    s_sp = tl.full([B], 0., tl.float64)
    s_odd = tl.full([B], 0., tl.float64)
    s_mult = tl.full([B], 0., tl.float64)
    sum_p = tl.full([B], -float("inf"), tl.float64)
    sum_b = tl.full([B], -float("inf"), tl.float64)
    omitted = tl.full([], 0, tl.int32)
    floor_p = tl.full([], -float("inf"), tl.float64) if PILOT else tl.load(Floors + f * 3)
    floor_b = tl.full([], -float("inf"), tl.float64) if PILOT else tl.load(Floors + f * 3 + 1)
    # Each lane carries only its two winners, never a trial-by-observation array.
    for di_loop in tl.range(1 if PILOT else ND, loop_unroll_factor=1):
        di = pilot_key // (NB // 2) % ND if PILOT else di_loop
        raw_k = tl.maximum(1, tl.floor(tl.load(Dur + di) * scale + .5).to(tl.int32))
        k = tl.minimum(raw_k, NB // 8 - 1)
        rho2 = tl.load(Rho + di) * tl.load(Rho + di)
        count = circular_sum(C, row + 3 * (NB + 1), jj, k, NB) \
                + circular_sum(C, row + 3 * (NB + 1), jj + NB // 2, k, NB)
        usable = (j < NB // 2) & (raw_k < NB // 8) & (count >= 3) & (count <= N - 3)
        for si_loop in tl.range(1 if PILOT else NS, loop_unroll_factor=1):
            si = pilot_key // ((NB // 2) * ND) if PILOT else si_loop
            ingress = tl.load(Ingress + si)
            ae = layered_sum(C, row, jj, k, ingress, NB, LAYERS, False)
            ao = layered_sum(C, row, jj + NB // 2, k, ingress, NB, LAYERS, False)
            pe = layered_sum(C, row + NB + 1, jj, k, ingress, NB, LAYERS, False)
            po = layered_sum(C, row + NB + 1, jj + NB // 2, k, ingress, NB, LAYERS, False)
            ne = layered_sum(C, row + 2 * (NB + 1), jj, k, ingress, NB, LAYERS, True)
            no = layered_sum(C, row + 2 * (NB + 1), jj + NB // 2, k, ingress, NB, LAYERS, True)
            qe, qo, qeo = ne - pe * pe / c, no - po * po / c, -pe * po / c
            for mode in range(RANK):
                mr = row + (4 + mode) * (NB + 1)
                ge = layered_sum(C, mr, jj, k, ingress, NB, LAYERS, False)
                go = layered_sum(C, mr, jj + NB // 2, k, ingress, NB, LAYERS, False)
                beta = tl.load(Beta + mode)
                qe -= beta * ge * ge
                qo -= beta * go * go
                qeo -= beta * ge * go
            ap, qp = ae + ao, qe + qo + 2. * qeo
            snr = ap / tl.sqrt(tl.maximum(qp * rho2, 1.e-300))
            depth = ap / tl.maximum(qp, 1.e-300)
            valid = usable & (ap > 0.) & (qp > 0.) & (snr >= tl.full([], THRESHOLD, tl.float64))
            if raw_k < NB // 8:
                lp = tl.full([B], -float("inf"), tl.float64)
                lb = tl.full([B], -float("inf"), tl.float64)
                component_best = tl.full([B], -float("inf"), tl.float64)
                sec_ratio = tl.full([B], 0., tl.float64)
                sec_phase = tl.full([B], .5, tl.float64)
                odd_ratio = tl.full([B], 1., tl.float64)
                binary_multiple = tl.full([B], 1., tl.float64)
                omitted_upper = tl.full([B], -float("inf"), tl.float64)
                for phase_index in tl.range(3, loop_unroll_factor=1):
                    offset = (phase_index + 1) * (NB // 8)
                    phase_weight = tl.where(phase_index == 1, tl.full([], .8, tl.float64), tl.full([], .1, tl.float64))
                    s0, s1 = jj + offset, jj + offset + NB // 2
                    ass = layered_sum(C, row, s0, k, ingress, NB, LAYERS, False) \
                          + layered_sum(C, row, s1, k, ingress, NB, LAYERS, False)
                    ps = layered_sum(C, row + NB + 1, s0, k, ingress, NB, LAYERS, False) \
                         + layered_sum(C, row + NB + 1, s1, k, ingress, NB, LAYERS, False)
                    ns = layered_sum(C, row + 2 * (NB + 1), s0, k, ingress, NB, LAYERS, True) \
                         + layered_sum(C, row + 2 * (NB + 1), s1, k, ingress, NB, LAYERS, True)
                    qs, qps = ns - ps * ps / c, -(pe + po) * ps / c
                    for mode in range(RANK):
                        mr = row + (4 + mode) * (NB + 1)
                        ge = layered_sum(C, mr, jj, k, ingress, NB, LAYERS, False)
                        go = layered_sum(C, mr, jj + NB // 2, k, ingress, NB, LAYERS, False)
                        gs = layered_sum(C, mr, s0, k, ingress, NB, LAYERS, False) \
                             + layered_sum(C, mr, s1, k, ingress, NB, LAYERS, False)
                        beta = tl.load(Beta + mode)
                        qs -= beta * gs * gs
                        qps -= beta * (ge + go) * gs
                    raw_upper = subspace_upper(ap, ass, qp, qs, qps, rho2)
                    upper = raw_upper + tl.load(DurationLogMass + di)
                    if tl.max(tl.where(j < NB // 2, upper, -float("inf")), 0) >= tl.minimum(floor_p, floor_b):
                        for ri in tl.range(5, loop_unroll_factor=1):
                            ratio = tl.load(Parameters + 6 + ri * 3)
                            wp = tl.load(Parameters + 7 + ri * 3)
                            wb = tl.load(Parameters + 8 + ri * 3)
                            aa = (ap + ratio * ass) / rho2
                            qq = (qp + 2. * ratio * qps + ratio * ratio * qs) / rho2
                            ep = positive_amplitude_evidence(aa, qq, tau_p) + tl.log(phase_weight * wp)
                            eb = log_add(positive_amplitude_evidence(aa, qq, tau_b),
                                         positive_amplitude_evidence(aa, qq, tau_d)) \
                                 + tl.log(.25 * phase_weight * wb)
                            lp, lb = log_add(lp, ep), log_add(lb, eb)
                            better_component = eb > component_best
                            sec_ratio = tl.where(better_component, ratio, sec_ratio)
                            sec_phase = tl.where(better_component, .25 * (phase_index + 1), sec_phase)
                            component_best = tl.maximum(component_best, eb)
                    else:
                        omitted += 1
                        omitted_upper = tl.maximum(omitted_upper, raw_upper)
                raw_upper = subspace_upper(ae, ao, qe, qo, qeo, rho2)
                upper = raw_upper + tl.load(DurationLogMass + di)
                if tl.max(tl.where(j < NB // 2, upper, -float("inf")), 0) >= floor_b:
                    for ri in tl.range(7, loop_unroll_factor=1):
                        ratio = tl.load(Parameters + 21 + ri)
                        aa = (ae + ratio * ao) / rho2
                        qq = (qe + 2. * ratio * qeo + ratio * ratio * qo) / rho2
                        eb = log_add(positive_amplitude_evidence(aa, qq, tau_b),
                                     positive_amplitude_evidence(aa, qq, tau_d)) + tl.log(.25 / tl.full([], 7., tl.float64))
                        lb = log_add(lb, eb)
                        better_component = eb > component_best
                        sec_ratio = tl.where(better_component, 0., sec_ratio)
                        sec_phase = tl.where(better_component, .5, sec_phase)
                        odd_ratio = tl.where(better_component, ratio, odd_ratio)
                        binary_multiple = tl.where(better_component, 2., binary_multiple)
                        component_best = tl.maximum(component_best, eb)
                else:
                    omitted += 1
                    omitted_upper = tl.maximum(omitted_upper, raw_upper)
                certain = component_best > omitted_upper
                sec_ratio = tl.where(certain, sec_ratio, float("nan"))
                sec_phase = tl.where(certain, sec_phase, float("nan"))
                odd_ratio = tl.where(certain, odd_ratio, float("nan"))
                binary_multiple = tl.where(certain, binary_multiple, float("nan"))
                lp += tl.load(ShapePrior + si * 2) + tl.load(DurationLogMass + di)
                lb += tl.load(ShapePrior + si * 2 + 1) + tl.load(DurationLogMass + di)
                sum_p = log_add(sum_p, tl.where(j < NB // 2, lp, -float("inf")))
                sum_b = log_add(sum_b, tl.where(j < NB // 2, lb, -float("inf")))
                joint = lp
                key = (si * ND + di).to(tl.int64) * (NB // 2) + jj
                bj = valid & ((joint > best_joint) | ((joint == best_joint) & (key < joint_key)))
                bs = valid & ((snr > best_signal) | ((snr == best_signal) & (key < signal_key)))
                # Interleave the two reductions; only tile winners reach global memory.
                for selection in tl.static_range(2):
                    better = bj if selection == 0 else bs
                    # Store the lane winners in scratch only after the bank scan below.
                    if selection == 0:
                        j_snr, j_depth = tl.where(better, snr, j_snr), tl.where(better, depth, j_depth)
                        j_lp, j_lb = tl.where(better, lp, j_lp), tl.where(better, lb, j_lb)
                        j_sr, j_sp = tl.where(better, sec_ratio, j_sr), tl.where(better, sec_phase, j_sp)
                        j_odd, j_mult = tl.where(better, odd_ratio, j_odd), tl.where(better, binary_multiple, j_mult)
                    else:
                        s_joint, s_depth = tl.where(better, joint, s_joint), tl.where(better, depth, s_depth)
                        s_lp, s_lb = tl.where(better, lp, s_lp), tl.where(better, lb, s_lb)
                        s_sr, s_sp = tl.where(better, sec_ratio, s_sr), tl.where(better, sec_phase, s_sp)
                        s_odd, s_mult = tl.where(better, odd_ratio, s_odd), tl.where(better, binary_multiple, s_mult)
                best_joint, joint_key = tl.where(bj, joint, best_joint), tl.where(bj, key, joint_key)
                best_signal, signal_key = tl.where(bs, snr, best_signal), tl.where(bs, key, signal_key)
    for selection in tl.static_range(2):
        merit = best_joint if selection == 0 else best_signal
        keys = joint_key if selection == 0 else signal_key
        maximum = tl.max(merit, 0)
        key = tl.min(tl.where(merit == maximum, keys, 9223372036854775807), 0)
        lane = tl.min(tl.where((merit == maximum) & (keys == key), tl.arange(0, B), 2147483647), 0)
        take = tl.arange(0, B) == lane
        out = Parts + ((f.to(tl.int64) * NT + tile) * 2 + selection) * 15
        tl.store(out, key.to(tl.float64))
        tl.store(out + 1, maximum)
        tl.store(out + 2, tl.sum(tl.where(take, j_snr if selection == 0 else best_signal, 0.), 0))
        tl.store(out + 3, tl.sum(tl.where(take, j_depth if selection == 0 else s_depth, 0.), 0))
        tl.store(out + 4, tl.sum(tl.where(take, j_lp if selection == 0 else s_lp, 0.), 0))
        tl.store(out + 5, tl.sum(tl.where(take, j_lb if selection == 0 else s_lb, 0.), 0))
        tl.store(out + 6, tl.sum(tl.where(take, j_sr if selection == 0 else s_sr, 0.), 0))
        tl.store(out + 7, tl.sum(tl.where(take, j_sp if selection == 0 else s_sp, 0.), 0))
        tl.store(out + 8, tl.sum(tl.where(take, j_odd if selection == 0 else s_odd, 0.), 0))
        tl.store(out + 9, tl.sum(tl.where(take, j_mult if selection == 0 else s_mult, 0.), 0))
        tl.store(out + 10, tl.sum(tl.where(take, best_joint if selection == 0 else s_joint, 0.), 0))
        tl.store(out + 11, log_sum(sum_p))
        tl.store(out + 12, log_sum(sum_b))
        tl.store(out + 13, omitted.to(tl.float64))


@triton.jit
def joint_merge(Parts, Period, Scale, Dur, Ingress, Parameters, LogMass, Table, NT: tl.constexpr,
                ND: tl.constexpr, NB: tl.constexpr, B: tl.constexpr):
    f, selection = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, B)
    base = ((f.to(tl.int64) * NT + i) * 2 + selection) * 15
    values = tl.load(Parts + base + 1, i < NT, other=-float("inf"))
    keys = tl.load(Parts + base, i < NT, other=float("inf"))
    maximum = tl.max(values, 0)
    key = tl.min(tl.where(values == maximum, keys, float("inf")), 0)
    tile = tl.min(tl.where((i < NT) & (values == maximum) & (keys == key), i, 2147483647), 0)
    part = Parts + ((f.to(tl.int64) * NT + tile) * 2 + selection) * 15
    lp = log_sum(tl.load(Parts + base + 11, i < NT, other=-float("inf"))) - tl.load(LogMass + f)
    lb = log_sum(tl.load(Parts + base + 12, i < NT, other=-float("inf"))) - tl.load(LogMass + f)
    joint = tl.load(Parameters + 3) + lp - log_add(tl.load(Parameters + 5), tl.load(Parameters + 4) + lb)
    # Empty periods have a finite harmless index and are removed by the host mask.
    decoded = tl.where(maximum > -float("inf"), key, 0.).to(tl.int64)
    j = decoded % (NB // 2)
    di = decoded // (NB // 2) % ND
    si = decoded // ((NB // 2) * ND)
    scale, period = tl.load(Scale + f), tl.load(Period + f)
    width = tl.maximum(1., tl.floor(tl.load(Dur + di) * scale + .5))
    out = Table + (f * 2 + selection) * 14
    tl.store(out, period)
    tl.store(out + 1, (j + width / 2.) / scale)
    tl.store(out + 2, width / scale)
    tl.store(out + 3, tl.where(maximum > -float("inf"), tl.load(part + 2), -float("inf")))
    tl.store(out + 4, tl.load(part + 3))
    tl.store(out + 5, joint)
    for field in tl.static_range(6):
        tl.store(out + 6 + field, tl.load(part + 4 + field))
    tl.store(out + 6, lp)
    tl.store(out + 7, lb)
    tl.store(out + 11, tl.load(part + 9) * period)
    tl.store(out + 12, tl.load(Ingress + si))
    tl.store(out + 13, 1. if selection == 0 else 2.)


@triton.jit
def evidence_peaks(Table, Mask, N, RADIUS: tl.constexpr, WINDOW: tl.constexpr, B: tl.constexpr):
    """Retain evidence maxima among ephemerides unresolved within the transit drift window."""
    i = tl.program_id(0)
    p, epoch = tl.load(Table + i * 14), tl.load(Table + i * 14 + 1)
    duration, evidence = tl.load(Table + i * 14 + 2), tl.load(Table + i * 14 + 6)
    superior = tl.full([B], False, tl.int1)
    for chunk in range(triton.cdiv(2 * WINDOW + 1, B)):
        offset = chunk * B + tl.arange(0, B)
        j = i + offset - WINDOW
        valid = (offset < 2 * WINDOW + 1) & (j >= 0) & (j < N)
        np = tl.load(Table + j * 14, valid, other=1.)
        ne = tl.load(Table + j * 14 + 1, valid, other=0.)
        nd = tl.load(Table + j * 14 + 2, valid, other=0.)
        score = tl.load(Table + j * 14 + 6, valid, other=-float("inf"))
        phase = ne - epoch + p / 2.
        phase -= tl.floor(phase / p) * p
        phase -= p / 2.
        compatible = (tl.abs(np / p - 1.) <= tl.full([], RADIUS, tl.float64)) & (tl.abs(phase) <= duration + nd)
        superior |= valid & compatible & ((score > evidence) | ((score == evidence) & (j < i)))
    tl.store(Mask + i, tl.sum(superior.to(tl.int32), 0) == 0)
