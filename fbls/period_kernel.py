"""Single-pass diagonal GLS period maxima on the existing finite shape bank."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .kernels import circular_sum, interval_sum


@triton.jit
def shape_sums(row, j, k, outer_a, outer_p, outer_w, ingress, Factors, si,
               LAYERS: tl.constexpr):
    a = tl.where(ingress == 0., outer_a, outer_a / LAYERS)
    hp = tl.where(ingress == 0., outer_p, outer_p / LAYERS)
    coefficient = tl.full([], 1. / (LAYERS * LAYERS), tl.float64)
    norm = tl.where(ingress == 0., outer_w, outer_w * coefficient)
    if ingress != 0.:
        for layer in tl.static_range(1, LAYERS):
            factor = tl.load(Factors + si * LAYERS + layer)
            width = tl.maximum(1, tl.floor(k.to(tl.float64) * factor + .5).to(tl.int32))
            start = j + (k - width) // 2
            a += interval_sum(row, 0, start, width) / LAYERS
            hp += interval_sum(row, 1, start, width) / LAYERS
            coefficient = tl.full([], (2. * layer + 1.) / (LAYERS * LAYERS), tl.float64)
            norm += interval_sum(row, 2, start, width) * coefficient
    return a, hp, norm


@triton.jit
def rank_template(a, hp, norm, rho, baseline, valid, key, di, best, best_key, best_duration,
                  bad, sensitive, unsafe, tiny, huge, guard, FAST: tl.constexpr):
    """Update the running per-lane maximum with one template; return the template's merit."""
    if FAST:
        # Rank by a^2/[rho^2(c*n-hp^2)] in float32; the exact float64 value is
        # recomputed for the winner, and tiles with near ties or abnormal
        # ranges are rescanned exactly.
        cn = (baseline * norm).to(tl.float32)
        hp32 = hp.to(tl.float32)
        qq = (rho * rho).to(tl.float32) * (cn - hp32 * hp32)
        a32 = a.to(tl.float32)
        merit = a32 * a32 / qq
        safe = ((cn > 0.) & (qq >= .001 * cn) & (merit >= tiny) & (merit <= huge))
        unsafe |= valid & ~safe
        near = (best > 0.) & (tl.abs(merit - best) <= guard * tl.maximum(merit, best))
        far_better = merit > best * (1. + 4. * guard)
        sensitive = tl.where(valid & safe & far_better, False, sensitive | (valid & safe & near))
        better = valid & safe & ((merit > best) | ((merit == best) & (key < best_key)))
        best = tl.where(better, merit, best)
        merit = tl.where(valid & safe, merit, -float("inf"))
    else:
        d = hp * hp / baseline
        q = norm - d
        bad |= valid & (q <= 0.)
        candidate = (a / rho) / tl.sqrt(tl.maximum(q, 1.e-300))
        merit = tl.where(valid & (q > 0.) & (candidate < float("inf")), candidate, -float("inf"))
        better = (merit > best) | ((merit == best) & (key < best_key))
        best = tl.where(better, merit, best)
    best_key = tl.where(better, key, best_key)
    best_duration = tl.where(better, di, best_duration)
    return best, best_key, best_duration, bad, sensitive, unsafe, merit


@triton.jit
def tile_choice(C, Scale, Dur, Rho, Ingress, Factors, BaseNorm, Gate, f, tile,
                N, NB, ND: tl.constexpr, NS: tl.constexpr,
                LAYERS: tl.constexpr, B: tl.constexpr, FAST: tl.constexpr):
    """Best template of one phase tile over durations and shapes.

    Shape index 0 must be the box (ingress 0). The remaining shapes are only
    evaluated when at least one lane's box statistic reaches `Gate`: a box
    recovers at least ~87% of the amplitude of any bank shape, so a gate well
    below the reporting threshold never removes a reportable template.
    """
    j = tile * B + tl.arange(0, B)
    jj = tl.minimum(j, NB - 1)
    scale = tl.load(Scale + f)
    baseline = tl.load(BaseNorm)
    row = C + f.to(tl.int64) * 4 * (2 * NB + 1)
    best = tl.full([B], -float("inf"), tl.float64)
    best_key = tl.full([B], 9223372036854775807, tl.int64)
    best_duration = tl.zeros([B], tl.int32)
    bad = tl.full([B], False, tl.int1)
    sensitive = tl.full([B], False, tl.int1)
    unsafe = tl.full([B], False, tl.int1)
    if FAST:
        best = tl.full([B], -float("inf"), tl.float32)
        tiny = tl.full([], 1.e-30, tl.float32)
        huge = tl.full([], 1.e30, tl.float32)
        guard = tl.full([], 1.e-5, tl.float32)
        # The float32 merit is score^2 / baseline_norm.
        gate = (Gate * Gate / baseline).to(tl.float32)
    else:
        tiny = tl.full([], 1.e-280, tl.float64)
        huge = tl.full([], 1.e280, tl.float64)
        guard = tl.full([], 1.e-10, tl.float64)
        gate = Gate + tl.zeros([], tl.float64)
    for di in range(ND):
        raw_k = tl.maximum(1, tl.floor(tl.load(Dur + di) * scale + .5).to(tl.int32))
        k = tl.minimum(raw_k, NB // 2)
        rho = tl.load(Rho + di)
        count = interval_sum(row, 3, jj, k)
        outer_a = interval_sum(row, 0, jj, k)
        outer_p = interval_sum(row, 1, jj, k)
        outer_w = interval_sum(row, 2, jj, k)
        usable = (j < NB) & (raw_k < NB // 2) & (count >= 3) & (count <= N - 3)
        valid = usable & (outer_a > 0.)
        key = k.to(tl.int64) * (NB + 1) + jj
        best, best_key, best_duration, bad, sensitive, unsafe, merit = rank_template(
            outer_a, outer_p, outer_w, rho, baseline, valid, key, di, best, best_key, best_duration,
            bad, sensitive, unsafe, tiny, huge, guard, FAST)
        passing = tl.sum((valid & (merit >= gate)).to(tl.int32), 0)
        if passing > 0:
            for si in tl.static_range(1, NS):
                ingress = tl.load(Ingress + si)
                a, hp, norm = shape_sums(row, jj, k, outer_a, outer_p, outer_w,
                                         ingress, Factors, si, LAYERS)
                valid = usable & (a > 0.)
                key = (si * (NB + 1) + k.to(tl.int64)) * (NB + 1) + jj
                best, best_key, best_duration, bad, sensitive, unsafe, merit = rank_template(
                    a, hp, norm, rho, baseline, valid, key, di, best, best_key, best_duration,
                    bad, sensitive, unsafe, tiny, huge, guard, FAST)
    if FAST:
        # Empty tiles still need a harmless decodable key; they are not returned.
        best_key = tl.where(best > 0., best_key, (NB + 1) + jj)
    value = tl.max(best, 0).to(tl.float64)
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
def diagonal_tiles(C, Period, Scale, Dur, Rho, Ingress, Factors, BaseNorm, Gate, Parts, Keys, Invalid, Retries,
                   N, NB, ND: tl.constexpr,
                   NS: tl.constexpr, LAYERS: tl.constexpr, NT: tl.constexpr,
                   B: tl.constexpr):
    tile = tl.program_id(0)
    f = tl.program_id(1)
    key, di, value, retry, bad = tile_choice(C, Scale, Dur, Rho, Ingress, Factors, BaseNorm, Gate,
                                            f, tile, N, NB, ND, NS, LAYERS, B, True)
    if retry:
        key, di, value, _, bad = tile_choice(C, Scale, Dur, Rho, Ingress, Factors, BaseNorm, Gate,
                                            f, tile, N, NB, ND, NS, LAYERS, B, False)
    # Reconstruct only the winner instead of carrying its fields per phase.
    phase = key % (NB + 1)
    width = key // (NB + 1) % (NB + 1)
    si = key // ((NB + 1) * (NB + 1))
    ingress = tl.load(Ingress + si)
    rho = tl.load(Rho + di)
    row = C + f.to(tl.int64) * 4 * (2 * NB + 1)
    baseline = tl.load(BaseNorm)
    phase32 = phase.to(tl.int32)
    width32 = width.to(tl.int32)
    outer_a = interval_sum(row, 0, phase32, width32)
    outer_p = interval_sum(row, 1, phase32, width32)
    outer_w = interval_sum(row, 2, phase32, width32)
    a, hp, norm = shape_sums(row, phase32, width32, outer_a, outer_p, outer_w,
                             ingress, Factors, si, LAYERS)
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
def fold_comparison(T, X, Scale, Out, N, NB,
                    NF: tl.constexpr, B: tl.constexpr):
    """Retain cycle parity by accumulating sufficient statistics over two periods."""
    f = tl.program_id(0)
    field = tl.program_id(1)
    i = tl.program_id(2) * B + tl.arange(0, B)
    cell = tl.floor(tl.load(T + i, i < N, other=0.) * tl.load(Scale + f)).to(tl.int64) % NB
    value = tl.load(X + field * N + i, i < N, other=0.)
    tl.atomic_add(Out + (f.to(tl.int64) * NF + field) * NB + cell, value, i < N, sem="relaxed")


@triton.jit
def fold_period_fields(T, X, Scale, Out, N, NB,
                       NF: tl.constexpr, FIRST: tl.constexpr, B: tl.constexpr):
    f, field = tl.program_id(0),tl.program_id(1)
    i=tl.program_id(2)*B+tl.arange(0,B)
    cell=tl.floor(tl.load(T+i,i<N,other=0.)*tl.load(Scale+f)).to(tl.int64)%NB
    value=tl.load(X+(f*4+field)*N+i,i<N,other=0.)
    tl.atomic_add(Out+(f.to(tl.int64)*NF+FIRST+field)*NB+cell,value,i<N,sem="relaxed")


@triton.jit
def layered_sum(C, row, start, width, Shape, si, NB, LAYERS: tl.constexpr,
                SQUARED: tl.constexpr):
    """Integrate a staircase or its square using nested interval sums."""
    start=start+tl.zeros_like(row)
    value = tl.full(start.shape, 0., tl.float64)
    for layer in tl.range(LAYERS,loop_unroll_factor=1):
        fraction = tl.load(Shape + (si * LAYERS + layer) * 3)
        inner = tl.maximum(1, tl.floor(width.to(tl.float64) * fraction + .5).to(tl.int32))
        coefficient = tl.load(Shape + (si * LAYERS + layer) * 3 + (2 if SQUARED else 1))
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
def comparison_screen(C, Beta, Scale, Dur, Rho, DurationLogMass, Ingress, Shape, ShapePrior, Parameters, BaseNorm, Scores, N,
                      NB, NF: tl.constexpr, RANK: tl.constexpr,
                      ND: tl.constexpr, NS: tl.constexpr, LAYERS: tl.constexpr,
                      NT: tl.constexpr, B: tl.constexpr):
    """Exact amplitude gate on the identical finite bank before full evidence integration."""
    tile, f = tl.program_id(0), tl.program_id(1)
    j = tile * B + tl.arange(0, B)
    jj = tl.minimum(j, NB // 2 - 1)
    row = f.to(tl.int64) * NF * (NB + 1)
    Parameters += f * 30
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
            a = layered_sum(C, row, jj, k, Shape, si, NB, LAYERS, False) \
                + layered_sum(C, row, jj + NB // 2, k, Shape, si, NB, LAYERS, False)
            pe = layered_sum(C, row + NB + 1, jj, k, Shape, si, NB, LAYERS, False)
            po = layered_sum(C, row + NB + 1, jj + NB // 2, k, Shape, si, NB, LAYERS, False)
            ne = layered_sum(C, row + 2 * (NB + 1), jj, k, Shape, si, NB, LAYERS, True)
            no = layered_sum(C, row + 2 * (NB + 1), jj + NB // 2, k, Shape, si, NB, LAYERS, True)
            qe, qo, qeo = ne - pe * pe / c, no - po * po / c, -pe * po / c
            for mode in range(RANK):
                mr = row + (4 + mode) * (NB + 1)
                ge = layered_sum(C, mr, jj, k, Shape, si, NB, LAYERS, False)
                go = layered_sum(C, mr, jj + NB // 2, k, Shape, si, NB, LAYERS, False)
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
def modulated_pair(a,b,q,r,cross,g0,g1,h0,h1,Modulation,f,di,family,rho2,ND:tl.constexpr):
    ptr=Modulation+((f*ND+di)*4+family)*6
    m0,m1,m2=tl.load(ptr),tl.load(ptr+1),tl.load(ptr+2)
    d0,d1,base=tl.load(ptr+3),tl.load(ptr+4),tl.load(ptr+5)
    aq=a/rho2-g0*d0-g1*d1
    bq=b/rho2-h0*d0-h1*d1
    qq=q/rho2-m0*g0*g0-2*m1*g0*g1-m2*g1*g1
    rr=r/rho2-m0*h0*h0-2*m1*h0*h1-m2*h1*h1
    xx=cross/rho2-m0*g0*h0-m1*(g0*h1+g1*h0)-m2*g1*h1
    return aq,bq,qq,rr,xx,base


@triton.jit
def harmonic_pair(C,row,j,s,k,ks,Shape,si,NB,LAYERS:tl.constexpr,
                  FIRST:tl.constexpr,PAIRED:tl.constexpr,ENABLED:tl.constexpr):
    g0=tl.full(j.shape,0.,tl.float64);g1=tl.full(j.shape,0.,tl.float64)
    h0=tl.full(j.shape,0.,tl.float64);h1=tl.full(j.shape,0.,tl.float64)
    if ENABLED:
        g0=layered_sum(C,row+FIRST*(NB+1),j,k,Shape,si,NB,LAYERS,False)
        g1=layered_sum(C,row+(FIRST+1)*(NB+1),j,k,Shape,si,NB,LAYERS,False)
        h0=layered_sum(C,row+FIRST*(NB+1),s,ks,Shape,si,NB,LAYERS,False)
        h1=layered_sum(C,row+(FIRST+1)*(NB+1),s,ks,Shape,si,NB,LAYERS,False)
        if PAIRED:
            g0+=layered_sum(C,row+FIRST*(NB+1),j+NB//2,k,Shape,si,NB,LAYERS,False)
            g1+=layered_sum(C,row+(FIRST+1)*(NB+1),j+NB//2,k,Shape,si,NB,LAYERS,False)
            h0+=layered_sum(C,row+FIRST*(NB+1),s+NB//2,ks,Shape,si,NB,LAYERS,False)
            h1+=layered_sum(C,row+(FIRST+1)*(NB+1),s+NB//2,ks,Shape,si,NB,LAYERS,False)
    return g0,g1,h0,h1


@triton.jit
def astrometric_pair(C,row,j,s,k,ks,Shape,si,Norm,NB,LAYERS:tl.constexpr,
                     FIRST:tl.constexpr,NG:tl.constexpr,PAIRED:tl.constexpr):
    axis=tl.arange(0,2)[:,None]
    jj=j[None,:]
    ss=s[None,:]
    a=tl.full((2,j.shape[0]),0.,tl.float64);b=tl.full((2,j.shape[0]),0.,tl.float64)
    q=tl.full((2,j.shape[0]),0.,tl.float64);r=tl.full((2,j.shape[0]),0.,tl.float64)
    cross=tl.full((2,j.shape[0]),0.,tl.float64)
    if NG>0:
        base=row+(FIRST+axis*(2+NG))*(NB+1)
        a=layered_sum(C,base,jj,k,Shape,si,NB,LAYERS,False)
        b=layered_sum(C,base,ss,ks,Shape,si,NB,LAYERS,False)
        q=layered_sum(C,base+NB+1,jj,k,Shape,si,NB,LAYERS,True)
        r=layered_sum(C,base+NB+1,ss,ks,Shape,si,NB,LAYERS,True)
        if PAIRED:
            a+=layered_sum(C,base,jj+NB//2,k,Shape,si,NB,LAYERS,False)
            b+=layered_sum(C,base,ss+NB//2,ks,Shape,si,NB,LAYERS,False)
            q+=layered_sum(C,base+NB+1,jj+NB//2,k,Shape,si,NB,LAYERS,True)
            r+=layered_sum(C,base+NB+1,ss+NB//2,ks,Shape,si,NB,LAYERS,True)
        for group in range(NG):
            c=tl.load(Norm+axis*NG+group)
            ga=layered_sum(C,base+(2+group)*(NB+1),jj,k,Shape,si,NB,LAYERS,False)
            gb=layered_sum(C,base+(2+group)*(NB+1),ss,ks,Shape,si,NB,LAYERS,False)
            if PAIRED:
                ga+=layered_sum(C,base+(2+group)*(NB+1),jj+NB//2,k,Shape,si,NB,LAYERS,False)
                gb+=layered_sum(C,base+(2+group)*(NB+1),ss+NB//2,ks,Shape,si,NB,LAYERS,False)
            q-=ga*ga/tl.maximum(c,1.e-300)
            r-=gb*gb/tl.maximum(c,1.e-300)
            cross-=ga*gb/tl.maximum(c,1.e-300)
    return a,b,q,r,cross


@triton.jit
def centroid_evidence(a,b,q,r,cross,ratio,sigma):
    aa=a+ratio*b
    qq=tl.maximum(q+2*ratio*cross+ratio*ratio*r,0.)
    v=1+sigma*sigma*qq
    return tl.sum(.5*sigma*sigma*aa*aa/v-.5*tl.log(v),0)


@triton.jit
def centroid_upper(a,b,q,r,cross):
    det=q*r-cross*cross
    safe=(q>0)&(r>0)&(det>tl.full([],1.e-8,tl.float64)*q*r)
    value=tl.where(safe,.5*(a*a*r+b*b*q-2*a*b*cross)/tl.maximum(det,1.e-300),float("inf"))
    value=tl.where((r==0)&(b==0)&(cross==0)&(q>0),.5*a*a/q,value)
    value=tl.where((q==0)&(a==0)&(cross==0)&(r>0),.5*b*b/r,value)
    value=tl.where((q==0)&(r==0)&(a==0)&(b==0),0.,value)
    return tl.sum(tl.maximum(value,0.),0)*tl.full([],1.000001,tl.float64)+tl.full([],1e-6,tl.float64)


@triton.jit
def log_sum(values):
    high = tl.max(values, 0)
    safe_high = tl.where(high > -float("inf"), high, 0.)
    return tl.where(high > -float("inf"), safe_high + tl.log(tl.sum(tl.exp(values - safe_high), 0)), high)


@triton.jit
def joint_tiles(C, Beta, Period, Scale, Dur, Rho, DurationLogMass, Ingress, Shape, ShapePrior, BaseNorm,
                Parameters, Orbit, Alias, Modulation, AstroNorm, Floors, Parts, N, NB, NF: tl.constexpr,
                RANK: tl.constexpr, ND: tl.constexpr, NS: tl.constexpr,
                LAYERS: tl.constexpr, NT: tl.constexpr, B: tl.constexpr,
                THRESHOLD: tl.constexpr, NORB: tl.constexpr, ASTART: tl.constexpr, NG: tl.constexpr,
                HSTART: tl.constexpr, HARMONIC: tl.constexpr, PILOT: tl.constexpr = False, PARTITIONED: tl.constexpr = False):
    """Compare normalized eclipse hypotheses at every ephemeris before either maximization."""
    part_index, f = tl.program_id(0), tl.program_id(1)
    phase_tiles=tl.cdiv(NB//2,B)
    tile=part_index%phase_tiles if PARTITIONED else part_index
    pilot_key = tl.load(Floors + f * 3 + 2).to(tl.int64)
    j = tl.full([B], 0, tl.int32) + pilot_key % (NB // 2) if PILOT else tile * B + tl.arange(0, B)
    jj = tl.minimum(j, NB // 2 - 1)
    row = f.to(tl.int64) * NF * (NB + 1)
    c = tl.load(BaseNorm)
    scale = tl.load(Scale + f)
    Parameters += f * 30
    tau_p, tau_b, tau_d = tl.load(Parameters), tl.load(Parameters + 1), tl.load(Parameters + 2)
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
    for di_loop in tl.range(1 if PILOT or PARTITIONED else ND, loop_unroll_factor=1):
        di = pilot_key // (NB // 2) % ND if PILOT else (part_index//phase_tiles)%ND if PARTITIONED else di_loop
        raw_k = tl.maximum(1, tl.floor(tl.load(Dur + di) * scale + .5).to(tl.int32))
        k = tl.minimum(raw_k, NB // 8 - 1)
        rho2 = tl.load(Rho + di) * tl.load(Rho + di)
        count = circular_sum(C, row + 3 * (NB + 1), jj, k, NB) \
                + circular_sum(C, row + 3 * (NB + 1), jj + NB // 2, k, NB)
        usable = (j < NB // 2) & (raw_k < NB // 8) & (count >= 3) & (count <= N - 3)
        for si_loop in tl.range(1 if PILOT or PARTITIONED else NS, loop_unroll_factor=1):
            si = pilot_key // ((NB // 2) * ND) if PILOT else part_index//(phase_tiles*ND) if PARTITIONED else si_loop
            ae = layered_sum(C, row, jj, k, Shape, si, NB, LAYERS, False)
            ao = layered_sum(C, row, jj + NB // 2, k, Shape, si, NB, LAYERS, False)
            pe = layered_sum(C, row + NB + 1, jj, k, Shape, si, NB, LAYERS, False)
            po = layered_sum(C, row + NB + 1, jj + NB // 2, k, Shape, si, NB, LAYERS, False)
            ne = layered_sum(C, row + 2 * (NB + 1), jj, k, Shape, si, NB, LAYERS, True)
            no = layered_sum(C, row + 2 * (NB + 1), jj + NB // 2, k, Shape, si, NB, LAYERS, True)
            qe, qo, qeo = ne - pe * pe / c, no - po * po / c, -pe * po / c
            for mode in range(RANK):
                mr = row + (4 + mode) * (NB + 1)
                ge = layered_sum(C, mr, jj, k, Shape, si, NB, LAYERS, False)
                go = layered_sum(C, mr, jj + NB // 2, k, Shape, si, NB, LAYERS, False)
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
                for phase_index in tl.range(NORB, loop_unroll_factor=1):
                    orbital=Orbit+((f*ND+di)*NORB+phase_index)*4
                    phase=tl.load(orbital)
                    duration_ratio=tl.load(orbital+1)
                    logwp,logwb=tl.load(orbital+2),tl.load(orbital+3)
                    offset=tl.floor(phase*(NB//2)+.5).to(tl.int32)
                    sk=tl.minimum(NB//2-1,tl.maximum(1,tl.floor(k*duration_ratio+.5).to(tl.int32)))
                    s0=jj+offset+tl.floor((k-sk).to(tl.float64)/2.).to(tl.int32)
                    s1=s0+NB//2
                    ass = layered_sum(C, row, s0, sk, Shape, si, NB, LAYERS, False) \
                          + layered_sum(C, row, s1, sk, Shape, si, NB, LAYERS, False)
                    ps = layered_sum(C, row + NB + 1, s0, sk, Shape, si, NB, LAYERS, False) \
                         + layered_sum(C, row + NB + 1, s1, sk, Shape, si, NB, LAYERS, False)
                    ns = layered_sum(C, row + 2 * (NB + 1), s0, sk, Shape, si, NB, LAYERS, True) \
                         + layered_sum(C, row + 2 * (NB + 1), s1, sk, Shape, si, NB, LAYERS, True)
                    qs, qps = ns - ps * ps / c, -(pe + po) * ps / c
                    for mode in range(RANK):
                        mr = row + (4 + mode) * (NB + 1)
                        ge = layered_sum(C, mr, jj, k, Shape, si, NB, LAYERS, False)
                        go = layered_sum(C, mr, jj + NB // 2, k, Shape, si, NB, LAYERS, False)
                        gs = layered_sum(C, mr, s0, sk, Shape, si, NB, LAYERS, False) \
                             + layered_sum(C, mr, s1, sk, Shape, si, NB, LAYERS, False)
                        beta = tl.load(Beta + mode)
                        qs -= beta * gs * gs
                        qps -= beta * (ge + go) * gs
                    g0,g1,h0,h1=harmonic_pair(C,row,jj,s0,k,sk,Shape,si,NB,LAYERS,HSTART+2,True,HARMONIC)
                    pa,pb,pq,pr,px,pbase=modulated_pair(ap,ass,qp,qs,qps,g0,g1,h0,h1,Modulation,f,di,0,rho2,ND)
                    ba,bb,bq,br,bx,bbase=modulated_pair(ap,ass,qp,qs,qps,g0,g1,h0,h1,Modulation,f,di,1,rho2,ND)
                    ca,cb,cq,cr,cx=astrometric_pair(C,row,jj,s0,k,sk,Shape,si,AstroNorm,NB,LAYERS,ASTART,NG,True)
                    cu=centroid_upper(ca,cb,cq,cr,cx)
                    pu=subspace_upper(pa,pb,pq,pr,px,1.)+pbase+logwp+cu
                    bu=tl.maximum(subspace_upper(ba,bb,bq,br,bx,1.)+bbase,subspace_upper(pa,pb,pq,pr,px,1.)+pbase)+logwb+cu
                    upper=tl.maximum(pu-floor_p,bu-floor_b)+tl.load(DurationLogMass+di)
                    upper=tl.where((logwp>-float("inf"))|(logwb>-float("inf")),upper,-float("inf"))
                    raw_upper=tl.where(logwb>-float("inf"),bu,-float("inf"))
                    if tl.max(tl.where(j<NB//2,upper,-float("inf")),0)>=0.:
                        for ri in tl.range(5,loop_unroll_factor=1):
                            ratio=tl.load(Parameters+6+ri*3)
                            wp,wb=tl.load(Parameters+7+ri*3),tl.load(Parameters+8+ri*3)
                            cp=centroid_evidence(ca,cb,cq,cr,cx,ratio,tl.load(Parameters+28))
                            cd=centroid_evidence(ca,cb,cq,cr,cx,ratio,tl.load(Parameters+29))
                            ep=positive_amplitude_evidence(pa+ratio*pb,pq+2*ratio*px+ratio*ratio*pr,tau_p)+pbase+cp+tl.log(wp)+logwp
                            if HARMONIC or NG>0:
                                diluted=log_add(positive_amplitude_evidence(ba+ratio*bb,bq+2*ratio*bx+ratio*ratio*br,tau_d)+bbase,
                                    positive_amplitude_evidence(pa+ratio*pb,pq+2*ratio*px+ratio*ratio*pr,tau_d)+pbase)-tl.log(tl.full([],2.,tl.float64))
                                centerd=log_add(cp,cd)-tl.log(tl.full([],2.,tl.float64))
                            else:
                                diluted=positive_amplitude_evidence(ba+ratio*bb,bq+2*ratio*bx+ratio*ratio*br,tau_d)+bbase
                                centerd=cd
                            eb=log_add(positive_amplitude_evidence(ba+ratio*bb,bq+2*ratio*bx+ratio*ratio*br,tau_b)+bbase+cp,
                                diluted+centerd)+tl.log(.25*wb)+logwb
                            lp,lb=log_add(lp,ep),log_add(lb,eb)
                            better_component=eb>component_best
                            sec_ratio=tl.where(better_component,ratio,sec_ratio)
                            sec_phase=tl.where(better_component,phase,sec_phase)
                            component_best=tl.maximum(component_best,eb)
                    else:
                        omitted += 1
                        omitted_upper = tl.maximum(omitted_upper, raw_upper)
                g0,g1,h0,h1=harmonic_pair(C,row,jj,jj+NB//2,k,k,Shape,si,NB,LAYERS,HSTART,False,HARMONIC)
                ba,bb,bq,br,bx,bbase=modulated_pair(ae,ao,qe,qo,qeo,g0,g1,h0,h1,Modulation,f,di,2,rho2,ND)
                ca,cb,cq,cr,cx=astrometric_pair(C,row,jj,jj+NB//2,k,k,Shape,si,AstroNorm,NB,LAYERS,ASTART,NG,False)
                pa,pb,pq,pr,px,pbase=modulated_pair(ae,ao,qe,qo,qeo,g0,g1,h0,h1,Modulation,f,di,3,rho2,ND)
                alias_mass=tl.load(Alias+f*ND+di)
                raw_upper=tl.maximum(subspace_upper(ba,bb,bq,br,bx,1.)+bbase,subspace_upper(pa,pb,pq,pr,px,1.)+pbase)+centroid_upper(ca,cb,cq,cr,cx)+alias_mass
                upper=raw_upper+tl.load(DurationLogMass+di)
                if tl.max(tl.where(j<NB//2,upper,-float("inf")),0)>=floor_b:
                    for ri in tl.range(7,loop_unroll_factor=1):
                        ratio=tl.load(Parameters+21+ri)
                        aa=ba+ratio*bb
                        qq=bq+2*ratio*bx+ratio*ratio*br
                        cp=centroid_evidence(ca,cb,cq,cr,cx,ratio,tl.load(Parameters+28))
                        cd=centroid_evidence(ca,cb,cq,cr,cx,ratio,tl.load(Parameters+29))
                        if HARMONIC or NG>0:
                            diluted=log_add(positive_amplitude_evidence(aa,qq,tau_d)+bbase,
                                positive_amplitude_evidence(pa+ratio*pb,pq+2*ratio*px+ratio*ratio*pr,tau_d)+pbase)-tl.log(tl.full([],2.,tl.float64))
                            centerd=log_add(cp,cd)-tl.log(tl.full([],2.,tl.float64))
                        else:
                            diluted=positive_amplitude_evidence(aa,qq,tau_d)+bbase
                            centerd=cd
                        eb=log_add(positive_amplitude_evidence(aa,qq,tau_b)+bbase+cp,diluted+centerd)+alias_mass+tl.log(.25/tl.full([],7.,tl.float64))
                        lb=log_add(lb,eb)
                        better_component=eb>component_best
                        sec_ratio=tl.where(better_component,0.,sec_ratio)
                        sec_phase=tl.where(better_component,.5,sec_phase)
                        odd_ratio=tl.where(better_component,ratio,odd_ratio)
                        binary_multiple=tl.where(better_component,2.,binary_multiple)
                        component_best=tl.maximum(component_best,eb)
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
        out = Parts + ((f.to(tl.int64) * NT + part_index) * 2 + selection) * 15
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
                ND: tl.constexpr, NB, B: tl.constexpr):
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
    joint = tl.load(Parameters + f * 30 + 3) + lp - log_add(tl.load(Parameters + f * 30 + 5), tl.load(Parameters + f * 30 + 4) + lb)
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
