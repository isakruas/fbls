"""Dense float64 reference checks for the joint planet/binary/noise evidence kernels."""
import math
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tess_search import SearchConfig
from tess_search.core import NoiseModel, compare_search, gpu64, prepare_astrometry, occulted_flux, profile_bank, orbit_bank

torch.manual_seed(7203)
torch.set_default_dtype(torch.float64)
device = "cuda"
cfg = SearchConfig()
comparison = {k: getattr(cfg, k) for k in ("planet_prior", "binary_prior", "planet_depth_scale",
    "binary_depth_scale", "diluted_binary_depth_scale", "stellar_radius_solar",
    "planet_radius_scale_rj", "target_flux_fraction")}


def evidence(a, q, tau):
    precision = 1 + tau**2 * q
    z = tau * a / precision.sqrt()
    return -.5 * precision.log() + .5 * z.square() + math.log(2) + torch.special.log_ndtr(z)


def reference(model, ps, ds, ingress, layers, oversample, rho, threshold):
    t, y, d, u, lam = model.time, model.deficit, model.variance, model.modes, model.eigenvalues
    root = d.sqrt()
    cov = torch.diag(d) + (root[:, None] * u * lam[None, :]) @ (root[:, None] * u).T
    precision = torch.linalg.solve(cov, torch.eye(len(t), device=device))
    p = precision.sum(1)
    K = precision - p[:, None] * p[None, :] / p.sum()
    null_mass = -.5 * ((len(t)-1) * rho.square().log() + (y @ K @ y) / rho.square())
    null_mass -= null_mass.max()
    nb = max(8, 4 * math.ceil(oversample * float(ps[-1]) / float(ds.min()) / 4))
    j = torch.arange(nb, device=device)
    shp = gpu64(ingress)
    shape_log = torch.stack((-4 * shp, 2 * shp), 1)
    shape_log -= torch.logsumexp(shape_log, 0)
    rows = []
    for period in ps:
        scale = nb / period
        cell = (t * scale).floor().long().remainder(2 * nb)
        trials = []
        for di, duration in enumerate(ds):
            k = max(1, math.floor(float(duration * scale) + .5))
            if k >= nb // 4:
                continue
            for si, ing in enumerate(ingress):
                def shape(start):
                    value = torch.zeros((nb, len(t)), device=device)
                    for l in range(layers):
                        width = max(1, math.floor(k * (1 - 2 * ing * (l / (layers-1) if layers > 1 else 0)) + .5))
                        begin = start + (k-width)//2
                        value += ((cell[None, :] - begin[:, None]).remainder(2*nb) < width) / layers
                    return value
                he, ho = shape(j), shape(j+nb)
                hp = he + ho
                a = hp @ K @ y
                q = (hp @ K * hp).sum(1)
                count = ((cell[None, :] - j[:, None]).remainder(nb) < k).sum(1)
                snr, depth = a / (q * rho[di]**2).sqrt(), a / q
                lp, lb, components = [], [], []
                for offset, weight in ((nb//4, .1), (nb//2, .8), (3*nb//4, .1)):
                    hs = shape(j+offset) + shape(j+offset+nb)
                    for ratio, wp, wb in zip((0., .01, .1, .5, 1.), (.75, .15, .08, .015, .005), (.1, .1, .2, .25, .35)):
                        h = hp + ratio * hs
                        aa, qq = (h @ K @ y)/rho[di]**2, (h @ K * h).sum(1)/rho[di]**2
                        lp.append(evidence(aa, qq, .01) + math.log(weight * wp))
                        eb = torch.logaddexp(evidence(aa, qq, .1), evidence(aa, qq, .01)) + math.log(.25*weight*wb)
                        lb.append(eb)
                        components.append((ratio, offset/nb, 1., float(period)))
                for ratio in (.25, .5, .75, 1., 4/3, 2., 4.):
                    h = he + ratio * ho
                    aa, qq = (h @ K @ y)/rho[di]**2, (h @ K * h).sum(1)/rho[di]**2
                    eb = torch.logaddexp(evidence(aa, qq, .1), evidence(aa, qq, .01)) + math.log(.25/7)
                    lb.append(eb)
                    components.append((0., .5, ratio, 2*float(period)))
                lp = torch.logsumexp(torch.stack(lp), 0) + shape_log[si, 0] + null_mass[di]
                lbb = torch.stack(lb)
                lb = torch.logsumexp(lbb, 0) + shape_log[si, 1] + null_mass[di]
                joint = math.log(.01) + lp - torch.logaddexp(torch.full_like(lb, math.log(.98)), math.log(.01)+lb)
                valid = (count >= 3) & (count <= len(t)-3) & (a > 0) & (q > 0) & (snr >= threshold)
                component_values = gpu64(components)[lbb.argmax(0)]
                block = torch.cat((torch.stack((period.expand(nb), (j+k/2)/scale+model.origin,
                    (k/scale).expand(nb), snr, depth, joint, lp, lb), 1), component_values,
                    torch.full((nb, 1), ing, device=device)), 1)
                block[~valid, 3] = -torch.inf
                block[~valid, 5] = -torch.inf
                trials.append(block)
        # Kernel tie order is shape, duration, phase, while this reference is duration first.
        bank = torch.cat(trials)
        valid_durations = (ds * scale + .5).floor().clamp_min(1) < nb // 4
        normalizer = math.log(nb) + torch.logsumexp(null_mass[valid_durations], 0)
        integrated_lp = torch.logsumexp(bank[:, 6], 0) - normalizer
        integrated_lb = torch.logsumexp(bank[:, 7], 0) - normalizer
        integrated_joint = math.log(.01) + integrated_lp - torch.logaddexp(
            gpu64(math.log(.98)), math.log(.01)+integrated_lb)
        result = []
        for col, flag in ((6, 1), (3, 2)):
            merit = torch.where(torch.isfinite(bank[:, 3]), bank[:, col], -torch.inf)
            maximum = merit.max()
            matches = bank[merit == maximum]
            key = matches[:, 12] * 1e6 + matches[:, 2] * 1e3 + matches[:, 1]
            row = matches[key.argmin()]
            if torch.isfinite(row[3]):
                row[5], row[6], row[7] = integrated_joint, integrated_lp, integrated_lb
                result.append(torch.cat((row, gpu64([flag]))))
        if len(result) == 2 and torch.equal(result[0][:13], result[1][:13]):
            result[0][-1] = 3
            result.pop()
        rows.extend(result)
    return torch.stack(rows) if rows else torch.empty((0, 14), device=device)




class JointEvidenceDenseTests(unittest.TestCase):
    def test_surrogate_bank_matches_dense_reference(self):
        results = []
        for rank in (0, 3):
            t = torch.arange(0., 9., .047, device=device)
            t = t[(t < 3.) | (t > 4.1)]
            n = len(t)
            d = (.002 * (.7 + torch.rand(n, device=device))).square()
            u = torch.linalg.qr(torch.randn((n, rank), device=device)).Q
            lam = gpu64([3., 1., .3][:rank])
            y = .002*torch.randn(n, device=device)
            y += .008*((t-.4).remainder(2.) < .15)
            y += .004*((t-1.4).remainder(2.) < .15)
            model = NoiseModel(t, y, d, u, lam, origin=2459000.)
            ps, ds, rho = gpu64([1.7, 2., 2.4]), gpu64([.10, .17]), gpu64([1., 1.3])
            actual, stats = compare_search(model, ps, ds, comparison, threshold=1., phase_oversample=2,
                ingress_fractions=(0., .2), shape_layers=3, noise_scale=rho)
            expected = reference(model, ps, ds, (0., .2), 3, 2, rho, 1.)
            assert actual.shape == expected.shape, (actual, expected)
            assert bool(torch.isfinite(actual[:, :8]).all() & torch.isfinite(actual[:, 12:]).all())
            errors = torch.nan_to_num((actual-expected).abs(), nan=0.).amax(0)
            assert float(errors.max()) < 1e-9, (actual, expected)
            results.append({"rank": rank, "rows": len(actual), "maximum_error": float(errors.max()), "columns": errors.tolist()})
            batched, _ = compare_search(model, ps, ds, comparison, threshold=1., phase_oversample=2,
                ingress_fractions=(0., .2), shape_layers=3, noise_scale=rho, period_batch=1)
            assert torch.allclose(actual, batched, atol=1e-10, rtol=1e-12, equal_nan=True)
            empty = NoiseModel(t, torch.zeros_like(t), d, u, lam, origin=2459000.)
            empty_table, _ = compare_search(empty, ps, ds, comparison, threshold=1., phase_oversample=2,
                ingress_fractions=(0., .2), shape_layers=3, noise_scale=rho)
            assert empty_table.shape == (0, 14)
            results[-1]["single_period_batch_matches"] = True
            results[-1]["empty_output_passed"] = True
            strong = NoiseModel(t, 10*y, d, u, lam, origin=2459000.)
            fast, fast_stats = compare_search(strong, ps, ds, comparison, threshold=1., phase_oversample=8,
                ingress_fractions=(0., .2), shape_layers=3, noise_scale=rho)
            exact, _ = compare_search(strong, ps, ds, dict(comparison, evidence_rtol=0.), threshold=1., phase_oversample=8,
                ingress_fractions=(0., .2), shape_layers=3, noise_scale=rho)
            dense = reference(strong, ps, ds, (0., .2), 3, 8, rho, 1.)
            assert fast.shape == exact.shape == dense.shape
            assert bool(torch.isfinite(fast[:, :8]).all() & torch.isfinite(fast[:, 12:]).all())
            bound_error = float(torch.nan_to_num((fast-exact).abs(), nan=0.).max())
            dense_error = float(torch.nan_to_num((fast-dense).abs(), nan=0.).max())
            assert bound_error < 1e-8 and dense_error < 1e-8, (fast,exact,dense)
            assert fast_stats["evidence_groups_omitted"] > 0
            results[-1].update(bound_error=bound_error, bounded_dense_error=dense_error,
                omitted_groups=fast_stats["evidence_groups_omitted"], total_groups=fast_stats["evidence_groups_total"])



class ExtendedPhysicsDenseTests(unittest.TestCase):
    def test_extended_bank_matches_dense_reference(self):
        torch.manual_seed(2104)
        z=gpu64([0.,.05,.5,.9,.95,1.,1.05,1.099,1.2])
        p=.1
        partial=(p*p*torch.acos(((z*z+p*p-1)/(2*z*p).clamp_min(1e-300)).clamp(-1,1))+
            torch.acos(((z*z+1-p*p)/(2*z).clamp_min(1e-300)).clamp(-1,1))-
            .5*((-z+p+1)*(z+p-1)*(z-p+1)*(z+p+1)).clamp_min(0).sqrt())/math.pi
        expected=torch.where(z<=1-p,p*p,torch.where(z>=1+p,0.,partial))
        error=float((occulted_flux(z,p,0.,0.)-expected).abs().max())
        self.assertLess(error, 1e-9)
        t=torch.arange(0.,8.,.071,device='cuda');t=t[(t<2.1)|(t>3.)]
        group=(t>=4).long();n=len(t)
        h=((t-.5).remainder(2.)<.16).double()
        y=.006*h+.0005*torch.cos(2*math.pi*t)+.001*torch.randn_like(t)
        d=torch.full_like(t,.001**2)
        cent=torch.stack((.004*h+.002*torch.randn_like(t)+10*group,
                         .001*h+.002*torch.randn_like(t)-4*group),1)
        astro=prepare_astrometry(t,cent,torch.full_like(cent,.002),group)
        config=SearchConfig(extended_physics=True,stellar_radius_solar=1.,stellar_mass_solar=1.,stellar_teff=None,eccentricity_nodes=1,periastron_nodes=4)
        comparison=asdict(config)
        rank=0
        u=torch.linalg.qr(torch.randn((n,rank),device='cuda')).Q
        lam=gpu64([3.,1.,.2][:rank])
        model=NoiseModel(t,y,d,u,lam)
        ps,ds,rho=gpu64([1.8,2.,2.2]),gpu64([.12,.18]),gpu64([1.,1.2])
        actual,stats=compare_search(model,ps,ds,comparison,threshold=1.,phase_oversample=2,ingress_fractions=(0.,.2),shape_layers=4,noise_scale=rho,astrometry=astro)
        assert len(actual)>0 and bool(torch.isfinite(actual[:,:8]).all())
        exact,_=compare_search(model,ps,ds,dict(comparison,evidence_rtol=0.),threshold=1.,phase_oversample=2,ingress_fractions=(0.,.2),shape_layers=4,noise_scale=rho,astrometry=astro)
        assert actual.shape==exact.shape
        delta=float(torch.nan_to_num((actual-exact).abs(),nan=0).max())
        assert delta<1e-8


        def evidence(a,q,tau):
            v=1+tau*tau*q.clamp_min(0)
            z=tau*a/v.sqrt()
            return -.5*v.log()+.5*z.square()+math.log(2)+torch.special.log_ndtr(z)

        def signed(h,sigma):
            result=torch.zeros(h.shape[0],device='cuda')
            for axis in range(2):
                w=torch.where(torch.isfinite(astro['variance'][:,axis]),1/astro['variance'][:,axis],0.)
                basis=torch.nn.functional.one_hot(group,2).double()*w[:,None]
                totals=basis.sum(0)
                K=torch.diag(w)-(basis/totals.clamp_min(1e-300)[None,:])@basis.T
                aa=h@K@astro['value'][:,axis]
                qq=(h@K*h).sum(1).clamp_min(0)
                vv=1+sigma*sigma*qq
                result+=.5*sigma*sigma*aa.square()/vv-.5*vv.log()
            return result

        nb=max(8,4*math.ceil(2*float(ps[-1])/float(ds.min())/4))
        shapes=profile_bank(gpu64([0.,.2]),4,comparison)
        shape_prior=torch.stack((-4*gpu64([0.,.2]),2*gpu64([0.,.2])),1)
        shape_prior-=torch.logsumexp(shape_prior,0)
        orbit,alias=orbit_bank(ps,ds,nb/ps,nb,comparison)
        base_cov=torch.diag(d)+(d.sqrt()[:,None]*model.modes*model.eigenvalues[None,:])@(d.sqrt()[:,None]*model.modes).T
        W=torch.linalg.solve(base_cov,torch.eye(n,device='cuda'));wp=W.sum(1);K=W-wp[:,None]*wp[None,:]/wp.sum()
        null=-.5*((n-1)*rho.square().log()+(y@K@y)/rho.square());null-=null.max()
        rows=[]
        for pi,period in enumerate(ps):
            cell=(t*nb/period).floor().long().remainder(2*nb)
            j=torch.arange(nb,device='cuda')
            angle=2*math.pi*t/period
            ar=(1.32712440018e20*(period*86400/(2*math.pi))**2)**(1/3)/6.957e8
            factor=1.5/ar**3
            sp=factor*.000954588;sb=factor*(.5/1.5)
            banks=[]
            for di,duration in enumerate(ds):
                k=max(1,math.floor(float(duration*nb/period)+.5))
                if k>=nb//4:continue
                covariance=base_cov*rho[di]**2
                def baseline(cov):
                    inv=torch.linalg.solve(cov,torch.eye(n,device='cuda'))
                    c=inv.sum();v=inv.sum(1)
                    return inv-v[:,None]*v[None,:]/c,torch.linalg.slogdet(cov)[1]+c.log()
                K0,logdet0=baseline(covariance)
                models=[]
                for frequency,sigma in ((2,sp),(2,sb),(1,sb/4),(1,sp/4)):
                    X=torch.stack((torch.cos(frequency*angle),torch.sin(frequency*angle)),1)
                    Kh,logdeth=baseline(covariance+sigma*sigma*(X@X.T))
                    base=.5*(y@K0@y-y@Kh@y)-.5*(logdeth-logdet0)
                    models.append((Kh,base))
                def flux(h,family,tau):
                    Kh,base=models[family]
                    return evidence(h@Kh@y,(h@Kh*h).sum(1),tau)+base
                for si,ing in enumerate((0.,.2)):
                    def shape(start,width):
                        value=torch.zeros((nb,n),device='cuda')
                        for layer in range(4):
                            inner=max(1,math.floor(width*float(shapes[si,layer,0])+.5))
                            begin=start+(width-inner)//2
                            value+=shapes[si,layer,1]*((cell[None,:]-begin[:,None]).remainder(2*nb)<inner)
                        return value
                    he,ho=shape(j,k),shape(j+nb,k);hp=he+ho
                    lp=[];lb=[];descriptions=[]
                    for oi in range(orbit.shape[2]):
                        ph,dr,lwp,lwb=orbit[pi,di,oi]
                        sk=max(1,math.floor(k*float(dr)+.5))
                        offset=math.floor(float(ph)*nb+.5)+(k-sk)//2
                        hs=shape(j+offset,sk)+shape(j+offset+nb,sk)
                        for ratio,pw,bw in zip((0.,.01,.1,.5,1.),(.75,.15,.08,.015,.005),(.1,.1,.2,.25,.35)):
                            h=hp+ratio*hs
                            cp,cd=signed(h,.001),signed(h,.05)
                            lp.append(flux(h,0,.10045**2)+cp+math.log(pw)+lwp)
                            diluted=torch.logaddexp(flux(h,1,.01),flux(h,0,.01))-math.log(2)
                            cd=torch.logaddexp(cp,cd)-math.log(2)
                            lb.append(torch.logaddexp(flux(h,1,.1)+cp,diluted+cd)+math.log(.25*bw)+lwb)
                            descriptions.append((ratio,float(ph),1.,float(period)))
                    for ratio in (.25,.5,.75,1.,4/3,2.,4.):
                        h=he+ratio*ho
                        cp,cd=signed(h,.001),signed(h,.05)
                        diluted=torch.logaddexp(flux(h,2,.01),flux(h,3,.01))-math.log(2)
                        cd=torch.logaddexp(cp,cd)-math.log(2)
                        lb.append(torch.logaddexp(flux(h,2,.1)+cp,diluted+cd)+alias[pi,di]+math.log(.25/7))
                        descriptions.append((0.,.5,ratio,2*float(period)))
                    component=torch.stack(lb)
                    lp=torch.logsumexp(torch.stack(lp),0)+shape_prior[si,0]+null[di]
                    lb=torch.logsumexp(component,0)+shape_prior[si,1]+null[di]
                    aa=hp@K0@y;qq=(hp@K0*hp).sum(1)
                    snr=aa/qq.clamp_min(1e-300).sqrt();depth=aa/qq.clamp_min(1e-300)
                    count=((cell[None,:]-j[:,None]).remainder(nb)<k).sum(1)
                    valid=(aa>0)&(qq>0)&(count>=3)&(count<=n-3)&(snr>=1.)
                    desc=gpu64(descriptions)[component.argmax(0)]
                    block=torch.cat((torch.stack((period.expand(nb),(j+k/2)*period/nb,(k*period/nb).expand(nb),snr,depth,lp,lb),1),desc,
                         torch.full((nb,1),ing,device='cuda')),1)
                    block[~valid,3]=-torch.inf
                    banks.append(block)
            bank=torch.cat(banks)
            lp=torch.logsumexp(bank[:,5],0)-math.log(nb)-torch.logsumexp(null,0)
            lb=torch.logsumexp(bank[:,6],0)-math.log(nb)-torch.logsumexp(null,0)
            odds=math.log(.01)+lp-torch.logaddexp(gpu64(math.log(.98)),math.log(.01)+lb)
            selected=[]
            for col,flag in ((5,1),(3,2)):
                merit=torch.where(torch.isfinite(bank[:,3]),bank[:,col],-torch.inf)
                winner=bank[int(merit.argmax())]
                if torch.isfinite(winner[3]):
                    selected.append(torch.cat((winner[:5],torch.stack((odds,lp,lb)),winner[7:],gpu64([flag]))))
            if len(selected)==2 and torch.equal(selected[0][:5],selected[1][:5]) and selected[0][12]==selected[1][12]:
                selected[0][-1]=3;selected.pop()
            rows.extend(selected)
        dense=torch.stack(rows)
        assert dense.shape==actual.shape,(dense,actual)
        err=torch.nan_to_num((dense-actual).abs(),nan=0).amax(0)
        assert float(err.max())<1e-7,(dense,actual)


if __name__ == "__main__":
    unittest.main()
