"""CPU tests of the survey layer: transit geometry derivation and CTOI row filling."""
import math
import unittest


from fbls.survey.ctoi import COLUMNS, ctoi_row
from fbls.survey.vetting import DAY, G, geometry


def durations(period, k, b, a_over_r):
    """Total and full durations (days) of a circular transit (Seager & Mallen-Ornelas 2003)."""
    inc = math.acos(b / a_over_r)
    t14 = period / math.pi * math.asin(math.sqrt((1 + k) ** 2 - b ** 2) / (a_over_r * math.sin(inc)))
    t23 = period / math.pi * math.asin(math.sqrt((1 - k) ** 2 - b ** 2) / (a_over_r * math.sin(inc)))
    return t14, t23


class GeometryTest(unittest.TestCase):
    period, k, b, a_over_r = 10., .1, .3, 20.

    def measured(self):
        t14, t23 = durations(self.period, self.k, self.b, self.a_over_r)
        ingress = (t14 - t23) / (2 * t14)
        return t14, ingress

    def test_recovers_injected_geometry(self):
        t14, ingress = self.measured()
        g = geometry(self.period, self.k ** 2, self.k ** 2 / 50, t14, .02 * t14, ingress, .1 * ingress,
                     cadence=2 / 1440, star={"teff": 5772.}, a_over_r_tic=None)
        self.assertTrue(g["reliable"])
        self.assertLess(abs(g["impact"] - self.b), 3 * g["impact_err"] + .02)
        self.assertLess(abs(g["a_over_rstar_fit"] / self.a_over_r - 1), .1)
        self.assertLess(abs(g["inclination_deg"] - math.degrees(math.acos(self.b / self.a_over_r))), 3 * g["inclination_err"] + .1)
        density = 3 * math.pi * self.a_over_r ** 3 / (G * (self.period * DAY) ** 2) / 1000
        self.assertLess(abs(g["fitted_density_gcc"] / density - 1), .3)
        self.assertLess(abs(g["teq_k"] - 5772 / math.sqrt(2 * self.a_over_r)), 3 * g["teq_err"])
        self.assertGreater(g["insolation_earth"], 0)

    def test_unresolved_ingress_leaves_geometry_empty_but_keeps_irradiation(self):
        t14, ingress = self.measured()
        g = geometry(self.period, self.k ** 2, self.k ** 2 / 50, t14, .02 * t14, ingress, ingress,
                     cadence=2 / 1440, star={"teff": 5772.}, a_over_r_tic=18.)
        self.assertFalse(g["reliable"])
        self.assertIn("ingress", g["unreliable"])
        self.assertIsNone(g["impact"])
        self.assertIsNone(g["fitted_density_gcc"])
        self.assertIsNotNone(g["teq_k"])
        self.assertIsNotNone(g["insolation_earth"])

    def test_weak_depth_is_rejected(self):
        t14, ingress = self.measured()
        g = geometry(self.period, self.k ** 2, self.k ** 2 / 3, t14, .02 * t14, ingress, .1 * ingress,
                     cadence=2 / 1440, star=None, a_over_r_tic=None)
        self.assertIn("depth", g["unreliable"])
        self.assertIsNone(g["teq_k"])


class CtoiRowTest(unittest.TestCase):
    def event(self, geometry_block):
        return {"tic": 1, "search": "periodic", "baseline_days": 300., "tier": "candidate",
                "peak": {"snr": 12.}, "metrics": {"period_days": 10., "epoch_bjd": 2459000., "duration_hours": 3.,
                                                   "depth_ppm": 1e4, "depth_err_ppm": 500., "snr": 11., "transits": 5,
                                                   "radius_earth": 10., "a_over_rstar": 18., "flags": {},
                                                   "fitted_duration_err_hours": .2, "geometry": geometry_block}}

    def test_reliable_geometry_fills_every_optional_column(self):
        row = ctoi_row(self.event({"reliable": True, "impact": .3, "impact_err": .1, "inclination_deg": 89., "inclination_err": .5,
                                   "a_over_rstar_fit": 20., "a_over_rstar_fit_err": 1., "fitted_density_gcc": 1.4, "fitted_density_err": .3,
                                   "insolation_earth": 50., "insolation_err": 5., "teq_k": 900., "teq_err": 40.}), "author")
        self.assertEqual(list(row), COLUMNS)
        self.assertEqual(row["a/Rad_s"], 20.)
        self.assertEqual(row["Impact Param"], .3)
        self.assertEqual(row["Fitted Stellar Density (g/cm^3)"], 1.4)
        self.assertEqual(row["Duration (hrs) Error"], .2)
        self.assertIn("trapezoid fit", row["Notes"])

    def test_unreliable_geometry_falls_back_to_tic_and_leaves_blanks(self):
        row = ctoi_row(self.event({"reliable": False, "unreliable": ["ingress"], "impact": None, "impact_err": None,
                                   "inclination_deg": None, "inclination_err": None, "a_over_rstar_fit": None,
                                   "a_over_rstar_fit_err": None, "fitted_density_gcc": None, "fitted_density_err": None,
                                   "insolation_earth": 60., "insolation_err": 8., "teq_k": 950., "teq_err": 60.}), "author")
        self.assertEqual(row["a/Rad_s"], 18.)
        self.assertAlmostEqual(row["a/Rad_s Error"], 1.8)
        self.assertIsNone(row["Impact Param"])
        self.assertIsNone(row["Fitted Stellar Density (g/cm^3)"])
        self.assertEqual(row["Equilibrium Temp (K)"], 950.)
        self.assertIn("TIC density", row["Notes"])

    def test_single_event_row_has_no_geometry(self):
        event = self.event(None)
        event["search"] = "single"
        row = ctoi_row(event, "author")
        self.assertIsNone(row["Period (days)"])
        self.assertIsNone(row["Impact Param"])
        self.assertIsNotNone(row["a/Rad_s"])



class TierTest(unittest.TestCase):
    def event(self, star, fpp=None):
        return {"search": "periodic", "peak": {"model_preferred": True}, "fpp": fpp,
                "metrics": {"verdict": "pass", "star": star}}

    def test_strong_needs_stellar_parameters(self):
        from fbls.survey.run import tier
        fpp = {"fpp": 1e-4, "nfpp": 0.}
        self.assertTrue(tier(self.event({"radius": 1., "teff": 5800.}, fpp)).startswith("strong"))
        self.assertEqual(tier(self.event({"radius": None, "teff": None}, fpp)), "candidate: no stellar parameters")
        self.assertEqual(tier(self.event({"radius": 1., "teff": 5800.})), "candidate: no FPP")



class FunnelTest(unittest.TestCase):
    def test_pixel_offset_removes_a_signal_from_the_next_stage(self):
        from fbls.survey.run import passing
        event = {"id": 0, "search": "periodic", "metrics": {"verdict": "pass", "flags": {}}}
        self.assertTrue(passing(event))
        self.assertTrue(passing(event, {"events": {"0": {"sigma": 1., "offset_pixels": .1}}}))
        self.assertFalse(passing(event, {"events": {"0": {"sigma": 9., "offset_pixels": 2.}}}))
        self.assertEqual(event["metrics"]["flags"], {})
        self.assertFalse(passing(dict(event, related={"period_days": 1.})))
        self.assertFalse(passing(dict(event, search="single")))



class FigureTest(unittest.TestCase):
    def test_candidate_figures_are_written(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from fbls.survey.figures import candidate_figures
        from fbls.survey.vetting import fold, trapezoid
        rng = np.random.default_rng(0)
        time = np.arange(0, 54, 2 / 1440)
        period, epoch, duration = 10., 3., 3 / 24
        phase = (time - epoch + period / 2) % period - period / 2
        flux = trapezoid(phase, 1e-3, duration, .2) + rng.normal(0, 3e-4, len(time))
        prep = fold(time, flux, np.zeros(len(time)), period, epoch, duration * 24)
        metrics = {"depth_ppm": 1000., "duration_hours": 3., "fitted_duration_hours": 3.1, "ingress_fraction": .2,
                   "odd_even_sigma": .3, "secondary_phase": .5, "secondary_sigma": .4}
        with tempfile.TemporaryDirectory() as folder:
            written = candidate_figures(prep, metrics, "test", Path(folder) / "1_0", (np.array([5., 10.]), np.array([7.5, 12.])))
            self.assertEqual([p.name for p in written],
                             ["light_curve.png", "folded.png", "transits.png", "odd_even.png", "period_maxima.png", "orbit.png"])
            self.assertTrue(all(p.stat().st_size > 5000 for p in written))

    def test_pixel_figure_is_written(self):
        import tempfile
        import numpy as np
        from fbls.survey.figures import pixel_figure
        image = np.zeros((11, 11))
        image[5, 5] = 1.
        images = [{"sector": 1, "transits": 2, "difference": image, "error": image + .1, "oot": image * 10, "target": (5., 5.),
                   "source": {"x": 5.2, "y": 4.9, "x_err": .1, "y_err": .1, "snr": 8.}}]
        with tempfile.TemporaryDirectory() as folder:
            path = pixel_figure(images, "test", folder)
            self.assertGreater(path.stat().st_size, 5000)
            self.assertIsNone(pixel_figure([dict(images[0], source=None)], "test", folder))


if __name__ == "__main__":
    unittest.main()
