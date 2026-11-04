"""CUDA integration tests: evidence cases, blind multi-signal recovery, families and noise scales."""
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tess_search import SearchConfig, search_prepared, search_transits, detrending_windows
from tess_search.core import NoiseModel, compare_search, correlated_noise_factors, gpu64


def box(t, period, epoch, duration):
    return (((t - epoch + period / 2).remainder(period) - period / 2).abs() < duration / 2).double()


class EvidenceCaseTests(unittest.TestCase):
    def test_surrogate_cases_prefer_expected_hypotheses(self):
        torch.manual_seed(90317)
        config = SearchConfig(threshold=7.1, bin_minutes=0., correlated_noise=False)
        comparison = {k: getattr(config, k) for k in ("planet_prior", "binary_prior", "planet_depth_scale",
                      "binary_depth_scale", "diluted_binary_depth_scale", "stellar_radius_solar",
                      "planet_radius_scale_rj", "target_flux_fraction")}
        t = torch.arange(0., 24., .031, device="cuda", dtype=torch.float64)
        ps, ds = gpu64([1.15, 1.85, 2.3, 3.7, 4.6, 7.4]), gpu64([.08, .12, .18])
        noise = .001 * torch.randn_like(t)
        planet = ((t - .4).remainder(2.3) < .12).double()
        primary = ((t - .6).remainder(3.7) < .12).double()
        secondary = ((t - .6 - 3.7 / 2).remainder(3.7) < .12).double()
        alternating = 1 + ((t - .4).div(2.3, rounding_mode="floor").long().remainder(2))
        cases = (("planet", .006 * planet + noise),
                 ("planet_occultation", .006 * planet + .0006 * ((t - .4 - 2.3 / 2).remainder(2.3) < .12) + noise),
                 ("binary", .012 * primary + .006 * secondary + noise),
                 ("alternating", .004 * planet * alternating + noise),
                 ("mixed", .006 * planet + .012 * primary + .006 * secondary + noise),
                 ("null", noise))
        for name, y in cases:
            model = NoiseModel(t, y, torch.full_like(t, .001 ** 2), torch.empty((len(t), 0), device="cuda"), gpu64([]))
            table, _ = compare_search(model, ps, ds, comparison, threshold=7.1)
            if name == "null":
                self.assertEqual(len(table), 0)
                continue
            joint = table[(table[:, 13].long() & 1) != 0]
            signal = table[(table[:, 13].long() & 2) != 0]
            self.assertTrue(len(joint) and len(signal), name)
            best = joint[joint[:, 5].argmax()]
            if name in ("planet", "planet_occultation", "mixed"):
                self.assertEqual(float(best[0]), 2.3, name)
                self.assertGreater(float(best[5]), 0, name)
            if name in ("binary", "alternating"):
                self.assertLess(float(best[5]), 0, name)
            if name == "mixed":
                self.assertEqual(float(signal[signal[:, 3].argmax(), 0]), 1.85)


class BlindRecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2611)
        self.t = torch.arange(0., 54., 10 / 1440, device="cuda", dtype=torch.float64)
        self.t = self.t[(self.t < 26.) | (self.t > 28.)] + 2459000.
        self.noise = .0004 * torch.randn_like(self.t)
        self.error = torch.full_like(self.t, .0004)
        self.config = SearchConfig(period_max=100., durations_hours=(1., 2., 3., 4.5), threshold=7.1,
                                   max_candidates=8, passes=3, min_events=2)

    def matches(self, peaks, period, epoch, duration_days):
        for rank, peak in enumerate(peaks, 1):
            cycles = round((peak["epoch_bjd"] - epoch) / period)
            if abs(peak["period_days"] / period - 1) < 1e-3 and abs(peak["epoch_bjd"] - epoch - cycles * period) < duration_days / 2:
                return rank
        return None

    def test_two_planets_recovered_blind_without_aliases(self):
        t = self.t
        deep = .009 * box(t, 10.6, 2459004.3, 4 / 24)
        shallow = .0012 * box(t, 4.77, 2459001.1, 2.5 / 24)
        flux = 1 + .01 * torch.sin(2 * math.pi * t / 7.3) - deep - shallow + self.noise
        result = search_transits(t, flux, self.error, torch.zeros_like(t, dtype=torch.long), config=self.config)
        peaks = result.candidates(ranking="signal")
        self.assertEqual(self.matches(peaks, 10.6, 2459004.3, 4 / 24), 1)
        self.assertIsNotNone(self.matches(peaks, 4.77, 2459001.1, 2.5 / 24))
        # Period aliases and harmonics of the deep signal are suppressed by isolated rescoring.
        for peak in peaks:
            ratio = peak["period_days"] / 10.6
            for harmonic in (.5, 2., 1 / 3, 3.):
                self.assertFalse(abs(ratio / harmonic - 1) < 2e-3, peak)
            self.assertGreaterEqual(peak["events"], 2)
            self.assertGreaterEqual(peak["snr"], 7.1)
        mapping = result.to_dict()
        json.dumps(mapping, allow_nan=False)
        self.assertEqual(mapping["schema_version"], 3)
        self.assertEqual(result.table.shape[1], 14)
        self.assertIn("single_events", mapping)
        self.assertEqual(result.candidates(0), [])

    def test_single_event_is_reported_without_a_period(self):
        t = self.t
        flux = 1 - .008 * box(t, 1e4, 2459013.2, 5 / 24) + self.noise
        result = search_transits(t, flux, self.error, config=self.config)
        self.assertTrue(result.single_events)
        self.assertLess(abs(result.single_events[0]["epoch_bjd"] - 2459013.2), 5 / 48)
        self.assertEqual(result.candidates(ranking="signal"), [])

    def test_prepared_input_recovers_the_same_planet(self):
        t = self.t
        flux = -.006 * box(t, 10.6, 2459004.3, 4 / 24) + self.noise
        result = search_prepared(t, flux, self.error, config=self.config)
        self.assertEqual(self.matches(result.candidates(ranking="signal"), 10.6, 2459004.3, 4 / 24), 1)
        self.assertEqual(len(result.statistics["families"]), 1)


class FamilyAndNoiseTests(unittest.TestCase):
    def test_duration_families_share_equal_windows(self):
        durations = gpu64([.5, .625, 1., 2.98, 3.73, 5.82, 27.76]) / 24
        families = detrending_windows(durations, .4, 3., 3.5)
        windows = [w for w, _ in families]
        self.assertEqual(families[0][1], [0, 1, 2, 3])
        self.assertAlmostEqual(windows[0], .4)
        self.assertAlmostEqual(windows[1], 3 * 3.73 / 24, places=5)
        self.assertAlmostEqual(windows[-1], 3.47, places=5)
        self.assertEqual(sorted(i for _, ids in families for i in ids), list(range(7)))

    def test_sparse_durations_inherit_conservative_noise_scales(self):
        torch.manual_seed(4106)
        t = torch.arange(0., 100., .02, device="cuda", dtype=torch.float64)
        sigma = torch.ones_like(t)
        noise = torch.randn_like(t)
        y = noise + 3 * torch.sin(t / 5)
        durations = gpu64([1., 4., 6., 8.])
        actual, detail = correlated_noise_factors(t, y, sigma, durations, return_details=True)
        self.assertEqual(detail["estimated_directly"], [True, True, False, False])
        self.assertTrue(bool((actual[2:] >= actual[1] * (durations[2:] / durations[1]).sqrt()).all()))
        permutation = torch.tensor([3, 0, 2, 1], device="cuda")
        shuffled = correlated_noise_factors(t, y, sigma, durations[permutation])
        self.assertTrue(torch.allclose(shuffled, actual[permutation], rtol=1e-13, atol=1e-13))
        for i in (0, 1):
            chunks = y.reshape(-1, round(float(durations[i]) / .02))
            z = chunks.mean(1) * chunks.shape[1] ** .5
            expected = (1.4826 * (z - z.median()).abs().median()).clamp_min(1.)
            self.assertTrue(torch.allclose(actual[i], expected, rtol=.03, atol=.03))
        short, short_details = correlated_noise_factors(t[:100], y[:100], sigma[:100], gpu64([1., 3.]), return_details=True)
        self.assertEqual(short_details["fallback_reference"], "unbinned_coherent_reference")
        self.assertTrue(bool(torch.isfinite(short).all() & (short > 1).all()))
        white = correlated_noise_factors(t, noise, sigma, gpu64([.1, .2, .5]))
        self.assertTrue(bool(((white >= 1) & (white < 1.3)).all()))


if __name__ == "__main__":
    unittest.main()
