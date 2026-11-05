"""Packaging, configuration and CLI contract tests with scalar input fixtures."""
import ast
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE))
from tess_search import SearchConfig, TransitSearchResult, __version__
from tess_search import cli


class ReleaseLayoutTests(unittest.TestCase):
    def test_runtime_packages_and_scripts_are_declared(self):
        import tomllib
        config = tomllib.loads((RELEASE / "pyproject.toml").read_text())
        self.assertEqual(config["tool"]["setuptools"]["packages"], ["tess_search", "tess_survey"])
        self.assertEqual(config["project"]["scripts"]["tess-search"], "tess_search.cli:main")
        self.assertEqual(config["project"]["scripts"]["tess-survey"], "tess_survey.run:main")
        self.assertEqual(config["project"]["version"], __version__)

    def test_imports_stay_inside_runtime_and_declared_dependencies(self):
        allowed = {"argparse", "dataclasses", "json", "math", "numpy", "pathlib", "sys", "time", "torch", "triton"}
        for path in (RELEASE / "tess_search").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    self.assertTrue(all(alias.name.split(".")[0] in allowed for alias in node.names), path)
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    self.assertIn(node.module.split(".")[0], allowed, path)


class ConfigTests(unittest.TestCase):
    def test_default_and_duration_normalization(self):
        self.assertEqual(SearchConfig().noise_degree, 0)
        self.assertEqual(SearchConfig(durations_hours=(2., .5, 2.)).durations_hours, (.5, 2.))

    def test_numeric_scalars_serialize_as_json(self):
        config = SearchConfig(period_min=np.float32(.5), threshold=np.float64(7.1))
        self.assertIs(type(config.period_min), float)
        json.dumps(asdict(config), allow_nan=False)

    def test_invalid_search_parameters(self):
        for kwargs in ({"period_min": 10., "period_max": 1.}, {"threshold": float("nan")},
                       {"durations_hours": ()}, {"durations_hours": (-1.,)},
                       {"durations_hours": (12.,)}, {"ingress_fractions": (.5,)},
                       {"shape_layers": 0}, {"max_candidates": -1}, {"period_batch": 0},
                       {"noise_degree": 5}, {"max_fold_mib": 0.}, {"planet_prior": 0.},
                       {"planet_prior": .6, "binary_prior": .5}, {"target_flux_fraction": 0.},
                       {"target_flux_fraction": 1.1}, {"stellar_radius_solar": -1.},
                       {"planet_depth_scale": 0.}, {"model_comparison": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SearchConfig(**kwargs)

    def test_invalid_physical_parameters(self):
        for kwargs in ({"limb_darkening":(1.,.2)},{"limb_darkening":(-.1,.2)},
                       {"eccentricity_max":1.},{"eccentricity_nodes":-1},
                       {"centroid_binary_scale":0.},{"stellar_mass_solar":0.}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                SearchConfig(**kwargs)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, name, **arrays):
        path = self.root / name
        np.savez(path, **arrays)
        return path

    def test_multiple_files_and_scalar_labels(self):
        a = self.save("a.npz", time_bjd=np.arange(4.), flux=np.ones(4), sector=np.int64(14))
        b = self.save("b.npz", time_bjd=np.arange(4., 8.), flux=np.ones(4))
        result = cli.read_inputs([a, b])
        self.assertEqual(len(result["flux"]), 8)
        self.assertEqual(result["sector"].tolist(), [14] * 4 + [0] * 4)
        self.assertEqual(result["file_index"].tolist(), [0] * 4 + [1] * 4)

    def test_rejects_mixed_error_columns(self):
        a = self.save("a.npz", time_bjd=np.arange(4.), flux=np.ones(4), flux_err=np.ones(4))
        b = self.save("b.npz", time_bjd=np.arange(4.), flux=np.ones(4))
        with self.assertRaisesRegex(ValueError, "All input"):
            cli.read_inputs([a, b])

    def test_rejects_column_shape_mismatch(self):
        path = self.save("bad.npz", time_bjd=np.arange(4.), flux=np.ones(3))
        with self.assertRaisesRegex(ValueError, "matching vectors"):
            cli.read_inputs([path])

    def test_rejects_missing_columns(self):
        path = self.save("bad.npz", flux=np.ones(4))
        with self.assertRaisesRegex(ValueError, "required columns"):
            cli.read_inputs([path])

    def test_centroid_aliases_and_missing_files(self):
        a=self.save("a.npz",time_bjd=np.arange(4.),flux=np.ones(4),centr1=np.ones(4),centr2=np.ones(4)*2)
        b=self.save("b.npz",time_bjd=np.arange(4.,8.),flux=np.ones(4))
        data=cli.read_inputs([a,b])
        self.assertEqual(data["centroid"].shape,(8,2))
        self.assertTrue(np.isnan(data["centroid"][4:]).all())
        self.assertTrue(np.isnan(data["centroid_err"]).all())


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.table = torch.tensor([
            [2., .3, .1, 12., .001, 2., 70., 65., 0., .5, 1., 2., .2, 3.],
            [3., .2, .1, 9., .001, 4., 35., 25., 0., .5, 1., 3., 0., 3.],
            [4., .1, .1, 50., .001, -1., 1000., 1001., .5, .5, 1., 4., 0., 3.],
        ], dtype=torch.float64)
        columns = ("planet_log_odds", "log_bf_planet_noise", "log_bf_binary_noise", "binary_secondary_ratio",
                   "binary_secondary_phase", "binary_odd_even_ratio", "binary_period_days", "ingress_fraction")
        self.peaks = []
        for row in self.table.tolist():
            peak = {"period_days": row[0], "epoch_bjd": row[1], "duration_hours": 24 * row[2], "snr": row[3],
                    "snr_folded": row[3], "depth": row[4], "depth_ppm": 1e6 * row[4], "events": 3, "family": 0, "pass": 1}
            peak.update(dict(zip(columns, row[5:13])))
            peak["model_preferred"] = row[5] >= 0.
            peak["binary_diagnostics_complete"] = True
            self.peaks.append(peak)
        self.result = TransitSearchResult(self.table, {}, SearchConfig(), self.peaks,
                                          [{"epoch_bjd": 5., "duration_hours": 3., "snr": 8., "depth": .001,
                                            "depth_ppm": 1000., "points": 7}])

    def test_rankings_order_resolved_peaks(self):
        planet = self.result.candidates(2)
        signal = self.result.candidates(2, ranking="signal")
        self.assertEqual([r["period_days"] for r in planet], [2., 3.])
        self.assertEqual([r["period_days"] for r in signal], [4., 2.])
        self.assertEqual([r["period_days"] for r in self.result.candidates(3)], [2., 3., 4.])

    def test_schema_carries_candidate_and_event_lists(self):
        value = self.result.to_dict()
        self.assertEqual(value["schema_version"], 3)
        self.assertIn("planet_log_odds", value["candidates"][0])
        self.assertIn("signal_candidates", value)
        self.assertEqual(len(value["single_events"]), 1)
        json.dumps(value, allow_nan=False)

    def test_rejects_unknown_rankings(self):
        with self.assertRaises(ValueError):
            self.result.candidates(ranking="unknown")

    def test_unresolved_diagnostics_serialize_as_null(self):
        self.peaks[0]["binary_secondary_ratio"] = None
        self.peaks[0]["binary_diagnostics_complete"] = False
        row = self.result.candidates(1)[0]
        self.assertIsNone(row["binary_secondary_ratio"])
        self.assertFalse(row["binary_diagnostics_complete"])
        json.dumps(self.result.to_dict(), allow_nan=False)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input = self.root / "input.npz"
        np.savez(self.input, time_bjd=np.arange(4.), flux=np.zeros(4), flux_err=np.ones(4))

    def tearDown(self):
        self.temporary.cleanup()

    def invoke_error(self, extra, gpu=False):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                patch.object(cli.torch.cuda, "is_available", return_value=gpu), self.assertRaises(SystemExit) as exit:
            cli.main(["--input", str(self.input), *extra])
        self.assertEqual(out.getvalue(), "")
        return exit.exception.code, err.getvalue()

    def test_reports_cuda_requirement_without_stdout_data(self):
        code, error = self.invoke_error([])
        self.assertEqual(code, 1)
        self.assertIn("CUDA", error)

    def test_config_error_uses_exit_two(self):
        code, error = self.invoke_error(["--pmin", "10", "--pmax", "1"])
        self.assertEqual(code, 2)
        self.assertIn("period_min", error)

    def test_rejects_input_overwrite(self):
        before = self.input.read_bytes()
        code, _ = self.invoke_error(["--output", str(self.input)])
        self.assertEqual(code, 2)
        self.assertEqual(self.input.read_bytes(), before)

    def test_rejects_output_collision(self):
        target = str(self.root / "same.npz")
        code, _ = self.invoke_error(["--output", target, "--save-table", target])
        self.assertEqual(code, 2)

    def test_prepared_input_requires_errors(self):
        np.savez(self.input, time_bjd=np.arange(4.), flux=np.zeros(4))
        code, error = self.invoke_error(["--prepared"], gpu=True)
        self.assertEqual(code, 2)
        self.assertIn("flux_err", error)

    def test_json_stdout_and_float64_table_export(self):
        class FixtureResult:
            table = torch.tensor([[2., 2457000., .1, 7. + 1e-10, .001, 2., 25., 20.,
                                   .1, .5, 1., 2., .2, 3.]], dtype=torch.float64)
            def to_dict(self):
                return {"schema_version": 2, "candidates": [{"period_days": 2.}]}
        target = self.root / "nested/table.npz"
        out = io.StringIO()
        with patch.object(cli.torch.cuda, "is_available", return_value=True), \
                patch.object(cli, "search_prepared", return_value=FixtureResult()) as search, \
                contextlib.redirect_stdout(out):
            cli.main(["--input", str(self.input), "--prepared", "--save-table", str(target)])
        search.assert_called_once()
        result = json.loads(out.getvalue())
        self.assertEqual(result["input_mode"], "prepared")
        with np.load(target, allow_pickle=False) as table:
            self.assertEqual(set(table.files), set(cli.TABLE_COLUMNS))
            self.assertTrue(all(table[key].dtype == np.float64 for key in table.files))
            self.assertGreater(float(table["snr"][0]), 7.)
        self.assertFalse(list(self.root.rglob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
