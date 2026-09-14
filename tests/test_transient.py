import bootstrap  # noqa: F401

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.cli import main
from namd_analysis.observables import (
    CLASSES,
    COUNTERFACTUAL,
    MODEL_INFERRED,
    OBSERVED,
    ObservableClassError,
    uniform,
    validate,
)
from namd_analysis.transient import (
    TransientError,
    Window,
    analyze,
    compare_regimes,
    flat_group_fields,
    parse_window,
    window_metrics,
)
from synthetic import kinetic_config

GROUPS = ["CBM", "BCF", "PCBM", "VBM"]


def _spike_then_flat(nsteps=1000, span_ns=10.0):
    """PCBM spikes early, falls back, then sits flat while VBM grows.

    The late window alone would report PCBM as unchanging.
    """
    time = np.linspace(0.0, span_ns, nsteps)
    pcbm = 0.6 * np.exp(-time / 0.05) * (1.0 - np.exp(-time / 0.01)) + 0.05
    vbm = 0.7 * (1.0 - np.exp(-time / 3.0))
    bcf = 0.2 * np.exp(-time / 2.0)
    cbm = np.clip(1.0 - pcbm - vbm - bcf, 0.0, None)
    total = cbm + bcf + pcbm + vbm
    stack = np.column_stack([cbm, bcf, pcbm, vbm]) / total[:, None]
    return time, stack


def _write_shprop(root, time_ns, populations, n_files=3, noise=0.0, seed=0):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    time_fs = time_ns * 1e6
    paths = []
    for index in range(n_files):
        values = populations.copy()
        if noise:
            values = np.clip(values + noise * rng.standard_normal(values.shape), 0, None)
            values = values / values.sum(axis=1, keepdims=True)
        table = np.column_stack([time_fs, np.full(time_ns.size, -1.5), values])
        path = root / f"SHPROP.{index + 1}"
        np.savetxt(path, table)
        paths.append(path)
    return paths


class WindowParsingTests(unittest.TestCase):
    def test_named_and_bare_forms(self):
        self.assertEqual(parse_window("early=0:0.1").name, "early")
        self.assertEqual(parse_window("early=0:0.1").start_ns, 0.0)
        self.assertEqual(parse_window("early=0:0.1").end_ns, 0.1)
        self.assertEqual(parse_window("1:5").start_ns, 1.0)

    def test_open_ends_resolve_to_the_data(self):
        time = np.linspace(0.0, 10.0, 101)
        self.assertEqual(parse_window("late=5:").resolve(time), (5.0, 10.0))
        self.assertEqual(parse_window("early=:2").resolve(time), (0.0, 2.0))
        self.assertEqual(Window("full").resolve(time), (0.0, 10.0))

    def test_malformed_windows_are_rejected(self):
        for text in ("0.1", "a:b", "5:1"):
            with self.assertRaises(TransientError):
                parse_window(text)

    def test_empty_window_is_rejected(self):
        time = np.linspace(0.0, 10.0, 101)
        with self.assertRaises(TransientError):
            Window("x", 5.0, 5.0).resolve(time)

    def test_window_with_too_few_samples_is_rejected(self):
        time = np.linspace(0.0, 10.0, 11)
        with self.assertRaises(TransientError) as ctx:
            Window("tiny", 0.0, 0.5).mask(time)
        self.assertIn("at least two", str(ctx.exception))

    def test_duplicate_window_names_are_rejected(self):
        time = np.linspace(0.0, 10.0, 101)
        values = np.linspace(1.0, 0.0, 101)
        with self.assertRaises(TransientError):
            analyze(time, [("A", values)], [Window("w", 0, 5), Window("w", 5, 10)])


class MetricTests(unittest.TestCase):
    def test_integral_of_a_constant_is_height_times_span(self):
        time = np.linspace(0.0, 4.0, 401)
        metrics = window_metrics(time, np.full(401, 0.25), "A", Window("full"))
        self.assertAlmostEqual(metrics.integrated_population_ns, 1.0, places=9)
        self.assertAlmostEqual(metrics.mean_population, 0.25, places=9)
        self.assertAlmostEqual(metrics.net_change, 0.0, places=12)

    def test_peak_and_its_time_are_reported(self):
        time = np.linspace(0.0, 4.0, 401)
        values = np.exp(-((time - 1.5) ** 2) / 0.05)
        metrics = window_metrics(time, values, "A", Window("full"))
        self.assertAlmostEqual(metrics.peak_population, 1.0, places=6)
        self.assertAlmostEqual(metrics.peak_time_ns, 1.5, places=2)

    def test_window_restricts_the_samples(self):
        time = np.linspace(0.0, 10.0, 1001)
        values = np.exp(-((time - 1.0) ** 2) / 0.02)
        full = window_metrics(time, values, "A", Window("full"))
        late = window_metrics(time, values, "A", Window("late", 5.0, 10.0))
        self.assertAlmostEqual(full.peak_population, 1.0, places=4)
        self.assertLess(late.peak_population, 1e-6)

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaises(TransientError):
            window_metrics(np.arange(10.0), np.arange(5.0), "A", Window("full"))

    def test_flat_fields_use_the_group_name(self):
        time = np.linspace(0.0, 4.0, 401)
        metrics = analyze(time, [("PCBM", np.full(401, 0.3))], [Window("full")])
        flat = flat_group_fields(metrics, "PCBM", "full")
        self.assertAlmostEqual(flat["pcbm_peak"], 0.3)
        self.assertIn("pcbm_integrated_population_ns", flat)
        self.assertIn("pcbm_peak_time_ns", flat)

    def test_every_metric_is_classified_as_observed(self):
        from namd_analysis.transient import METRIC_CLASSES

        self.assertTrue(METRIC_CLASSES)
        for value in METRIC_CLASSES.values():
            self.assertEqual(value, OBSERVED)


class EarlySpikeTests(unittest.TestCase):
    """The reason this module exists."""

    def test_the_early_spike_survives_a_late_only_view(self):
        time, stack = _spike_then_flat()
        pcbm = stack[:, 2]
        series = [("PCBM", pcbm)]
        early = window_metrics(time, pcbm, "PCBM", Window("transient", 0.0, 0.5))
        late = window_metrics(time, pcbm, "PCBM", Window("late", 1.0, 10.0))

        # The late window really does look flat.
        self.assertLess(abs(late.net_change), 0.01)
        self.assertLess(late.peak_population - late.minimum_population, 0.01)
        # The early window does not, and the full-trajectory view keeps it.
        self.assertGreater(early.peak_population, 3 * late.peak_population)
        full = window_metrics(time, pcbm, "PCBM", Window("full"))
        self.assertAlmostEqual(
            full.peak_population, early.peak_population, places=6
        )
        self.assertLess(full.peak_time_ns, 0.5)
        del series

    def test_regime_comparison_flags_the_missed_transient(self):
        time, stack = _spike_then_flat()
        series = [(name, stack[:, i]) for i, name in enumerate(GROUPS)]
        payload = compare_regimes(
            time, series, Window("transient", 0.0, 0.5), Window("late", 1.0, 10.0)
        )
        record = next(r for r in payload["records"] if r["group"] == "PCBM")
        self.assertGreater(record["transient_peak_population"], 0.3)
        self.assertLess(abs(record["late_net_change"]), 0.02)
        self.assertGreater(record["peak_ratio_transient_over_late"], 1.5)
        self.assertTrue(
            any("must not be described" in note for note in payload["notes"])
        )

    def test_integrated_population_is_not_called_a_flux(self):
        time, stack = _spike_then_flat()
        series = [(name, stack[:, i]) for i, name in enumerate(GROUPS)]
        payload = compare_regimes(
            time, series, Window("transient", 0.0, 0.5), Window("late", 1.0, 10.0)
        )
        for value in payload["observable_class"].values():
            self.assertEqual(value, OBSERVED)


class ObservableClassTests(unittest.TestCase):
    def test_the_three_classes_are_distinct(self):
        self.assertEqual(len(set(CLASSES)), 3)
        self.assertIn(OBSERVED, CLASSES)
        self.assertIn(MODEL_INFERRED, CLASSES)
        self.assertIn(COUNTERFACTUAL, CLASSES)

    def test_validate_rejects_an_unknown_class(self):
        with self.assertRaises(ObservableClassError):
            validate({"x": "measured"})
        self.assertEqual(validate({"x": OBSERVED}), {"x": OBSERVED})

    def test_uniform_rejects_an_unknown_class(self):
        with self.assertRaises(ObservableClassError):
            uniform(["a"], "measured")
        self.assertEqual(uniform(["a", "b"], OBSERVED), {"a": OBSERVED, "b": OBSERVED})


class TransientCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.time, self.stack = _spike_then_flat(nsteps=600)
        _write_shprop(self.root / "run", self.time, self.stack, n_files=3)
        self.config = self.root / "map.json"
        self.config.write_text(
            json.dumps(kinetic_config(GROUPS, recombined="VBM")), encoding="utf-8"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _report(self, out):
        return json.loads((out / "report.json").read_text(encoding="utf-8"))

    def test_windows_and_regime_table_are_written(self):
        out = self.root / "out"
        code = main(
            [
                "transient-populations",
                "--files", str(self.root / "run" / "SHPROP.*"),
                "--config", str(self.config),
                "--transient-window", "0:0.5",
                "--late-window", "1:10",
                "--out", str(out),
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue((out / "transient_populations.csv").is_file())
        self.assertTrue((out / "regime_comparison.csv").is_file())
        report = self._report(out)
        self.assertEqual(len(report["windows"]), 2)
        pcbm_transient = next(
            m for m in report["metrics"]
            if m["group"] == "PCBM" and m["window"] == "transient"
        )
        self.assertGreater(pcbm_transient["peak_population"], 0.3)
        self.assertTrue(report["regime_comparison"]["notes"])
        for value in report["observable_class"].values():
            self.assertEqual(value, OBSERVED)

    def test_repeatable_named_windows(self):
        out = self.root / "out_named"
        code = main(
            [
                "transient-populations",
                "--files", str(self.root / "run" / "SHPROP.*"),
                "--config", str(self.config),
                "--window", "first=0:1",
                "--window", "second=1:5",
                "--window", "rest=5:",
                "--out", str(out),
            ]
        )
        self.assertEqual(code, 0)
        names = {w["name"] for w in self._report(out)["windows"]}
        self.assertEqual(names, {"first", "second", "rest"})

    def test_default_is_the_whole_trajectory(self):
        out = self.root / "out_full"
        self.assertEqual(
            main([
                "transient-populations",
                "--files", str(self.root / "run" / "SHPROP.*"),
                "--config", str(self.config),
                "--out", str(out),
            ]),
            0,
        )
        report = self._report(out)
        self.assertEqual(len(report["windows"]), 1)
        self.assertEqual(report["windows"][0]["name"], "full")

    def test_one_regime_window_without_the_other_is_refused(self):
        with self.assertRaises(SystemExit):
            main([
                "transient-populations",
                "--files", str(self.root / "run" / "SHPROP.*"),
                "--config", str(self.config),
                "--transient-window", "0:1",
                "--out", str(self.root / "out_half"),
            ])

    def test_malformed_window_is_reported(self):
        code = main([
            "transient-populations",
            "--files", str(self.root / "run" / "SHPROP.*"),
            "--config", str(self.config),
            "--window", "bad=9:1",
            "--out", str(self.root / "out_bad"),
        ])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
