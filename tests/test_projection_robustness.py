"""The projection-robustness audit (Issue #4).

The central requirement: the audit must tell apart four behaviours that a
normalized fraction alone conflates --

* genuinely mixed raw weights,
* tiny raw weights that normalization inflates into apparent mixing,
* high-capture pure states,
* low-capture states with no mixing at all --

and it must do so without discarding, repairing or reweighting a sample.
"""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from robustness_fixtures import (
    GROUP_NAMES,
    SCENARIO_BAND,
    scenario_raw,
    write_campaign,
)

from namd_analysis.crossings import read_projection_character
from namd_analysis.dispatch import main as dispatch_main
from namd_analysis.projection_robustness import (
    UNCAPTURED,
    ProjectionTable,
    RobustnessError,
    allocation_class,
    audit,
    automatic_marks,
    mixing_levels,
    parse_grid,
    parse_mark,
    sample_header,
    sample_rows,
    sensitivity_rows,
)


def _table_from_raw(raw, frames=None, bands=None, total=None):
    """A ProjectionTable straight from raw weights (nf, nb, ng)."""
    raw = np.asarray(raw, dtype=float)
    captured = raw.sum(axis=2)
    nf, nb, _ = raw.shape
    return ProjectionTable(
        frames=frames if frames is not None else np.arange(1, nf + 1),
        bands=bands if bands is not None else np.arange(1, nb + 1),
        groups=list(GROUP_NAMES),
        weights=raw / captured[..., None],
        captured=captured,
        total=captured if total is None else total,
    )


def _rows(result):
    header = sample_header(result.table.groups)
    return [dict(zip(header, row)) for row in sample_rows(result.table, result.pair, result.arrays)]


def _run(argv):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = dispatch_main(argv)
    return code, stream.getvalue()


class FourScenarioTests(unittest.TestCase):
    """Each scenario on its own band, over several frames, through the real CSV."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.campaign = write_campaign(Path(cls.tmp.name), scenario_raw())
        series = read_projection_character(cls.campaign["projection_character"])
        cls.table = ProjectionTable.from_character_series(series)
        cls.result = audit(cls.table, quality_threshold=0.5)
        cls.rows = _rows(cls.result)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def band_rows(self, name):
        band = SCENARIO_BAND[name]
        return [r for r in self.rows if r["band"] == band]

    def band_summary(self, name):
        return self.result.band(SCENARIO_BAND[name])

    def threshold_entry(self, name, tau=0.10):
        for entry in self.band_summary(name)["mixing_by_threshold"]:
            if abs(entry["threshold"] - tau) < 1e-12:
                return entry
        raise AssertionError(tau)

    def test_genuinely_mixed_raw_weights_survive_every_test(self):
        rows = self.band_rows("genuinely_mixed")
        for r in rows:
            self.assertGreaterEqual(r["pair_min_normalized"], 0.10)
            self.assertGreaterEqual(r["pair_min_worst_case"], 0.25)
            self.assertNotEqual(r["dominant_before_normalization"], UNCAPTURED)
            self.assertFalse(r["dominance_changed_by_normalization"])
        entry = self.threshold_entry("genuinely_mixed")
        self.assertEqual(entry["n_mixed_normalized"], len(rows))
        self.assertEqual(entry["n_normalized_mixed_that_survive_worst_case"], len(rows))
        self.assertEqual(entry["n_mixed_under_every_allocation"], len(rows))

    def test_tiny_raw_weights_are_mixed_only_after_normalization(self):
        rows = self.band_rows("tiny_but_normalized")
        for r in rows:
            # Normalization makes it look strongly mixed ...
            self.assertGreaterEqual(r["pair_min_normalized"], 0.30)
            # ... but the pair holds under three percent of the band ...
            self.assertLess(r["pair_min_worst_case"], 0.03)
            # ... and more of the band is uncaptured than any fragment holds.
            self.assertEqual(r["dominant_before_normalization"], UNCAPTURED)
            self.assertIn(r["dominant_after_normalization"], ("BCF", "PCBM"))
            self.assertTrue(r["dominance_changed_by_normalization"])
            self.assertLess(r["captured_projection"], 0.06)
        entry = self.threshold_entry("tiny_but_normalized")
        self.assertEqual(entry["n_mixed_normalized"], len(rows))
        self.assertEqual(entry["n_normalized_mixed_that_survive_worst_case"], 0)
        self.assertEqual(entry["n_mixed_under_every_allocation"], 0)
        self.assertEqual(entry["n_mixed_under_some_allocation"], len(rows))

    def test_high_capture_pure_state_is_unmixed_under_any_allocation(self):
        rows = self.band_rows("high_capture_pure")
        for r in rows:
            self.assertEqual(r["pair_min_normalized"], 0.0)
            self.assertEqual(r["dominant_before_normalization"], "perovskite")
            self.assertEqual(r["dominant_after_normalization"], "perovskite")
            self.assertEqual(
                allocation_class(r["pair_min_worst_case"], r["pair_min_best_case"], 0.10),
                "not_mixed_under_any_allocation",
            )
        summary = self.band_summary("high_capture_pure")
        self.assertEqual(summary["capture"]["samples_below_existing_threshold"], 0)
        for entry in summary["mixing_by_threshold"]:
            self.assertEqual(entry["n_mixed_normalized"], 0)

    def test_low_capture_without_mixing_is_not_called_mixed(self):
        rows = self.band_rows("low_capture_unmixed")
        summary = self.band_summary("low_capture_unmixed")
        for r in rows:
            self.assertEqual(r["pair_min_normalized"], 0.0)
            self.assertEqual(r["dominant_after_normalization"], "perovskite")
            self.assertEqual(r["dominant_before_normalization"], UNCAPTURED)
            # Low capture leaves room for mixing the projection cannot exclude;
            # the audit says so, and still does not call the sample mixed.
            self.assertEqual(
                allocation_class(r["pair_min_worst_case"], r["pair_min_best_case"], 0.10),
                "mixed_under_some_allocation",
            )
        self.assertEqual(summary["capture"]["samples_below_existing_threshold"], len(rows))
        for entry in summary["mixing_by_threshold"]:
            self.assertEqual(entry["n_mixed_normalized"], 0)

    def test_every_sample_carries_its_allocation_class(self):
        header = sample_header(self.table.groups, [0.10])
        self.assertEqual(header[-1], "allocation_class_at_0.1")
        rows = [dict(zip(header, row)) for row in
                sample_rows(self.table, self.result.pair, self.result.arrays, [0.10])]
        expected = {
            "genuinely_mixed": "mixed_under_every_allocation",
            "tiny_but_normalized": "mixed_under_some_allocation",
            "high_capture_pure": "not_mixed_under_any_allocation",
            "low_capture_unmixed": "mixed_under_some_allocation",
        }
        for name, cls in expected.items():
            classes = {r["allocation_class_at_0.1"] for r in rows
                       if r["band"] == SCENARIO_BAND[name]}
            self.assertEqual(classes, {cls}, name)

    def test_four_scenarios_are_distinct_on_the_audited_axes(self):
        signature = {}
        for name in SCENARIO_BAND:
            r = self.band_rows(name)[0]
            signature[name] = (
                r["pair_min_normalized"] >= 0.10,
                r["pair_min_worst_case"] >= 0.10,
                r["captured_projection"] >= 0.5,
            )
        self.assertEqual(len(set(signature.values())), 4, signature)

    def test_raw_minimum_grid_separates_genuine_from_inflated(self):
        rows = sensitivity_rows(
            self.table, self.result.pair, self.result.arrays, [0.10], [0.0], [0.0, 0.03]
        )
        counts = {(r[0], r[3]): r[7] for r in rows}
        n = len(self.table.frames)
        genuine, tiny = SCENARIO_BAND["genuinely_mixed"], SCENARIO_BAND["tiny_but_normalized"]
        self.assertEqual(counts[(genuine, 0.0)], n)
        self.assertEqual(counts[(genuine, 0.03)], n)
        self.assertEqual(counts[(tiny, 0.0)], n)
        self.assertEqual(counts[(tiny, 0.03)], 0)

    def test_capture_condition_is_reported_not_applied_silently(self):
        rows = sensitivity_rows(
            self.table, self.result.pair, self.result.arrays, [0.10], [0.0, 0.1], [0.0]
        )
        tiny = SCENARIO_BAND["tiny_but_normalized"]
        by_cmin = {r[2]: r for r in rows if r[0] == tiny}
        n = len(self.table.frames)
        self.assertEqual(by_cmin[0.0][5], n)  # all samples meet c_min = 0
        self.assertEqual(by_cmin[0.0][7], n)
        self.assertEqual(by_cmin[0.1][6], n)  # every one set aside, and counted
        self.assertEqual(by_cmin[0.1][7], 0)
        for row in rows:
            self.assertEqual(row[5] + row[6], row[4])

    def test_every_sample_is_in_the_sample_table(self):
        self.assertEqual(len(self.rows), len(self.table.frames) * len(self.table.bands))
        low = [r for r in self.rows if r["captured_projection"] < 0.5]
        self.assertTrue(low)

    def test_raw_weights_reconstruct_the_procar_sums(self):
        raw = scenario_raw()
        index = {name: i for i, name in enumerate(self.table.groups)}
        for r in self.rows:
            expected = raw[r["frame"]][r["band"]]
            for name, value in zip(GROUP_NAMES, expected):
                # PROCAR values are printed to six decimals per ion.
                self.assertAlmostEqual(r[f"raw_{name}"], value, places=5)
            self.assertIn(r["dominant_raw_declared_only"], index)

    def test_series_and_csv_give_the_same_audit(self):
        from_series = ProjectionTable.from_projection_series(self.campaign["series"])
        np.testing.assert_allclose(from_series.raw_weights(), self.table.raw_weights(), atol=1e-12)
        np.testing.assert_allclose(
            self.campaign["series"].raw_weights(), self.table.raw_weights(), atol=1e-12
        )
        other = audit(from_series, quality_threshold=0.5)
        self.assertEqual(
            [b["mixing_by_threshold"] for b in other.per_band],
            [b["mixing_by_threshold"] for b in self.result.per_band],
        )

    def test_total_projection_is_read_from_the_csv(self):
        self.assertIsNotNone(self.table.total)
        np.testing.assert_allclose(self.table.total, self.table.captured)
        self.assertEqual(self.result.campaign["declared_argmax_invariance_violations"], 0)


class MixingLevelTests(unittest.TestCase):
    def test_worst_normalized_best_are_ordered(self):
        rng = np.random.default_rng(3)
        raw = rng.dirichlet(np.ones(4), size=500)[:, :3] * rng.uniform(0.05, 1.0, (500, 1))
        levels = mixing_levels(raw[:, 1], raw[:, 2], raw.sum(axis=1))
        self.assertTrue(np.all(levels["worst_case"] <= levels["normalized"] + 1e-15))
        self.assertTrue(np.all(levels["normalized"] <= levels["best_case"] + 1e-15))

    def test_best_case_matches_brute_force(self):
        rng = np.random.default_rng(11)
        for _ in range(200):
            w_a, w_b, other = rng.uniform(0, 0.3, 3)
            captured = w_a + w_b + other
            budget = 1.0 - captured
            share = np.linspace(0.0, budget, 2001)
            brute = np.max(np.minimum(w_a + share, w_b + (budget - share)))
            best = mixing_levels(np.array(w_a), np.array(w_b), np.array(captured))["best_case"]
            self.assertAlmostEqual(float(best), float(brute), places=3)

    def test_capture_above_one_leaves_bracket_undefined(self):
        levels = mixing_levels(np.array(0.6), np.array(0.5), np.array(1.2))
        self.assertTrue(np.isnan(levels["best_case"]))
        self.assertEqual(allocation_class(0.5, float(levels["best_case"]), 0.1), "bracket_undefined")

    def test_pair_balance_is_normalization_invariant(self):
        levels = mixing_levels(np.array(0.02), np.array(0.04), np.array(0.1))
        self.assertAlmostEqual(float(levels["balance"]), 0.5)


class QuestionTests(unittest.TestCase):
    """'Mixed only when the absolute weight is small?' -- both answers must be visible."""

    def _band_table(self, mixed_at_low_capture_only):
        rng = np.random.default_rng(5)
        n = 400
        raw = np.zeros((n, 1, 3))
        for i in range(n):
            captured = rng.uniform(0.40, 0.60)
            if mixed_at_low_capture_only:
                # Mixed only well inside the lowest capture quartile.
                pair = 0.03 if captured < 0.44 else 0.0
            else:
                pair = 0.12
            raw[i, 0] = (captured - 2 * pair, pair, pair)
        return _table_from_raw(raw, bands=[981])

    def test_artefact_like_band_mixes_only_in_lowest_stratum_and_fails_raw(self):
        result = audit(self._band_table(True))
        entry = [e for e in result.band(981)["mixing_by_threshold"] if e["threshold"] == 0.05][0]
        self.assertGreater(entry["n_mixed_normalized"], 0)
        self.assertEqual(entry["n_mixed_in_lowest_capture_stratum"], entry["n_mixed_normalized"])
        self.assertEqual(entry["n_normalized_mixed_that_survive_worst_case"], 0)
        self.assertLess(entry["median_capture_mixed"], entry["median_capture_not_mixed"])
        spearman = result.band(981)["correlations"]["capture_vs_pair_min_normalized"]["spearman"]
        self.assertLess(spearman, -0.5)

    def test_genuine_band_mixes_in_every_stratum_and_survives_raw(self):
        result = audit(self._band_table(False))
        entry = [e for e in result.band(981)["mixing_by_threshold"] if e["threshold"] == 0.10][0]
        self.assertEqual(entry["n_mixed_normalized"], 400)
        self.assertEqual(entry["n_normalized_mixed_that_survive_worst_case"], 400)
        for stratum in result.band(981)["capture_strata"]:
            self.assertEqual(stratum["mixed_normalized"]["0.1"], stratum["n_samples"])

    def test_constant_series_gives_null_correlation_with_reason(self):
        raw = np.tile(np.array([[[0.4, 0.05, 0.05]]]), (10, 1, 1))
        result = audit(_table_from_raw(raw))
        record = result.band(1)["correlations"]["capture_vs_normalized"]["BCF"]
        self.assertIsNone(record["spearman"])
        self.assertIn("constant", record["undefined_because"])


class InputTests(unittest.TestCase):
    def test_pair_must_name_two_table_groups(self):
        table = _table_from_raw(np.full((2, 1, 3), 0.2))
        with self.assertRaises(RobustnessError):
            audit(table, pair=("BCF", "fullerene"))
        with self.assertRaises(RobustnessError):
            audit(table, pair=("BCF", "BCF"))

    def test_grid_parsing(self):
        self.assertEqual(parse_grid("0.1,0.05,0.1", (0.2,), "x"), [0.05, 0.1])
        self.assertEqual(parse_grid("0.40:0.46:0.02", (0.2,), "x"), [0.4, 0.42, 0.44, 0.46])
        self.assertEqual(parse_grid(None, (0.2, 0.1), "x"), [0.1, 0.2])
        with self.assertRaises(RobustnessError):
            parse_grid("0.1,-1", (0.2,), "x")
        with self.assertRaises(RobustnessError):
            parse_grid("0.5:0.1:0.1", (0.2,), "x")

    def test_zero_normalized_threshold_is_refused(self):
        with self.assertRaises(RobustnessError):
            audit(_table_from_raw(np.full((2, 1, 3), 0.2)), normalized_thresholds=[0.0])

    def test_existing_threshold_and_zero_are_always_capture_grid_points(self):
        result = audit(_table_from_raw(np.full((2, 1, 3), 0.2)),
                       quality_threshold=0.37, capture_minima=[0.45])
        self.assertEqual(result.grids["capture_minima"], [0.0, 0.37, 0.45])

    def test_mark_parsing(self):
        mark = parse_mark("issue4_low_capture=1273,1286,1302")
        self.assertEqual(mark.frames, [1273, 1286, 1302])
        for bad in ("nolabel", "=12", "bad label=1", "x=a"):
            with self.assertRaises(RobustnessError):
                parse_mark(bad)

    def test_automatic_marks_follow_their_rules(self):
        captured = np.array([0.5, 0.2, 0.8, 0.4, 0.6])
        raw = np.stack([captured - 0.02, np.full(5, 0.01), np.full(5, 0.01)], axis=1)[:, None, :]
        raw[3, 0] = (0.3, 0.05, 0.05)
        table = _table_from_raw(raw, frames=[10, 20, 30, 40, 50])
        marks = {m.label: m for m in automatic_marks(table, audit(table).arrays, 1, count=1)}
        self.assertEqual(marks["lowest_capture"].frames, [20])
        self.assertEqual(marks["highest_capture"].frames, [30])
        self.assertEqual(marks["median_capture"].frames, [10])
        self.assertEqual(marks["most_mixed_normalized"].frames, [40])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        raw = {}
        rng = np.random.default_rng(2)
        for frame in range(1, 41):
            captured = 0.45 + 0.02 * np.sin(frame / 4.0)
            pair = 0.03 if frame in (12, 13, 14) else rng.uniform(0.0, 0.01)
            raw[frame] = {
                980: (0.0, 0.0, 0.51),
                981: (captured - 2 * pair, pair, pair),
            }
        self.campaign = write_campaign(self.root, raw, decimals=3)
        profile = {
            "configurations": {
                "A": {"episodes": {"crossing_A": "20:24"}, "control_episodes": {}},
                "B": {"episodes": {"crossing_B1": "30:32"},
                      "control_episodes": {"closest_B": "5:6"}},
            }
        }
        self.profile = self.root / "profile.json"
        self.profile.write_text(json.dumps(profile))

    def tearDown(self):
        self.tmp.cleanup()

    def test_focus_report_marks_only_this_configurations_windows(self):
        out = self.root / "robust"
        code, text = _run([
            "character-robustness",
            "--projection-character", str(self.campaign["projection_character"]),
            "--focus-band", "981",
            "--mark", "issue4_low_capture=12,13,14",
            "--profile", str(self.profile), "--configuration", "A",
            "--vasp-dir", str(self.root / "production" / "0001"),
            "--projection-manifest", str(self.campaign["manifest"]),
            "--atom-groups", str(self.campaign["atom_groups"]),
            "--out", str(out),
        ])
        self.assertEqual(code, 0, text)
        for name in ("robustness_samples.csv", "robustness_by_band.csv",
                     "mixing_sensitivity.csv", "robustness_summary.json",
                     "band_981_focus.csv", "band_981_focus.json", "band_981_focus.png",
                     "band_981_mixing_sensitivity.png", "procar_resolution.csv"):
            self.assertTrue((out / name).is_file(), name)
        summary = json.loads((out / "robustness_summary.json").read_text())
        self.assertEqual([w["name"] for w in summary["windows"]], ["crossing_A"])
        self.assertEqual(summary["projection_method"]["method"], "paw_projectors")
        self.assertEqual(summary["projection_method"]["LORBIT"], 11)
        self.assertFalse(summary["projection_method"]["rwigs_sweep_tests_these_weights"])
        self.assertEqual(summary["existing_reporting_threshold"]["value"], 0.5)
        focus = json.loads((out / "band_981_focus.json").read_text())
        labels = [m["label"] for m in focus["marks"]]
        self.assertIn("issue4_low_capture", labels)
        self.assertIn("highest_capture", labels)
        issue = [m for m in focus["marks"] if m["label"] == "issue4_low_capture"][0]
        self.assertEqual([f["frame"] for f in issue["frames"]], [12, 13, 14])
        with (out / "band_981_focus.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 40)
        self.assertEqual({r["windows"] for r in rows if 20 <= int(r["frame"]) <= 24},
                         {"crossing_A"})
        self.assertTrue(all("crossing_B1" not in r["windows"] for r in rows))
        with (out / "robustness_samples.csv").open() as handle:
            self.assertEqual(sum(1 for _ in handle) - 1, 80)
        self.assertEqual(summary["procar_resolution"]["frames"],
                         sorted({f for m in focus["marks"] for f in
                                 [x["frame"] for x in m["frames"]]}))

    def test_configuration_b_gets_b_windows_and_its_control(self):
        out = self.root / "robust_b"
        code, text = _run([
            "character-robustness",
            "--projection-character", str(self.campaign["projection_character"]),
            "--focus-band", "981", "--no-figures",
            "--profile", str(self.profile), "--configuration", "B",
            "--out", str(out),
        ])
        self.assertEqual(code, 0, text)
        focus = json.loads((out / "band_981_focus.json").read_text())
        roles = {w["name"]: w["role"] for w in focus["windows"]}
        self.assertEqual(roles, {"crossing_B1": "crossing", "closest_B": "control"})
        summary = json.loads((out / "robustness_summary.json").read_text())
        self.assertEqual(summary["projection_method"]["method"], "undetermined")
        self.assertIsNone(summary["projection_method"]["rwigs_sweep_tests_these_weights"])

    def test_lorbit_is_read_from_a_manifest_frame_when_no_directory_is_named(self):
        out = self.root / "robust_default"
        code, text = _run([
            "character-robustness",
            "--projection-character", str(self.campaign["projection_character"]),
            "--focus-band", "981", "--no-figures",
            "--projection-manifest", str(self.campaign["manifest"]),
            "--atom-groups", str(self.campaign["atom_groups"]),
            "--out", str(out),
        ])
        self.assertEqual(code, 0, text)
        method = json.loads((out / "robustness_summary.json").read_text())["projection_method"]
        self.assertEqual(method["method"], "paw_projectors")
        self.assertTrue(method["searched_directory"].endswith("0001"))

    def test_the_input_table_is_never_modified(self):
        path = self.campaign["projection_character"]
        before = path.read_bytes()
        code, text = _run([
            "character-robustness", "--projection-character", str(path),
            "--focus-band", "981", "--no-figures", "--out", str(self.root / "r"),
        ])
        self.assertEqual(code, 0, text)
        self.assertEqual(path.read_bytes(), before)

    def test_refusals(self):
        base = ["character-robustness", "--projection-character",
                str(self.campaign["projection_character"]), "--no-figures"]
        cases = [
            ["--focus-band", "981", "--mark", "x=999"],
            ["--focus-band", "977"],
            ["--configuration", "A"],
            ["--profile", str(self.profile), "--configuration", "Z"],
            ["--episode", "outside=500:510"],
            ["--pair", "BCF,fullerene"],
            ["--projection-manifest", str(self.campaign["manifest"]), "--focus-band", "981"],
        ]
        for i, extra in enumerate(cases):
            code, text = _run(base + extra + ["--out", str(self.root / f"bad{i}")])
            self.assertEqual(code, 2, (extra, text))
            self.assertIn("error:", text)

    def test_listed_in_help(self):
        code, text = _run(["--help"])
        self.assertEqual(code, 0)
        for command in ("character-robustness", "character-fullspace-prepare",
                        "character-fullspace-compare", "character-ensemble"):
            self.assertIn(command, text)


if __name__ == "__main__":
    unittest.main()
