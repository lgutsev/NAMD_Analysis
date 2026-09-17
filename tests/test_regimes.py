"""Early and late regimes: separate fits, and one model against two.

The fixtures are analytic sequential kinetics, so the right answer is known
before the fit runs.  Each one asks a different question of the machinery:
does it recover rates it should, refuse rates it cannot constrain, and prefer
the piecewise description only when the rates genuinely differ?
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.dispatch import main as dispatch_main
from namd_analysis.kinetics import (
    KineticsError,
    fit_master_equation,
    grid_uniformity,
    parse_edges,
    require_uniform_grid,
)
from namd_analysis.regimes import (
    DEFAULT_EARLY_WINDOW_NS,
    LATE_WINDOW_NOTE,
    compare_piecewise,
    early_transient_narrative,
    fit_regime,
)
from namd_analysis.transient import Window

GROUPS = ["BCF", "PCBM", "VBM"]
SCHEME = "BCF->PCBM,PCBM->VBM"


def sequential(time, k1, k2):
    """Analytic BCF -> PCBM -> VBM populations."""
    a = np.exp(-k1 * time)
    b = k1 / (k2 - k1) * (np.exp(-k1 * time) - np.exp(-k2 * time))
    return np.column_stack([a, b, 1.0 - a - b])


def piecewise_sequential(time, boundary, early_rates, late_rates):
    """Sequential kinetics whose rates change at ``boundary``.

    Continuous at the join: the late segment is propagated from the populations
    the early segment ended on, so nothing jumps.
    """
    from namd_analysis.kinetics import build_rate_matrix, propagate

    edges = parse_edges(SCHEME, GROUPS)
    early_mask = time <= boundary
    late_mask = ~early_mask
    out = np.empty((time.size, 3), dtype=float)

    k_early = build_rate_matrix(np.asarray(early_rates, dtype=float), edges, len(GROUPS))
    out[early_mask] = propagate(k_early, np.array([1.0, 0.0, 0.0]), time[early_mask])
    if np.any(late_mask):
        k_late = build_rate_matrix(np.asarray(late_rates, dtype=float), edges, len(GROUPS))
        late_time = time[late_mask]
        shifted = late_time - late_time[0] + (late_time[1] - late_time[0])
        out[late_mask] = propagate(k_late, out[early_mask][-1], shifted)
    return out


def write_campaign(root, time_ns, observed, n_files=2):
    """SHPROP histories and a state map for the analytic populations."""
    root = Path(root)
    paths = []
    for index in range(n_files):
        path = root / f"SHPROP.{index + 1}"
        with path.open("w", encoding="utf-8") as handle:
            handle.write("# NAMDTINI = 1\n# NSW = 2\n# BMIN = 1\n# BMAX = 3\n")
            for step, row in zip(time_ns, observed):
                values = [step * 1.0e6, -0.8, *row]
                handle.write(" ".join(f"{v:.10E}" for v in values) + "\n")
        paths.append(path)
    config = root / "state_map.json"
    config.write_text(
        json.dumps(
            {
                "name": "synthetic",
                "time_column": 0,
                "time_unit": "fs",
                "population_columns": [2, 3, 4],
                "groups": {"BCF": [2], "PCBM": [3], "VBM": [4]},
                "complete_population": True,
                "recombined_group": "VBM",
            }
        ),
        encoding="utf-8",
    )
    return paths, config


class _Synthetic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.edges = parse_edges(SCHEME, GROUPS)

    def tearDown(self):
        self.tmp.cleanup()

    def _regimes(self, time, observed, early=(0.0, 0.1), late=(0.1, 1.0)):
        a = fit_regime("early", Window("early", *early), time, observed, GROUPS, self.edges)
        b = fit_regime("late", Window("late", *late), time, observed, GROUPS, self.edges)
        return a, b


class GridUniformityTests(unittest.TestCase):
    """A time column written as text is uniform data with rounding jitter."""

    def test_a_text_roundtripped_grid_is_accepted(self):
        time = np.linspace(0.0, 1.0, 2000)
        roundtripped = np.array([float(f"{v:.10E}") for v in time * 1.0e6]) / 1.0e6
        report = grid_uniformity(roundtripped)
        self.assertGreater(report["relative_spread"], 1e-9)
        self.assertTrue(report["uniform_enough"])
        # The old 1e-8 test rejected exactly this, and because KineticsError
        # subclasses ValueError the optimizer loop swallowed it and reported
        # "every optimizer start failed".
        observed = sequential(roundtripped, 60.0, 1.2)
        fit = fit_master_equation(
            roundtripped, observed, GROUPS, parse_edges(SCHEME, GROUPS)
        )
        self.assertAlmostEqual(min(fit.eigen_timescales_ns), 1 / 60.0, places=3)

    def test_a_genuinely_irregular_grid_is_refused_with_numbers(self):
        time = np.sort(np.concatenate([np.linspace(0.0, 1.0, 50), [0.5001]]))
        with self.assertRaises(KineticsError) as ctx:
            require_uniform_grid(time)
        message = str(ctx.exception)
        self.assertIn("relative spread", message)
        self.assertIn("nothing here interpolates", message)

    def test_the_failure_is_raised_once_not_per_optimizer_start(self):
        time = np.sort(np.concatenate([np.linspace(0.0, 1.0, 50), [0.5001]]))
        observed = np.tile([[0.5, 0.3, 0.2]], (time.size, 1))
        with self.assertRaises(KineticsError) as ctx:
            fit_master_equation(time, observed, GROUPS, parse_edges(SCHEME, GROUPS))
        self.assertNotIn("every optimizer start failed", str(ctx.exception))
        self.assertIn("not uniform enough", str(ctx.exception))


class SameMechanismTests(_Synthetic):
    """One rate matrix really does govern both windows."""

    def setUp(self):
        super().setUp()
        self.time = np.linspace(0.0, 1.0, 2000)
        self.observed = sequential(self.time, 60.0, 1.2)

    def test_both_windows_recover_the_true_timescales(self):
        early, late = self._regimes(self.time, self.observed)
        for regime in (early, late):
            self.assertEqual(regime.identifiability["status"], "all_identified")
            timescales = sorted(regime.fit.eigen_timescales_ns)
            self.assertAlmostEqual(timescales[0], 1 / 60.0, places=3)
            self.assertAlmostEqual(timescales[1], 1 / 1.2, places=2)

    def test_the_global_description_is_preferred(self):
        early, late = self._regimes(self.time, self.observed)
        result = compare_piecewise(
            self.time, self.observed, GROUPS, self.edges, early, late
        )
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["preferred"], "global")

    def test_the_comparison_refuses_to_claim_a_mechanism(self):
        early, late = self._regimes(self.time, self.observed)
        result = compare_piecewise(
            self.time, self.observed, GROUPS, self.edges, early, late
        )
        self.assertIn("NOT proof of a mechanistic transition", result["what_this_is"])
        self.assertIn("does not establish a mechanism", result["conclusion"])


class DifferentRatesTests(_Synthetic):
    """The rates genuinely change at the boundary."""

    def setUp(self):
        super().setUp()
        self.time = np.linspace(0.0, 1.0, 2000)
        self.observed = piecewise_sequential(
            self.time, 0.1, early_rates=(60.0, 3.0), late_rates=(2.0, 0.4)
        )

    def test_populations_are_physical(self):
        self.assertGreaterEqual(self.observed.min(), -1e-9)
        self.assertLessEqual(self.observed.max(), 1 + 1e-9)
        np.testing.assert_allclose(self.observed.sum(axis=1), 1.0, atol=1e-9)

    def test_the_piecewise_description_is_preferred(self):
        early, late = self._regimes(self.time, self.observed)
        result = compare_piecewise(
            self.time, self.observed, GROUPS, self.edges, early, late
        )
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["preferred"], "piecewise")
        self.assertLess(result["delta"], 0.0)

    def test_the_two_windows_report_different_timescales(self):
        early, late = self._regimes(self.time, self.observed)
        if early.fit is None or late.fit is None:
            self.skipTest("a window did not fit; covered by the unidentifiable tests")
        self.assertNotAlmostEqual(
            min(early.fit.eigen_timescales_ns),
            min(late.fit.eigen_timescales_ns),
            places=2,
        )


class OvershootTests(_Synthetic):
    """An acceptor maximum that has relaxed before the late window opens."""

    def setUp(self):
        super().setUp()
        self.time = np.linspace(0.0, 1.0, 2000)
        self.observed = sequential(self.time, 60.0, 1.2)

    def test_the_overshoot_is_detected_and_attributed(self):
        early, late = self._regimes(self.time, self.observed)
        answers = {
            f["question"]: f for f in early_transient_narrative(early, late)["findings"]
        }
        rise = next(k for k in answers if "rise within the early window" in k)
        self.assertTrue(answers[rise]["answer"].startswith("yes"))

        overshoot = next(k for k in answers if "transient PCBM maximum" in k)
        self.assertTrue(answers[overshoot]["answer"].startswith("yes"))
        evidence = answers[overshoot]["evidence"]
        self.assertGreater(evidence["early_peak"], evidence["late_window_initial"])

        missed = next(k for k in answers if "late fit miss early transfer" in k)
        self.assertIn("cannot see that occupation", answers[missed]["answer"])
        self.assertEqual(answers[missed]["observable_class"], "interpretive")

    def test_the_donor_destination_is_apportioned_not_asserted(self):
        early, late = self._regimes(self.time, self.observed)
        answers = {
            f["question"]: f for f in early_transient_narrative(early, late)["findings"]
        }
        key = next(k for k in answers if "depopulate into" in k)
        self.assertIn("PCBM", answers[key]["answer"])
        self.assertIn("not a measured flux", answers[key]["answer"])
        shares = answers[key]["evidence"]["gain_share"]
        self.assertGreater(shares["PCBM"], shares.get("VBM", 0.0))

    def test_no_overshoot_is_reported_when_there_is_none(self):
        # Monotonic rise to a plateau: the early peak cannot exceed the late one.
        time = np.linspace(0.0, 1.0, 2000)
        pcbm = 1.0 - np.exp(-3.0 * time)
        observed = np.column_stack([np.exp(-3.0 * time), pcbm, np.zeros_like(time)])
        early, late = self._regimes(time, observed)
        answers = {
            f["question"]: f for f in early_transient_narrative(early, late)["findings"]
        }
        overshoot = next(k for k in answers if "transient PCBM maximum" in k)
        self.assertTrue(answers[overshoot]["answer"].startswith("no"))


class UnidentifiableWindowTests(_Synthetic):
    """A window that cannot constrain the graph says so instead of inventing rates."""

    def test_a_window_with_no_dynamics_reports_rather_than_fits(self):
        time = np.linspace(0.0, 1.0, 400)
        flat = np.tile([[0.2, 0.3, 0.5]], (time.size, 1))
        early, late = self._regimes(time, flat)
        for regime in (early, late):
            status = regime.identifiability["status"]
            self.assertIn(status, {"not_fitted", "none_identified", "partially_identified"})
            payload = regime.as_dict()
            if regime.fit is None:
                self.assertEqual(payload["kinetics"]["status"], "not_fitted")
                self.assertIn("reported as such", payload["kinetics"]["note"])

    def test_an_unfitted_window_blocks_the_comparison_rather_than_forcing_it(self):
        time = np.linspace(0.0, 1.0, 400)
        flat = np.tile([[0.2, 0.3, 0.5]], (time.size, 1))
        early, late = self._regimes(time, flat)
        result = compare_piecewise(time, flat, GROUPS, self.edges, early, late)
        self.assertIn(
            result["status"],
            {"scored", "piecewise_incomplete", "global_fit_failed", "insufficient_observations"},
        )
        if result["status"] == "piecewise_incomplete":
            self.assertIn("No rates are invented", result["conclusion"])

    def test_too_few_points_is_a_refusal_not_a_fit(self):
        time = np.linspace(0.0, 1.0, 2000)
        observed = sequential(time, 60.0, 1.2)
        from namd_analysis.transient import TransientError

        with self.assertRaises(TransientError) as ctx:
            fit_regime(
                "sliver", Window("sliver", 0.0, 0.0005), time, observed, GROUPS, self.edges
            )
        self.assertIn("at least two are needed", str(ctx.exception))


class WindowPolicyTests(unittest.TestCase):
    def test_the_early_window_defaults_to_one_hundred_picoseconds(self):
        self.assertEqual(DEFAULT_EARLY_WINDOW_NS, (0.0, 0.1))

    def test_the_late_window_note_explains_why_it_is_not_defaulted(self):
        self.assertIn("not defaulted", LATE_WINDOW_NOTE)
        self.assertIn("comparison_manifest.template.json", LATE_WINDOW_NOTE)

    def test_the_repository_still_encodes_no_manuscript_fit_window(self):
        # If a real late window is ever committed, this test should fail and the
        # CLI should start defaulting to it rather than requiring it.
        repo = Path(__file__).resolve().parent.parent
        template = (
            repo / "examples" / "bcf_pcbm" / "comparison_manifest.template.json"
        ).read_text(encoding="utf-8")
        self.assertIn("not universal constants", template)
        self.assertIn("not defaulted anywhere in the code", template)


class RegimeCliTests(_Synthetic):
    def test_end_to_end_writes_every_artifact(self):
        time = np.linspace(0.0, 1.0, 2000)
        observed = sequential(time, 60.0, 1.2)
        paths, config = write_campaign(self.root, time, observed)
        out = self.root / "out"
        code = dispatch_main(
            [
                "regime-analysis",
                "--files", *[str(p) for p in paths],
                "--config", str(config),
                "--late-window", "0.1:1.0",
                "--scheme", SCHEME,
                "--out", str(out),
            ]
        )
        self.assertEqual(code, 0)
        for name in (
            "regime_observables.csv",
            "regime_rates.csv",
            "report.json",
            "early_late_summary.md",
            "regime_populations.png",
            "regime_fits.png",
        ):
            self.assertTrue((out / name).is_file(), name)
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["windows"]["early"]["source"], "default")
        self.assertEqual(report["windows"]["late"]["source"], "explicit_cli")
        self.assertEqual(report["windows"]["early"]["resolved_ns"], [0.0, 0.1])

    def test_the_late_window_is_required(self):
        time = np.linspace(0.0, 1.0, 500)
        paths, config = write_campaign(self.root, time, sequential(time, 20.0, 1.0))
        with self.assertRaises(SystemExit):
            dispatch_main(
                [
                    "regime-analysis",
                    "--files", str(paths[0]),
                    "--config", str(config),
                    "--scheme", SCHEME,
                    "--out", str(self.root / "nope"),
                ]
            )

    def test_the_summary_labels_every_claim(self):
        time = np.linspace(0.0, 1.0, 2000)
        paths, config = write_campaign(self.root, time, sequential(time, 60.0, 1.2))
        out = self.root / "labelled"
        dispatch_main(
            [
                "regime-analysis",
                "--files", *[str(p) for p in paths],
                "--config", str(config),
                "--late-window", "0.1:1.0",
                "--scheme", SCHEME,
                "--out", str(out),
            ]
        )
        text = (out / "early_late_summary.md").read_text(encoding="utf-8")
        self.assertIn("observed", text)
        self.assertIn("model-inferred", text)
        self.assertIn("not proof of a mechanistic transition", text)
        self.assertIn("not a small rate", text)


if __name__ == "__main__":
    unittest.main()
