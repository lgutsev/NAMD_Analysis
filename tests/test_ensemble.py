"""The hierarchy: passes inside histories inside runs.

What these pin is mostly what the statistics must *refuse* to do -- treat
passes as samples, treat 300 histories as 300 nuclear realizations, or reduce
three runs to a mean that hides their disagreement.
"""

import unittest

import numpy as np

from namd_analysis.ensemble import (
    BOOKKEEPING_NOTE,
    HIERARCHY_NOTE,
    EnsembleError,
    HistoryResult,
    compare_runs,
    ensemble_curve,
    history_header,
    reviewer_table,
    summarize_run,
)

GROUPS = ["perovskite", "BCF", "PCBM"]


def history(run, k, bcf, pcbm, verdict="essentially_reversible"):
    return HistoryResult(
        run=run, history=f"SHPROP.{k}", namdtini=k,
        n_rows=10_000_000, n_passes=5003,
        net_full={"perovskite": -(bcf + pcbm), "BCF": bcf, "PCBM": pcbm},
        net_late={"BCF": bcf / 10.0, "PCBM": pcbm / 10.0},
        range_late={"BCF": 0.84, "PCBM": 0.84},
        occupation_redistribution={"BCF": bcf * 0.8, "PCBM": pcbm * 0.1},
        character_evolution={"BCF": bcf * 0.2, "PCBM": pcbm * 0.9},
        episode_net_per_pass={"BCF": 1.8e-3, "PCBM": -2.6e-4},
        episode_occupation_per_pass={"BCF": 2.4e-5, "PCBM": -2.2e-5},
        episode_verdict=verdict,
        decomposition_residual=3.33e-16,
    )


def run_of(name, bcfs, pcbms, verdicts=None):
    hs = [
        history(name, k, b, p, (verdicts or {}).get(k, "essentially_reversible"))
        for k, (b, p) in enumerate(zip(bcfs, pcbms))
    ]
    return summarize_run(name, hs, GROUPS)


class RunSummaryTests(unittest.TestCase):
    def test_sign_fractions_count_gains_losses_and_flats(self):
        # 3 gains, 2 losses, 1 flat on PCBM.
        summary = run_of(
            "run1",
            [0.5] * 6,
            [0.4, 0.3, 0.2, -0.3, -0.4, 0.0],
        )
        block = summary["quantities"]["net_full"]["PCBM"]
        self.assertEqual(block["n"], 6)
        self.assertEqual(block["n_gain"], 3)
        self.assertEqual(block["n_loss"], 2)
        self.assertEqual(block["n_flat"], 1)
        self.assertAlmostEqual(block["fraction_gain"], 0.5)

    def test_quantiles_and_median_are_reported(self):
        summary = run_of("run1", [0.1] * 100, np.linspace(-1.0, 1.0, 100))
        block = summary["quantities"]["net_full"]["PCBM"]
        for key in ("q05", "q25", "q50", "q75", "q95"):
            self.assertIn(key, block)
        self.assertAlmostEqual(block["median"], block["q50"], places=12)
        self.assertLess(block["q05"], block["q95"])

    def test_the_spread_is_labelled_as_between_histories(self):
        summary = run_of("run1", [0.5, 0.6, 0.4], [0.1, 0.2, 0.0])
        self.assertIn("std_between_histories", summary["quantities"]["net_full"]["BCF"])
        self.assertIn("fixed nuclear trajectory", summary["spread_means"])
        self.assertIn("not an uncertainty", summary["spread_means"])

    def test_the_hierarchy_and_bookkeeping_notes_travel_with_the_summary(self):
        summary = run_of("run1", [0.5], [0.1])
        self.assertEqual(summary["hierarchy_note"], HIERARCHY_NOTE)
        self.assertEqual(summary["bookkeeping_note"], BOOKKEEPING_NOTE)
        self.assertIn("not independent samples", summary["hierarchy_note"])
        self.assertIn("NOT physical branching fractions", summary["bookkeeping_note"])

    def test_episode_verdicts_are_tallied(self):
        summary = run_of(
            "run1", [0.5] * 4, [0.1] * 4,
            verdicts={0: "persistent_acceptor_gain", 1: "persistent_donor_gain"},
        )
        self.assertEqual(summary["episode_verdicts"]["persistent_acceptor_gain"], 1)
        self.assertEqual(summary["episode_verdicts"]["essentially_reversible"], 2)

    def test_a_history_from_another_run_is_refused(self):
        mixed = [history("run1", 0, 0.5, 0.1), history("run2", 1, 0.5, 0.1)]
        with self.assertRaises(EnsembleError) as ctx:
            summarize_run("run1", mixed, GROUPS)
        self.assertIn("run2", str(ctx.exception))

    def test_an_empty_run_is_refused(self):
        with self.assertRaises(EnsembleError):
            summarize_run("run1", [], GROUPS)

    def test_the_per_history_row_and_header_line_up(self):
        header = history_header(GROUPS)
        row = history("run1", 0, 0.5, 0.1).as_row(GROUPS)
        self.assertEqual(len(header), len(row))
        self.assertEqual(header[0], "run")
        self.assertIn("net_late_PCBM", header)
        self.assertIn("decomposition_residual", header)


class AcrossRunTests(unittest.TestCase):
    def test_runs_disagreeing_in_sign_are_reported_as_such(self):
        runs = [
            run_of("run1", [0.5] * 20, [-0.4] * 20),
            run_of("run2", [0.3] * 20, [-0.2] * 20),
            run_of("run3", [0.1] * 20, [+0.3] * 20),
        ]
        out = compare_runs(runs, GROUPS)
        self.assertFalse(
            out["groups"]["PCBM"]["runs_agree_in_sign"],
            "run3 has the opposite sign and that must show",
        )
        self.assertTrue(out["groups"]["BCF"]["runs_agree_in_sign"])

    def test_the_range_of_run_medians_is_the_headline_not_the_mean(self):
        runs = [
            run_of("run1", [0.5] * 10, [-0.4] * 10),
            run_of("run2", [0.5] * 10, [-0.2] * 10),
            run_of("run3", [0.5] * 10, [+0.3] * 10),
        ]
        out = compare_runs(runs, GROUPS)
        self.assertAlmostEqual(out["groups"]["PCBM"]["range_of_run_medians"], 0.7, places=6)
        self.assertIn("read the spread and not the mean", out["note"])

    def test_per_run_values_are_keyed_by_run_name(self):
        runs = [run_of("run1", [0.5] * 5, [0.1] * 5),
                run_of("run2", [0.2] * 5, [0.3] * 5)]
        out = compare_runs(runs, GROUPS)
        self.assertEqual(
            sorted(out["groups"]["BCF"]["per_run_median"]), ["run1", "run2"]
        )
        self.assertAlmostEqual(out["groups"]["BCF"]["per_run_median"]["run2"], 0.2)

    def test_no_runs_is_refused(self):
        with self.assertRaises(EnsembleError):
            compare_runs([], GROUPS)


class ReviewerTableTests(unittest.TestCase):
    def test_counts_carry_their_denominator(self):
        runs = [run_of("run1", [0.5] * 100, [-0.4] * 60 + [0.4] * 40)]
        table = reviewer_table(runs)
        row = next(r for r in table["rows"] if "PCBM gain" in r["row"])
        self.assertEqual(row["values"]["run1"], "40/100")

    def test_medians_appear_as_numbers(self):
        runs = [run_of("run1", [0.5] * 10, [-0.4] * 10)]
        table = reviewer_table(runs)
        row = next(r for r in table["rows"] if "median dP_PCBM" in r["row"])
        self.assertAlmostEqual(row["values"]["run1"], -0.4, places=9)

    def test_the_late_window_can_be_tabulated_too(self):
        runs = [run_of("run1", [0.5] * 10, [-0.4] * 10)]
        table = reviewer_table(runs, quantity="net_late")
        row = next(r for r in table["rows"] if "median dP_PCBM" in r["row"])
        self.assertAlmostEqual(row["values"]["run1"], -0.04, places=9)

    def test_the_fraction_is_of_electronic_conditions_not_nuclear_ones(self):
        runs = [run_of("run1", [0.5] * 10, [0.1] * 10)]
        self.assertIn("not of nuclear", reviewer_table(runs)["note"])


class EnsembleCurveTests(unittest.TestCase):
    def test_a_quantile_band_is_produced_across_histories(self):
        series = [np.linspace(0.0, 1.0, 50) + k * 0.1 for k in range(9)]
        out = ensemble_curve(series)
        self.assertEqual(out["n_histories"], 9)
        self.assertEqual(out["median"].size, 50)
        self.assertTrue(np.all(out["q25"] <= out["median"] + 1e-12))
        self.assertTrue(np.all(out["median"] <= out["q75"] + 1e-12))
        self.assertTrue(np.all(out["min"] <= out["max"]))

    def test_differing_lengths_are_refused_rather_than_interpolated(self):
        with self.assertRaises(EnsembleError) as ctx:
            ensemble_curve([np.zeros(10), np.zeros(11)])
        self.assertIn("no interpolation", str(ctx.exception))

    def test_no_series_is_refused(self):
        with self.assertRaises(EnsembleError):
            ensemble_curve([])


if __name__ == "__main__":
    unittest.main()
