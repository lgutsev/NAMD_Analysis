import bootstrap  # noqa: F401

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.cli import main
from namd_analysis.kinetics import (
    KineticsError,
    bootstrap_rates,
    build_rate_matrix,
    edge_names,
    fit_master_equation,
    parse_edges,
    propagate,
    sink_sweep,
)
from synthetic import kinetic_config, write_kinetic_shprop_set

GROUPS = ["CBM", "BCF", "PCBM", "VBM"]
SCHEME = "CBM->BCF,BCF->CBM,BCF->PCBM,PCBM->VBM"
TRUE_RATES = [8.0, 2.0, 3.0, 0.5]  # per ns


def _true_matrix():
    return build_rate_matrix(TRUE_RATES, parse_edges(SCHEME, GROUPS), len(GROUPS))


class SchemeTests(unittest.TestCase):
    def test_presets_resolve_to_the_expected_edges(self):
        self.assertEqual(parse_edges("sequential", ["A", "B", "C"]), [(0, 1), (1, 2)])
        self.assertEqual(
            parse_edges("reversible", ["A", "B", "C"]),
            [(0, 1), (1, 0), (1, 2), (2, 1)],
        )
        self.assertEqual(len(parse_edges("dense", ["A", "B", "C"])), 6)

    def test_explicit_edges_resolve_by_name(self):
        edges = parse_edges("B->A, A->C", ["A", "B", "C"])
        self.assertEqual(edges, [(1, 0), (0, 2)])
        self.assertEqual(edge_names(edges, ["A", "B", "C"]), ["B->A", "A->C"])

    def test_unknown_group_is_rejected(self):
        with self.assertRaises(KineticsError) as ctx:
            parse_edges("A->Z", ["A", "B"])
        self.assertIn("not a declared group", str(ctx.exception))

    def test_malformed_self_loop_and_duplicate_are_rejected(self):
        with self.assertRaises(KineticsError):
            parse_edges("A B", ["A", "B"])
        with self.assertRaises(KineticsError):
            parse_edges("A->A", ["A", "B"])
        with self.assertRaises(KineticsError):
            parse_edges("A->B,A->B", ["A", "B"])


class RateMatrixTests(unittest.TestCase):
    def test_columns_sum_to_zero(self):
        K = _true_matrix()
        np.testing.assert_allclose(K.sum(axis=0), 0.0, atol=1e-12)

    def test_offdiagonal_entry_is_the_rate_from_source_to_target(self):
        K = build_rate_matrix([2.5], [(0, 1)], 3)
        self.assertAlmostEqual(K[1, 0], 2.5)
        self.assertAlmostEqual(K[0, 0], -2.5)

    def test_propagation_conserves_population(self):
        time = np.linspace(0.0, 2.0, 401)
        trajectory = propagate(_true_matrix(), np.array([1.0, 0.0, 0.0, 0.0]), time)
        np.testing.assert_allclose(trajectory.sum(axis=1), 1.0, atol=1e-10)

    def test_propagation_matches_a_two_state_analytic_solution(self):
        # A -> B at rate k: P_A = exp(-k t).
        time = np.linspace(0.0, 3.0, 301)
        trajectory = propagate(
            build_rate_matrix([1.7], [(0, 1)], 2), np.array([1.0, 0.0]), time
        )
        np.testing.assert_allclose(trajectory[:, 0], np.exp(-1.7 * time), atol=1e-10)

    def test_non_uniform_grid_is_rejected(self):
        time = np.array([0.0, 1.0, 3.0, 4.0])
        with self.assertRaises(KineticsError) as ctx:
            propagate(_true_matrix(), np.array([1.0, 0.0, 0.0, 0.0]), time)
        self.assertIn("not uniform", str(ctx.exception))

    def test_rate_count_must_match_edge_count(self):
        with self.assertRaises(KineticsError):
            build_rate_matrix([1.0, 2.0], [(0, 1)], 2)


class FitTests(unittest.TestCase):
    def setUp(self):
        self.time = np.linspace(0.0, 2.0, 801)
        self.edges = parse_edges(SCHEME, GROUPS)
        self.observed = propagate(
            _true_matrix(), np.array([1.0, 0.0, 0.0, 0.0]), self.time
        )

    def test_recovers_the_true_rates_from_noiseless_data(self):
        fit = fit_master_equation(self.time, self.observed, GROUPS, self.edges)
        for estimate, truth in zip(fit.rates, TRUE_RATES):
            self.assertAlmostEqual(estimate.rate_per_ns, truth, places=5)
            self.assertTrue(estimate.identified)
        self.assertGreater(fit.r_squared_total, 0.999999)
        self.assertEqual(fit.warnings, [])

    def test_initial_condition_comes_from_the_data(self):
        fit = fit_master_equation(self.time, self.observed, GROUPS, self.edges)
        np.testing.assert_allclose(fit.p0, self.observed[0])

    def test_lifetime_is_the_reciprocal_rate(self):
        fit = fit_master_equation(self.time, self.observed, GROUPS, self.edges)
        for estimate in fit.rates:
            self.assertAlmostEqual(
                estimate.lifetime_ns, 1.0 / estimate.rate_per_ns, places=9
            )

    def test_eigen_timescales_match_the_true_matrix(self):
        fit = fit_master_equation(self.time, self.observed, GROUPS, self.edges)
        eigenvalues = np.linalg.eigvals(_true_matrix())
        expected = sorted(
            -1.0 / value.real for value in eigenvalues if value.real < -1e-12
        )
        np.testing.assert_allclose(
            sorted(fit.eigen_timescales_ns), expected, rtol=1e-4
        )

    def test_noisy_data_still_recovers_rates_within_a_few_percent(self):
        rng = np.random.default_rng(3)
        noisy = self.observed + 0.003 * rng.standard_normal(self.observed.shape)
        fit = fit_master_equation(self.time, noisy, GROUPS, self.edges)
        for estimate, truth in zip(fit.rates, TRUE_RATES):
            self.assertLess(abs(estimate.rate_per_ns - truth) / truth, 0.1,
                            f"{estimate.name}: {estimate.rate_per_ns} vs {truth}")

    def test_over_parameterized_scheme_is_flagged_as_unidentified(self):
        # The dense scheme fits the same data just as well but its individual
        # rates are not determined. That must be visible, not hidden.
        rng = np.random.default_rng(4)
        noisy = self.observed + 0.003 * rng.standard_normal(self.observed.shape)
        dense = parse_edges("dense", GROUPS)
        fit = fit_master_equation(self.time, noisy, GROUPS, dense)
        self.assertGreater(fit.r_squared_total, 0.99)
        unidentified = [r for r in fit.rates if not r.identified]
        self.assertGreater(len(unidentified), 0)
        self.assertGreater(fit.condition_number, 1e8)
        self.assertTrue(any("do not determine" in w for w in fit.warnings))

    def test_degenerate_partners_are_named(self):
        rng = np.random.default_rng(6)
        noisy = self.observed + 0.002 * rng.standard_normal(self.observed.shape)
        fit = fit_master_equation(self.time, noisy, GROUPS, parse_edges("dense", GROUPS))
        partnered = [r for r in fit.rates if r.degenerate_with]
        self.assertTrue(partnered)
        for estimate in partnered:
            self.assertFalse(estimate.identified)
            for partner in estimate.degenerate_with:
                self.assertIn(partner, [r.name for r in fit.rates])

    def test_wrong_scheme_gives_a_poor_fit_and_says_so(self):
        # Force a scheme with no route to VBM: the model cannot reproduce the
        # observed VBM growth.
        edges = parse_edges("CBM->BCF,BCF->PCBM", GROUPS)
        fit = fit_master_equation(self.time, self.observed, GROUPS, edges)
        self.assertLess(fit.r_squared_per_group["VBM"], 0.5)
        self.assertTrue(any("does not describe" in w for w in fit.warnings))

    def test_too_few_points_for_the_parameter_count_is_rejected(self):
        with self.assertRaises(KineticsError):
            fit_master_equation(
                self.time[:3], self.observed[:3], GROUPS, parse_edges("dense", GROUPS)
            )

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(KineticsError):
            fit_master_equation(self.time, self.observed[:, :2], GROUPS, self.edges)

    def test_report_dict_is_serializable_and_carries_the_assumptions(self):
        fit = fit_master_equation(self.time, self.observed, GROUPS, self.edges)
        payload = fit.as_dict()
        json.dumps(payload)
        self.assertIn("model_assumptions", payload)
        self.assertIn("eigen_timescales_note", payload)
        self.assertEqual(len(payload["rates"]), len(TRUE_RATES))


class BootstrapTests(unittest.TestCase):
    def test_interval_brackets_the_true_rate(self):
        time = np.linspace(0.0, 2.0, 201)
        exact = propagate(_true_matrix(), np.array([1.0, 0.0, 0.0, 0.0]), time)
        rng = np.random.default_rng(7)
        per_file = np.stack(
            [exact + 0.004 * rng.standard_normal(exact.shape) for _ in range(6)]
        )
        edges = parse_edges(SCHEME, GROUPS)
        intervals = bootstrap_rates(time, per_file, GROUPS, edges, n_resamples=40, seed=1)
        self.assertIn("CBM->BCF", intervals)
        low, high = intervals["CBM->BCF"]
        self.assertLess(low, high)
        self.assertLess(low, 8.0 * 1.5)
        self.assertGreater(high, 8.0 * 0.5)

    def test_single_file_is_rejected(self):
        time = np.linspace(0.0, 1.0, 51)
        exact = propagate(_true_matrix(), np.array([1.0, 0.0, 0.0, 0.0]), time)
        with self.assertRaises(KineticsError):
            bootstrap_rates(
                time, exact[None, ...], GROUPS, parse_edges(SCHEME, GROUPS)
            )


class SinkTests(unittest.TestCase):
    def setUp(self):
        self.time = np.linspace(0.0, 2.0, 401)
        self.edges = parse_edges(SCHEME, GROUPS)
        observed = propagate(_true_matrix(), np.array([1.0, 0.0, 0.0, 0.0]), self.time)
        self.fit = fit_master_equation(self.time, observed, GROUPS, self.edges)

    def test_zero_escape_collects_nothing_and_leaves_the_dynamics_unchanged(self):
        point = sink_sweep(self.fit, "PCBM", [0.0], recombined_group="VBM")[0]
        self.assertAlmostEqual(point.collected_final, 0.0, places=10)
        self.assertAlmostEqual(point.remaining_final, 1.0, places=8)

    def test_escaped_population_leaves_the_dynamics(self):
        # Total (still in the interface + collected) must stay at one: the sink
        # is a channel in the propagated system, not a post-hoc integral.
        for point in sink_sweep(self.fit, "PCBM", [0.1, 1.0, 10.0, 100.0],
                                recombined_group="VBM"):
            self.assertAlmostEqual(
                point.collected_final + point.remaining_final, 1.0, places=8
            )

    def test_faster_escape_collects_more_and_recombines_less(self):
        points = sink_sweep(
            self.fit, "PCBM", [0.1, 1.0, 10.0, 100.0], recombined_group="VBM"
        )
        collected = [p.collected_final for p in points]
        recombined = [p.recombined_final for p in points]
        self.assertEqual(collected, sorted(collected))
        self.assertEqual(recombined, sorted(recombined, reverse=True))

    def test_unknown_groups_are_rejected(self):
        with self.assertRaises(KineticsError):
            sink_sweep(self.fit, "nope", [1.0])
        with self.assertRaises(KineticsError):
            sink_sweep(self.fit, "PCBM", [1.0], recombined_group="nope")

    def test_negative_escape_rate_is_rejected(self):
        with self.assertRaises(KineticsError):
            sink_sweep(self.fit, "PCBM", [-1.0])


class KineticsCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = self.root / "map.json"
        self.config.write_text(
            json.dumps(kinetic_config(GROUPS, recombined="VBM")), encoding="utf-8"
        )
        write_kinetic_shprop_set(
            self.root / "run",
            _true_matrix(),
            [1.0, 0.0, 0.0, 0.0],
            n_files=4,
            nsteps=300,
            dt_fs=5000.0,
            noise=0.002,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _report(self, out):
        return json.loads((out / "report.json").read_text(encoding="utf-8"))

    def test_end_to_end_recovers_the_rates(self):
        out = self.root / "out"
        code = main(
            [
                "kinetics",
                "--files", str(self.root / "run" / "SHPROP.*"),
                "--config", str(self.config),
                "--scheme", SCHEME,
                "--out", str(out),
            ]
        )
        self.assertEqual(code, 0)
        for name in ("report.json", "rates.csv", "kinetics_curves.csv",
                     "kinetics.png", "kinetics.pdf"):
            self.assertTrue((out / name).is_file(), name)
        report = self._report(out)
        rates = {r["transition"]: r["rate_per_ns"] for r in report["fit"]["rates"]}
        for name, truth in zip(edge_names(parse_edges(SCHEME, GROUPS), GROUPS), TRUE_RATES):
            self.assertLess(abs(rates[name] - truth) / truth, 0.15, name)
        self.assertGreater(report["fit"]["fit_quality"]["r_squared_total"], 0.99)

    def test_sink_sweep_is_written(self):
        out = self.root / "out_sink"
        code = main(
            [
                "kinetics",
                "--files", str(self.root / "run" / "SHPROP.*"),
                "--config", str(self.config),
                "--scheme", SCHEME,
                "--sink-group", "PCBM",
                "--sink-rates", "0.1:100:5",
                "--out", str(out),
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue((out / "sink_sweep.csv").is_file())
        self.assertTrue((out / "sink_sweep.png").is_file())
        report = self._report(out)
        points = report["sink_sweep"]["points"]
        self.assertEqual(len(points), 5)
        self.assertLess(points[0]["collected_final"], points[-1]["collected_final"])

    def test_bootstrap_intervals_are_written(self):
        out = self.root / "out_boot"
        code = main(
            [
                "kinetics",
                "--files", str(self.root / "run" / "SHPROP.*"),
                "--config", str(self.config),
                "--scheme", SCHEME,
                "--bootstrap", "20",
                "--out", str(out),
            ]
        )
        self.assertEqual(code, 0)
        report = self._report(out)
        self.assertTrue(report["bootstrap"]["intervals_per_ns"])
        header = (out / "rates.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("bootstrap_low_per_ns", header)

    def test_partial_state_map_is_refused(self):
        partial = self.root / "partial.json"
        payload = kinetic_config(GROUPS)
        payload["complete_population"] = False
        payload["recombined_group"] = None
        partial.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SystemExit) as ctx:
            main(
                [
                    "kinetics",
                    "--files", str(self.root / "run" / "SHPROP.*"),
                    "--config", str(partial),
                    "--scheme", SCHEME,
                    "--out", str(self.root / "out_partial"),
                ]
            )
        self.assertIn("closed system", str(ctx.exception))

    def test_unknown_sink_group_is_refused(self):
        with self.assertRaises(SystemExit):
            main(
                [
                    "kinetics",
                    "--files", str(self.root / "run" / "SHPROP.*"),
                    "--config", str(self.config),
                    "--scheme", SCHEME,
                    "--sink-group", "ETL",
                    "--out", str(self.root / "out_bad_sink"),
                ]
            )

    def test_malformed_scheme_is_reported(self):
        code = main(
            [
                "kinetics",
                "--files", str(self.root / "run" / "SHPROP.*"),
                "--config", str(self.config),
                "--scheme", "CBM=>BCF",
                "--out", str(self.root / "out_bad_scheme"),
            ]
        )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
