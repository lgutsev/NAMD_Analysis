import bootstrap  # noqa: F401

import unittest

import numpy as np

from namd_analysis.branching import (
    EXTRACTED,
    IDENTIFIED,
    NOT_IDENTIFIABLE,
    WEAKLY_IDENTIFIED,
    BranchingError,
    absorption_report,
    augment_with_sink,
    bootstrap_branch_probability,
    branch_probability,
    extraction_competition,
    hitting_probabilities,
    local_branching_ratio,
)
from namd_analysis.kinetics import build_rate_matrix, fit_master_equation, parse_edges, propagate

GROUPS = ["CBM", "BCF", "PCBM", "VBM"]


def _matrix(spec, rates, groups=GROUPS):
    return build_rate_matrix(rates, parse_edges(spec, groups), len(groups))


class HittingProbabilityTests(unittest.TestCase):
    def test_two_competing_channels_match_a_over_a_plus_b(self):
        # The one case with a closed form: BCF -> PCBM at a, BCF -> VBM at b.
        for a, b in ((3.0, 1.0), (1.0, 4.0), (0.5, 0.5), (10.0, 0.01), (1e-3, 7.0)):
            K = _matrix("BCF->PCBM,BCF->VBM", [a, b])
            h = hitting_probabilities(K, GROUPS, ["PCBM"], ["VBM"])
            self.assertAlmostEqual(h["BCF"], a / (a + b), places=12)

    def test_back_transfer_network_matches_an_independent_linear_solve(self):
        K = _matrix(
            "CBM->BCF,BCF->CBM,BCF->PCBM,PCBM->BCF,BCF->VBM",
            [5.0, 2.0, 3.0, 1.5, 0.7],
        )
        h = hitting_probabilities(K, GROUPS, ["PCBM"], ["VBM"])
        # Solve Q h = 0 on the transient block independently of the module.
        Q = K.T
        transient = [0, 1]
        expected = np.linalg.solve(
            Q[np.ix_(transient, transient)],
            -Q[np.ix_(transient, [2])].sum(axis=1),
        )
        self.assertAlmostEqual(h["CBM"], expected[0], places=12)
        self.assertAlmostEqual(h["BCF"], expected[1], places=12)

    def test_boundary_conditions_hold(self):
        K = _matrix("BCF->PCBM,BCF->VBM", [2.0, 1.0])
        h = hitting_probabilities(K, GROUPS, ["PCBM"], ["VBM"])
        self.assertEqual(h["PCBM"], 1.0)
        self.assertEqual(h["VBM"], 0.0)

    def test_source_already_in_the_success_set(self):
        K = _matrix("BCF->PCBM,BCF->VBM", [2.0, 1.0])
        h = hitting_probabilities(K, GROUPS, ["PCBM"], ["VBM"])
        self.assertEqual(h["PCBM"], 1.0)

    def test_source_already_in_the_failure_set(self):
        K = _matrix("BCF->PCBM,BCF->VBM", [2.0, 1.0])
        h = hitting_probabilities(K, GROUPS, ["PCBM"], ["VBM"])
        self.assertEqual(h["VBM"], 0.0)

    def test_multiple_states_in_either_absorbing_set(self):
        groups = ["S", "A1", "A2", "F1", "F2"]
        K = build_rate_matrix(
            [1.0, 3.0, 2.0, 4.0],
            parse_edges("S->A1,S->A2,S->F1,S->F2", groups),
            len(groups),
        )
        h = hitting_probabilities(K, groups, ["A1", "A2"], ["F1", "F2"])
        self.assertAlmostEqual(h["S"], (1.0 + 3.0) / (1.0 + 3.0 + 2.0 + 4.0), places=12)

    def test_disconnected_target_gives_zero_not_an_error(self):
        # PCBM is unreachable: BCF only decays to VBM.
        K = _matrix("BCF->VBM", [1.0])
        h = hitting_probabilities(K, GROUPS, ["PCBM"], ["VBM"])
        self.assertEqual(h["BCF"], 0.0)

    def test_a_state_that_reaches_neither_outcome_is_not_an_error(self):
        # CBM is isolated; the question is still well posed for BCF.
        K = _matrix("BCF->PCBM,BCF->VBM", [3.0, 1.0])
        report = absorption_report(K, GROUPS, ["PCBM"], ["VBM"])
        self.assertEqual(report["stranded_groups"], ["CBM"])
        self.assertAlmostEqual(report["success"]["BCF"], 0.75, places=12)
        self.assertAlmostEqual(report["unresolved"]["CBM"], 1.0, places=12)
        self.assertAlmostEqual(report["unresolved"]["BCF"], 0.0, places=12)

    def test_success_and_failure_probabilities_sum_to_one_when_all_absorb(self):
        K = _matrix(
            "CBM->BCF,BCF->CBM,BCF->PCBM,PCBM->BCF,BCF->VBM",
            [5.0, 2.0, 3.0, 1.5, 0.7],
        )
        report = absorption_report(K, GROUPS, ["PCBM"], ["VBM"])
        for name in GROUPS:
            self.assertAlmostEqual(
                report["success"][name] + report["failure"][name], 1.0, places=10
            )

    def test_overlapping_sets_are_rejected(self):
        K = _matrix("BCF->PCBM,BCF->VBM", [1.0, 1.0])
        with self.assertRaises(BranchingError) as ctx:
            hitting_probabilities(K, GROUPS, ["PCBM"], ["PCBM"])
        self.assertIn("cannot be both", str(ctx.exception))

    def test_empty_and_unknown_sets_are_rejected(self):
        K = _matrix("BCF->PCBM,BCF->VBM", [1.0, 1.0])
        with self.assertRaises(BranchingError):
            hitting_probabilities(K, GROUPS, [], ["VBM"])
        with self.assertRaises(BranchingError):
            hitting_probabilities(K, GROUPS, ["PCBM"], [])
        with self.assertRaises(BranchingError):
            hitting_probabilities(K, GROUPS, ["nope"], ["VBM"])

    def test_non_finite_matrix_is_rejected(self):
        K = _matrix("BCF->PCBM,BCF->VBM", [1.0, 1.0])
        K[0, 0] = np.nan
        with self.assertRaises(BranchingError):
            hitting_probabilities(K, GROUPS, ["PCBM"], ["VBM"])

    def test_inert_transient_states_do_not_break_the_solve(self):
        # Every transient state is inert; the absorbing boundary still holds.
        K = np.zeros((4, 4))
        h = hitting_probabilities(K, GROUPS, ["PCBM"], ["VBM"])
        self.assertEqual(h["PCBM"], 1.0)
        self.assertEqual(h["VBM"], 0.0)
        self.assertEqual(h["BCF"], 0.0)

    def test_monte_carlo_agrees_with_the_solver(self):
        K = _matrix(
            "CBM->BCF,BCF->CBM,BCF->PCBM,PCBM->BCF,BCF->VBM",
            [5.0, 2.0, 3.0, 1.5, 0.7],
        )
        exact = hitting_probabilities(K, GROUPS, ["PCBM"], ["VBM"])["BCF"]
        rng = np.random.default_rng(0)
        Q = K.T
        hits = 0
        trials = 20000
        for _ in range(trials):
            state = 1
            for _step in range(5000):
                out = Q[state].copy()
                out[state] = 0.0
                total = out.sum()
                if total <= 0:
                    break
                state = int(rng.choice(len(GROUPS), p=out / total))
                if state == 2:
                    hits += 1
                    break
                if state == 3:
                    break
        estimate = hits / trials
        stderr = np.sqrt(exact * (1 - exact) / trials)
        self.assertLess(abs(estimate - exact), 5 * stderr)


class LocalRatioTests(unittest.TestCase):
    def test_exact_when_every_exit_lands_in_an_absorbing_set(self):
        K = _matrix("BCF->PCBM,BCF->VBM", [3.0, 1.0])
        ratio = local_branching_ratio(K, GROUPS, "BCF", ["PCBM"], ["VBM"])
        self.assertTrue(ratio["available"])
        self.assertAlmostEqual(ratio["ratio"], 0.75, places=12)
        self.assertAlmostEqual(ratio["complement"], 0.25, places=12)
        exact = hitting_probabilities(K, GROUPS, ["PCBM"], ["VBM"])["BCF"]
        self.assertAlmostEqual(ratio["ratio"], exact, places=12)

    def test_refused_when_the_source_has_another_exit(self):
        K = _matrix("BCF->PCBM,BCF->VBM,BCF->CBM", [3.0, 1.0, 2.0])
        ratio = local_branching_ratio(K, GROUPS, "BCF", ["PCBM"], ["VBM"])
        self.assertFalse(ratio["available"])
        self.assertIsNone(ratio["ratio"])
        self.assertIn("CBM", ratio["other_exits"])
        self.assertIn("first-passage", ratio["reason"])

    def test_refused_when_nothing_leaves_the_source(self):
        K = _matrix("CBM->PCBM", [1.0])
        ratio = local_branching_ratio(K, GROUPS, "BCF", ["PCBM"], ["VBM"])
        self.assertFalse(ratio["available"])
        self.assertIn("no transition leaves", ratio["reason"])


class BranchStatusTests(unittest.TestCase):
    def _fit(self, spec, rates, noise=0.0, seed=0, n=401, span=2.0):
        edges = parse_edges(spec, GROUPS)
        K = build_rate_matrix(rates, edges, len(GROUPS))
        time = np.linspace(0.0, span, n)
        observed = propagate(K, np.array([1.0, 0.0, 0.0, 0.0]), time)
        if noise:
            rng = np.random.default_rng(seed)
            observed = observed + noise * rng.standard_normal(observed.shape)
        return fit_master_equation(time, observed, GROUPS, edges)

    def test_identified_model_gives_an_identified_branch(self):
        fit = self._fit("CBM->BCF,BCF->PCBM,BCF->VBM", [6.0, 3.0, 1.0])
        result = branch_probability(fit, "BCF", ["PCBM"], ["VBM"])
        self.assertEqual(result.status, IDENTIFIED)
        self.assertAlmostEqual(result.probability, 0.75, places=4)
        self.assertAlmostEqual(result.complement, 0.25, places=4)
        self.assertAlmostEqual(result.unresolved, 0.0, places=8)

    def test_an_unidentifiable_contributing_rate_fails_the_branch(self):
        # Fitting only the late window, after the fast BCF <-> PCBM pair has
        # equilibrated, leaves BCF->PCBM in the null space of the Jacobian. It
        # is a contributing rate, so the branch cannot be trusted.
        edges = parse_edges("CBM->BCF,BCF->PCBM,PCBM->BCF,BCF->VBM", GROUPS)
        K = build_rate_matrix([40.0, 20.0, 12.0, 0.4], edges, len(GROUPS))
        time = np.linspace(0.0, 10.0, 401)
        observed = propagate(K, np.array([1.0, 0.0, 0.0, 0.0]), time)
        late = time >= 1.0
        fit = fit_master_equation(time[late], observed[late], GROUPS, edges)
        result = branch_probability(fit, "BCF", ["PCBM"], ["VBM"])
        self.assertIn("BCF->PCBM", result.contributing_rates)
        self.assertEqual(result.status, NOT_IDENTIFIABLE)
        self.assertIn("does not determine", result.status_reason)

    def test_a_blind_rate_behind_an_absorbing_state_does_not_contaminate(self):
        # PCBM is never populated, so PCBM->BCF is blind. It is also downstream
        # of an absorbing state for this question, so it cannot change the
        # answer and must not be counted against it.
        fit = self._fit("CBM->BCF,BCF->VBM,PCBM->BCF", [6.0, 1.0, 2.0])
        result = branch_probability(fit, "BCF", ["PCBM"], ["VBM"])
        self.assertEqual(result.contributing_rates, ["BCF->VBM"])
        self.assertEqual(result.status, IDENTIFIED)
        self.assertEqual(result.probability, 0.0)

    def test_source_in_an_absorbing_set_depends_on_no_rate(self):
        fit = self._fit("CBM->BCF,BCF->PCBM,BCF->VBM", [6.0, 3.0, 1.0])
        result = branch_probability(fit, "PCBM", ["PCBM"], ["VBM"])
        self.assertEqual(result.probability, 1.0)
        self.assertEqual(result.contributing_rates, [])
        self.assertEqual(result.status, IDENTIFIED)
        other = branch_probability(fit, "VBM", ["PCBM"], ["VBM"])
        self.assertEqual(other.probability, 0.0)

    def test_status_is_one_of_the_three_verdicts(self):
        fit = self._fit("CBM->BCF,BCF->PCBM,BCF->VBM", [6.0, 3.0, 1.0], noise=0.004)
        result = branch_probability(fit, "BCF", ["PCBM"], ["VBM"])
        self.assertIn(
            result.status, (IDENTIFIED, WEAKLY_IDENTIFIED, NOT_IDENTIFIABLE)
        )

    def test_report_dict_never_calls_it_an_extraction_efficiency(self):
        fit = self._fit("CBM->BCF,BCF->PCBM,BCF->VBM", [6.0, 3.0, 1.0])
        payload = branch_probability(fit, "BCF", ["PCBM"], ["VBM"]).as_dict()
        text = payload["interpretation"].lower()
        self.assertIn("not", text)
        self.assertIn("device", text)
        self.assertEqual(
            payload["observable_class"]["probability_success_before_failure"],
            "model_inferred",
        )


class BootstrapBranchTests(unittest.TestCase):
    def _per_file(self, spec, rates, n_files=6, noise=0.004, seed=1):
        edges = parse_edges(spec, GROUPS)
        K = build_rate_matrix(rates, edges, len(GROUPS))
        time = np.linspace(0.0, 2.0, 201)
        exact = propagate(K, np.array([1.0, 0.0, 0.0, 0.0]), time)
        rng = np.random.default_rng(seed)
        stack = np.stack(
            [exact + noise * rng.standard_normal(exact.shape) for _ in range(n_files)]
        )
        return time, stack, edges

    def test_interval_brackets_the_analytic_branch(self):
        spec = "CBM->BCF,BCF->PCBM,BCF->VBM"
        time, stack, edges = self._per_file(spec, [6.0, 3.0, 1.0])
        result = bootstrap_branch_probability(
            time, stack, GROUPS, edges, "BCF", ["PCBM"], ["VBM"],
            n_resamples=40, seed=2,
        )
        self.assertEqual(result["branch_ci_status"], "reported")
        self.assertLessEqual(result["branch_ci_low"], 0.75)
        self.assertGreaterEqual(result["branch_ci_high"], 0.75)
        self.assertEqual(
            result["bootstrap_branch_successes"], result["bootstrap_fit_successes"]
        )

    def test_interval_is_suppressed_when_the_branch_is_rarely_identifiable(self):
        # Late-window fits leave the fast BCF <-> PCBM pair unidentifiable in
        # every resample, so no interval may be reported for the branch.
        edges = parse_edges("CBM->BCF,BCF->PCBM,PCBM->BCF,BCF->VBM", GROUPS)
        K = build_rate_matrix([40.0, 20.0, 12.0, 0.4], edges, len(GROUPS))
        time = np.linspace(0.0, 10.0, 401)
        exact = propagate(K, np.array([1.0, 0.0, 0.0, 0.0]), time)
        rng = np.random.default_rng(9)
        stack = np.stack(
            [exact + 0.003 * rng.standard_normal(exact.shape) for _ in range(6)]
        )
        late = time >= 1.0
        result = bootstrap_branch_probability(
            time[late], stack[:, late, :], GROUPS, edges, "BCF", ["PCBM"], ["VBM"],
            n_resamples=25, seed=3,
        )
        self.assertNotEqual(result["branch_ci_status"], "reported")
        self.assertIsNone(result["branch_ci_low"])
        self.assertIsNone(result["branch_ci_high"])
        self.assertIn("suppressed", result["branch_ci_status"])

    def test_dropped_resamples_are_counted_not_hidden(self):
        spec = "CBM->BCF,BCF->PCBM,BCF->VBM"
        time, stack, edges = self._per_file(spec, [6.0, 3.0, 1.0])
        result = bootstrap_branch_probability(
            time, stack, GROUPS, edges, "BCF", ["PCBM"], ["VBM"],
            n_resamples=10, seed=4,
        )
        self.assertEqual(result["bootstrap_requested"], 10)
        self.assertIn("bootstrap_fit_successes", result)
        self.assertIn("branch_status_counts", result)
        # Ten resamples is below the floor, so no interval may be claimed.
        self.assertIsNone(result["branch_ci_low"])

    def test_one_file_is_rejected(self):
        spec = "CBM->BCF,BCF->PCBM,BCF->VBM"
        time, stack, edges = self._per_file(spec, [6.0, 3.0, 1.0], n_files=1)
        with self.assertRaises(BranchingError):
            bootstrap_branch_probability(
                time, stack, GROUPS, edges, "BCF", ["PCBM"], ["VBM"], n_resamples=5
            )


class ExtractionCompetitionTests(unittest.TestCase):
    def _fit(self):
        spec = "CBM->BCF,BCF->PCBM,PCBM->BCF,BCF->VBM"
        edges = parse_edges(spec, GROUPS)
        K = build_rate_matrix([20.0, 10.0, 6.0, 0.5], edges, len(GROUPS))
        time = np.linspace(0.0, 4.0, 401)
        observed = propagate(K, np.array([1.0, 0.0, 0.0, 0.0]), time)
        return fit_master_equation(time, observed, GROUPS, edges)

    def test_sink_augmentation_keeps_columns_summing_to_zero(self):
        K = _matrix("BCF->PCBM,BCF->VBM", [2.0, 1.0])
        augmented, names = augment_with_sink(K, GROUPS, "PCBM", 3.0)
        self.assertEqual(names[-1], EXTRACTED)
        np.testing.assert_allclose(augmented.sum(axis=0), 0.0, atol=1e-12)

    def test_yields_are_conserved_and_monotonic(self):
        fit = self._fit()
        payload = extraction_competition(
            fit, "PCBM", ["VBM"], [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        )
        extracted = [p["extracted_yield"] for p in payload["points"]]
        for point in payload["points"]:
            self.assertAlmostEqual(
                point["extracted_yield"] + point["recombined_yield"], 1.0, places=8
            )
            self.assertLess(point["unresolved"], 1e-8)
        self.assertEqual(extracted, sorted(extracted))
        self.assertTrue(payload["extracted_yield_is_monotonic_in_k_escape"])

    def test_zero_escape_extracts_nothing(self):
        fit = self._fit()
        payload = extraction_competition(fit, "PCBM", ["VBM"], [0.0, 1.0])
        self.assertAlmostEqual(payload["points"][0]["extracted_yield"], 0.0, places=10)
        self.assertAlmostEqual(payload["points"][0]["recombined_yield"], 1.0, places=8)

    def test_crossover_is_where_the_yields_are_equal(self):
        fit = self._fit()
        payload = extraction_competition(
            fit, "PCBM", ["VBM"], list(np.geomspace(1e-3, 1e3, 30))
        )
        rate = payload["required_escape_rate_per_ns"]
        self.assertIsNotNone(rate)
        self.assertAlmostEqual(
            payload["required_escape_time_ns"], 1.0 / rate, places=10
        )
        single = extraction_competition(fit, "PCBM", ["VBM"], [rate])["points"][0]
        self.assertAlmostEqual(
            single["extracted_yield"], single["recombined_yield"], places=7
        )

    def test_no_crossover_is_reported_rather_than_invented(self):
        fit = self._fit()
        payload = extraction_competition(fit, "PCBM", ["VBM"], [1e-6, 1e-5])
        self.assertIsNone(payload["required_escape_rate_per_ns"])
        self.assertIn("does not overtake", payload["crossover_note"])

    def test_sink_group_cannot_also_be_the_recombination_outcome(self):
        fit = self._fit()
        with self.assertRaises(BranchingError):
            extraction_competition(fit, "PCBM", ["PCBM"], [1.0])

    def test_negative_escape_rate_is_rejected(self):
        fit = self._fit()
        with self.assertRaises(BranchingError):
            extraction_competition(fit, "PCBM", ["VBM"], [-1.0])

    def test_every_field_is_labelled_counterfactual(self):
        fit = self._fit()
        payload = extraction_competition(fit, "PCBM", ["VBM"], [0.1, 1.0])
        for value in payload["observable_class"].values():
            self.assertEqual(value, "counterfactual")
        self.assertIn("not a device collection efficiency", payload["interpretation"])


if __name__ == "__main__":
    unittest.main()
