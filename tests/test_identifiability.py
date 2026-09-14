"""Identifiability diagnostics: null space, optimizer bounds, bootstrap, dof.

Refusing to quote a rate the data did not determine is this package's main
scientific safeguard, so each of the four checks that can fire gets its own
known-answer case, and the well-posed control is re-checked every time to make
sure a tightened check has not started rejecting good rates.
"""

import bootstrap  # noqa: F401

import unittest

import numpy as np
from scipy.linalg import helmert

from namd_analysis.kinetics import (
    BOOTSTRAP_MIN_IDENTIFIED_FRACTION,
    NULL_PARTICIPATION_LIMIT,
    bootstrap_rates,
    build_rate_matrix,
    fit_master_equation,
    independent_residual_count,
    null_space_analysis,
    parse_edges,
    propagate,
)

GROUPS = ["CBM", "BCF", "PCBM", "VBM"]
SCHEME = "CBM->BCF,BCF->CBM,BCF->PCBM,PCBM->VBM"
TRUE_RATES = [8.0, 2.0, 3.0, 0.5]


def _observed(time=None, scheme=SCHEME, rates=None, groups=GROUPS, p0=None):
    time = np.linspace(0.0, 2.0, 801) if time is None else time
    rates = TRUE_RATES if rates is None else rates
    edges = parse_edges(scheme, groups)
    K = build_rate_matrix(rates, edges, len(groups))
    start = np.zeros(len(groups)) if p0 is None else np.asarray(p0, float)
    if p0 is None:
        start[0] = 1.0
    return time, propagate(K, start, time), edges


class NullSpaceUnitTests(unittest.TestCase):
    """The SVD analysis itself, on matrices whose null space is known."""

    def test_a_full_rank_jacobian_has_no_null_space(self):
        rng = np.random.default_rng(0)
        jacobian = rng.standard_normal((200, 4))
        result = null_space_analysis(jacobian)
        self.assertEqual(result["rank"], 4)
        self.assertEqual(result["nullity"], 0)
        np.testing.assert_allclose(result["participation"], 0.0)

    def test_a_single_blind_column_participates_fully(self):
        rng = np.random.default_rng(1)
        jacobian = rng.standard_normal((200, 4))
        jacobian[:, 2] = 0.0
        result = null_space_analysis(jacobian)
        self.assertEqual(result["rank"], 3)
        self.assertEqual(result["nullity"], 1)
        self.assertAlmostEqual(result["participation"][2], 1.0, places=10)
        for index in (0, 1, 3):
            self.assertLess(result["participation"][index], 1e-10)

    def test_a_linear_combination_degeneracy_with_no_zero_column(self):
        # Every column is large, no pair is collinear, and yet the combination
        # c = (1, 1, -1) leaves the residuals exactly unchanged. Nothing in a
        # column norm or a pairwise correlation sees this; the null space does.
        rng = np.random.default_rng(2)
        u = rng.standard_normal(300)
        v = rng.standard_normal(300)
        jacobian = np.column_stack([u, v, u + v])
        result = null_space_analysis(jacobian)
        self.assertEqual(result["rank"], 2)
        self.assertEqual(result["nullity"], 1)
        for index in range(3):
            self.assertGreater(np.linalg.norm(jacobian[:, index]), 1.0)
            self.assertAlmostEqual(
                result["participation"][index], 1.0 / np.sqrt(3.0), places=8
            )
            self.assertGreater(result["participation"][index], NULL_PARTICIPATION_LIMIT)

    def test_a_two_way_degeneracy_splits_participation_evenly(self):
        rng = np.random.default_rng(3)
        u = rng.standard_normal(300)
        w = rng.standard_normal(300)
        jacobian = np.column_stack([u, u, w])
        result = null_space_analysis(jacobian)
        self.assertEqual(result["nullity"], 1)
        np.testing.assert_allclose(
            result["participation"][:2], 1.0 / np.sqrt(2.0), atol=1e-8
        )
        self.assertLess(result["participation"][2], 1e-8)

    def test_participation_is_bounded_and_the_tolerance_scales(self):
        rng = np.random.default_rng(4)
        jacobian = rng.standard_normal((120, 5))
        small = null_space_analysis(jacobian)
        large = null_space_analysis(jacobian * 1e6)
        self.assertAlmostEqual(large["tolerance"] / small["tolerance"], 1e6, places=3)
        self.assertEqual(small["rank"], large["rank"])
        self.assertTrue(np.all(small["participation"] >= 0))
        self.assertTrue(np.all(small["participation"] <= 1))


class NullSpaceFitTests(unittest.TestCase):
    """The same analysis reaching the per-rate flags of a real fit."""

    def test_a_well_posed_sparse_fit_stays_fully_identified(self):
        time, observed, edges = _observed()
        fit = fit_master_equation(time, observed, GROUPS, edges)
        self.assertEqual(fit.jacobian_rank, len(edges))
        self.assertEqual(fit.jacobian_nullity, 0)
        for estimate, truth in zip(fit.rates, TRUE_RATES):
            self.assertTrue(estimate.identified, estimate.unidentified_reason)
            self.assertTrue(estimate.null_space_identifiable)
            self.assertLess(estimate.null_space_participation, 1e-8)
            self.assertAlmostEqual(estimate.rate_per_ns, truth, places=5)

    def test_noise_does_not_make_a_well_posed_fit_null_deficient(self):
        rng = np.random.default_rng(11)
        time, observed, edges = _observed()
        noisy = observed + 0.003 * rng.standard_normal(observed.shape)
        fit = fit_master_equation(time, noisy, GROUPS, edges)
        self.assertEqual(fit.jacobian_nullity, 0)
        self.assertTrue(all(r.identified for r in fit.rates))

    def test_a_blind_rate_is_caught_by_the_null_space_too(self):
        edges = parse_edges("CBM->BCF,BCF->VBM", GROUPS)
        K = build_rate_matrix([6.0, 1.0], edges, len(GROUPS))
        time = np.linspace(0.0, 1.5, 300)
        observed = propagate(K, np.array([1.0, 0.0, 0.0, 0.0]), time)
        fit = fit_master_equation(
            time, observed, GROUPS, parse_edges("CBM->BCF,BCF->VBM,PCBM->BCF", GROUPS)
        )
        self.assertEqual(fit.jacobian_nullity, 1)
        blind = next(r for r in fit.rates if r.name == "PCBM->BCF")
        self.assertFalse(blind.identified)
        self.assertFalse(blind.null_space_identifiable)
        self.assertAlmostEqual(blind.null_space_participation, 1.0, places=6)
        for name in ("CBM->BCF", "BCF->VBM"):
            self.assertTrue(next(r for r in fit.rates if r.name == name).identified)

    def test_a_rate_with_no_flow_in_the_data_is_never_identified(self):
        # Before the null-space check this came back identified with a standard
        # error of exactly zero: its Jacobian column is small but not zero, so
        # the blind-column test missed it, and the pseudo-inverse reports ZERO
        # variance for a direction it discarded rather than infinite.
        groups = ["A", "B", "C"]
        time = np.linspace(0.0, 2.0, 401)
        K = build_rate_matrix([5.0, 1.0], parse_edges("A->B,B->C", groups), 3)
        observed = propagate(K, np.array([1.0, 0.0, 0.0]), time)
        fit = fit_master_equation(
            time, observed, groups, parse_edges("A->B,B->C,C->B", groups)
        )
        reverse = next(r for r in fit.rates if r.name == "C->B")
        self.assertFalse(reverse.identified)
        self.assertIsNone(reverse.stderr_per_ns)
        self.assertIn("null-space", reverse.unidentified_reason)
        self.assertGreater(reverse.null_space_participation, NULL_PARTICIPATION_LIMIT)
        for name in ("A->B", "B->C"):
            self.assertTrue(next(r for r in fit.rates if r.name == name).identified)

    def test_a_dense_overparameterized_scheme_is_rank_deficient(self):
        rng = np.random.default_rng(4)
        time, observed, _ = _observed()
        noisy = observed + 0.003 * rng.standard_normal(observed.shape)
        dense = parse_edges("dense", GROUPS)
        fit = fit_master_equation(time, noisy, GROUPS, dense)
        self.assertLess(fit.jacobian_rank, len(dense))
        self.assertGreater(fit.jacobian_nullity, 0)
        self.assertTrue(
            any(r.null_space_participation > NULL_PARTICIPATION_LIMIT for r in fit.rates)
        )
        self.assertTrue(any("nullity" in w for w in fit.warnings), fit.warnings)
        # The existing checks are supplements, not replacements.
        self.assertTrue(any(r.degenerate_with for r in fit.rates))

    def test_the_report_carries_the_null_space_fields(self):
        time, observed, edges = _observed()
        payload = fit_master_equation(time, observed, GROUPS, edges).as_dict()
        block = payload["identifiability"]
        for key in ("jacobian_rank", "jacobian_nullity", "null_space_tolerance",
                    "null_space_participation", "null_space_identifiable"):
            self.assertIn(key, block)
        self.assertEqual(len(block["null_space_participation"]), len(edges))
        self.assertIn("supplements", block["null_space_note"])


class OptimizerBoundTests(unittest.TestCase):
    """A rate pinned to the edge of the allowed range is not an estimate."""

    def _pinned(self, bound_decades=1.0):
        groups = ["A", "B"]
        time = np.linspace(0.0, 1.0, 201)
        K = build_rate_matrix([3.0], parse_edges("A->B", groups), 2)
        observed = propagate(K, np.array([1.0, 0.0]), time)
        return groups, time, fit_master_equation(
            time, observed, groups, parse_edges("A->B,B->A", groups),
            bound_decades=bound_decades,
        )

    def test_a_rate_pinned_to_the_lower_bound_is_not_identified(self):
        # The data holds no reverse flow at all, so the reverse rate runs to
        # the edge of the allowed range and stops there. The covariance is a
        # local quadratic picture that does not describe an edge.
        _, _, fit = self._pinned()
        reverse = next(r for r in fit.rates if r.name == "B->A")
        self.assertTrue(reverse.at_optimizer_bound)
        self.assertEqual(reverse.optimizer_bound, "lower")
        self.assertFalse(reverse.identified)
        self.assertIn("boundary", reverse.unidentified_reason)
        self.assertIn("do not determine an interior estimate",
                      reverse.unidentified_reason)

    def test_the_bound_hit_survives_a_small_standard_error(self):
        # This is the whole point: the rate sits exactly on the bound with a
        # perfectly ordinary-looking relative standard error, and is refused
        # anyway. Before this check it would have been quoted.
        _, _, fit = self._pinned()
        reverse = next(r for r in fit.rates if r.name == "B->A")
        self.assertIsNotNone(reverse.relative_stderr)
        self.assertFalse(reverse.identified)

    def test_the_bound_is_reported_in_the_fit_payload_and_warnings(self):
        _, _, fit = self._pinned()
        payload = fit.as_dict()["identifiability"]
        self.assertEqual(payload["rates_at_an_optimizer_bound"], ["B->A"])
        entry = next(r for r in payload_rates(fit) if r["transition"] == "B->A")
        self.assertTrue(entry["at_optimizer_bound"])
        self.assertEqual(entry["optimizer_bound"], "lower")
        self.assertTrue(
            any("edge of the allowed range" in w for w in fit.warnings), fit.warnings
        )

    def test_the_freely_determined_rate_beside_it_is_still_identified(self):
        _, _, fit = self._pinned()
        forward = next(r for r in fit.rates if r.name == "A->B")
        self.assertFalse(forward.at_optimizer_bound)
        self.assertIsNone(forward.optimizer_bound)
        self.assertTrue(forward.identified, forward.unidentified_reason)

    def test_an_interior_optimum_reports_no_bound(self):
        time, observed, edges = _observed()
        fit = fit_master_equation(time, observed, GROUPS, edges)
        for estimate in fit.rates:
            self.assertFalse(estimate.at_optimizer_bound)
            self.assertIsNone(estimate.optimizer_bound)
        self.assertEqual(
            fit.as_dict()["identifiability"]["rates_at_an_optimizer_bound"], []
        )

    def test_widening_the_range_lets_the_same_rate_leave_the_bound(self):
        # The flag tracks the optimizer, not a fixed magnitude: with room to
        # run, the same rate stops interior and is judged on other grounds.
        _, _, wide = self._pinned(bound_decades=9.0)
        reverse = next(r for r in wide.rates if r.name == "B->A")
        self.assertFalse(reverse.at_optimizer_bound)


def payload_rates(fit):
    return fit.as_dict()["rates"]


class BootstrapIdentifiabilityTests(unittest.TestCase):
    def _per_file(self, n_files=6, noise=0.004, seed=7, n_points=201):
        time, exact, edges = _observed(np.linspace(0.0, 2.0, n_points))
        rng = np.random.default_rng(seed)
        per_file = np.stack(
            [exact + noise * rng.standard_normal(exact.shape) for _ in range(n_files)]
        )
        return time, per_file, edges

    def test_a_well_determined_rate_keeps_its_interval(self):
        time, per_file, edges = self._per_file()
        intervals, info = bootstrap_rates(
            time, per_file, GROUPS, edges, n_resamples=40, seed=1
        )
        self.assertEqual(info["converged_resamples"], 40)
        self.assertIn("CBM->BCF", intervals)
        record = info["per_rate"]["CBM->BCF"]
        self.assertEqual(record["bootstrap_ci_status"], "reported")
        self.assertEqual(record["bootstrap_identified_fraction"], 1.0)
        self.assertEqual(record["bootstrap_identified"], 40)
        low, high = intervals["CBM->BCF"]
        self.assertLess(low, high)

    def test_every_rate_reports_its_identified_counts(self):
        time, per_file, edges = self._per_file()
        _, info = bootstrap_rates(
            time, per_file, GROUPS, edges, n_resamples=25, seed=2
        )
        self.assertEqual(set(info["per_rate"]), {
            "CBM->BCF", "BCF->CBM", "BCF->PCBM", "PCBM->VBM"
        })
        for record in info["per_rate"].values():
            self.assertEqual(record["bootstrap_successes"], 25)
            self.assertIn("bootstrap_identified", record)
            self.assertIn("bootstrap_identified_fraction", record)
            self.assertIn("bootstrap_ci_status", record)
            self.assertIn("raw_optimizer_percentiles_per_ns", record)
        self.assertEqual(
            info["minimum_identified_fraction"], BOOTSTRAP_MIN_IDENTIFIED_FRACTION
        )

    def test_an_unidentified_rate_gets_no_interval_and_says_why(self):
        # The dense scheme converges on every resample while determining almost
        # nothing. Keeping those optimizer stopping points would produce a tight
        # percentile band around a number the data never fixed.
        time, per_file, _ = self._per_file(noise=0.004, seed=8)
        dense = parse_edges("dense", GROUPS)
        intervals, info = bootstrap_rates(
            time, per_file, GROUPS, dense, n_resamples=25, seed=3
        )
        self.assertGreater(info["converged_resamples"], 20)
        suppressed = info.get("suppressed_for_low_identified_fraction", [])
        self.assertTrue(suppressed, info["per_rate"])
        for name in suppressed:
            self.assertNotIn(name, intervals)
            record = info["per_rate"][name]
            self.assertIn(
                record["bootstrap_ci_status"],
                ("suppressed_low_identified_fraction", "suppressed_never_identified"),
            )
            self.assertLess(
                record["bootstrap_identified_fraction"],
                BOOTSTRAP_MIN_IDENTIFIED_FRACTION,
            )
            # The dropped draws are counted, not quietly discarded.
            self.assertEqual(record["raw_distribution_n"], info["converged_resamples"])
            self.assertIn("resamples", record["note"])
            self.assertIn("raw_optimizer_percentiles_per_ns", record)

    def test_a_reported_interval_comes_from_the_identified_draws_only(self):
        time, per_file, _ = self._per_file(noise=0.004, seed=8)
        dense = parse_edges("dense", GROUPS)
        intervals, info = bootstrap_rates(
            time, per_file, GROUPS, dense, n_resamples=25, seed=3,
            min_identified_fraction=0.0,
        )
        for name, record in info["per_rate"].items():
            if record["bootstrap_ci_status"] != "reported":
                continue
            if record["bootstrap_identified"] == record["raw_distribution_n"]:
                continue
            # A subset was used, so the interval must differ from the raw one.
            self.assertNotEqual(
                intervals[name], record["raw_optimizer_percentiles_per_ns"]
            )

    def test_too_few_successes_suppresses_every_interval(self):
        time, per_file, edges = self._per_file(n_files=4)
        intervals, info = bootstrap_rates(
            time, per_file, GROUPS, edges, n_resamples=5, seed=3
        )
        self.assertEqual(intervals, {})
        for record in info["per_rate"].values():
            self.assertEqual(
                record["bootstrap_ci_status"], "insufficient_successful_resamples"
            )
        self.assertIn("fewer than", info["note"])

    def test_the_threshold_is_exposed_as_metadata(self):
        time, per_file, edges = self._per_file()
        _, info = bootstrap_rates(
            time, per_file, GROUPS, edges, n_resamples=25, seed=4,
            min_identified_fraction=0.5,
        )
        self.assertEqual(info["minimum_identified_fraction"], 0.5)
        self.assertIn("identified in at least", info["policy"])


class ResidualDimensionTests(unittest.TestCase):
    def test_conserved_populations_lose_one_coordinate_per_time(self):
        time, observed, _ = _observed(np.linspace(0.0, 1.0, 51))
        count, per_time, conserved = independent_residual_count(observed)
        self.assertTrue(conserved)
        self.assertEqual(per_time, len(GROUPS) - 1)
        self.assertEqual(count, 50 * 3)

    def test_unconserved_populations_keep_every_coordinate(self):
        time, observed, _ = _observed(np.linspace(0.0, 1.0, 51))
        partial = observed[:, :2]  # a subset no longer sums to a constant
        count, per_time, conserved = independent_residual_count(partial)
        self.assertFalse(conserved)
        self.assertEqual(per_time, 2)
        self.assertEqual(count, 50 * 2)

    def test_helmert_contrasts_preserve_the_sum_of_squares(self):
        # The conservation-subspace representation changes the observation
        # COUNT, not the residual magnitude: the contrasts are orthonormal.
        time, observed, edges = _observed(np.linspace(0.0, 1.0, 101))
        rng = np.random.default_rng(5)
        noisy = observed + 0.002 * rng.standard_normal(observed.shape)
        noisy = noisy / noisy.sum(axis=1, keepdims=True)  # keep it conserved
        fit = fit_master_equation(time, noisy, GROUPS, edges)
        residual = fit.model - noisy
        contrast = helmert(len(GROUPS))
        redundant = float(np.sum(residual[1:] ** 2))
        transformed = float(np.sum((residual[1:] @ contrast.T) ** 2))
        self.assertAlmostEqual(redundant, transformed, places=12)
        # ... and the fit counted the reduced dimension, not the redundant one.
        self.assertTrue(fit.population_conserved)
        self.assertEqual(fit.independent_coordinates_per_time, len(GROUPS) - 1)
        self.assertEqual(fit.n_independent_residuals, (len(time) - 1) * 3)

    def test_the_conditioned_first_row_is_excluded(self):
        time, observed, edges = _observed(np.linspace(0.0, 1.0, 101))
        fit = fit_master_equation(time, observed, GROUPS, edges)
        np.testing.assert_allclose(fit.model[0], observed[0])
        self.assertEqual(fit.n_independent_residuals, (fit.n_points - 1) * 3)
        block = fit.as_dict()["residual_dimension"]
        self.assertTrue(block["conditioned_initial_row_excluded"])
        self.assertTrue(block["population_conservation_detected"])
        self.assertIn("Helmert", block["note"])

    def test_dropping_the_redundant_direction_widens_the_standard_errors(self):
        # Counting the conservation direction as independent information
        # divides the residual variance by observations that were never free,
        # which understates errors that are already a lower bound.
        time, observed, edges = _observed(np.linspace(0.0, 1.0, 101))
        rng = np.random.default_rng(6)
        noisy = observed + 0.002 * rng.standard_normal(observed.shape)
        noisy = noisy / noisy.sum(axis=1, keepdims=True)
        fit = fit_master_equation(time, noisy, GROUPS, edges)
        n_groups = len(GROUPS)
        redundant_dof = fit.n_points * n_groups - fit.n_parameters - (n_groups - 1)
        honest_dof = (
            fit.n_independent_residuals - fit.n_parameters - (n_groups - 1)
        )
        self.assertLess(honest_dof, redundant_dof)
        widening = np.sqrt(redundant_dof / honest_dof)
        self.assertGreater(widening, 1.0)
        for estimate in fit.rates:
            self.assertIsNotNone(estimate.stderr_per_ns)
            self.assertGreater(estimate.stderr_per_ns, 0.0)

    def test_compare_schemes_counts_the_same_independent_observations(self):
        # compare-schemes scores in the Helmert subspace; the kinetic covariance
        # scales by the same dimension. The two must not disagree about how much
        # independent information the data holds.
        from namd_analysis.comparison import compare_schemes

        time, observed, _ = _observed(np.linspace(0.0, 1.0, 61))
        payload, fits = compare_schemes(
            time,
            observed,
            GROUPS,
            {"sparse": SCHEME, "sequential": "CBM->BCF,BCF->PCBM,PCBM->VBM"},
            n_starts=2,
        )
        expected = (len(time) - 1) * (len(GROUPS) - 1)
        self.assertEqual(payload["observation_count"]["n"], expected)
        self.assertTrue(
            payload["observation_count"]["matches_kinetics_covariance_dimension"]
        )
        for fit in fits.values():
            self.assertEqual(fit.n_independent_residuals, expected)

    def test_the_standard_error_note_stays_a_local_approximation(self):
        time, observed, edges = _observed(np.linspace(0.0, 1.0, 101))
        payload = fit_master_equation(time, observed, GROUPS, edges).as_dict()
        self.assertIn("asymptotic", payload["standard_errors"]["method"])
        self.assertIn("LOWER BOUND", payload["standard_errors"]["caveat"])


if __name__ == "__main__":
    unittest.main()
