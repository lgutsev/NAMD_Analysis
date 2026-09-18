"""Threshold-based transfer counting, and what it must refuse to say.

The cutoff is a reporting convention, not a measurement (Toldo et al., PCCP 25,
8293-8316, 2023).  These tests pin the two consequences: a count that depends
on the cutoff is visible as such, and a band relabelling can never register as
transfer because only the occupation-driven component is ever consumed.
"""

import unittest

import numpy as np

from namd_analysis.transfer import (
    DEFAULT_THRESHOLD_PAIRS,
    TOLDO_2023,
    TransferError,
    analyze_transfer,
    continuous_change,
    count_threshold_events,
    occupation_driven_trajectory,
    population_decay,
)

GROUPS = ["perovskite", "BCF", "PCBM"]


def clean_transfer(n=200):
    """BCF -> PCBM, occupation only: donor 0.95 -> 0.02, acceptor the mirror."""
    donor = np.linspace(0.95, 0.02, n)
    acceptor = 0.97 - donor
    perovskite = np.full(n, 0.03)
    return np.column_stack([perovskite, donor, acceptor])


class TrajectoryTests(unittest.TestCase):
    def test_per_step_changes_accumulate_from_the_initial_population(self):
        driven = np.zeros((4, 3))
        driven[1] = [0.0, -0.1, 0.1]
        driven[2] = [0.0, -0.2, 0.2]
        traj = occupation_driven_trajectory([0.0, 1.0, 0.0], driven)
        np.testing.assert_allclose(traj[0], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(traj[2], [0.0, 0.7, 0.3])
        np.testing.assert_allclose(traj[3], [0.0, 0.7, 0.3])

    def test_a_mismatched_initial_vector_is_refused(self):
        with self.assertRaises(TransferError):
            occupation_driven_trajectory([0.0, 1.0], np.zeros((4, 3)))


class ThresholdSweepTests(unittest.TestCase):
    def test_a_clean_transfer_is_counted_once_at_every_cutoff(self):
        traj = clean_transfer()
        rows = count_threshold_events(traj[:, 1], traj[:, 2])
        self.assertEqual([r.n_events for r in rows], [1, 1, 1])

    def test_one_decay_is_not_counted_many_times(self):
        # Jitter around the cutoff must not re-arm the counter.
        donor = np.concatenate([
            np.full(20, 0.95),
            np.array([0.09, 0.11, 0.08, 0.12, 0.07]),
            np.full(20, 0.05),
        ])
        acceptor = 1.0 - donor
        rows = count_threshold_events(donor, acceptor, [(0.9, 0.1)])
        self.assertEqual(rows[0].n_events, 1)

    def test_a_donor_that_recovers_can_transfer_again(self):
        donor = np.concatenate([
            np.full(10, 0.95), np.full(10, 0.05),
            np.full(10, 0.95), np.full(10, 0.05),
        ])
        acceptor = 1.0 - donor
        rows = count_threshold_events(donor, acceptor, [(0.9, 0.1)])
        self.assertEqual(rows[0].n_events, 2)

    def test_the_acceptor_must_confirm_the_transfer(self):
        # Donor decays, but into something that is not the acceptor.
        donor = np.concatenate([np.full(10, 0.95), np.full(10, 0.02)])
        acceptor = np.full(20, 0.0)
        rows = count_threshold_events(donor, acceptor, [(0.9, 0.1)])
        self.assertEqual(
            rows[0].n_events, 0, "a donor emptying elsewhere is not transfer here"
        )

    def test_a_count_that_depends_on_the_cutoff_is_visible(self):
        # Donor only ever falls to 0.25, so the strict cutoff sees nothing.
        donor = np.concatenate([np.full(20, 0.95), np.full(20, 0.25)])
        acceptor = 1.0 - donor
        analysis = analyze_transfer(
            "BCF", "PCBM", GROUPS,
            np.column_stack([np.zeros(40), donor, acceptor]),
        )
        self.assertEqual(analysis.counts, [0, 0, 1])
        self.assertTrue(analysis.threshold_dependent)

    def test_a_stable_count_is_reported_as_stable(self):
        analysis = analyze_transfer("BCF", "PCBM", GROUPS, clean_transfer())
        self.assertFalse(analysis.threshold_dependent)

    def test_an_invalid_threshold_pair_is_refused(self):
        traj = clean_transfer()
        for bad in [(0.1, 0.9), (0.5, 0.5), (1.5, 0.1)]:
            with self.assertRaises(TransferError):
                count_threshold_events(traj[:, 1], traj[:, 2], [bad])

    def test_the_sweep_default_is_three_pairs(self):
        self.assertEqual(DEFAULT_THRESHOLD_PAIRS, ((0.9, 0.1), (0.8, 0.2), (0.7, 0.3)))


class DelocalizedTests(unittest.TestCase):
    """Zero events is not evidence of no transfer."""

    def test_occupation_can_move_while_every_count_is_zero(self):
        # A delocalized carrier: 0.45 -> 0.20 on the donor. Real movement,
        # never near 0.9, so no transition is ever counted.
        donor = np.linspace(0.45, 0.20, 100)
        acceptor = 0.65 - donor
        analysis = analyze_transfer(
            "BCF", "PCBM", GROUPS,
            np.column_stack([np.full(100, 0.35), donor, acceptor]),
            time_ns=np.linspace(0.0, 1.0, 100),
        )
        self.assertEqual(analysis.counts, [0, 0, 0])
        self.assertTrue(analysis.continuous.donor_loss_with_acceptor_gain)
        self.assertAlmostEqual(analysis.continuous.donor_net, -0.25, places=6)
        self.assertAlmostEqual(analysis.continuous.acceptor_net, 0.25, places=6)
        # The decay fit is the observable that still works here.
        self.assertTrue(analysis.decay.fitted)
        self.assertIn("delocalization_note", analysis.to_dict())

    def test_the_report_says_zero_counts_may_mean_delocalized(self):
        payload = analyze_transfer("BCF", "PCBM", GROUPS, clean_transfer()).to_dict()
        self.assertIn("not evidence of no transfer", payload["delocalization_note"])
        self.assertIn("arbitrary", payload["note"])
        self.assertEqual(payload["citation"], TOLDO_2023)


class ContinuousTests(unittest.TestCase):
    def test_concomitant_loss_and_gain_is_measured_not_assumed(self):
        traj = clean_transfer(50)
        change = continuous_change(traj[:, 1], traj[:, 2])
        self.assertTrue(change.donor_loss_with_acceptor_gain)
        self.assertAlmostEqual(change.concomitant, 0.93, places=6)

    def test_a_donor_and_acceptor_both_rising_is_not_concomitant(self):
        donor = np.linspace(0.1, 0.4, 50)
        acceptor = np.linspace(0.1, 0.4, 50)
        change = continuous_change(donor, acceptor)
        self.assertFalse(change.donor_loss_with_acceptor_gain)
        self.assertAlmostEqual(change.concomitant, 0.0, places=12)

    def test_the_raw_change_is_reported_even_with_no_events(self):
        donor = np.linspace(0.45, 0.44, 30)
        payload = analyze_transfer(
            "BCF", "PCBM", GROUPS,
            np.column_stack([np.zeros(30), donor, 1.0 - donor]),
        ).to_dict()
        self.assertEqual(sum(r["n_events"] for r in payload["threshold_sweep"]), 0)
        self.assertAlmostEqual(payload["continuous"]["donor_net"], -0.01, places=6)


class DecayTests(unittest.TestCase):
    def test_a_known_timescale_is_recovered(self):
        t = np.linspace(0.0, 2.0, 400)
        tau = 0.25
        donor = 0.9 * np.exp(-t / tau) + 0.05
        decay = population_decay(t, donor)
        self.assertTrue(decay.fitted)
        self.assertAlmostEqual(decay.tau_ns, tau, places=3)
        self.assertGreater(decay.r_squared, 0.999)

    def test_a_flat_donor_is_refused_rather_than_fitted(self):
        t = np.linspace(0.0, 1.0, 50)
        decay = population_decay(t, np.full(50, 0.4))
        self.assertFalse(decay.fitted)
        self.assertIn("flat", decay.reason)

    def test_a_rising_donor_is_not_called_a_decay(self):
        t = np.linspace(0.0, 1.0, 100)
        decay = population_decay(t, 0.2 + 0.5 * t)
        if decay.fitted:
            self.assertGreater(decay.tau_ns, 0.0)
        else:
            self.assertTrue(decay.reason)

    def test_too_few_samples_is_refused(self):
        decay = population_decay(np.arange(3.0), np.array([1.0, 0.5, 0.2]))
        self.assertFalse(decay.fitted)
        self.assertIn("four", decay.reason)


class GuardTests(unittest.TestCase):
    def test_an_unknown_group_is_refused(self):
        with self.assertRaises(TransferError):
            analyze_transfer("BCF", "nope", GROUPS, clean_transfer())

    def test_donor_and_acceptor_must_differ(self):
        with self.assertRaises(TransferError):
            analyze_transfer("BCF", "BCF", GROUPS, clean_transfer())

    def test_the_quantity_is_named_as_occupation_only(self):
        payload = analyze_transfer("BCF", "PCBM", GROUPS, clean_transfer()).to_dict()
        self.assertIn("occupation-driven", payload["quantity"])
        self.assertIn("character-driven component is excluded", payload["quantity"])


if __name__ == "__main__":
    unittest.main()
