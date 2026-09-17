"""Multistate manifolds, band participation, and frame alignment.

A region where three states approach each other is not a sequence of two-state
crossings, and a state can sit in the middle of one while being irrelevant to
*fragment* transfer.  Both distinctions are load-bearing for the mechanism, so
both are pinned here.
"""

import unittest

import numpy as np

from namd_analysis.crossings import CharacterSeries, CouplingSeries
from namd_analysis.episodes import (
    EpisodeError,
    alignment_scan,
    band_participation,
    manifold_table,
)

GROUPS = ["perovskite", "BCF", "PCBM"]
PEROV, BCF, PCBM = 0, 1, 2


def build(nframes=61, third_offset=1.0, third_group=PCBM, coupling=0.02):
    """A real two-state avoided crossing, plus a third state nearby.

    The mixing has to be *graded* -- a binary swap makes min(BCF, PCBM)
    constant, and then gap closure has nothing to correlate against. Real
    character varies continuously through the crossing, so the fixture does
    too: cos^2/sin^2 of the mixing angle, with the gap minimal where they meet.
    """
    detuning = np.linspace(-1.0, 1.0, nframes) * 0.5
    theta = 0.5 * np.arctan2(2.0 * coupling, detuning)
    half = np.sqrt((detuning / 2.0) ** 2 + coupling**2)

    w = np.zeros((nframes, 3, 3))
    w[:, 0, BCF] = np.cos(theta) ** 2
    w[:, 0, PCBM] = np.sin(theta) ** 2
    w[:, 1, BCF] = np.sin(theta) ** 2
    w[:, 1, PCBM] = np.cos(theta) ** 2
    w[:, 2, third_group] = 1.0
    energies = np.column_stack([-half, half, half + third_offset])
    character = CharacterSeries(
        frames=np.arange(1, nframes + 1),
        bands=np.array([977, 978, 979]),
        groups=list(GROUPS),
        weights=w,
        captured=np.full((nframes, 3), 0.51),
    )
    couplings = CouplingSeries(
        frames=character.frames.copy(), energies=energies,
        nac=np.zeros((nframes, 3, 3)), nac_unit="eV", dt_fs=1.0, ceiling_ev=None,
    )
    return character, couplings


class ManifoldTableTests(unittest.TestCase):
    def test_every_state_pair_and_both_weight_conventions_appear(self):
        character, couplings = build()
        header, rows = manifold_table(character, couplings, [977, 978, 979], 10, 40)
        self.assertEqual(len(rows), 31)
        for column in ("E_977_eV", "E_979_eV", "gap_977_978_eV", "gap_978_979_eV",
                       "gap_977_979_eV", "abs_nac_977_978_eV",
                       "977_BCF", "977_raw_BCF", "979_captured", "979_dominant"):
            self.assertIn(column, header)

    def test_raw_weights_are_the_normalized_ones_times_captured(self):
        character, couplings = build()
        header, rows = manifold_table(character, couplings, [977, 978, 979], 10, 40)
        raw = header.index("977_raw_PCBM")
        norm = header.index("977_PCBM")
        cap = header.index("977_captured")
        for row in rows:
            self.assertAlmostEqual(row[raw], row[norm] * row[cap], places=12)

    def test_occupations_are_blank_rather_than_invented(self):
        character, couplings = build()
        header, rows = manifold_table(character, couplings, [977, 978], 10, 20)
        self.assertNotIn("P_977", header)
        header2, rows2 = manifold_table(
            character, couplings, [977, 978], 10, 20,
            occupations={977: np.full(61, 0.4)},
        )
        self.assertIn("P_977", header2)
        self.assertIn("P_978", header2)
        p977 = header2.index("P_977")
        p978 = header2.index("P_978")
        self.assertAlmostEqual(rows2[0][p977], 0.4)
        self.assertEqual(rows2[0][p978], "", "a band with no occupation stays blank")

    def test_a_band_outside_the_table_is_refused(self):
        character, couplings = build()
        with self.assertRaises(EpisodeError):
            manifold_table(character, couplings, [977, 999], 10, 20)


class ParticipationTests(unittest.TestCase):
    def test_a_pure_acceptor_state_that_crowds_the_pair_is_energetic_only(self):
        # This is the real band-979 case: it comes closer than the pair does to
        # itself, but its character never leaves PCBM, so it cannot carry
        # BCF<->PCBM transfer.
        character, couplings = build(third_offset=0.002, third_group=PCBM)
        out = band_participation(character, couplings, 979, (977, 978), 20, 40)
        self.assertEqual(out["verdict"], "energetically_involved_only")
        self.assertTrue(out["energetic"]["comes_closer_than_the_pair_does"])
        self.assertEqual(out["character"]["dominant_fragments_seen"], ["PCBM"])
        self.assertAlmostEqual(out["character"]["BCF_range"], 0.0, places=12)
        self.assertIn("cannot by itself carry fragment transfer", out["why"])

    def test_a_distant_unchanging_state_is_not_involved(self):
        character, couplings = build(third_offset=5.0, third_group=PCBM)
        out = band_participation(character, couplings, 979, (977, 978), 20, 40)
        self.assertEqual(out["verdict"], "not_involved")
        self.assertFalse(out["energetic"]["comes_closer_than_the_pair_does"])

    def test_a_state_that_exchanges_between_the_fragments_does_participate(self):
        character, couplings = build(third_offset=5.0)
        # Make 979 itself swap BCF <-> PCBM.
        character.weights[:30, 2] = [0.0, 0.9, 0.1]
        character.weights[30:, 2] = [0.0, 0.1, 0.9]
        out = band_participation(character, couplings, 979, (977, 978), 1, 61)
        self.assertEqual(out["verdict"], "participates_in_fragment_transfer")
        self.assertTrue(out["character"]["exchanges_between_donor_and_acceptor"])

    def test_energetic_proximity_is_not_occupation(self):
        character, couplings = build(third_offset=0.002)
        out = band_participation(character, couplings, 979, (977, 978), 20, 40)
        self.assertIsNone(out["occupation"])
        self.assertIn("are not occupation", out["occupation_note"])

    def test_a_supplied_occupation_is_summarized(self):
        character, couplings = build(third_offset=0.002)
        out = band_participation(
            character, couplings, 979, (977, 978), 20, 40,
            occupation=np.linspace(0.1, 0.5, 21),
        )
        self.assertAlmostEqual(out["occupation"]["net_change"], 0.4, places=9)
        self.assertNotIn("occupation_note", out)

    def test_an_empty_window_is_refused(self):
        character, couplings = build()
        with self.assertRaises(EpisodeError):
            band_participation(character, couplings, 979, (977, 978), 500, 600)


class AlignmentScanTests(unittest.TestCase):
    def test_an_aligned_pair_peaks_at_offset_zero(self):
        character, couplings = build()
        out = alignment_scan(character, couplings, 977, 978)
        self.assertTrue(out["ran"])
        self.assertEqual(out["best_offset"], 0)
        self.assertTrue(out["aligned"])
        self.assertTrue(out["unique"])

    def test_a_deliberately_shifted_axis_does_not_peak_at_zero(self):
        character, couplings = build()
        shift = 4
        couplings.energies = np.roll(couplings.energies, shift, axis=0)
        out = alignment_scan(character, couplings, 977, 978)
        self.assertTrue(out["ran"])
        self.assertNotEqual(
            out["best_offset"], 0,
            "a shifted energy axis must not report itself aligned",
        )
        self.assertFalse(out["aligned"])

    def test_a_pair_with_no_mixing_variation_refuses_rather_than_scoring(self):
        # Both states pure and unchanging: gap closure has nothing to correlate
        # against, and a number here would be meaningless.
        n = 40
        w = np.zeros((n, 2, 3))
        w[:, 0] = [0.0, 1.0, 0.0]
        w[:, 1] = [0.0, 0.0, 1.0]
        character = CharacterSeries(
            frames=np.arange(1, n + 1), bands=np.array([977, 978]),
            groups=list(GROUPS), weights=w, captured=np.full((n, 2), 0.5),
        )
        couplings = CouplingSeries(
            frames=character.frames.copy(),
            energies=np.column_stack([np.zeros(n), np.full(n, 0.02)]),
            nac=None, nac_unit="eV", dt_fs=1.0, ceiling_ev=None,
        )
        out = alignment_scan(character, couplings, 977, 978)
        self.assertFalse(out["ran"])
        self.assertIn("no variation", out["reason"])

    def test_the_margin_over_the_next_offset_is_reported(self):
        character, couplings = build()
        out = alignment_scan(character, couplings, 977, 978)
        self.assertIn("margin_over_second", out)
        self.assertGreaterEqual(out["margin_over_second"], 0.0)
        self.assertIn("not a confirmation either", out["note"])


if __name__ == "__main__":
    unittest.main()
