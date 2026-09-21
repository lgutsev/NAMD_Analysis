"""Crossing episodes, and the three things they must refuse to conflate.

An episode is the unit at which "did charge follow the character?" is a
well-posed question: a passage through an avoided crossing spans several
frames, and a single finite difference cannot answer it.
"""

import unittest

import numpy as np

from namd_analysis.crossings import CharacterSeries, CouplingSeries
from namd_analysis.episodes import (
    NO_HOP_RECORD,
    CrossingEpisode,
    EpisodeError,
    classify_episode,
    find_episodes,
    fixed_vs_dynamic_late,
    isolation_check,
    local_crossing_table,
    raw_weight_check,
)

GROUPS = ["perovskite", "BCF", "PCBM"]


def there_and_back(nframes=40, swap_in=15, swap_out=20):
    """977 is BCF, 978 is PCBM; they exchange between swap_in and swap_out."""
    w = np.zeros((nframes, 2, 3))
    for k in range(nframes):
        exchanged = swap_in <= k < swap_out
        w[k, 0] = [0.0, 0.05, 0.95] if exchanged else [0.0, 0.95, 0.05]
        w[k, 1] = [0.0, 0.95, 0.05] if exchanged else [0.0, 0.05, 0.95]
    return CharacterSeries(
        frames=np.arange(1, nframes + 1),
        bands=np.array([977, 978]),
        groups=list(GROUPS),
        weights=w,
        captured=np.full((nframes, 2), 0.51),
    )


def couplings_for(character, gap=0.02):
    n = character.frames.size
    e = np.zeros((n, 2))
    e[:, 1] = gap
    return CouplingSeries(
        frames=character.frames.copy(), energies=e, nac=None,
        nac_unit="eV", dt_fs=1.0, ceiling_ev=None,
    )


class EpisodeGroupingTests(unittest.TestCase):
    def test_a_there_and_back_passage_is_one_episode_not_two(self):
        # Counting the exchange and the exchange back as two events would
        # double every rate derived from the crossing.
        character = there_and_back()
        episodes = find_episodes(character, couplings_for(character), 977, 978)
        self.assertEqual(len(episodes), 1)
        self.assertLessEqual(episodes[0].first_frame, 15)
        self.assertGreaterEqual(episodes[0].last_frame, 20)

    def test_the_net_character_can_return_to_where_it_started(self):
        character = there_and_back()
        ep = find_episodes(character, couplings_for(character), 977, 978)[0]
        self.assertEqual(ep.dominant_i_before, ep.dominant_i_after)
        self.assertEqual(ep.dominant_j_before, ep.dominant_j_after)

    def test_well_separated_passages_stay_separate(self):
        w = np.zeros((80, 2, 3))
        for k in range(80):
            exchanged = (10 <= k < 14) or (60 <= k < 64)
            w[k, 0] = [0.0, 0.05, 0.95] if exchanged else [0.0, 0.95, 0.05]
            w[k, 1] = [0.0, 0.95, 0.05] if exchanged else [0.0, 0.05, 0.95]
        character = CharacterSeries(
            frames=np.arange(1, 81), bands=np.array([977, 978]),
            groups=list(GROUPS), weights=w, captured=np.full((80, 2), 0.5),
        )
        self.assertEqual(
            len(find_episodes(character, couplings_for(character), 977, 978)), 2
        )

    def test_no_exchange_gives_no_episode(self):
        w = np.zeros((30, 2, 3))
        w[:, 0] = [0.0, 0.95, 0.05]
        w[:, 1] = [0.0, 0.05, 0.95]
        character = CharacterSeries(
            frames=np.arange(1, 31), bands=np.array([977, 978]),
            groups=list(GROUPS), weights=w, captured=np.full((30, 2), 0.5),
        )
        self.assertEqual(find_episodes(character, couplings_for(character), 977, 978), [])

    def test_a_band_not_in_the_table_is_refused(self):
        character = there_and_back()
        with self.assertRaises(EpisodeError):
            find_episodes(character, couplings_for(character), 977, 999)


def episode_with(net, occ, char):
    return CrossingEpisode(
        band_i=977, band_j=978, first_frame=1, last_frame=10, start=0, stop=10,
        min_gap_ev=0.01, max_abs_nac_ev=0.1,
        net_change=net, occupation_redistribution=occ, character_evolution=char,
    )


class ClassificationTests(unittest.TestCase):
    def test_occupation_dominated_is_named_and_not_called_hopping(self):
        ep = classify_episode(
            episode_with({"BCF": -0.3, "PCBM": 0.3}, {"PCBM": 0.28}, {"PCBM": 0.02})
        )
        self.assertEqual(ep.classification, "occupation_redistribution_dominated")
        self.assertEqual(ep.direction, "BCF->PCBM")
        self.assertNotIn("hop", ep.note.split(NO_HOP_RECORD[:20])[0].lower())
        self.assertIn("no hop record", ep.note)

    def test_character_following_transfer_is_named(self):
        ep = classify_episode(
            episode_with({"BCF": -0.3, "PCBM": 0.3}, {"PCBM": 0.02}, {"PCBM": 0.28})
        )
        self.assertEqual(ep.classification, "character_following_transfer")
        self.assertIn("adiabatic", ep.note)

    def test_a_balanced_episode_is_mixed(self):
        ep = classify_episode(
            episode_with({"BCF": -0.3, "PCBM": 0.3}, {"PCBM": 0.15}, {"PCBM": 0.15})
        )
        self.assertEqual(ep.classification, "mixed")

    def test_character_swap_with_no_net_movement_is_named_as_such(self):
        ep = classify_episode(
            episode_with({"BCF": 0.0, "PCBM": 0.0}, {"PCBM": 0.0}, {"PCBM": 0.0})
        )
        self.assertEqual(ep.classification, "no_net_fragment_transfer")
        self.assertEqual(ep.direction, "no_net_transfer")
        self.assertIn("ended the episode where it began", ep.note)
        # A net of zero over the episode is not a claim that nothing moved
        # inside it, and the note must not be readable as one.
        self.assertIn("may have moved and returned within it", ep.note)

    def test_the_reverse_direction_is_reported(self):
        ep = classify_episode(
            episode_with({"BCF": 0.3, "PCBM": -0.3}, {"PCBM": -0.28}, {"PCBM": -0.02})
        )
        self.assertEqual(ep.direction, "PCBM->BCF")

    def test_without_shprop_nothing_is_classified(self):
        ep = classify_episode(
            CrossingEpisode(977, 978, 1, 10, 0, 10, 0.01, 0.1)
        )
        self.assertEqual(ep.classification, "not_classified")
        self.assertIn("whether charge followed it is not", ep.note)


class RawWeightTests(unittest.TestCase):
    def test_a_swap_inside_the_window_is_found_even_when_it_returns(self):
        # Comparing only the endpoints of a there-and-back passage finds
        # nothing; the check must look inside the window.
        character = there_and_back()
        out = raw_weight_check(character, 977, 978, 1, 40)
        self.assertTrue(out["bands"][977]["normalized_swaps"])
        self.assertTrue(out["bands"][977]["raw_swaps"])
        self.assertTrue(out["all_agree"])

    def test_a_swap_created_only_by_the_denominator_is_caught(self):
        # Raw weights that never cross, with a captured fraction that moves a
        # lot: normalized swaps, raw does not, and they must disagree.
        n = 20
        raw = np.zeros((n, 1, 3))
        raw[:, 0, 1] = 0.30          # BCF raw, constant
        raw[:, 0, 2] = 0.20          # PCBM raw, constant
        captured = raw.sum(axis=2)
        w = raw / captured[:, :, None]
        character = CharacterSeries(
            frames=np.arange(1, n + 1), bands=np.array([977]),
            groups=list(GROUPS), weights=w, captured=captured,
        )
        out = raw_weight_check(character, 977, 977, 1, n)
        self.assertFalse(out["bands"][977]["raw_swaps"])
        self.assertFalse(out["bands"][977]["normalized_swaps"])

    def test_a_table_without_captured_is_refused(self):
        character = there_and_back()
        character.captured = None
        with self.assertRaises(EpisodeError):
            raw_weight_check(character, 977, 978, 1, 40)


class IsolationTests(unittest.TestCase):
    def _three_state(self, third_offset):
        n = 20
        e = np.zeros((n, 3))
        e[:, 1] = 0.02
        e[:, 2] = 0.02 + third_offset
        character = CharacterSeries(
            frames=np.arange(1, n + 1), bands=np.array([977, 978, 979]),
            groups=list(GROUPS), weights=np.zeros((n, 3, 3)),
            captured=np.full((n, 3), 0.5),
        )
        return CouplingSeries(
        frames=character.frames.copy(), energies=e, nac=None,
        nac_unit="eV", dt_fs=1.0, ceiling_ev=None,
    ), character

    def test_a_far_neighbour_gives_a_large_isolation_ratio(self):
        couplings, character = self._three_state(1.0)
        out = isolation_check(couplings, character, 977, 978, [979], 1, 20)
        self.assertGreater(out["isolation_ratio"], 10.0)

    def test_a_near_neighbour_gives_a_small_one(self):
        couplings, character = self._three_state(0.005)
        out = isolation_check(couplings, character, 977, 978, [979], 1, 20)
        self.assertLess(out["isolation_ratio"], 1.0)
        self.assertIn("reported rather than thresholded", out["note"])

    def test_an_empty_window_is_refused(self):
        couplings, character = self._three_state(1.0)
        with self.assertRaises(EpisodeError):
            isolation_check(couplings, character, 977, 978, [979], 500, 600)


class LocalTableTests(unittest.TestCase):
    def test_the_table_carries_energies_gap_character_and_raw_weights(self):
        character = there_and_back()
        header, rows = local_crossing_table(
            character, couplings_for(character), 977, 978, 10, 30
        )
        self.assertEqual(len(rows), 21)
        for column in ("frame", "gap_977_978_eV", "977_BCF", "978_PCBM",
                       "977_raw_BCF", "978_raw_PCBM", "977_captured"):
            self.assertIn(column, header)
        raw_bcf = header.index("977_raw_BCF")
        norm_bcf = header.index("977_BCF")
        cap = header.index("977_captured")
        for row in rows:
            self.assertAlmostEqual(row[raw_bcf], row[norm_bcf] * row[cap], places=12)


class LateWindowTests(unittest.TestCase):
    def test_a_fixed_flat_population_that_is_not_flat_dynamically(self):
        t = np.linspace(0.0, 1.0, 200)
        fixed = {"PCBM": np.full(200, 0.30)}
        dynamic = {"PCBM": 0.30 + 0.20 * (t - 0.1) / 0.9}
        out = fixed_vs_dynamic_late(t, fixed, dynamic, start_ns=0.1)
        pcbm = out["groups"]["PCBM"]
        self.assertAlmostEqual(pcbm["fixed"]["net_change"], 0.0, places=12)
        self.assertGreater(pcbm["dynamic"]["net_change"], 0.15)
        self.assertEqual(pcbm["range_ratio_dynamic_over_fixed"], float("inf"))

    def test_the_window_starts_where_it_is_told(self):
        t = np.linspace(0.0, 1.0, 101)
        out = fixed_vs_dynamic_late(
            t, {"A": t.copy()}, {"A": t.copy()}, start_ns=0.1
        )
        self.assertAlmostEqual(out["groups"]["A"]["fixed"]["initial"], 0.1, places=9)
        self.assertEqual(out["window_ns"][0], 0.1)

    def test_a_window_with_too_few_samples_is_refused(self):
        t = np.linspace(0.0, 1.0, 10)
        with self.assertRaises(EpisodeError):
            fixed_vs_dynamic_late(t, {"A": t}, {"A": t}, start_ns=0.99)


if __name__ == "__main__":
    unittest.main()
