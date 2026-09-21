"""Character exchange, population transfer, and telling them apart.

The fixtures are built from the two-state avoided-crossing model, so what the
detector should find is known before it runs:

    |phi_1> =  cos(theta)|A> + sin(theta)|B>
    |phi_2> = -sin(theta)|A> + cos(theta)|B>      tan(2 theta) = 2V / (E_A - E_B)

A PROCAR projection onto A's atoms measures approximately cos^2(theta), so
sweeping the diabatic gap through zero sweeps theta through pi/4 and exchanges
the fragment character of the two adiabatic states.  Crucially, that exchange
happens whether or not any population moves: the three cases below separate
them.
"""

import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from namd_analysis.crossings import (
    DEFAULT_THRESHOLDS,
    NO_PROJECTED_CHANGE,
    NOT_EVALUATED,
    PROJECTED_CHANGE,
    CrossingError,
    detect_events,
    load_couplings,
    nac_ceiling_note,
    read_projection_character,
    sensitivity,
)
from namd_analysis.dispatch import main as dispatch_main

V_EV = 0.02


def avoided_crossing(nframes=41, coupling=V_EV, sweep=0.5):
    """Adiabatic energies, |NAC| and cos^2/sin^2 fragment weights."""
    detuning = np.linspace(-1.0, 1.0, nframes) * sweep
    theta = 0.5 * np.arctan2(2.0 * coupling, detuning)
    half_gap = np.sqrt((detuning / 2.0) ** 2 + coupling**2)
    energies = np.column_stack([-half_gap, half_gap])
    nac = np.zeros((nframes, 2, 2))
    # d_12 ~ V / (2 * (E+ - E-)^2) * dE/dt; the shape is what matters here.
    nac[:, 0, 1] = nac[:, 1, 0] = coupling / (2.0 * half_gap) * 0.05
    weights = np.stack([np.cos(theta) ** 2, np.sin(theta) ** 2], axis=1)
    return energies, nac, weights


def write_character(path, frames, weights_by_band, groups=("BCF", "PCBM")):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "band", "group", "normalized_weight"])
        for index, frame in enumerate(frames):
            for band, table in weights_by_band.items():
                for gi, group in enumerate(groups):
                    writer.writerow([int(frame), int(band), group, float(table[index, gi])])
    return path


def write_couplings(root, energies, nac):
    root = Path(root)
    np.savetxt(root / "EIGTXT", energies)
    np.savetxt(root / "NATXT", nac.reshape(nac.shape[0], -1))
    return root / "EIGTXT", root / "NATXT"


class _Crossing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.nframes = 41
        self.frames = np.arange(1, self.nframes + 1)
        self.energies, self.nac, self.weights = avoided_crossing(self.nframes)

    def tearDown(self):
        self.tmp.cleanup()

    def _series(self, weights_by_band):
        path = write_character(
            self.root / "projection_character.csv", self.frames, weights_by_band
        )
        return read_projection_character(path)

    def _couplings(self):
        eig, nat = write_couplings(self.root, self.energies, self.nac)
        return load_couplings(eig, nat, frames=self.frames, nac_unit="eV", dt_fs=1.0)


class CharacterSwapWithoutPopulationHopTests(_Crossing):
    """The band labels exchange; no fragment population moves."""

    def test_the_swap_is_found_and_is_not_called_transfer(self):
        swapped = np.column_stack([self.weights[:, 1], self.weights[:, 0]])
        character = self._series({976: self.weights, 977: swapped})
        events = detect_events(character, self._couplings(), [(976, 977)])
        swaps = [e for e in events if e.character_swap]
        self.assertTrue(swaps, "the character exchange must be detected")
        for event in swaps:
            # No fragment population was supplied, so nothing may be called
            # transfer on the strength of the character alone -- and equally,
            # nothing may be called *not* transfer. The question was not asked.
            self.assertEqual(
                event.classification, "character_swap_population_not_evaluated"
            )
            self.assertFalse(event.population_evaluated)
            self.assertIn("NOT EVALUATED", event.note)
            self.assertIn("nothing here rules it out", event.note)

    def test_the_swap_lands_where_theta_crosses_forty_five_degrees(self):
        swapped = np.column_stack([self.weights[:, 1], self.weights[:, 0]])
        character = self._series({976: self.weights, 977: swapped})
        events = detect_events(character, self._couplings(), [(976, 977)])
        swaps = [e for e in events if e.character_swap]
        centre = int(self.frames[np.argmin(np.abs(self.weights[:, 0] - 0.5))])
        self.assertTrue(
            min(abs(e.frame - centre) for e in swaps) <= 2,
            "the exchange must be found at the crossing, not elsewhere",
        )

    def test_a_population_that_does_not_move_keeps_the_label_honest(self):
        swapped = np.column_stack([self.weights[:, 1], self.weights[:, 0]])
        character = self._series({976: self.weights, 977: swapped})
        flat = {int(f): {"BCF": 0.5, "PCBM": 0.5} for f in self.frames}
        events = detect_events(
            character, self._couplings(), [(976, 977)], fragment_population=flat
        )
        for event in (e for e in events if e.character_swap):
            self.assertEqual(
                event.classification, NO_PROJECTED_CHANGE
            )
            self.assertEqual(event.fragment_population_change, 0.0)


class ConfigurationOnlyScanClaimsNothingTests(_Crossing):
    """A scan with no SHPROP cannot report the absence of fragment transfer.

    ``character-crossings`` runs on ``projection_character.csv`` + EIGTXT +
    NATXT alone. None of those carries a SHPROP population, so fragment
    transfer is **not evaluated** in that mode. The old behaviour collapsed
    ``fragment_population_change = None`` into "the population did not follow",
    which turns a missing input into a scientific finding. These pin the
    separation at every layer it has to survive: the classifier, the event
    record, the CSV, the JSON report and the referee summary.
    """

    def _swapping_character(self):
        swapped = np.column_stack([self.weights[:, 1], self.weights[:, 0]])
        return self._series({976: self.weights, 977: swapped})

    def _run_cli(self, name):
        swapped = np.column_stack([self.weights[:, 1], self.weights[:, 0]])
        character_path = write_character(
            self.root / "projection_character.csv", self.frames,
            {976: self.weights, 977: swapped},
        )
        eig, nat = write_couplings(self.root, self.energies, self.nac)
        out = self.root / name
        code = dispatch_main([
            "character-crossings",
            "--projection-character", str(character_path),
            "--eigtxt", str(eig), "--natxt", str(nat), "--dt-fs", "1.0",
            "--out", str(out),
        ])
        self.assertEqual(code, 0)
        return out

    def test_an_unsupplied_population_is_not_evaluated_not_zero(self):
        events = detect_events(
            self._swapping_character(), self._couplings(), [(976, 977)]
        )
        swaps = [e for e in events if e.character_swap]
        self.assertTrue(swaps)
        for event in swaps:
            self.assertIsNone(event.fragment_population_change)
            self.assertFalse(event.population_evaluated)
            self.assertEqual(event.classification, NOT_EVALUATED)
            self.assertNotEqual(event.classification, NO_PROJECTED_CHANGE)

    def test_a_supplied_zero_change_is_still_a_measured_absence(self):
        # The other half of the distinction: a P_g that WAS read and did not
        # move is a measured absence, and keeps the stronger label.
        flat = {int(f): {"BCF": 0.5, "PCBM": 0.5} for f in self.frames}
        events = detect_events(
            self._swapping_character(), self._couplings(), [(976, 977)],
            fragment_population=flat,
        )
        swaps = [e for e in events if e.character_swap]
        self.assertTrue(swaps)
        for event in swaps:
            self.assertTrue(event.population_evaluated)
            self.assertEqual(event.fragment_population_change, 0.0)
            self.assertEqual(event.classification, NO_PROJECTED_CHANGE)
            self.assertIn("did not move beyond tolerance", event.note)
            self.assertIn("sums over every band of the basis", event.note)

    def test_the_two_cases_never_share_a_classification(self):
        self.assertNotEqual(NOT_EVALUATED, NO_PROJECTED_CHANGE)
        flat = {int(f): {"BCF": 0.5, "PCBM": 0.5} for f in self.frames}
        character = self._swapping_character()
        without = detect_events(character, self._couplings(), [(976, 977)])
        with_pop = detect_events(
            character, self._couplings(), [(976, 977)], fragment_population=flat
        )
        labels_without = {e.classification for e in without if e.character_swap}
        labels_with = {e.classification for e in with_pop if e.character_swap}
        self.assertEqual(labels_without, {NOT_EVALUATED})
        self.assertEqual(labels_with, {NO_PROJECTED_CHANGE})
        self.assertFalse(labels_without & labels_with)

    def test_a_metric_only_event_does_not_claim_the_population_was_flat(self):
        # The same mistake hides in the non-swap branches: their notes used to
        # say "no change of character or fragment population" whatever was
        # supplied.
        for event in detect_events(
            self._swapping_character(), self._couplings(), [(976, 977)]
        ):
            if event.character_swap:
                continue
            self.assertIn("NOT EVALUATED", event.note)

    def test_the_event_table_carries_an_explicit_evaluated_column(self):
        # A blank CSV cell is ambiguous to a reader; a False is not.
        out = self._run_cli("not_evaluated_csv")
        with (out / "crossing_events.csv").open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertIn("population_evaluated", rows[0])
        for row in rows:
            self.assertEqual(row["population_evaluated"], "False")
            self.assertEqual(row["projected_fragment_population_change"], "")
        swaps = [r for r in rows if r["character_swap"] == "True"]
        self.assertTrue(swaps)
        for row in swaps:
            self.assertEqual(row["classification"], NOT_EVALUATED)

    def test_the_report_states_that_population_was_not_evaluated(self):
        out = self._run_cli("not_evaluated_json")
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        block = report["fragment_population"]
        self.assertFalse(block["evaluated"])
        self.assertEqual(block["n_events_population_evaluated"], 0)
        self.assertEqual(block["n_events_population_not_evaluated"], report["n_events"])
        self.assertIn("NOT EVALUATED", block["meaning_of_null"])
        self.assertEqual(block["not_evaluated_classification"], NOT_EVALUATED)
        self.assertNotIn(NO_PROJECTED_CHANGE, report["event_counts"])

    def test_the_summary_refuses_to_announce_an_absence_it_never_measured(self):
        out = self._run_cli("not_evaluated_summary")
        text = (out / "crossing_summary.md").read_text(encoding="utf-8")
        # The exact sentences the old report produced from a no-SHPROP run.
        self.assertNotIn(
            "No character exchange in this run was accompanied by a change in "
            "projection-weighted fragment population.",
            text,
        )
        self.assertNotIn(
            "No character exchange in this run was accompanied by a fragment "
            "population change in the direction the swap implies.",
            text,
        )
        self.assertNotIn(NO_PROJECTED_CHANGE + "` |", text)
        self.assertIn("NOT EVALUATED", text)
        self.assertIn("not** a finding that no charge moved", text)
        self.assertIn("Neither a change nor its absence may be claimed", text)

    def test_an_evaluated_run_may_still_state_the_absence(self):
        # The guard must not have silenced the real finding: with a population
        # supplied, the summary says plainly that nothing moved.
        from namd_analysis.crossing_summary import render

        flat = {int(f): {"BCF": 0.5, "PCBM": 0.5} for f in self.frames}
        events = detect_events(
            self._swapping_character(), self._couplings(), [(976, 977)],
            fragment_population=flat,
        )
        text = render({"n_events": len(events)}, events)
        self.assertIn("none moved `P_g` beyond tolerance", text)
        self.assertIn("projected occupied density stayed where it was", text)
        self.assertNotIn("NOT EVALUATED", text)


class PopulationHopWithStableCharacterTests(_Crossing):
    """Occupation moves between fragments while every band keeps its character."""

    def test_transfer_without_a_swap_is_labelled_as_such(self):
        stable_a = np.tile([[0.95, 0.05]], (self.nframes, 1))
        stable_b = np.tile([[0.05, 0.95]], (self.nframes, 1))
        character = self._series({976: stable_a, 977: stable_b})
        # Fragment population migrates from BCF to PCBM over the trajectory.
        ramp = np.linspace(0.0, 1.0, self.nframes)
        population = {
            int(f): {"BCF": 1.0 - ramp[i], "PCBM": ramp[i]}
            for i, f in enumerate(self.frames)
        }
        events = detect_events(
            character,
            self._couplings(),
            [(976, 977)],
            fragment_population=population,
        )
        self.assertTrue(events, "a close/coupled pair must still be flagged")
        self.assertFalse(
            any(e.character_swap for e in events),
            "no band changed its dominant fragment",
        )
        labels = {e.classification for e in events}
        self.assertIn("fragment_population_change_without_character_swap", labels)

    def test_a_stable_character_is_never_reported_as_an_exchange(self):
        stable_a = np.tile([[0.95, 0.05]], (self.nframes, 1))
        stable_b = np.tile([[0.05, 0.95]], (self.nframes, 1))
        character = self._series({976: stable_a, 977: stable_b})
        events = detect_events(character, self._couplings(), [(976, 977)])
        for event in events:
            self.assertEqual(event.dominant_i_before, event.dominant_i_after)
            self.assertEqual(event.dominant_j_before, event.dominant_j_after)


class SwapDirectionTests(_Crossing):
    """The direction test is a descriptor, and it runs on ``P_g`` itself."""

    def test_a_symmetric_exchange_leaves_the_direction_undecided(self):
        # detect_events receives the contracted P_g. That is the observable the
        # classification rests on, so a change in it IS classified as a change;
        # what stays undecided is only which fragment the swap points at, when
        # both bands exchange character at once.
        moving = np.array([[0.9, 0.1]] * 20 + [[0.1, 0.9]] * 21)
        stable = np.tile([[0.05, 0.95]], (self.nframes, 1))
        character = self._series({976: moving, 977: stable})
        ramp = np.linspace(0.9, 0.1, self.nframes)
        population = {
            int(f): {"BCF": ramp[i], "PCBM": 1.0 - ramp[i]}
            for i, f in enumerate(self.frames)
        }
        events = detect_events(
            character, self._couplings(), [(976, 977)], fragment_population=population
        )
        moved = [e for e in events if e.character_swap and e.fragment_population_change]
        self.assertTrue(moved)
        for event in moved:
            self.assertEqual(event.classification, PROJECTED_CHANGE)
            self.assertIsNotNone(event.dominant_fragment)
            # The split is not available on this path, and is not guessed --
            # which costs the classification nothing, because it never used it.
            self.assertIsNone(event.occupation_redistribution)
            self.assertIsNone(event.character_evolution)
            self.assertIsNone(event.decomposition_descriptor)

    def test_the_helper_tests_the_projected_population_not_the_split(self):
        from namd_analysis.crossings import _direction_agrees, _swap_moves

        # One band moves BCF -> PCBM, the other stands still: a single direction.
        moves = _swap_moves("BCF", "PCBM", "PCBM", "PCBM")
        self.assertIs(
            _direction_agrees(moves, {"BCF": -0.4, "PCBM": 0.4}, 1e-3), True
        )
        self.assertIs(
            _direction_agrees(moves, {"BCF": 0.4, "PCBM": -0.4}, 1e-3), False
        )
        # P_g did not move at all: nothing went the way the swap points.
        self.assertIs(
            _direction_agrees(moves, {"BCF": 0.0, "PCBM": 0.0}, 1e-3), False
        )

    def test_the_helper_leaves_an_undecidable_question_open(self):
        from namd_analysis.crossings import _direction_agrees, _swap_moves

        delta = {"BCF": -0.4, "PCBM": 0.4}
        # Both bands swap, oppositely: either direction would match one of
        # them, so this is undecidable -- None, and emphatically not False.
        both = _swap_moves("BCF", "PCBM", "PCBM", "BCF")
        self.assertIsNone(_direction_agrees(both, delta, 1e-3))
        # No band moved its dominant fragment: no direction to test.
        self.assertIsNone(_direction_agrees([], delta, 1e-3))
        # No per-fragment change supplied.
        self.assertIsNone(_direction_agrees(both, None, 1e-3))

    def test_the_direction_test_never_decides_the_classification(self):
        # A change in P_g that runs opposite to the swap is still a change in
        # P_g. The descriptor records the mismatch; the label does not downgrade.
        from namd_analysis.crossings import _classify

        for agrees in (True, False, None):
            label, _ = _classify(True, True, True, 0.5, 1e-3, agrees)
            self.assertEqual(label, PROJECTED_CHANGE)

    def test_the_projection_note_says_why_p_g_is_the_observable(self):
        from namd_analysis.crossings import PROJECTION_NOTE

        self.assertIn("sum_i P_i w_ig", PROJECTION_NOTE)
        self.assertIn("EVERY band of the basis", PROJECTION_NOTE)
        self.assertIn("not a labelling artifact", PROJECTION_NOTE)
        # The correction: character evolution is not a relabelling.
        self.assertIn("NOT mere relabelling", PROJECTION_NOTE)
        self.assertIn("'no charge moved' is wrong", PROJECTION_NOTE)
        # And the limit that travels with it: P_g is diagonal, so neither term
        # is a quantity of charge.
        self.assertIn("projection-weighted DIAGONAL", PROJECTION_NOTE)
        self.assertIn("NOT exact fragment charge", PROJECTION_NOTE)
        self.assertIn("one of infinitely many exact splits", PROJECTION_NOTE)


class SimultaneousEventTests(_Crossing):
    """Small gap, enhanced NAC and character exchange at the same frame."""

    def test_all_three_coincide_at_the_crossing(self):
        swapped = np.column_stack([self.weights[:, 1], self.weights[:, 0]])
        character = self._series({976: self.weights, 977: swapped})
        couplings = self._couplings()
        events = detect_events(character, couplings, [(976, 977)])
        together = [
            e for e in events if e.small_gap and e.strong_nac and e.character_swap
        ]
        self.assertTrue(
            together,
            "the fixture puts the minimum gap, the maximum coupling and the "
            "character exchange at the same frame; the detector must see all three",
        )
        for event in together:
            self.assertLessEqual(event.gap_ev, DEFAULT_THRESHOLDS["gap_ev"])
            self.assertGreaterEqual(event.nac_ev, DEFAULT_THRESHOLDS["nac_ev"])

    def test_the_minimum_gap_is_twice_the_coupling(self):
        # A property of the model, and a check that the fixture is the physics
        # it claims to be: at resonance the adiabatic gap is 2V.
        gaps = np.abs(self.energies[:, 1] - self.energies[:, 0])
        self.assertAlmostEqual(gaps.min(), 2.0 * V_EV, places=6)

    def test_raw_metrics_travel_with_every_event(self):
        swapped = np.column_stack([self.weights[:, 1], self.weights[:, 0]])
        character = self._series({976: self.weights, 977: swapped})
        events = detect_events(character, self._couplings(), [(976, 977)])
        for event in events:
            self.assertTrue(np.isfinite(event.gap_ev))
            self.assertIsNotNone(event.nac_ev)
            self.assertGreaterEqual(event.character_change_i, 0.0)


class ThresholdTests(_Crossing):
    def test_the_sensitivity_sweep_reports_how_much_the_cutoff_chose(self):
        swapped = np.column_stack([self.weights[:, 1], self.weights[:, 0]])
        character = self._series({976: self.weights, 977: swapped})
        sweep = sensitivity(character, self._couplings(), [(976, 977)])
        totals = [row["total_events"] for row in sweep["rows"]]
        self.assertEqual(len(totals), len(sweep["factors"]))
        self.assertGreater(max(totals), min(totals), "counts must depend on the cutoff")
        self.assertIn("the threshold chose", sweep["note"])
        for row in sweep["rows"]:
            self.assertEqual(
                set(row["thresholds"]), set(DEFAULT_THRESHOLDS),
                "every threshold is scaled, none silently fixed",
            )

    def test_a_tighter_gap_threshold_flags_fewer_close_frames(self):
        swapped = np.column_stack([self.weights[:, 1], self.weights[:, 0]])
        character = self._series({976: self.weights, 977: swapped})
        couplings = self._couplings()
        loose = detect_events(character, couplings, [(976, 977)], {"gap_ev": 1.0})
        tight = detect_events(character, couplings, [(976, 977)], {"gap_ev": 0.001})
        self.assertGreater(
            sum(e.small_gap for e in loose), sum(e.small_gap for e in tight)
        )


class NacInterpretationTests(_Crossing):
    """The engineered-ceiling reading survives."""

    def test_a_repeated_magnitude_is_called_upstream_policy_not_clipping(self):
        nac = self.nac.copy()
        nac[:, 0, 1] = nac[:, 1, 0] = 0.6  # an imposed ceiling, repeated exactly
        eig, nat = write_couplings(self.root, self.energies, nac)
        couplings = load_couplings(eig, nat, frames=self.frames, nac_unit="eV", dt_fs=1.0)
        note = nac_ceiling_note(couplings)
        self.assertIn("repeated_magnitude_note", note)
        message = note["repeated_magnitude_note"]
        self.assertIn("did not come out of the dynamics", message)
        self.assertIn("NOT described as", message)
        self.assertIn("accidental clipping", message)

    def test_couplings_near_the_limit_are_called_pathological_not_giant(self):
        couplings = self._couplings()
        note = nac_ceiling_note(couplings)
        self.assertIn("numerically pathological", note["interpretation"])
        self.assertIn("NOT a measurement", note["interpretation"])
        self.assertAlmostEqual(note["hbar_over_dt_ev"], 0.6582119569509066, places=6)


class FrameSynchronizationTests(_Crossing):
    """Everything is joined on the MD frame, never on a row index."""

    def test_frames_are_the_join_key(self):
        swapped = np.column_stack([self.weights[:, 1], self.weights[:, 0]])
        character = self._series({976: self.weights, 977: swapped})
        self.assertEqual(list(character.frames), list(self.frames))
        # Offsetting the coupling frames must move which frame each metric
        # lands on -- proving the join is by frame, not by position.
        eig, nat = write_couplings(self.root, self.energies, self.nac)
        shifted = load_couplings(
            eig, nat, frames=self.frames + 100, nac_unit="eV", dt_fs=1.0
        )
        events = detect_events(character, shifted, [(976, 977)])
        self.assertTrue(
            all(not np.isfinite(e.gap_ev) for e in events),
            "no character frame matches a coupling frame, so no gap is attached",
        )

    def test_a_ragged_character_table_is_refused(self):
        path = self.root / "ragged.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "band", "group", "normalized_weight"])
            writer.writerow([1, 976, "BCF", 0.9])
            writer.writerow([1, 976, "PCBM", 0.1])
            writer.writerow([2, 976, "BCF", 0.8])  # PCBM missing at frame 2
        with self.assertRaises(CrossingError) as ctx:
            read_projection_character(path)
        self.assertIn("rectangular", str(ctx.exception))

    def test_a_table_without_the_required_columns_is_refused(self):
        path = self.root / "wrong.csv"
        path.write_text("time_ns,group,population\n0,BCF,1.0\n", encoding="utf-8")
        with self.assertRaises(CrossingError) as ctx:
            read_projection_character(path)
        self.assertIn("projection_character.csv", str(ctx.exception))


def write_history(root, name, nsw, rows, ramp_from=0.0, ramp_to=1.0, step_fs=10000.0):
    """A SHPROP history whose fragment population ramps BCF -> PCBM."""
    path = Path(root) / name
    ramp = np.linspace(ramp_from, ramp_to, rows)
    with path.open("w", encoding="utf-8") as handle:
        print(f"# NSW = {nsw}", file=handle)
        for index in range(rows):
            values = [(index + 1) * step_fs, -0.8, 1.0 - ramp[index], ramp[index]]
            print(" ".join(f"{v:.10E}" for v in values), file=handle)
    return path


def write_state_map(root):
    path = Path(root) / "state_map.json"
    path.write_text(
        json.dumps(
            {
                "name": "synthetic",
                "time_column": 0,
                "time_unit": "fs",
                "population_columns": [2, 3],
                "groups": {"BCF": [2], "PCBM": [3]},
                "complete_population": True,
            }
        ),
        encoding="utf-8",
    )
    return path


class PerHistoryTests(_Crossing):
    """Histories that start at a different NAMDTINI are never merged first.

    Under a cyclic mapping a history revisits a frame many times, at a
    different population each time, and two histories occupy different frames
    at the same row.  So there is no ensemble "population at frame f" to
    classify against, and each history is walked on its own trajectory.
    """

    def setUp(self):
        super().setUp()
        self.nframes = 8
        self.frames = np.arange(1, self.nframes + 1)
        self.energies, self.nac, self.weights = avoided_crossing(self.nframes)
        # 976 flips BCF -> PCBM halfway round the cycle; 977 stays PCBM. Only
        # one band's dominance moves, so the swap names a single direction and
        # the population can actually be tested against it.
        moving = np.array([[0.9, 0.1]] * 4 + [[0.1, 0.9]] * 4)
        stable = np.tile([[0.05, 0.95]], (self.nframes, 1))
        self.character_path = write_character(
            self.root / "projection_character.csv",
            self.frames,
            {976: moving, 977: stable},
        )
        self.eig, self.nat = write_couplings(self.root, self.energies, self.nac)
        self.state_map_path = write_state_map(self.root)
        self.manifest = self.root / "manifest.json"
        for frame in self.frames:
            # Only the cycle length is read here, but the manifest still
            # insists every PROCAR it names exists.
            (self.root / f"PROCAR.{int(frame)}").write_text("", encoding="utf-8")
        self.manifest.write_text(
            json.dumps(
                {
                    "frames": [
                        {"frame": int(f), "procar": f"PROCAR.{int(f)}"}
                        for f in self.frames
                    ],
                    "cycle_length": self.nframes,
                }
            ),
            encoding="utf-8",
        )
        self.histories = [
            write_history(self.root, "SHPROP.1", self.nframes + 1, 24),
            write_history(self.root, "SHPROP.5", self.nframes + 1, 24),
        ]
        self.runs = 0

    def _run(self, extra=(), histories=None):
        self.runs += 1
        out = self.root / f"out{self.runs}"
        paths = [str(p) for p in (histories if histories is not None else self.histories)]
        code = dispatch_main(
            [
                "character-crossings",
                "--projection-character", str(self.character_path),
                "--eigtxt", str(self.eig),
                "--natxt", str(self.nat),
                "--dt-fs", "1.0",
                "--shprop", *paths,
                "--state-map", str(self.state_map_path),
                "--projection-manifest", str(self.manifest),
                "--frame-mode", "dish-cyclic",
                "--out", str(out),
                *extra,
            ]
        )
        self.assertEqual(code, 0)
        return out, json.loads((out / "report.json").read_text(encoding="utf-8"))

    def test_each_history_uses_its_own_resolved_frame_mapping(self):
        _, report = self._run()
        per = report["per_history"]
        self.assertEqual(per["distinct_namdtini"], [1, 5])
        by_name = {r["file"]: r for r in per["per_history"]}
        # NAMDTINI 1 opens on frame 1, NAMDTINI 5 on frame 5.  Were the two
        # sharing a mapping -- or correlated by row -- these would agree.
        self.assertEqual(by_name["SHPROP.1"]["first_frame"], 1)
        self.assertEqual(by_name["SHPROP.5"]["first_frame"], 5)
        self.assertNotEqual(
            by_name["SHPROP.1"]["last_frame"], by_name["SHPROP.5"]["last_frame"]
        )

    def test_a_history_wraps_the_cycle_from_its_own_start(self):
        from namd_analysis.character import aligned_frames

        frames = aligned_frames(
            {"NAMDTINI": 5, "NSW": self.nframes + 1},
            24,
            "dish-cyclic",
            self.histories[1],
            cycle_length=self.nframes,
        )
        # 5,6,7,8,1,2,...  The wrap is what makes one shared mapping wrong.
        self.assertEqual([int(f) for f in frames[:5]], [5, 6, 7, 8, 1])
        self.assertEqual(int(frames[-1]), 4)

    def test_classification_happens_per_history_and_only_then_aggregates(self):
        _, report = self._run()
        per = report["per_history"]
        summed = {}
        for record in per["per_history"]:
            for label, count in record["by_classification"].items():
                summed[label] = summed.get(label, 0) + count
        self.assertEqual(per["totals_by_classification"], summed)
        self.assertTrue(all(r["n_events"] for r in per["per_history"]))
        self.assertIn("nothing was averaged before classification", per["note"])
        self.assertIn("correlated by row number", per["note"])
        self.assertIn("opposite directions", per["why_not_averaged"])

    def test_a_per_history_population_is_what_allows_any_verdict_at_all(self):
        # Without a per-history P_g nothing could be said either way. With one,
        # a swap whose P_g moved is recorded as having moved it.
        _, report = self._run()
        per = report["per_history"]
        totals = per["totals_by_classification"]
        self.assertIn(PROJECTED_CHANGE, totals)
        self.assertGreater(totals[PROJECTED_CHANGE], 0)
        self.assertNotIn(NOT_EVALUATED, totals)
        # And the split describes those changes without claiming a mechanism.
        self.assertTrue(per["totals_by_decomposition_descriptor"])
        note = per["decomposition_descriptor_note"]
        self.assertIn("NOT physical branching fractions", note)
        self.assertIn("NOT mechanisms", note)
        self.assertIn("by exactly as much as an occupation_dominated one did", note)

    def test_two_histories_moving_oppositely_do_not_cancel(self):
        # Averaged first, a rise and a matching fall leave a flat population,
        # no population change at all, and therefore no transfer anywhere.
        rising = write_history(self.root, "SHPROP.1", self.nframes + 1, 24, 0.0, 1.0)
        falling = write_history(self.root, "SHPROP.5", self.nframes + 1, 24, 1.0, 0.0)
        out, report = self._run(histories=[rising, falling])
        by_name = {
            r["file"]: r["by_classification"]
            for r in report["per_history"]["per_history"]
        }
        # Each history moves P_g at its own steps. Both must record changes of
        # their own -- which an average taken first could not show.
        self.assertGreater(by_name["SHPROP.1"].get(PROJECTED_CHANGE, 0), 0)
        self.assertGreater(by_name["SHPROP.5"].get(PROJECTED_CHANGE, 0), 0)

        # They share a nuclear trajectory, so at a given frame the character
        # term is common to both; what runs oppositely is the occupation. That
        # difference must survive to the event table -- and, crucially, must
        # NOT change either history's classification. A change whose occupation
        # runs the other way is still a change in P_g.
        with (out / "per_history_events.csv").open("r", encoding="utf-8") as handle:
            rows = [
                r for r in csv.DictReader(handle)
                if r["classification"] == PROJECTED_CHANGE
            ]
        signs = {}
        for row in rows:
            key = (row["frame"], row["dominant_fragment"])
            signs.setdefault(key, {})[row["file"]] = np.sign(
                float(row["occupation_redistribution"])
            )
        opposed = [
            key for key, per_file in signs.items()
            if len(per_file) == 2 and len(set(per_file.values())) == 2
        ]
        self.assertTrue(
            opposed,
            "the two histories redistribute occupation oppositely at a shared "
            "frame; an average taken first would have shown neither",
        )
        # The mean of the two is flat, so an average taken first would have
        # shown neither.
        mean = 0.5 * (np.linspace(0.0, 1.0, 24) + np.linspace(1.0, 0.0, 24))
        self.assertLess(float(np.max(np.abs(np.diff(mean)))), 1.0e-12)

    def test_the_per_history_event_table_names_the_history(self):
        out, _ = self._run(extra=["--late-window", "0.1:0.3"])
        with (out / "per_history_events.csv").open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        self.assertEqual({r["file"] for r in rows}, {"SHPROP.1", "SHPROP.5"})
        for row in rows:
            self.assertIn(row["NAMDTINI"], {"1", "5"})
            self.assertIn(row["window"], {"early", "late", "outside"})

    def test_early_and_late_event_counts_are_compared(self):
        _, report = self._run(extra=["--late-window", "0.1:0.3"])
        comparison = report["early_vs_late_events"]
        self.assertEqual(comparison["early_window"], "early")
        self.assertEqual(comparison["late_window"], "late")
        by_window = report["per_history"]["totals_by_window"]
        self.assertEqual(comparison["early_total"], sum(by_window["early"].values()))
        self.assertEqual(comparison["late_total"], sum(by_window["late"].values()))
        self.assertGreater(comparison["early_total"], 0)
        self.assertGreater(comparison["late_total"], 0)
        for row in comparison["rows"]:
            self.assertEqual(row["difference"], row["early"] - row["late"])

    def test_the_comparison_refuses_to_read_a_raw_count_as_a_rate(self):
        _, report = self._run(extra=["--late-window", "0.1:0.3"])
        note = report["early_vs_late_events"]["note"]
        self.assertIn("different length", note)
        self.assertIn("not by itself a higher rate", note)

    def test_the_early_window_defaults_and_the_late_one_does_not(self):
        _, report = self._run()
        # No --late-window: there is nothing to compare against, and none is
        # invented.
        self.assertIsNone(report["early_vs_late_events"])
        self.assertEqual(list(report["per_history"]["totals_by_window"]), ["early"])

    def test_the_referee_summary_reports_each_history_before_the_total(self):
        out, report = self._run(extra=["--late-window", "0.1:0.3"])
        text = (out / "crossing_summary.md").read_text(encoding="utf-8")
        self.assertIn("## Per history", text)
        self.assertIn("SHPROP.1", text)
        self.assertIn("SHPROP.5", text)
        self.assertIn("nothing was averaged before classification", text)
        self.assertIn("### Early against late", text)
        # The per-history column must appear beside the total, not instead of it.
        self.assertIn("| total |", text)
        for record in report["per_history"]["per_history"]:
            self.assertIn(str(record["first_frame"]), text)

    def test_the_summary_says_no_late_window_rather_than_inventing_one(self):
        out, _ = self._run()
        text = (out / "crossing_summary.md").read_text(encoding="utf-8")
        self.assertIn("No late window was supplied", text)
        self.assertIn("not invented here", text)
        self.assertNotIn("### Early against late", text)

    def test_shprop_without_a_state_map_is_refused(self):
        code = dispatch_main(
            [
                "character-crossings",
                "--projection-character", str(self.character_path),
                "--eigtxt", str(self.eig),
                "--natxt", str(self.nat),
                "--dt-fs", "1.0",
                "--shprop", str(self.histories[0]),
                "--frame-mode", "dish-cyclic",
                "--out", str(self.root / "refused"),
            ]
        )
        self.assertEqual(code, 2)

    def test_a_history_visiting_an_uncovered_frame_is_refused(self):
        from namd_analysis.crossings import history_fragment_population
        from namd_analysis.populations import StateMap

        character = read_projection_character(self.character_path)
        with self.assertRaises(CrossingError) as ctx:
            history_fragment_population(
                self.histories[0],
                StateMap.from_json(self.state_map_path),
                character,
                [976, 977],
                np.full(24, 999),  # a frame the character table does not cover
            )
        self.assertIn("does not cover", str(ctx.exception))

    def test_the_population_is_the_projection_weighted_diagonal_contraction(self):
        from namd_analysis.crossings import history_fragment_population
        from namd_analysis.populations import StateMap

        character = read_projection_character(self.character_path)
        frames = np.array([1, 2, 3] * 8)
        split = history_fragment_population(
            self.histories[0],
            StateMap.from_json(self.state_map_path),
            character,
            [976, 977],
            frames,
        )
        population, times = split.total, split.time_raw
        raw = np.loadtxt(self.histories[0])
        band_at = character.band_index()
        frame_at = character.frame_index()
        for step in (0, 7, 23):
            weights = character.weights[frame_at[int(frames[step])]]
            expected = (
                raw[step, 2] * weights[band_at[976]]
                + raw[step, 3] * weights[band_at[977]]
            )
            np.testing.assert_allclose(population[step], expected, rtol=0, atol=1e-12)
        # The time axis is the file's own column, not a grid rebuilt from the
        # first and last rows.
        np.testing.assert_allclose(times, raw[:, 0], rtol=0, atol=1e-9)

    def test_an_uneven_time_column_is_taken_as_written(self):
        from namd_analysis.crossings import history_fragment_population
        from namd_analysis.populations import StateMap

        # A grid rebuilt from first and last rows would put row 1 at 20000 fs;
        # the file says 15000, and the file is what decides the window.
        path = self.root / "SHPROP.1"
        text = path.read_text(encoding="utf-8").splitlines()
        fields = text[2].split()
        fields[0] = f"{15000.0:.10E}"
        text[2] = " ".join(fields)
        path.write_text("\n".join(text) + "\n", encoding="utf-8")

        character = read_projection_character(self.character_path)
        split = history_fragment_population(
            path,
            StateMap.from_json(self.state_map_path),
            character,
            [976, 977],
            np.array([1, 2, 3] * 8),
        )
        self.assertAlmostEqual(float(split.time_raw[1]), 15000.0, places=6)

    def test_the_change_splits_exactly_into_population_and_character_parts(self):
        from namd_analysis.crossings import history_fragment_population
        from namd_analysis.populations import StateMap

        character = read_projection_character(self.character_path)
        frames = np.array([((t) % self.nframes) + 1 for t in range(24)])
        split = history_fragment_population(
            self.histories[0],
            StateMap.from_json(self.state_map_path),
            character,
            [976, 977],
            frames,
        )
        # dP = dP_pop + dP_char, to floating-point exactness.
        np.testing.assert_allclose(
            np.diff(split.total, axis=0),
            (split.occupation_redistribution + split.character_evolution)[1:],
            rtol=0,
            atol=1e-12,
        )
        # The first row has no previous step and is not invented.
        np.testing.assert_array_equal(split.occupation_redistribution[0], 0.0)
        np.testing.assert_array_equal(split.character_evolution[0], 0.0)

    def test_the_split_is_the_symmetric_midpoint_form_not_an_endpoint_one(self):
        """Which exact split the code uses, pinned against the alternative.

        Both forms sum to the same dP_g, so the closure test above passes
        either way. They differ in how they *apportion* that change, which is
        precisely what ``occupation_redistribution`` and ``character_evolution``
        report -- by up to half a step's movement. This fixture has both P_i
        and w_ig changing at the same step, so the two forms disagree, and the
        code must match the midpoint one.
        """
        from namd_analysis.crossings import history_fragment_population
        from namd_analysis.populations import StateMap

        character = read_projection_character(self.character_path)
        frames = np.array([((t) % self.nframes) + 1 for t in range(24)])
        state_map = StateMap.from_json(self.state_map_path)
        split = history_fragment_population(
            self.histories[0], state_map, character, [976, 977], frames,
        )

        raw = np.loadtxt(self.histories[0])
        pops = raw[:, np.asarray(state_map.population_columns, dtype=int)]
        band_at = character.band_index()
        frame_at = character.frame_index()
        rows = np.asarray([frame_at[int(f)] for f in frames], dtype=int)
        weights = character.weights[:, [band_at[976], band_at[977]], :][rows]

        dP = pops[1:] - pops[:-1]
        dW = weights[1:] - weights[:-1]
        midpoint_pop = np.einsum("ts,tsg->tg", dP, 0.5 * (weights[1:] + weights[:-1]))
        midpoint_char = np.einsum("ts,tsg->tg", 0.5 * (pops[1:] + pops[:-1]), dW)
        endpoint_pop = np.einsum("ts,tsg->tg", dP, weights[1:])
        endpoint_char = np.einsum("ts,tsg->tg", pops[:-1], dW)

        # The fixture must actually distinguish them, or this pins nothing.
        self.assertGreater(
            float(np.max(np.abs(midpoint_pop - endpoint_pop))), 1e-6,
            "the fixture cannot tell the two forms apart",
        )
        # Both are exact in the sum -- which is why a closure test cannot
        # choose between them and this test has to.
        np.testing.assert_allclose(
            midpoint_pop + midpoint_char, endpoint_pop + endpoint_char, atol=1e-12
        )

        np.testing.assert_allclose(
            split.occupation_redistribution[1:], midpoint_pop, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            split.character_evolution[1:], midpoint_char, rtol=0, atol=1e-12
        )
        self.assertGreater(
            float(np.max(np.abs(split.occupation_redistribution[1:] - endpoint_pop))),
            1e-6,
            "the code is computing the endpoint-biased split",
        )

    def test_the_split_does_not_depend_on_where_the_chunks_fall(self):
        from namd_analysis.crossings import history_fragment_population
        from namd_analysis.populations import StateMap

        character = read_projection_character(self.character_path)
        state_map = StateMap.from_json(self.state_map_path)
        frames = np.array([((t) % self.nframes) + 1 for t in range(24)])
        whole = history_fragment_population(
            self.histories[0], state_map, character, [976, 977], frames
        )
        for chunk_rows in (1, 2, 5, 7, 23, 24):
            part = history_fragment_population(
                self.histories[0], state_map, character, [976, 977], frames,
                chunk_rows=chunk_rows,
            )
            for name in ("total", "occupation_redistribution", "character_evolution"):
                np.testing.assert_allclose(
                    getattr(part, name), getattr(whole, name), rtol=0, atol=1e-12,
                    err_msg=f"{name} changed at chunk_rows={chunk_rows}",
                )

    def test_a_character_driven_shift_is_not_reported_as_occupation_moving(self):
        from namd_analysis.crossings import history_fragment_population
        from namd_analysis.populations import StateMap

        # A history whose per-band populations never move. Every change in the
        # fragment population is then character-driven by construction.
        path = self.root / "SHPROP.9"
        with path.open("w", encoding="utf-8") as handle:
            print(f"# NSW = {self.nframes + 1}", file=handle)
            for index in range(24):
                values = [(index + 1) * 10000.0, -0.8, 0.5, 0.5]
                print(" ".join(f"{v:.10E}" for v in values), file=handle)

        character = read_projection_character(self.character_path)
        frames = np.array([((t) % self.nframes) + 1 for t in range(24)])
        split = history_fragment_population(
            path, StateMap.from_json(self.state_map_path), character, [976, 977], frames
        )
        self.assertLess(float(np.max(np.abs(split.occupation_redistribution))), 1e-12)
        self.assertGreater(float(np.max(np.abs(split.character_evolution))), 0.1)

    def test_a_purely_character_driven_change_is_a_real_projected_change(self):
        """The correction this test exists to hold.

        This history keeps every adiabatic population dead flat at 0.5/0.5, so
        ``occupation_redistribution`` is exactly zero and the whole movement of
        ``P_g`` is ``character_evolution``. An earlier version of the
        classifier called that ``character_swap_without_fragment_transfer`` --
        "the band label now points at a different orbital; no charge moved
        between fragments". That is wrong. ``P_g`` sums over the whole basis,
        so a relabelling cannot move it at all; what moved it here is an
        occupied state's own composition turning from one fragment to the
        other, which carries its density with it.

        So the change must be **retained as a real projected fragment-population
        change**, and the mechanism must be left unresolved rather than denied.
        """
        path = self.root / "SHPROP.9"
        with path.open("w", encoding="utf-8") as handle:
            print(f"# NSW = {self.nframes + 1}", file=handle)
            for index in range(24):
                values = [(index + 1) * 10000.0, -0.8, 0.5, 0.5]
                print(" ".join(f"{v:.10E}" for v in values), file=handle)

        out, report = self._run(histories=[path])
        counts = report["per_history"]["totals_by_classification"]
        self.assertGreater(counts.get(PROJECTED_CHANGE, 0), 0)
        # The label that used to be produced here, and must not be any more.
        self.assertNotIn(NO_PROJECTED_CHANGE, counts)
        self.assertNotIn("character_swap_without_fragment_transfer", counts)

        with (out / "per_history_events.csv").open("r", encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if r["character_swap"] == "True"]
        self.assertTrue(rows)
        for row in rows:
            # Occupation is exactly flat; character carries all of it.
            self.assertLess(abs(float(row["occupation_redistribution"])), 1e-9)
            self.assertGreater(abs(float(row["character_evolution"])), 0.1)
            # And P_g moved, which is what the classification rests on.
            self.assertGreater(
                abs(float(row["projected_fragment_population_change"])), 0.1)
            self.assertEqual(row["classification"], PROJECTED_CHANGE)
            # Described, not explained.
            self.assertEqual(row["decomposition_descriptor"], "character_dominated")
            self.assertTrue(row["dominant_fragment"])

    def test_the_split_terms_close_on_the_fragment_they_are_reported_for(self):
        # dP_g = occupation + character, for the fragment the row names, so the
        # descriptor is a share of one quantity rather than of two unrelated
        # maxima.
        out, _ = self._run()
        with (out / "per_history_events.csv").open("r", encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if r["occupation_redistribution"]]
        self.assertTrue(rows)
        for row in rows:
            total = (float(row["occupation_redistribution"])
                     + float(row["character_evolution"]))
            self.assertAlmostEqual(
                abs(total),
                float(row["projected_fragment_population_change"]),
                places=9,
                msg=f"frame {row['frame']}: the split does not close on "
                    f"{row['dominant_fragment']}",
            )

    def test_a_character_driven_change_is_never_called_no_charge_moved(self):
        path = self.root / "SHPROP.9"
        with path.open("w", encoding="utf-8") as handle:
            print(f"# NSW = {self.nframes + 1}", file=handle)
            for index in range(24):
                values = [(index + 1) * 10000.0, -0.8, 0.5, 0.5]
                print(" ".join(f"{v:.10E}" for v in values), file=handle)

        out, _ = self._run(histories=[path])
        text = (out / "crossing_summary.md").read_text(encoding="utf-8")
        for forbidden in (
            "no charge moved between fragments",
            "The band labels moved; the charge did not follow",
            "the whole change is character-driven",
        ):
            self.assertNotIn(forbidden, text)
        self.assertIn("must not be reported as", text)
        self.assertIn("can correspond to a spatial redistribution", text)
        # And the opposite overstatement is absent too: the descriptor is not
        # a quantity of charge.
        self.assertIn("not a scale of how much charge moved", text)
        # And the bookkeeping caveat still travels with it.
        self.assertIn("one of infinitely many exact splits", text)
        self.assertIn("not physical branching fractions", text.lower())

    def test_the_descriptor_never_decides_whether_a_change_occurred(self):
        from namd_analysis.crossings import _classify, decomposition_descriptor

        # Character carries everything, occupation nothing.
        self.assertEqual(
            decomposition_descriptor(0.0, 0.5, 1e-3), "character_dominated")
        self.assertEqual(
            decomposition_descriptor(0.5, 0.0, 1e-3), "occupation_dominated")
        self.assertEqual(decomposition_descriptor(0.25, 0.25, 1e-3), "mixed")
        self.assertEqual(decomposition_descriptor(0.0, 0.0, 1e-3), "no_movement")
        self.assertIsNone(decomposition_descriptor(None, 0.5, 1e-3))
        # Whatever the descriptor says, |dP_g| decides the label.
        for descriptor in (
            "character_dominated", "occupation_dominated", "mixed", None,
        ):
            self.assertEqual(
                _classify(True, True, True, 0.5, 1e-3, None, descriptor)[0],
                PROJECTED_CHANGE,
            )
            self.assertEqual(
                _classify(True, True, True, 0.0, 1e-3, None, descriptor)[0],
                NO_PROJECTED_CHANGE,
            )


class CrossingCliTests(_Crossing):
    def test_end_to_end_writes_every_artifact(self):
        swapped = np.column_stack([self.weights[:, 1], self.weights[:, 0]])
        character_path = write_character(
            self.root / "projection_character.csv",
            self.frames,
            {976: self.weights, 977: swapped},
        )
        eig, nat = write_couplings(self.root, self.energies, self.nac)
        out = self.root / "out"
        code = dispatch_main(
            [
                "character-crossings",
                "--projection-character", str(character_path),
                "--eigtxt", str(eig),
                "--natxt", str(nat),
                "--dt-fs", "1.0",
                "--out", str(out),
            ]
        )
        self.assertEqual(code, 0)
        for name in (
            "crossing_events.csv",
            "crossing_metrics.csv",
            "crossing_summary.md",
            "report.json",
            "crossing_overlay.png",
            "character_heatmap.png",
        ):
            self.assertTrue((out / name).is_file(), name)

        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["synchronization"]["join_key"], "MD frame")
        self.assertIn("never on a SHPROP row index", report["synchronization"]["note"])
        self.assertEqual(set(report["thresholds"]["values"]), set(DEFAULT_THRESHOLDS))
        self.assertIn("not performed anywhere", report["vocabulary"]["true_diabatic_transformation"])

    def test_the_alias_is_the_same_command(self):
        swapped = np.column_stack([self.weights[:, 1], self.weights[:, 0]])
        character_path = write_character(
            self.root / "projection_character.csv",
            self.frames,
            {976: self.weights, 977: swapped},
        )
        eig, _ = write_couplings(self.root, self.energies, self.nac)
        code = dispatch_main(
            [
                "adiabatic-character-events",
                "--projection-character", str(character_path),
                "--eigtxt", str(eig),
                "--out", str(self.root / "alias"),
            ]
        )
        self.assertEqual(code, 0)

    def test_the_summary_keeps_the_vocabulary_apart(self):
        swapped = np.column_stack([self.weights[:, 1], self.weights[:, 0]])
        character_path = write_character(
            self.root / "projection_character.csv",
            self.frames,
            {976: self.weights, 977: swapped},
        )
        eig, nat = write_couplings(self.root, self.energies, self.nac)
        out = self.root / "summary"
        dispatch_main(
            [
                "character-crossings",
                "--projection-character", str(character_path),
                "--eigtxt", str(eig), "--natxt", str(nat), "--dt-fs", "1.0",
                "--out", str(out),
            ]
        )
        text = (out / "crossing_summary.md").read_text(encoding="utf-8")
        for phrase in (
            "A character swap is not a surface hop",
            "not** a diabatization",
            "not performed anywhere in this package",
            "changes** the fragment identity",
            # No SHPROP was supplied here, so the summary must report the
            # transfer question as unasked rather than answered.
            "NOT EVALUATED",
        ):
            self.assertIn(phrase, text)

    def test_classification_counts_reach_the_report(self):
        swapped = np.column_stack([self.weights[:, 1], self.weights[:, 0]])
        character_path = write_character(
            self.root / "projection_character.csv",
            self.frames,
            {976: self.weights, 977: swapped},
        )
        eig, nat = write_couplings(self.root, self.energies, self.nac)
        out = self.root / "counts"
        dispatch_main(
            [
                "character-crossings",
                "--projection-character", str(character_path),
                "--eigtxt", str(eig), "--natxt", str(nat), "--dt-fs", "1.0",
                "--out", str(out),
            ]
        )
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        counts = report["event_counts"]
        self.assertEqual(sum(counts.values()), report["n_events"])
        with (out / "crossing_events.csv").open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), report["n_events"])
        self.assertEqual(
            Counter(r["classification"] for r in rows), Counter(counts)
        )


if __name__ == "__main__":
    unittest.main()
