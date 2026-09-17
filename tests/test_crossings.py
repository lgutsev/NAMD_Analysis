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
            # transfer on the strength of the character alone.
            self.assertEqual(
                event.classification, "character_swap_without_fragment_transfer"
            )
            self.assertIn("NOT a charge-transfer event", event.note)

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
                event.classification, "character_swap_without_fragment_transfer"
            )
            self.assertEqual(event.fragment_population_change, 0.0)


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
            "not charge-transfer events",
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
