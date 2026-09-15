"""v0.5.1 hardening: selective loading, preflight, diagnostics, PROCAR dialects."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from character_fixtures import (
    expected_swap_populations,
    swap_campaign,
    write_atom_groups,
    write_manifest,
    write_poison_procar,
    write_procar,
    write_shprop,
    write_state_map,
)
from namd_analysis.character import (
    AtomGroupMap,
    CharacterError,
    aligned_frames,
    character_populations,
    fixed_vs_projected_summary,
    load_projection_series,
    plan_analysis,
    preflight_report,
)
from namd_analysis.dispatch import main as dispatch_main
from namd_analysis.io.procar import ProcarFormatError, procar_structure, read_procar_ion_totals
from namd_analysis.populations import StateMap


class _Campaign(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.campaign = swap_campaign(self.root)
        self.state_map = StateMap.from_json(self.campaign["state_map"])
        self.atom_groups = AtomGroupMap.from_json(self.campaign["atom_groups"])

    def tearDown(self):
        self.tmp.cleanup()

    def _plan(self, mode="dish-cyclic"):
        return plan_analysis(
            self.campaign["shprop"], self.state_map, self.campaign["manifest"], mode
        )

    def _run(self, mode="dish-cyclic"):
        return character_populations(
            self.campaign["shprop"],
            self.state_map,
            self.campaign["manifest"],
            self.atom_groups,
            mode,
        )


class ScientificInvariantTests(_Campaign):
    def test_projection_is_applied_per_file_before_averaging(self):
        result = self._run()
        expected = expected_swap_populations(self.campaign)
        np.testing.assert_allclose(result.mean, expected, atol=1e-12)

    def test_averaging_first_would_give_a_different_answer(self):
        # Guards the ordering: if the code ever averaged the adiabatic
        # populations and then projected with one frame series, this campaign
        # would move. The two histories visit the cycle in opposite order.
        result = self._run()
        pops_a = self.campaign["populations"]["SHPROP.1"]
        pops_b = self.campaign["populations"]["SHPROP.3"]
        averaged_first = 0.5 * (pops_a + pops_b)
        # Project the average with history 1's frames only (the wrong thing).
        ntime = averaged_first.shape[0]
        wrong = np.zeros((ntime, 2))
        for t, frame in enumerate([1, 2, 3, 4]):
            if frame in (1, 2):
                wrong[t] = averaged_first[t, 0], averaged_first[t, 1]
            else:
                wrong[t] = averaged_first[t, 1], averaged_first[t, 0]
        self.assertFalse(np.allclose(result.mean, wrong, atol=1e-6))

    def test_population_is_conserved(self):
        result = self._run()
        np.testing.assert_allclose(result.mean.sum(axis=1), 1.0, atol=1e-12)
        self.assertLess(result.conservation["max_abs_deviation_from_one"], 1e-12)

    def test_sem_is_across_files(self):
        result = self._run()
        self.assertIsNotNone(result.sem)
        expected = result.per_file.std(axis=0, ddof=1) / np.sqrt(2)
        np.testing.assert_allclose(result.sem, expected, atol=1e-12)


class FrameAlignmentTests(_Campaign):
    def test_different_namdtini_select_different_frames(self):
        plan = self._plan()
        first, second = plan.frames_by_file
        self.assertEqual(list(first), [1, 2, 3, 4])
        self.assertEqual(list(second), [3, 4, 1, 2])
        self.assertNotEqual(list(first), list(second))

    def test_linear_mode_counts_up_from_namdtini(self):
        frames = aligned_frames({"NAMDTINI": 7, "NSW": 100}, 5, "linear", Path("x"))
        self.assertEqual(list(frames), [7, 8, 9, 10, 11])

    def test_cyclic_wrap_is_exact_across_the_boundary(self):
        # Start two steps before the end of a 6-frame cycle and run past it, so
        # the sequence must read 5, 6, 1, 2, 3 with no gap and no repeat.
        frames = aligned_frames(
            {"NAMDTINI": 5, "NSW": 7}, 5, "dish-cyclic", Path("x")
        )
        self.assertEqual(list(frames), [5, 6, 1, 2, 3])

    def test_cyclic_wrap_never_emits_zero_or_exceeds_the_period(self):
        for start in range(1, 9):
            frames = aligned_frames(
                {"NAMDTINI": start, "NSW": 9}, 40, "dish-cyclic", Path("x")
            )
            self.assertTrue(np.all(frames >= 1))
            self.assertTrue(np.all(frames <= 8))
            # Consecutive steps advance by one, modulo the period.
            steps = np.diff(frames)
            self.assertTrue(np.all((steps == 1) | (steps == -(8 - 1))))

    def test_explicit_cycle_length_overrides_nsw_and_records_mismatch(self):
        # Manifest says 4; the SHPROP headers say NSW-1 = 4 as well, so rebuild
        # a manifest declaring a different period and confirm it wins.
        frames = {f: self.campaign["frames"][f] for f in (1, 2, 3)}
        manifest = write_manifest(self.root / "p3.json", frames, cycle_length=3)
        plan = plan_analysis(
            self.campaign["shprop"], self.state_map, manifest, "dish-cyclic"
        )
        self.assertEqual(list(plan.frames_by_file[0]), [1, 2, 3, 1])
        record = plan.alignments[0]
        self.assertEqual(record["cycle_length_used"], 3)
        self.assertEqual(record["cycle_length_source"], "projection_manifest")
        self.assertEqual(record["header_cycle_length"], 4)
        self.assertTrue(record["header_cycle_mismatch"])

    def test_alignment_record_is_human_readable(self):
        record = self._plan().alignments[0]
        self.assertEqual(record["first_five_frames"], [1, 2, 3, 4])
        self.assertEqual(record["last_five_frames"], [1, 2, 3, 4])
        self.assertEqual(record["unique_projection_frames_used"], 4)
        self.assertEqual(record["wrap_count"], 0)
        self.assertEqual(record["n_time_points"], 4)
        second = self._plan().alignments[1]
        self.assertEqual(second["first_five_frames"], [3, 4, 1, 2])
        self.assertEqual(second["wrap_count"], 1)

    def test_missing_required_frame_fails_with_the_range(self):
        frames = {f: self.campaign["frames"][f] for f in (1, 2, 3)}
        manifest = write_manifest(self.root / "gap.json", frames)
        with self.assertRaises(CharacterError) as ctx:
            character_populations(
                self.campaign["shprop"], self.state_map, manifest,
                self.atom_groups, "dish-cyclic",
            )
        message = str(ctx.exception)
        self.assertIn("required electronic frames", message)
        self.assertIn("[4]", message)
        self.assertIn("1..3", message)

    def test_cycle_length_larger_than_the_manifest_is_rejected(self):
        frames = {f: self.campaign["frames"][f] for f in (1, 2)}
        manifest = write_manifest(self.root / "short.json", frames, cycle_length=6)
        with self.assertRaises(CharacterError) as ctx:
            plan_analysis(
                self.campaign["shprop"], self.state_map, manifest, "dish-cyclic"
            )
        self.assertIn("cycle_length=6", str(ctx.exception))

    def test_missing_namdtini_names_the_file_and_the_key(self):
        path = write_shprop(self.root / "SHPROP.bad", 1, np.full((4, 2), 0.5))
        text = path.read_text(encoding="utf-8").replace("# NAMDTINI = 1\n", "")
        path.write_text(text, encoding="utf-8")
        with self.assertRaises(CharacterError) as ctx:
            plan_analysis(
                [path], self.state_map, self.campaign["manifest"], "dish-cyclic"
            )
        message = str(ctx.exception)
        self.assertIn("SHPROP.bad", message)
        self.assertIn("NAMDTINI", message)

    def test_basis_size_mismatch_names_both_counts(self):
        state_map = write_state_map(self.root / "wide.json", bmin=10, bmax=12)
        with self.assertRaises(CharacterError) as ctx:
            plan_analysis(
                self.campaign["shprop"],
                StateMap.from_json(state_map),
                self.campaign["manifest"],
                "dish-cyclic",
            )
        message = str(ctx.exception)
        self.assertIn("10:11", message)
        self.assertIn("2 states", message)
        self.assertIn("3 population columns", message)
        self.assertIn("fixable", message.lower())


class SelectiveLoadingTests(_Campaign):
    def test_only_required_frames_are_parsed(self):
        # Frames 5..12 are declared but never visited. They are written as
        # unparseable rubbish: if the loader opened them, this would raise.
        frames = dict(self.campaign["frames"])
        for extra in range(5, 13):
            frames[extra] = write_poison_procar(self.root / f"PROCAR.poison{extra}")
        manifest = write_manifest(self.root / "wide.json", frames, cycle_length=4)
        plan = plan_analysis(
            self.campaign["shprop"], self.state_map, manifest, "dish-cyclic"
        )
        self.assertEqual(plan.required_frames, [1, 2, 3, 4])
        self.assertEqual(plan.manifest_frame_count, 12)

        result = character_populations(
            self.campaign["shprop"], self.state_map, manifest,
            self.atom_groups, "dish-cyclic", plan=plan,
        )
        self.assertEqual(len(result.projection.frames), 4)
        np.testing.assert_allclose(
            result.mean, expected_swap_populations(self.campaign), atol=1e-12
        )

    def test_consumption_reports_declared_and_consumed(self):
        frames = dict(self.campaign["frames"])
        for extra in range(5, 13):
            frames[extra] = write_poison_procar(self.root / f"PROCAR.poison{extra}")
        manifest = write_manifest(self.root / "wide.json", frames, cycle_length=4)
        consumption = plan_analysis(
            self.campaign["shprop"], self.state_map, manifest, "dish-cyclic"
        ).consumption()
        self.assertEqual(consumption["manifest_declared_frames"], 12)
        self.assertEqual(consumption["frames_required_by_shprop"], 4)
        self.assertEqual(consumption["procars_parsed"], 4)
        self.assertEqual(consumption["procars_skipped"], 8)

    def test_manifest_validation_is_not_weakened_by_selection(self):
        # A declared-but-unused PROCAR that does not exist is still an error:
        # skipping it to save time would let a real gap go unnoticed.
        frames = dict(self.campaign["frames"])
        frames[9] = self.root / "PROCAR.absent"
        manifest = write_manifest(self.root / "absent.json", frames, cycle_length=4)
        with self.assertRaises(CharacterError) as ctx:
            plan_analysis(
                self.campaign["shprop"], self.state_map, manifest, "dish-cyclic"
            )
        self.assertIn("missing PROCAR files", str(ctx.exception))

    def test_load_projection_series_honours_a_frame_subset(self):
        series = load_projection_series(
            self.campaign["manifest"], self.atom_groups, [10, 11], required_frames=[2, 3]
        )
        self.assertEqual(list(series.frames), [2, 3])


class QualityDiagnosticTests(_Campaign):
    def test_quality_summary_reports_percentiles_and_per_band(self):
        quality = self._run().projection.quality_summary()
        for key in (
            "minimum_captured_projection",
            "p1_captured_projection",
            "p5_captured_projection",
            "median_captured_projection",
            "p95_captured_projection",
        ):
            self.assertIn(key, quality)
        self.assertEqual(len(quality["per_band"]), 2)
        for entry in quality["per_band"]:
            self.assertIn("median_captured_projection", entry)
            self.assertIn("frame_of_minimum", entry)
        self.assertTrue(quality["worst_frame_band_samples"])
        self.assertTrue(quality["worst_frames_by_minimum_capture"])

    def test_low_capture_is_reported_not_deleted(self):
        # Ions 3 and 4 carry almost nothing for band 11 at frame 1, so the
        # declared groups capture little of it. The sample must survive.
        frames = dict(self.campaign["frames"])
        frames[1] = write_procar(
            self.root / "PROCAR.weak", {10: [1.0, 1.0, 0.0, 0.0], 11: [0.01, 0.0, 0.0, 0.0]}
        )
        manifest = write_manifest(self.root / "weak.json", frames, cycle_length=4)
        result = character_populations(
            self.campaign["shprop"], self.state_map, manifest,
            self.atom_groups, "dish-cyclic",
        )
        quality = result.projection.quality_summary()
        self.assertLess(quality["minimum_captured_projection"], 0.5)
        self.assertGreaterEqual(quality["samples_below_threshold"], 1)
        # Still four frames: nothing was dropped.
        self.assertEqual(len(result.projection.frames), 4)
        self.assertIn("Nothing is discarded", quality["note"])

    def test_zero_capture_is_a_loud_error_not_a_divide_by_zero(self):
        frames = dict(self.campaign["frames"])
        frames[1] = write_procar(
            self.root / "PROCAR.zero", {10: [1.0, 1.0, 0.0, 0.0], 11: [0.0, 0.0, 0.0, 0.0]}
        )
        manifest = write_manifest(self.root / "zero.json", frames, cycle_length=4)
        with self.assertRaises(CharacterError) as ctx:
            character_populations(
                self.campaign["shprop"], self.state_map, manifest,
                self.atom_groups, "dish-cyclic",
            )
        message = str(ctx.exception)
        self.assertIn("essentially zero projection", message)
        self.assertIn("11", message)

    def test_dominance_summary_counts_swaps_per_band(self):
        dominance = self._run().projection.dominance_summary(0.6)
        self.assertEqual(len(dominance["per_band"]), 2)
        for entry in dominance["per_band"]:
            # Each band is BCF for frames 1-2 and PCBM for 3-4: exactly one swap.
            self.assertEqual(entry["dominant_character_swaps"], 1)
            self.assertAlmostEqual(
                sum(entry["dominant_occupancy_fraction"].values()), 1.0, places=12
            )


class DiscrepancyTests(_Campaign):
    def test_summary_matches_a_hand_computed_difference(self):
        groups = {"BCF": [2], "PCBM": [3]}
        state_map = StateMap.from_json(
            write_state_map(self.root / "named.json", groups=groups)
        )
        result = character_populations(
            self.campaign["shprop"], state_map, self.campaign["manifest"],
            self.atom_groups, "dish-cyclic",
        )
        summary = fixed_vs_projected_summary(result)
        self.assertEqual(sorted(summary["shared_groups"]), ["BCF", "PCBM"])

        lookup = {name: i for i, name in enumerate(result.group_names)}
        for record in summary["per_group"]:
            fixed = np.asarray(result.fixed_mean[record["group"]])
            projected = result.mean[:, lookup[record["group"]]]
            difference = projected - fixed
            self.assertAlmostEqual(
                record["max_abs_difference"], float(np.max(np.abs(difference))), places=12
            )
            self.assertAlmostEqual(
                record["rms_difference"],
                float(np.sqrt(np.mean(difference**2))),
                places=12,
            )
            peak = int(np.argmax(np.abs(difference)))
            self.assertAlmostEqual(
                record["time_of_max_ns"], float(result.time_ns[peak]), places=12
            )

    def test_identical_maps_give_zero_discrepancy(self):
        # Make every PROCAR frame put band 10 entirely on BCF and band 11
        # entirely on PCBM, so projected and fixed labels agree exactly.
        frames = {
            frame: write_procar(
                self.root / f"PROCAR.flat{frame}",
                {10: [1.0, 1.0, 0.0, 0.0], 11: [0.0, 0.0, 1.0, 1.0]},
            )
            for frame in (1, 2, 3, 4)
        }
        manifest = write_manifest(self.root / "flat.json", frames, cycle_length=4)
        state_map = StateMap.from_json(
            write_state_map(self.root / "flat_map.json", groups={"BCF": [2], "PCBM": [3]})
        )
        result = character_populations(
            self.campaign["shprop"], state_map, manifest, self.atom_groups, "dish-cyclic"
        )
        summary = fixed_vs_projected_summary(result)
        for record in summary["per_group"]:
            self.assertLess(record["max_abs_difference"], 1e-12)
            self.assertLess(record["rms_difference"], 1e-12)

    def test_groups_present_in_only_one_map_are_listed(self):
        result = self._run()
        summary = fixed_vs_projected_summary(result)
        self.assertEqual(summary["shared_groups"], [])
        self.assertEqual(summary["groups_only_in_projection"], ["BCF", "PCBM"])
        self.assertEqual(sorted(summary["groups_only_in_fixed_map"]), ["band10", "band11"])


class PreflightTests(_Campaign):
    def test_preflight_reports_the_plan_without_parsing_projections(self):
        frames = dict(self.campaign["frames"])
        for extra in range(5, 9):
            frames[extra] = write_poison_procar(self.root / f"PROCAR.poison{extra}")
        manifest = write_manifest(self.root / "wide.json", frames, cycle_length=4)
        report = preflight_report(
            self.campaign["shprop"], self.state_map, manifest,
            self.atom_groups, "dish-cyclic",
        )
        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["n_shprop_files"], 2)
        self.assertEqual(report["distinct_namdtini"], [1, 3])
        self.assertEqual(report["band_window"]["basis_size"], 2)
        self.assertEqual(report["frames"]["frames_required_by_shprop"], 4)
        self.assertEqual(report["frames"]["procars_skipped"], 4)
        self.assertEqual(report["representative_procar"]["n_ions"], 4)
        self.assertEqual(report["atom_coverage"]["assigned_ions"], 4)

    def test_preflight_catches_bad_configuration_without_the_full_run(self):
        groups = write_atom_groups(
            self.root / "partial.json", {"BCF": [1, 2]}, complete_atoms=True
        )
        report = preflight_report(
            self.campaign["shprop"], self.state_map, self.campaign["manifest"],
            AtomGroupMap.from_json(groups), "dish-cyclic",
        )
        self.assertFalse(report["ok"])
        joined = " ".join(report["problems"])
        self.assertIn("complete_atoms", joined)
        self.assertIn("2 of the 4", joined)
        self.assertIn("Fixable", joined)

    def test_preflight_reports_a_missing_frame(self):
        frames = {f: self.campaign["frames"][f] for f in (1, 2, 3)}
        manifest = write_manifest(self.root / "gap.json", frames)
        report = preflight_report(
            self.campaign["shprop"], self.state_map, manifest,
            self.atom_groups, "dish-cyclic",
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["n_missing_required_frames"], 1)
        self.assertIn(4, report["missing_required_frames"])

    def test_preflight_warns_on_cycle_disagreement(self):
        frames = {f: self.campaign["frames"][f] for f in (1, 2, 3)}
        manifest = write_manifest(self.root / "p3.json", frames, cycle_length=3)
        report = preflight_report(
            self.campaign["shprop"], self.state_map, manifest,
            self.atom_groups, "dish-cyclic",
        )
        self.assertTrue(report["cycle"]["disagree"])
        self.assertTrue(any("cycle_length=3" in w for w in report["warnings"]))

    def test_preflight_survives_a_planning_failure(self):
        state_map = StateMap.from_json(write_state_map(self.root / "wide2.json", bmax=12))
        report = preflight_report(
            self.campaign["shprop"], state_map, self.campaign["manifest"],
            self.atom_groups, "dish-cyclic",
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["stage"], "planning")
        self.assertEqual(len(report["problems"]), 1)


class ProcarDialectTests(unittest.TestCase):
    """Real VASP dialects must be read correctly or rejected, never mis-read."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.totals = {10: [0.8, 0.2, 0.05, 0.0], 11: [0.1, 0.0, 0.5, 0.4]}

    def tearDown(self):
        self.tmp.cleanup()

    def _read(self, **kwargs):
        path = write_procar(self.root / "PROCAR", self.totals, **kwargs)
        return read_procar_ion_totals(path)

    def test_plain_single_kpoint(self):
        result = self._read()
        np.testing.assert_allclose(result.ion_totals[0], self.totals[10])
        np.testing.assert_allclose(result.ion_totals[1], self.totals[11])

    def test_tot_summary_row_is_not_read_as_an_ion(self):
        result = self._read()
        self.assertEqual(result.ion_totals.shape, (2, 4))

    def test_lm_decomposed_columns(self):
        result = self._read(orbitals="spd")
        np.testing.assert_allclose(result.ion_totals[0], self.totals[10], atol=1e-9)

    def test_scientific_notation(self):
        result = self._read(scientific=True)
        np.testing.assert_allclose(result.ion_totals[0], self.totals[10], atol=1e-9)

    def test_extra_whitespace(self):
        result = self._read(pad="        ")
        np.testing.assert_allclose(result.ion_totals[0], self.totals[10], atol=1e-9)

    def test_crlf_and_bom(self):
        result = self._read(newline="\r\n", bom=True)
        np.testing.assert_allclose(result.ion_totals[0], self.totals[10], atol=1e-9)

    def test_phase_table_is_not_read_instead_of_the_charge_table(self):
        # LORBIT=12 writes a second ion table per band holding phase factors.
        # Reading it would silently halve and negate every weight.
        result = self._read(tables_per_band=2)
        np.testing.assert_allclose(result.ion_totals[0], self.totals[10], atol=1e-9)
        self.assertTrue(np.all(result.ion_totals >= 0.0))

    def test_soc_style_four_tables_per_band(self):
        result = self._read(tables_per_band=4)
        np.testing.assert_allclose(result.ion_totals[0], self.totals[10], atol=1e-9)

    def test_multiple_kpoints_rejected(self):
        with self.assertRaises(ProcarFormatError) as ctx:
            self._read(nkpoints=2)
        self.assertIn("k-points", str(ctx.exception))

    def test_multiple_spins_rejected(self):
        with self.assertRaises(ProcarFormatError) as ctx:
            self._read(spin_blocks=2)
        self.assertIn("spin components", str(ctx.exception))

    def test_truncated_ion_table_rejected(self):
        path = write_procar(self.root / "PROCAR", self.totals, truncate_after=9)
        with self.assertRaises(ProcarFormatError) as ctx:
            read_procar_ion_totals(path)
        self.assertIn("band", str(ctx.exception))

    def test_duplicate_ion_row_rejected(self):
        with self.assertRaises(ProcarFormatError) as ctx:
            self._read(duplicate_ion=True)
        self.assertIn("in order", str(ctx.exception))

    def test_declared_band_count_mismatch_rejected(self):
        with self.assertRaises(ProcarFormatError) as ctx:
            self._read(declared_bands=5)
        self.assertIn("declares", str(ctx.exception))

    def test_missing_required_band_names_it(self):
        path = write_procar(self.root / "PROCAR", self.totals)
        with self.assertRaises(ProcarFormatError) as ctx:
            read_procar_ion_totals(path, bands=[10, 99])
        self.assertIn("99", str(ctx.exception))

    def test_band_filter_only_retains_requested_bands(self):
        path = write_procar(self.root / "PROCAR", self.totals)
        result = read_procar_ion_totals(path, bands=[11])
        self.assertEqual(list(result.bands), [11])
        self.assertEqual(result.ion_totals.shape, (1, 4))
        self.assertEqual(result.bands_seen, 2)

    def test_not_a_procar_rejected(self):
        path = self.root / "PROCAR"
        path.write_text("hello\n" * 50, encoding="utf-8")
        with self.assertRaises(ProcarFormatError) as ctx:
            read_procar_ion_totals(path)
        self.assertIn("does not look like a PROCAR", str(ctx.exception))

    def test_structure_probe_reads_only_the_header(self):
        path = write_procar(self.root / "PROCAR", self.totals, orbitals="spd")
        structure = procar_structure(path)
        self.assertEqual(structure["n_ions"], 4)
        self.assertEqual(structure["n_bands"], 2)
        self.assertEqual(structure["n_kpoints"], 1)
        self.assertTrue(structure["single_kpoint"])
        self.assertIn("tot", structure["ion_header_columns"])


class CliSmokeTests(_Campaign):
    """End to end through the installed command spelling."""

    def test_character_populations_end_to_end(self):
        out = self.root / "out"
        code = dispatch_main(
            [
                "character-populations",
                "--files", str(self.campaign["shprop"][0]), str(self.campaign["shprop"][1]),
                "--config", str(self.campaign["state_map"]),
                "--projection-manifest", str(self.campaign["manifest"]),
                "--atom-groups", str(self.campaign["atom_groups"]),
                "--frame-mode", "dish-cyclic",
                "--out", str(out),
            ]
        )
        self.assertEqual(code, 0)
        for name in (
            "character_populations.csv",
            "fixed_vs_projected.csv",
            "fixed_vs_projected_summary.csv",
            "character_swaps.csv",
            "projection_character.csv",
            "projection_quality_by_band.csv",
            "shprop_alignment.csv",
            "report.json",
            "character_populations.png",
            "character_populations.pdf",
        ):
            self.assertTrue((out / name).is_file(), name)

        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["command"], "character-populations")
        self.assertEqual(report["frame_mode"], "dish-cyclic")
        self.assertEqual(report["frame_consumption"]["frames_required_by_shprop"], 4)
        self.assertEqual(len(report["inputs"]), 2 + 3 + 4)
        for record in report["inputs"]:
            self.assertEqual(len(record["sha256"]), 64)
        self.assertIn("fixed_vs_projected_summary", report)
        self.assertIn("dominance", report)
        self.assertIn("p5_captured_projection", report["projection_quality"])

        alignment = report["shprop_alignment"]
        self.assertEqual(alignment[0]["first_five_frames"], [1, 2, 3, 4])
        self.assertEqual(alignment[1]["first_five_frames"], [3, 4, 1, 2])

        expected = expected_swap_populations(self.campaign)
        rows = (out / "character_populations.csv").read_text(encoding="utf-8").splitlines()[1:]
        values = {}
        for row in rows:
            time, group, population = row.split(",")[:3]
            values[(float(time), group)] = float(population)
        for ti, time in enumerate(np.arange(4) * 1000.0 / 1e6):
            self.assertAlmostEqual(values[(time, "BCF")], expected[ti, 0], places=10)
            self.assertAlmostEqual(values[(time, "PCBM")], expected[ti, 1], places=10)

    def test_preflight_flag_writes_no_populations(self):
        out = self.root / "pre"
        code = dispatch_main(
            [
                "character-populations",
                "--files", str(self.campaign["shprop"][0]), str(self.campaign["shprop"][1]),
                "--config", str(self.campaign["state_map"]),
                "--projection-manifest", str(self.campaign["manifest"]),
                "--atom-groups", str(self.campaign["atom_groups"]),
                "--frame-mode", "dish-cyclic",
                "--preflight",
                "--out", str(out),
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue((out / "preflight.json").is_file())
        self.assertFalse((out / "character_populations.csv").exists())
        payload = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
        self.assertTrue(payload["preflight"]["ok"])

    def test_preflight_alias_needs_no_output_directory(self):
        code = dispatch_main(
            [
                "character-preflight",
                "--files", str(self.campaign["shprop"][0]),
                "--config", str(self.campaign["state_map"]),
                "--projection-manifest", str(self.campaign["manifest"]),
                "--atom-groups", str(self.campaign["atom_groups"]),
                "--frame-mode", "dish-cyclic",
            ]
        )
        self.assertEqual(code, 0)

    def test_preflight_returns_nonzero_on_a_blocking_problem(self):
        groups = write_atom_groups(
            self.root / "partial.json", {"BCF": [1, 2]}, complete_atoms=True
        )
        code = dispatch_main(
            [
                "character-preflight",
                "--files", str(self.campaign["shprop"][0]),
                "--config", str(self.campaign["state_map"]),
                "--projection-manifest", str(self.campaign["manifest"]),
                "--atom-groups", str(groups),
                "--frame-mode", "dish-cyclic",
            ]
        )
        self.assertEqual(code, 2)

    def test_missing_out_without_preflight_is_refused(self):
        code = dispatch_main(
            [
                "character-populations",
                "--files", str(self.campaign["shprop"][0]),
                "--config", str(self.campaign["state_map"]),
                "--projection-manifest", str(self.campaign["manifest"]),
                "--atom-groups", str(self.campaign["atom_groups"]),
                "--frame-mode", "dish-cyclic",
            ]
        )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
