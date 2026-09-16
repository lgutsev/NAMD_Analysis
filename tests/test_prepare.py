"""v0.5.2: campaign preparation, presets, Slurm generation and versioning."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis import __version__
from namd_analysis.character import AtomGroupMap, _load_projection_manifest
from namd_analysis.dispatch import main as dispatch_main
from namd_analysis.populations import StateMap
from namd_analysis.prepare import (
    FROM_CLI,
    FROM_DIRECTORY,
    PrepareError,
    SlurmOptions,
    build_manifest_payload,
    discover_frames,
    infer_columns,
    prepare_campaign,
    propose_cycle_length,
    survey_shprop,
    validate_atom_groups,
)
from namd_analysis.presets import PresetError, load_preset
from prepare_fixtures import (
    A_IONS,
    A_PARTITION,
    build_campaign,
    six_state_populations,
    write_campaign_procar,
    write_campaign_shprop,
)

REPO = Path(__file__).resolve().parent.parent


class _Prepared(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.campaign = build_campaign(self.root)
        self.out = self.root / "cfg"

    def tearDown(self):
        self.tmp.cleanup()

    def _prepare(self, **kwargs):
        options = dict(
            shprop_paths=self.campaign["shprop"],
            projection_dir=self.campaign["projection_dir"],
            out_dir=self.out,
            frame_mode="dish-cyclic",
            preset="bcf_pcbm",
            campaign="A",
        )
        options.update(kwargs)
        return prepare_campaign(**options)


class VersionTests(unittest.TestCase):
    def test_package_and_metadata_versions_agree_at_0_5_2(self):
        self.assertEqual(__version__, "0.5.2")
        text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('version = "0.5.2"', text)

    def test_provenance_reports_the_package_version(self):
        from namd_analysis.provenance import environment

        self.assertEqual(environment()["version"], __version__)


class ColumnInferenceTests(_Prepared):
    def test_six_population_columns_are_inferred_from_time_energy_six(self):
        survey = survey_shprop(self.campaign["shprop"])
        self.assertEqual(survey.n_states, 6)
        self.assertEqual(survey.n_columns, 8)
        columns, time_column, rationale = infer_columns(survey)
        self.assertEqual(columns, [2, 3, 4, 5, 6, 7])
        self.assertEqual(time_column, 0)
        self.assertEqual(rationale["chosen"], [2, 3, 4, 5, 6, 7])

    def test_the_energy_column_is_rejected_on_evidence_not_position(self):
        survey = survey_shprop(self.campaign["shprop"])
        _, _, rationale = infer_columns(survey)
        rejected = [c for c in rationale["candidates"] if not c["accepted"]]
        self.assertTrue(rejected)
        block = next(c for c in rejected if c["population_columns"] == [1, 2, 3, 4, 5, 6])
        self.assertTrue(
            any("row sums" in reason or "outside [0,1]" in reason for reason in block["reasons"])
        )

    def test_an_ambiguous_layout_is_refused(self):
        # Duplicate the population block so two readings fit equally well.
        rows = 10
        pops = six_state_populations(rows, seed=3)
        path = self.root / "amb" / "SHPROP.1"
        path.parent.mkdir(parents=True)
        table = np.column_stack(
            [np.arange(1, rows + 1, dtype=float)] + [pops[:, i] for i in range(6)]
            + [pops[:, i] for i in range(6)]
        )
        with path.open("w", encoding="utf-8") as handle:
            handle.write("# NAMDTINI = 1\n# NSW = 21\n# BMIN = 10\n# BMAX = 15\n")
            for row in table:
                handle.write(" ".join(f"{v:.10E}" for v in row) + "\n")
        survey = survey_shprop([path])
        with self.assertRaises(PrepareError) as ctx:
            infer_columns(survey)
        message = str(ctx.exception)
        self.assertIn("fit equally well", message)
        self.assertIn("[1, 2, 3, 4, 5, 6]", message)
        self.assertIn("[7, 8, 9, 10, 11, 12]", message)

    def test_no_fitting_layout_is_refused_with_the_evidence(self):
        rows = 8
        path = self.root / "bad" / "SHPROP.1"
        path.parent.mkdir(parents=True)
        table = np.column_stack(
            [np.arange(1, rows + 1, dtype=float)] + [np.full(rows, 5.0)] * 7
        )
        with path.open("w", encoding="utf-8") as handle:
            handle.write("# NAMDTINI = 1\n# NSW = 21\n# BMIN = 10\n# BMAX = 15\n")
            for row in table:
                handle.write(" ".join(f"{v:.10E}" for v in row) + "\n")
        with self.assertRaises(PrepareError) as ctx:
            infer_columns(survey_shprop([path]))
        self.assertIn("behaves like a complete population", str(ctx.exception))

    def test_the_rationale_states_how_much_of_the_file_was_sampled(self):
        # The sample is bounded so preparation costs the same on a 900 MB
        # history as on a small one. That bound is a real blind spot and the
        # report says so rather than implying the whole file was checked.
        survey = survey_shprop(self.campaign["shprop"])
        _, _, rationale = infer_columns(survey, sample_rows=5)
        self.assertEqual(rationale["sampled_rows_per_file"], 5)
        self.assertEqual(rationale["total_rows_per_file"], survey.n_rows)
        self.assertFalse(rationale["sample_covers_whole_file"])
        self.assertIn("validates every row", rationale["sampling_caveat"])

    def test_a_full_sample_is_reported_as_covering_the_file(self):
        survey = survey_shprop(self.campaign["shprop"])
        _, _, rationale = infer_columns(survey)
        self.assertTrue(rationale["sample_covers_whole_file"])

    def test_inconsistent_table_structure_is_refused(self):
        odd = write_campaign_shprop(
            self.root / "NuTest" / "SHPROP.99",
            9,
            six_state_populations(12, seed=7),
            extra_columns=2,
        )
        with self.assertRaises(PrepareError) as ctx:
            survey_shprop(self.campaign["shprop"] + [odd])
        self.assertIn("different column counts", str(ctx.exception))

    def test_inconsistent_row_counts_are_refused(self):
        odd = write_campaign_shprop(
            self.root / "NuTest" / "SHPROP.98", 9, six_state_populations(5, seed=8)
        )
        with self.assertRaises(PrepareError) as ctx:
            survey_shprop(self.campaign["shprop"] + [odd])
        self.assertIn("different row counts", str(ctx.exception))

    def test_a_missing_header_is_refused_not_inferred(self):
        path = self.root / "NuTest" / "SHPROP.noheader"
        rows = six_state_populations(6, seed=2)
        write_campaign_shprop(path, 1, rows)
        text = path.read_text(encoding="utf-8").replace("# BMIN = 10\n", "")
        path.write_text(text, encoding="utf-8")
        with self.assertRaises(PrepareError) as ctx:
            survey_shprop([path])
        self.assertIn("no integer BMIN", str(ctx.exception))


class PresetTests(_Prepared):
    def test_campaign_a_state_map_is_generated_exactly(self):
        prepared = self._prepare()
        payload = json.loads(prepared.state_map_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["name"], "FAPI_001_BCF_PCBM_A")
        self.assertEqual(payload["time_column"], 0)
        self.assertEqual(payload["time_unit"], "fs")
        self.assertEqual(payload["population_columns"], [2, 3, 4, 5, 6, 7])
        self.assertEqual(
            payload["groups"], {"VBM": [2], "BCF": [3], "PCBM": [4, 5, 6], "CBM": [7]}
        )
        self.assertTrue(payload["complete_population"])
        self.assertEqual(payload["recombined_group"], "VBM")
        # It must load as a real state map, not merely look like one.
        StateMap.from_json(prepared.state_map_path)

    def test_campaign_a_atom_groups_are_generated_exactly(self):
        prepared = self._prepare()
        payload = json.loads(prepared.atom_groups_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["groups"], A_PARTITION)
        self.assertTrue(payload["complete_atoms"])
        self.assertEqual(payload["min_projection_weight"], 0.5)
        AtomGroupMap.from_json(prepared.atom_groups_path)

    def test_the_partition_covers_exactly_one_to_446(self):
        prepared = self._prepare()
        validation = prepared.report["atom_groups"]["validation"]
        self.assertEqual(validation["assigned_ions"], A_IONS)
        self.assertEqual(validation["procar_ions"], A_IONS)
        self.assertEqual(validation["lowest_ion"], 1)
        self.assertEqual(validation["highest_ion"], A_IONS)
        self.assertEqual(validation["n_unassigned_ions"], 0)
        self.assertTrue(validation["covers_exactly"])

    def test_a_different_ion_count_fails_loudly(self):
        other = build_campaign(self.root / "other", n_ions=400)
        with self.assertRaises(PrepareError) as ctx:
            prepare_campaign(
                other["shprop"],
                other["projection_dir"],
                self.root / "other_cfg",
                "dish-cyclic",
                preset="bcf_pcbm",
                campaign="A",
            )
        message = str(ctx.exception)
        self.assertIn("446-ion", message)
        self.assertIn("400 ions", message)
        self.assertIn("not rescaled or trimmed", message)

    def test_a_known_but_unresolved_campaign_is_not_substituted(self):
        # B and C exist, but their provenance was never established. The error
        # must say that, and must not hand back A's bands or A's partition.
        for campaign in ("B", "C"):
            with self.assertRaises(PresetError) as ctx:
                load_preset("bcf_pcbm", campaign)
            message = str(ctx.exception)
            self.assertIn("known but unresolved", message)
            self.assertIn("none of them transfer from A", message)
            self.assertNotIn("976", message)

    def test_an_entirely_unknown_campaign_is_a_different_error(self):
        with self.assertRaises(PresetError) as ctx:
            load_preset("bcf_pcbm", "Z")
        message = str(ctx.exception)
        self.assertIn("no campaign 'Z'", message)
        self.assertIn("['A']", message)

    def test_campaign_a_bands_are_the_confirmed_vasp_numbers(self):
        preset = load_preset("bcf_pcbm", "A")
        self.assertEqual(preset.band_numbers, [976, 977, 978, 979, 980, 981])
        self.assertEqual(
            len(preset.band_numbers), len(preset.population_columns)
        )

    def test_no_other_registered_campaign_carries_As_bands(self):
        from namd_analysis.presets import PRESETS

        for preset_name, campaigns in PRESETS.items():
            for campaign_name, entry in campaigns.items():
                if (preset_name, campaign_name) == ("bcf_pcbm", "A"):
                    continue
                self.assertNotEqual(
                    entry.band_numbers,
                    [976, 977, 978, 979, 980, 981],
                    f"{preset_name}/{campaign_name} must not inherit A's basis",
                )

    def test_without_a_preset_or_atom_groups_nothing_is_invented(self):
        with self.assertRaises(PrepareError) as ctx:
            self._prepare(preset=None, campaign=None)
        message = str(ctx.exception)
        self.assertIn("never inferred", message)
        self.assertIn("contiguity are not evidence", message)

    def test_atom_group_validation_rejects_an_overreaching_partition(self):
        with self.assertRaises(PrepareError) as ctx:
            validate_atom_groups({"groups": {"a": ["1-500"]}}, A_IONS, "test")
        self.assertIn("only 446 ions", str(ctx.exception))

    def test_atom_group_validation_rejects_overlap(self):
        with self.assertRaises(PrepareError) as ctx:
            validate_atom_groups(
                {"groups": {"a": ["1-10"], "b": ["5-20"]}}, A_IONS, "test"
            )
        self.assertIn("must be disjoint", str(ctx.exception))


class FrameDiscoveryTests(_Prepared):
    def test_numeric_frame_directories_are_discovered(self):
        discovery = discover_frames(self.campaign["projection_dir"])
        self.assertEqual(discovery.count, 20)
        self.assertEqual((discovery.first, discovery.last), (1, 20))
        self.assertEqual(discovery.missing, [])
        self.assertEqual(discovery.padding, 4)
        self.assertTrue(discovery.regular)

    def test_a_missing_frame_directory_is_detected(self):
        import shutil

        shutil.rmtree(self.campaign["projection_dir"] / "0007")
        discovery = discover_frames(self.campaign["projection_dir"])
        self.assertEqual(discovery.missing, [7])
        self.assertFalse(discovery.regular)
        self.assertIn("missing", discovery.irregular_reason)

    def test_a_numeric_directory_without_a_procar_is_reported_not_counted(self):
        (self.campaign["projection_dir"] / "0099").mkdir()
        discovery = discover_frames(self.campaign["projection_dir"])
        self.assertNotIn(99, discovery.frames)
        self.assertIn("0099", discovery.numeric_dirs_without_procar)

    def test_duplicate_frame_spellings_are_refused(self):
        write_campaign_procar(
            self.campaign["projection_dir"] / "7" / "PROCAR", A_IONS, range(10, 16)
        )
        with self.assertRaises(PrepareError) as ctx:
            discover_frames(self.campaign["projection_dir"])
        message = str(ctx.exception)
        self.assertIn("more than one directory spelling", message)
        self.assertIn("frame 7", message)

    def test_a_regular_tree_gives_a_pattern_manifest(self):
        discovery = discover_frames(self.campaign["projection_dir"])
        payload = build_manifest_payload(discovery, 20)
        self.assertIn("procar_pattern", payload)
        self.assertIn("{frame:04d}", payload["procar_pattern"])
        self.assertEqual(payload["first_frame"], 1)
        self.assertEqual(payload["last_frame"], 20)
        self.assertEqual(payload["frame_step"], 1)
        self.assertEqual(payload["cycle_length"], 20)

    def test_an_unpadded_tree_is_regular_and_uses_a_plain_frame_field(self):
        # Names "1".."12" have mixed widths but are still exactly reproduced by
        # "{frame}": a width check alone would wrongly call this irregular and
        # emit 1999 explicit entries for a real campaign.
        campaign = build_campaign(self.root / "flat", frames=range(1, 13), padding=1)
        discovery = discover_frames(campaign["projection_dir"])
        self.assertTrue(discovery.regular)
        self.assertEqual(discovery.padding, 1)
        payload = build_manifest_payload(discovery, 12)
        self.assertIn("{frame}/", payload["procar_pattern"])
        self.assertNotIn("{frame:0", payload["procar_pattern"])

    def test_generated_patterns_round_trip_through_the_real_loader(self):
        for padding in (1, 4):
            campaign = build_campaign(
                self.root / f"rt{padding}", frames=range(1, 13), padding=padding
            )
            discovery = discover_frames(campaign["projection_dir"])
            payload = build_manifest_payload(discovery, 12)
            path = self.root / f"rt{padding}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = _load_projection_manifest(path)
            self.assertEqual(len(manifest.frames), 12)
            self.assertEqual(manifest.cycle_length, 12)
            self.assertTrue(all(p.is_file() for _, p in manifest.frames))
            self.assertEqual(sorted(f for f, _ in manifest.frames), list(range(1, 13)))

    def test_a_genuinely_mixed_tree_is_irregular(self):
        campaign = build_campaign(self.root / "mix", frames=range(1, 10), padding=4)
        write_campaign_procar(
            campaign["projection_dir"] / "12" / "PROCAR", A_IONS, range(10, 16)
        )
        discovery = discover_frames(campaign["projection_dir"])
        self.assertFalse(discovery.regular)
        self.assertIn("mixed widths", discovery.irregular_reason)
        payload = build_manifest_payload(discovery, None)
        self.assertNotIn("procar_pattern", payload)

    def test_explicit_entries_also_round_trip(self):
        campaign = build_campaign(self.root / "exp", frames=range(1, 10), padding=4)
        write_campaign_procar(
            campaign["projection_dir"] / "12" / "PROCAR", A_IONS, range(10, 16)
        )
        discovery = discover_frames(campaign["projection_dir"])
        payload = build_manifest_payload(discovery, None)
        path = self.root / "exp.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        manifest = _load_projection_manifest(path)
        self.assertEqual(len(manifest.frames), 10)
        self.assertTrue(all(p.is_file() for _, p in manifest.frames))

    def test_an_irregular_tree_falls_back_to_explicit_entries(self):
        write_campaign_procar(
            self.campaign["projection_dir"] / "21" / "PROCAR", A_IONS, range(10, 16)
        )
        discovery = discover_frames(self.campaign["projection_dir"])
        self.assertFalse(discovery.regular)
        payload = build_manifest_payload(discovery, None)
        self.assertNotIn("procar_pattern", payload)
        self.assertEqual(len(payload["frames"]), 21)
        self.assertEqual(payload["frames"][0]["frame"], 1)

    def test_an_empty_projection_directory_is_refused(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(PrepareError) as ctx:
            discover_frames(empty)
        self.assertIn("no numbered subdirectory", str(ctx.exception))


class CycleLengthTests(_Prepared):
    def _survey_and_frames(self, nsw=21):
        campaign = build_campaign(self.root / f"c{nsw}", nsw=nsw)
        return (
            survey_shprop(campaign["shprop"]),
            discover_frames(campaign["projection_dir"]),
        )

    def test_agreement_proposes_the_covered_range(self):
        survey, discovery = self._survey_and_frames(nsw=21)
        record = propose_cycle_length(survey, discovery, "dish-cyclic")
        self.assertEqual(record["coverage_derived_cycle_length"], 20)
        self.assertEqual(record["header_derived_periods_nsw_minus_1"], [20])
        self.assertEqual(record["selected"], 20)
        self.assertEqual(record["source"], FROM_DIRECTORY)
        self.assertTrue(record["agrees"])

    def test_disagreement_is_recorded_and_nothing_is_chosen(self):
        survey, discovery = self._survey_and_frames(nsw=2000)
        record = propose_cycle_length(survey, discovery, "dish-cyclic")
        self.assertEqual(record["coverage_derived_cycle_length"], 20)
        self.assertEqual(record["header_derived_periods_nsw_minus_1"], [1999])
        self.assertIsNone(record["selected"])
        self.assertEqual(record["source"], "ambiguous")
        self.assertIn("disagree", record["note"])
        self.assertIn("Nothing is chosen", record["note"])

    def test_explicit_cycle_length_overrides_the_proposal(self):
        survey, discovery = self._survey_and_frames(nsw=2000)
        record = propose_cycle_length(survey, discovery, "dish-cyclic", explicit=1999)
        self.assertEqual(record["selected"], 1999)
        self.assertEqual(record["source"], FROM_CLI)
        self.assertEqual(record["coverage_derived_cycle_length"], 20)

    def test_an_uncoverable_explicit_period_is_flagged_with_the_real_count(self):
        # The manifest cannot cover 1999 frames when 20 exist. Preflight would
        # catch it, but --run-preflight is optional, so prepare says it too.
        campaign = build_campaign(self.root / "short", nsw=2000)
        prepared = prepare_campaign(
            campaign["shprop"],
            campaign["projection_dir"],
            self.root / "short_cfg",
            "dish-cyclic",
            preset="bcf_pcbm",
            campaign="A",
            cycle_length=1999,
        )
        self.assertFalse(prepared.ok)
        message = " ".join(prepared.unresolved)
        self.assertIn("cycle_length 1999", message)
        self.assertIn("holds 20 frame(s)", message)
        self.assertIn("1979 of them have no PROCAR directory", message)

    def test_linear_mode_writes_no_period(self):
        survey, discovery = self._survey_and_frames()
        record = propose_cycle_length(survey, discovery, "linear")
        self.assertIsNone(record["selected"])
        self.assertEqual(record["source"], "not_applicable")
        payload = build_manifest_payload(discovery, record["selected"])
        self.assertNotIn("cycle_length", payload)

    def test_a_disagreeing_campaign_prepares_but_is_not_ok(self):
        campaign = build_campaign(self.root / "mismatch", nsw=2000)
        prepared = prepare_campaign(
            campaign["shprop"],
            campaign["projection_dir"],
            self.root / "mismatch_cfg",
            "dish-cyclic",
            preset="bcf_pcbm",
            campaign="A",
        )
        self.assertFalse(prepared.ok)
        self.assertTrue(any("disagree" in item for item in prepared.unresolved))
        payload = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
        self.assertNotIn("cycle_length", payload)


class SbatchTests(_Prepared):
    def _script(self, **kwargs):
        prepared = self._prepare(write_sbatch=True, **kwargs)
        return prepared, prepared.sbatch_path.read_text(encoding="utf-8")

    def test_a_script_is_generated_with_the_requested_resources(self):
        _, script = self._script(
            slurm=SlurmOptions(
                account="loni_perovsk27", partition="workq", memory="64G", time="08:00:00", cpus=4
            )
        )
        self.assertIn("#SBATCH --account=loni_perovsk27", script)
        self.assertIn("#SBATCH --partition=workq", script)
        self.assertIn("#SBATCH --mem=64G", script)
        self.assertIn("#SBATCH --time=08:00:00", script)
        self.assertIn("#SBATCH --cpus-per-task=4", script)

    def test_defaults_are_the_documented_proposal(self):
        _, script = self._script()
        self.assertIn("#SBATCH --nodes=1", script)
        self.assertIn("#SBATCH --ntasks=1", script)
        self.assertIn("#SBATCH --cpus-per-task=1", script)
        self.assertIn("#SBATCH --mem=32G", script)
        self.assertIn("#SBATCH --time=04:00:00", script)

    def test_the_script_uses_strict_bash(self):
        _, script = self._script()
        self.assertIn("set -euo pipefail", script)

    def test_the_script_wraps_the_analysis_in_usr_bin_time(self):
        _, script = self._script()
        self.assertIn("/usr/bin/time -v namd-analysis character-populations", script)

    def test_preflight_runs_before_the_full_analysis(self):
        _, script = self._script()
        pre = script.index("character-preflight")
        full = script.index("/usr/bin/time -v namd-analysis character-populations")
        self.assertLess(pre, full)
        # set -euo pipefail is what makes a non-zero preflight abort the job.
        self.assertLess(script.index("set -euo pipefail"), pre)

    def test_the_script_records_its_own_provenance(self):
        _, script = self._script()
        for marker in ("hostname", "date -Is", "pwd", "SLURM_JOB_ID",
                       "command -v namd-analysis", "__version__",
                       "'rev-parse', 'HEAD'", "not an editable checkout"):
            self.assertIn(marker, script)

    def test_the_script_lists_inputs_and_outputs(self):
        prepared, script = self._script()
        for path in self.campaign["shprop"]:
            self.assertIn(path.resolve().as_posix(), script)
        self.assertIn(prepared.state_map_path.resolve().as_posix(), script)
        self.assertIn("--- generated files ---", script)

    def test_chunk_rows_are_passed_through_to_the_script(self):
        prepared = self._prepare(write_sbatch=True, extra_run_args=["--shprop-chunk-rows 50000"])
        self.assertIn("--shprop-chunk-rows 50000", prepared.sbatch_path.read_text(encoding="utf-8"))


class ReportProvenanceTests(_Prepared):
    def test_every_inferred_field_carries_a_source(self):
        prepared = self._prepare()
        report = prepared.report
        self.assertEqual(report["population_columns"]["source"], "inferred_from_shprop_table")
        self.assertEqual(report["time_column"]["source"], "inferred_from_shprop_table")
        self.assertEqual(report["frame_mode"]["source"], FROM_CLI)
        self.assertEqual(report["manifest"]["source"], FROM_DIRECTORY)
        self.assertEqual(report["cycle_length"]["source"], FROM_DIRECTORY)
        self.assertTrue(report["state_map"]["source"].startswith("preset_"))
        self.assertTrue(report["atom_groups"]["source"].startswith("preset_"))

    def test_the_report_names_what_it_refuses_to_automate(self):
        report = self._prepare().report
        joined = " ".join(report["refusals"])
        self.assertIn("never inferred from a structure", joined)
        self.assertIn("exactly one reading fits", joined)
        self.assertIn("never inferred from filenames", joined)
        self.assertIn("never chosen when directory coverage and NSW-1 disagree", joined)

    def test_the_rationale_records_every_candidate_considered(self):
        report = self._prepare().report
        rationale = report["population_column_rationale"]
        self.assertEqual(len(rationale["candidates"]), 2)
        self.assertEqual(sum(1 for c in rationale["candidates"] if c["accepted"]), 1)
        self.assertIn("rule", rationale)

    def test_no_shprop_table_is_materialized_while_preparing(self):
        # The preparation path must not reach for the whole-file reader: that
        # is what exhausted memory on the real ~889 MB histories.
        import namd_analysis.prepare as prepare_module

        calls = []
        original = prepare_module.shprop_structure

        def spy(path):
            calls.append(Path(path).name)
            return original(path)

        prepare_module.shprop_structure = spy
        try:
            import namd_analysis.io.tables as tables

            def forbidden(*args, **kwargs):
                raise AssertionError("preparation materialized a whole SHPROP table")

            saved = tables.read_numeric_table
            tables.read_numeric_table = forbidden
            try:
                self._prepare()
            finally:
                tables.read_numeric_table = saved
        finally:
            prepare_module.shprop_structure = original
        self.assertEqual(sorted(calls), ["SHPROP.1", "SHPROP.5"])


class PrepareCliTests(_Prepared):
    def _argv(self, *extra):
        return [
            "character-prepare",
            "--shprop-dir", str(self.campaign["shprop_dir"]),
            "--projection-dir", str(self.campaign["projection_dir"]),
            "--preset", "bcf_pcbm", "--campaign", "A",
            "--frame-mode", "dish-cyclic",
            "--out", str(self.out),
            *extra,
        ]

    def test_end_to_end_produces_every_expected_file(self):
        code = dispatch_main(self._argv("--write-sbatch"))
        self.assertEqual(code, 0)
        for name in (
            "state_map.json",
            "atom_groups.json",
            "projection_manifest.json",
            "prepare_report.json",
            "run_character_test.sbatch",
        ):
            self.assertTrue((self.out / name).is_file(), name)
        report = json.loads((self.out / "prepare_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["command"], "character-prepare")
        self.assertIn("environment", report)
        self.assertTrue(report["ok"])

    def test_character_init_is_the_same_command(self):
        code = dispatch_main(["character-init", *self._argv()[1:]])
        self.assertEqual(code, 0)
        self.assertTrue((self.out / "state_map.json").is_file())

    def test_character_sbatch_implies_write_sbatch(self):
        code = dispatch_main(["character-sbatch", *self._argv()[1:]])
        self.assertEqual(code, 0)
        self.assertTrue((self.out / "run_character_test.sbatch").is_file())

    def test_run_preflight_uses_the_generated_files(self):
        code = dispatch_main(self._argv("--run-preflight"))
        self.assertEqual(code, 0)

    def test_named_files_are_honoured(self):
        code = dispatch_main(
            self._argv("--files", "SHPROP.1", "--run-preflight")
        )
        self.assertEqual(code, 0)
        report = json.loads((self.out / "prepare_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["shprop"]["n_files"], 1)

    def test_a_master_file_is_refused(self):
        (self.campaign["shprop_dir"] / "SHPROP.master").write_text("1 2\n", encoding="utf-8")
        code = dispatch_main(self._argv())
        self.assertEqual(code, 2)

    def test_frame_mode_is_required(self):
        with self.assertRaises(SystemExit):
            dispatch_main([
                "character-prepare",
                "--shprop-dir", str(self.campaign["shprop_dir"]),
                "--projection-dir", str(self.campaign["projection_dir"]),
                "--out", str(self.out),
            ])

    def test_unresolved_configuration_exits_nonzero(self):
        campaign = build_campaign(self.root / "mismatch", nsw=2000)
        code = dispatch_main([
            "character-prepare",
            "--shprop-dir", str(campaign["shprop_dir"]),
            "--projection-dir", str(campaign["projection_dir"]),
            "--preset", "bcf_pcbm", "--campaign", "A",
            "--frame-mode", "dish-cyclic",
            "--out", str(self.root / "mismatch_cfg"),
        ])
        self.assertEqual(code, 3)

    def test_the_generated_configuration_drives_a_real_analysis(self):
        self.assertEqual(dispatch_main(self._argv()), 0)
        code = dispatch_main([
            "character-populations",
            "--files", *[str(p) for p in self.campaign["shprop"]],
            "--config", str(self.out / "state_map.json"),
            "--projection-manifest", str(self.out / "projection_manifest.json"),
            "--atom-groups", str(self.out / "atom_groups.json"),
            "--frame-mode", "dish-cyclic",
            "--out", str(self.root / "results"),
        ])
        self.assertEqual(code, 0)
        self.assertTrue((self.root / "results" / "character_populations.csv").is_file())


class ExampleRegistryTests(unittest.TestCase):
    def test_the_shipped_example_matches_the_preset(self):
        root = REPO / "examples" / "bcf_pcbm" / "FAPI_001_A"
        preset = load_preset("bcf_pcbm", "A")
        state_map = json.loads((root / "state_map.json").read_text(encoding="utf-8"))
        atoms = json.loads((root / "atom_groups.json").read_text(encoding="utf-8"))
        self.assertEqual(state_map, preset.state_map_payload())
        self.assertEqual(atoms, preset.atom_groups_payload())

    def test_the_example_readme_explains_the_collapses(self):
        text = (REPO / "examples" / "bcf_pcbm" / "FAPI_001_A" / "README.md").read_text(
            encoding="utf-8"
        )
        for phrase in (
            "PCBM1",
            "perovskite / BCF / PCBM",
            "cannot separate them",
            "reference-only",
            "Do not reuse this for B or C",
        ):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
