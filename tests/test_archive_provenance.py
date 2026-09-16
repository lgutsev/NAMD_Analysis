"""Provenance for real archives: where the basis, origin and period come from.

Production SHPROP files are plain numeric tables.  ``BMIN``/``BMAX`` are
properties of the NAMD input and need not appear in one at all, and
``NAMDTINI`` often survives only in the historical ``SHPROP.<start-frame>``
filename.  Each quantity therefore has an ordered list of independent sources.

These are unittest cases on purpose.  They were pytest-style functions using
the ``tmp_path`` fixture, which ``python -m unittest discover`` -- the command
CI runs -- collects as zero tests, so the entire provenance model was unverified
in CI while CI reported green.

Required precedence:

* bands:   ``state_map.band_numbers`` > registered campaign > agreeing SHPROP BMIN/BMAX
* origin:  SHPROP ``NAMDTINI`` metadata > validated ``SHPROP.<integer>`` suffix
* period:  projection-manifest ``cycle_length`` > SHPROP ``NSW - 1``
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.character import (
    CharacterError,
    band_provenance_conflicts,
    plan_analysis,
    resolve_band_numbers,
    resolve_cycle_period,
    resolve_namdtini,
)
from namd_analysis.io.hefei import shprop_structure
from namd_analysis.populations import ConfigError, StateMap
from namd_analysis.presets import bands_for_campaign_name

A_BANDS = [976, 977, 978, 979, 980, 981]


def write_plain_shprop(path, rows=4, header_lines=(), populations=None):
    """A SHPROP table with whatever header lines are asked for -- possibly none."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if populations is None:
        populations = np.tile(np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]), (rows, 1))
    lines = list(header_lines)
    for step in range(rows):
        values = [float(step + 1), -0.8] + [float(v) for v in populations[step]]
        lines.append(" ".join(f"{v:.10E}" for v in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_manifest(tmp, cycle_length=4, frames=None):
    tmp = Path(tmp)
    if frames is None:
        frames = range(1, (cycle_length or 4) + 1)
    frames = list(frames)
    entries = []
    for frame in frames:
        procar = tmp / f"PROCAR.{frame}"
        procar.write_text("placeholder\n", encoding="utf-8")
        entries.append({"frame": frame, "procar": str(procar)})
    path = tmp / "manifest.json"
    payload = {"frames": entries}
    if cycle_length is not None:
        payload["cycle_length"] = cycle_length
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def state_map(name="FAPI_001_BCF_PCBM_A", band_numbers=None):
    payload = {
        "name": name,
        "time_column": 0,
        "time_unit": "fs",
        "population_columns": [2, 3, 4, 5, 6, 7],
        "groups": {"VBM": [2], "BCF": [3], "PCBM": [4, 5, 6], "CBM": [7]},
        "complete_population": True,
        "recombined_group": "VBM",
    }
    if band_numbers is not None:
        payload["band_numbers"] = band_numbers
    return StateMap.from_dict(payload)


class _Temp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class BandProvenanceTests(_Temp):
    """bands: state_map.band_numbers > registered campaign > agreeing BMIN/BMAX."""

    def _records(self, header_lines=(), names=("SHPROP.37",)):
        return [
            shprop_structure(write_plain_shprop(self.root / name, header_lines=header_lines))
            for name in names
        ]

    def test_explicit_band_numbers_win(self):
        records = self._records(("# BMIN = 10", "# BMAX = 15"))
        bands, source = resolve_band_numbers(state_map(band_numbers=A_BANDS), records)
        self.assertEqual(bands, A_BANDS)
        self.assertEqual(source, "state_map.band_numbers")

    def test_registered_campaign_used_when_no_explicit_bands(self):
        bands, source = resolve_band_numbers(state_map(), self._records())
        self.assertEqual(bands, A_BANDS)
        self.assertEqual(source, "preset_bcf_pcbm_A")

    def test_explicit_bands_override_the_registered_campaign(self):
        other = [100, 101, 102, 103, 104, 105]
        bands, source = resolve_band_numbers(state_map(band_numbers=other), self._records())
        self.assertEqual(bands, other)
        self.assertEqual(source, "state_map.band_numbers")

    def test_agreeing_bmin_bmax_used_when_nothing_higher_exists(self):
        records = self._records(("# BMIN = 10", "# BMAX = 15"))
        bands, source = resolve_band_numbers(state_map(name="unregistered"), records)
        self.assertEqual(bands, [10, 11, 12, 13, 14, 15])
        self.assertEqual(source, "SHPROP_BMIN_BMAX_metadata")

    def test_no_source_at_all_is_refused_not_guessed(self):
        with self.assertRaises(CharacterError) as ctx:
            resolve_band_numbers(state_map(name="unregistered"), self._records())
        message = str(ctx.exception)
        self.assertIn("not recoverable", message)
        self.assertIn("band_numbers", message)
        self.assertIn("registered campaign preset", message)

    def test_disagreeing_bmin_bmax_across_files_is_refused(self):
        records = [
            shprop_structure(
                write_plain_shprop(self.root / "SHPROP.1", header_lines=("# BMIN = 10", "# BMAX = 15"))
            ),
            shprop_structure(
                write_plain_shprop(self.root / "SHPROP.2", header_lines=("# BMIN = 20", "# BMAX = 25"))
            ),
        ]
        with self.assertRaises(CharacterError) as ctx:
            resolve_band_numbers(state_map(name="unregistered"), records)
        self.assertIn("different optional BMIN/BMAX windows", str(ctx.exception))

    def test_bmin_bmax_on_only_some_files_is_refused(self):
        records = [
            shprop_structure(
                write_plain_shprop(self.root / "SHPROP.1", header_lines=("# BMIN = 10", "# BMAX = 15"))
            ),
            shprop_structure(write_plain_shprop(self.root / "SHPROP.2")),
        ]
        with self.assertRaises(CharacterError) as ctx:
            resolve_band_numbers(state_map(name="unregistered"), records)
        self.assertIn("carry no BMIN/BMAX metadata while others do", str(ctx.exception))

    def test_inverted_window_is_refused(self):
        records = self._records(("# BMIN = 15", "# BMAX = 10"))
        with self.assertRaises(CharacterError) as ctx:
            resolve_band_numbers(state_map(name="unregistered"), records)
        self.assertIn("BMAX=10 < BMIN=15", str(ctx.exception))

    def test_wrong_length_band_numbers_are_refused(self):
        with self.assertRaises(ConfigError):
            state_map(band_numbers=[976, 977])

    def test_duplicate_band_numbers_are_refused(self):
        with self.assertRaises(ConfigError):
            state_map(band_numbers=[976, 976, 978, 979, 980, 981])

    def test_non_positive_band_numbers_are_refused(self):
        for bad in ([0, 977, 978, 979, 980, 981], [-1, 977, 978, 979, 980, 981]):
            with self.assertRaises(ConfigError):
                state_map(band_numbers=bad)

    def test_non_integer_band_numbers_are_refused(self):
        with self.assertRaises(ConfigError):
            state_map(band_numbers=["976", 977, 978, 979, 980, 981])

    def test_campaign_b_and_c_do_not_inherit_a_bands(self):
        for name in ("FAPI_001_BCF_PCBM_B", "FAPI_001_BCF_PCBM_C"):
            self.assertIsNone(bands_for_campaign_name(name))


class BandConflictReportingTests(_Temp):
    """A higher-precedence basis must still be compared with what the files say.

    Bands are the one quantity nothing downstream can detect as wrong: a
    plausible population comes out either way. So precedence decides which
    value is used, but a disagreement is always reported.
    """

    def _records(self, header_lines):
        return [
            shprop_structure(
                write_plain_shprop(self.root / "SHPROP.37", header_lines=header_lines)
            )
        ]

    def test_a_disagreement_with_the_headers_is_reported(self):
        records = self._records(("# BMIN = 500", "# BMAX = 505"))
        conflicts = band_provenance_conflicts(A_BANDS, "state_map.band_numbers", records)
        self.assertEqual(len(conflicts), 1)
        self.assertIn("976..981", conflicts[0])
        self.assertIn("500:505", conflicts[0])
        self.assertIn("state_map.band_numbers", conflicts[0])

    def test_agreement_reports_nothing(self):
        records = self._records(("# BMIN = 976", "# BMAX = 981"))
        self.assertEqual(
            band_provenance_conflicts(A_BANDS, "state_map.band_numbers", records), []
        )

    def test_headerless_files_report_nothing(self):
        self.assertEqual(
            band_provenance_conflicts(A_BANDS, "preset_bcf_pcbm_A", self._records(())), []
        )

    def test_the_headers_cannot_conflict_with_themselves(self):
        records = self._records(("# BMIN = 10", "# BMAX = 15"))
        self.assertEqual(
            band_provenance_conflicts(
                [10, 11, 12, 13, 14, 15], "SHPROP_BMIN_BMAX_metadata", records
            ),
            [],
        )

    def test_the_plan_carries_the_conflict_and_still_uses_precedence(self):
        paths = [
            write_plain_shprop(
                self.root / "SHPROP.37", header_lines=("# BMIN = 500", "# BMAX = 505")
            )
        ]
        plan = plan_analysis(
            paths, state_map(band_numbers=A_BANDS), write_manifest(self.root, 4), "dish-cyclic"
        )
        self.assertEqual(plan.bands, A_BANDS)
        self.assertEqual(plan.band_numbers_source, "state_map.band_numbers")
        self.assertEqual(len(plan.band_provenance_conflicts), 1)
        self.assertIn("500:505", plan.band_provenance_conflicts[0])

    def test_a_preset_basis_conflicting_with_headers_is_reported(self):
        paths = [
            write_plain_shprop(
                self.root / "SHPROP.37", header_lines=("# BMIN = 10", "# BMAX = 15")
            )
        ]
        plan = plan_analysis(
            paths, state_map(), write_manifest(self.root, 4), "dish-cyclic"
        )
        self.assertEqual(plan.bands, A_BANDS)
        self.assertEqual(plan.band_numbers_source, "preset_bcf_pcbm_A")
        self.assertTrue(plan.band_provenance_conflicts)
        self.assertIn("10:15", plan.band_provenance_conflicts[0])


class MalformedMetadataTests(_Temp):
    """A key that is present must be readable, not quietly treated as absent.

    Demoting an unreadable value to the next provenance source means a file
    that states something wrong is handled as though it stated nothing.
    """

    def _record(self, name, header_lines):
        return shprop_structure(
            write_plain_shprop(self.root / name, header_lines=header_lines)
        )

    def test_unreadable_namdtini_is_refused_not_demoted_to_the_filename(self):
        for bad in ("37.5", "abc", "", "1,5"):
            with self.assertRaises(CharacterError) as ctx:
                resolve_namdtini(self._record("SHPROP.37", (f"# NAMDTINI = {bad}",)))
            message = str(ctx.exception)
            self.assertIn("not an integer", message)
            self.assertIn("silently fall through", message)

    def test_unreadable_nsw_is_refused(self):
        from namd_analysis.character import _load_projection_manifest

        manifest = _load_projection_manifest(write_manifest(self.root, cycle_length=None))
        with self.assertRaises(CharacterError) as ctx:
            resolve_cycle_period(
                self._record("SHPROP.1", ("# NSW = later",)), manifest, "dish-cyclic"
            )
        self.assertIn("not an integer", str(ctx.exception))

    def test_unreadable_bmin_is_refused(self):
        with self.assertRaises(CharacterError) as ctx:
            resolve_band_numbers(
                state_map(name="unregistered"),
                [self._record("SHPROP.1", ("# BMIN = ten", "# BMAX = 15"))],
            )
        self.assertIn("not an integer", str(ctx.exception))

    def test_an_absent_key_still_falls_through_normally(self):
        value, source = resolve_namdtini(self._record("SHPROP.37", ()))
        self.assertEqual((value, source), (37, "SHPROP_filename_suffix"))


class NamdtiniProvenanceTests(_Temp):
    """origin: SHPROP NAMDTINI metadata > validated SHPROP.<integer> suffix."""

    def _record(self, name, header_lines=()):
        return shprop_structure(write_plain_shprop(self.root / name, header_lines=header_lines))

    def test_filename_suffix_alone(self):
        value, source = resolve_namdtini(self._record("SHPROP.37"))
        self.assertEqual(value, 37)
        self.assertEqual(source, "SHPROP_filename_suffix")

    def test_every_real_test_file_name_resolves(self):
        for start in (37, 171, 425, 848, 1625):
            value, source = resolve_namdtini(self._record(f"SHPROP.{start}"))
            self.assertEqual(value, start)
            self.assertEqual(source, "SHPROP_filename_suffix")

    def test_metadata_alone(self):
        value, source = resolve_namdtini(
            self._record("history.txt", header_lines=("# NAMDTINI = 42",))
        )
        self.assertEqual(value, 42)
        self.assertEqual(source, "SHPROP_NAMDTINI_metadata")

    def test_metadata_wins_over_a_matching_suffix_and_is_labelled_as_such(self):
        # The label must say metadata, not filename: a report that misstates
        # which source was used is worse than one that omits it.
        value, source = resolve_namdtini(
            self._record("SHPROP.37", header_lines=("# NAMDTINI = 37",))
        )
        self.assertEqual(value, 37)
        self.assertEqual(source, "SHPROP_NAMDTINI_metadata")

    def test_suffix_and_metadata_disagreement_is_refused(self):
        with self.assertRaises(CharacterError) as ctx:
            resolve_namdtini(self._record("SHPROP.37", header_lines=("# NAMDTINI = 99",)))
        message = str(ctx.exception)
        self.assertIn("NAMDTINI=99", message)
        self.assertIn("filename suffix says 37", message)
        self.assertIn("refused", message)

    def test_malformed_suffix_with_no_metadata_is_refused(self):
        for name in ("SHPROP.abc", "SHPROP", "SHPROP.", "SHPROP.12x", "shprop_37"):
            with self.assertRaises(CharacterError) as ctx:
                resolve_namdtini(self._record(name))
            self.assertIn("never guessed", str(ctx.exception))

    def test_zero_suffix_is_refused(self):
        with self.assertRaises(CharacterError) as ctx:
            resolve_namdtini(self._record("SHPROP.0"))
        self.assertIn("one-based", str(ctx.exception))

    def test_non_positive_metadata_is_refused(self):
        with self.assertRaises(CharacterError) as ctx:
            resolve_namdtini(self._record("history.txt", header_lines=("# NAMDTINI = 0",)))
        self.assertIn("one-based", str(ctx.exception))

    def test_case_insensitive_suffix_is_accepted(self):
        value, source = resolve_namdtini(self._record("shprop.55"))
        self.assertEqual(value, 55)
        self.assertEqual(source, "SHPROP_filename_suffix")


class CyclePeriodProvenanceTests(_Temp):
    """period: projection-manifest cycle_length > SHPROP NSW-1."""

    def _record(self, header_lines=()):
        return shprop_structure(
            write_plain_shprop(self.root / "SHPROP.1", header_lines=header_lines)
        )

    def _manifest(self, cycle_length):
        from namd_analysis.character import _load_projection_manifest

        return _load_projection_manifest(write_manifest(self.root, cycle_length=cycle_length))

    def test_manifest_cycle_length_wins(self):
        period, source, header = resolve_cycle_period(
            self._record(("# NSW = 2000",)), self._manifest(4), "dish-cyclic"
        )
        self.assertEqual(period, 4)
        self.assertEqual(source, "projection_manifest")
        self.assertEqual(header, 1999)

    def test_nsw_minus_one_used_when_the_manifest_is_silent(self):
        period, source, header = resolve_cycle_period(
            self._record(("# NSW = 21",)), self._manifest(None), "dish-cyclic"
        )
        self.assertEqual(period, 20)
        self.assertEqual(source, "SHPROP_NSW_minus_1_metadata")
        self.assertEqual(header, 20)

    def test_neither_source_is_refused(self):
        with self.assertRaises(CharacterError) as ctx:
            resolve_cycle_period(self._record(), self._manifest(None), "dish-cyclic")
        message = str(ctx.exception)
        self.assertIn("never inferred from how many frames happen to exist", message)

    def test_linear_mode_needs_no_period(self):
        period, source, header = resolve_cycle_period(
            self._record(), self._manifest(None), "linear"
        )
        self.assertIsNone(period)
        self.assertEqual(source, "not_applicable")


class HeaderlessEndToEndTests(_Temp):
    """A completely headerless archive must plan correctly, or refuse clearly."""

    def _plan(self, names, header_lines=(), name="FAPI_001_BCF_PCBM_A", bands=None):
        paths = [
            write_plain_shprop(self.root / n, header_lines=header_lines) for n in names
        ]
        manifest = write_manifest(self.root, cycle_length=4)
        return plan_analysis(paths, state_map(name=name, band_numbers=bands), manifest, "dish-cyclic")

    def test_headerless_files_plan_from_filename_and_preset(self):
        plan = self._plan(["SHPROP.37", "SHPROP.171"])
        self.assertEqual(plan.bands, A_BANDS)
        self.assertEqual([r["NAMDTINI"] for r in plan.alignments], [37, 171])
        self.assertEqual(
            [r["NAMDTINI_source"] for r in plan.alignments],
            ["SHPROP_filename_suffix", "SHPROP_filename_suffix"],
        )
        self.assertEqual(
            [r["band_numbers_source"] for r in plan.alignments],
            ["preset_bcf_pcbm_A", "preset_bcf_pcbm_A"],
        )
        self.assertEqual(
            [r["cycle_length_source"] for r in plan.alignments],
            ["projection_manifest", "projection_manifest"],
        )

    def test_headerless_files_plan_from_filename_and_explicit_bands(self):
        plan = self._plan(["SHPROP.37"], name="unregistered", bands=A_BANDS)
        self.assertEqual(plan.bands, A_BANDS)
        self.assertEqual(plan.alignments[0]["band_numbers_source"], "state_map.band_numbers")

    def test_headerless_files_without_basis_provenance_are_refused(self):
        with self.assertRaises(CharacterError) as ctx:
            self._plan(["SHPROP.37"], name="unregistered")
        self.assertIn("not recoverable", str(ctx.exception))

    def test_headerless_file_with_an_unusable_name_is_refused(self):
        with self.assertRaises(CharacterError) as ctx:
            self._plan(["history.dat"])
        self.assertIn("never guessed", str(ctx.exception))

    def test_frames_match_the_documented_cyclic_mapping(self):
        plan = self._plan(["SHPROP.37"])
        # period 4, NAMDTINI 37, four rows: mod(t + 37 - 1, 4) with 0 -> 4.
        expected = [((step + 36) % 4) or 4 for step in range(1, 5)]
        self.assertEqual([int(f) for f in plan.frames_by_file[0]], expected)

    def test_header_and_filename_agreement_is_accepted(self):
        plan = self._plan(["SHPROP.37"], header_lines=("# NAMDTINI = 37",))
        self.assertEqual(plan.alignments[0]["NAMDTINI"], 37)
        self.assertEqual(
            plan.alignments[0]["NAMDTINI_source"], "SHPROP_NAMDTINI_metadata"
        )

    def test_header_and_filename_disagreement_is_refused(self):
        with self.assertRaises(CharacterError) as ctx:
            self._plan(["SHPROP.37"], header_lines=("# NAMDTINI = 99",))
        self.assertIn("disagree", str(ctx.exception))


class InstalledShimTests(unittest.TestCase):
    """The old install() hook must remain importable and harmless."""

    def test_install_is_a_noop_and_resolvers_are_re_exported(self):
        from namd_analysis import archive_provenance, character

        archive_provenance.install()
        self.assertIs(archive_provenance.resolve_namdtini, character.resolve_namdtini)
        self.assertIs(
            archive_provenance.resolve_band_numbers, character.resolve_band_numbers
        )
        self.assertIs(character.plan_analysis.__module__, character.__name__)

    def test_the_provenance_model_does_not_depend_on_import_order(self):
        # It used to: install() replaced character.preflight_report, but
        # character_cli binds that name at its own import time, so importing
        # character_cli first left the CLI with the unpatched function.
        import subprocess
        import sys

        for order in (
            "import namd_analysis.character_cli, namd_analysis.dispatch",
            "import namd_analysis.dispatch, namd_analysis.character_cli",
            "import namd_analysis.character",
        ):
            code = (
                f"{order}\n"
                "from namd_analysis import character\n"
                "assert character.plan_analysis.__module__ == 'namd_analysis.character'\n"
                "assert hasattr(character, 'resolve_band_numbers')\n"
                "print('ok')\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
