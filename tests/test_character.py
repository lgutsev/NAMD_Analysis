import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.character import (
    AtomGroupMap,
    CharacterError,
    character_populations,
    character_swap_rows,
    load_projection_series,
)
from namd_analysis.io.hefei import read_shprop_with_metadata
from namd_analysis.io.procar import ProcarFormatError, read_procar_ion_totals
from namd_analysis.populations import StateMap


def _procar(path, band10, band11, nkpoints=1):
    def table(band, values):
        return (
            f" band {band} # energy 0.0 # occ. 0.0\n"
            " ion      s      tot\n"
            f"   1   {values[0]:.6f}   {values[0]:.6f}\n"
            f"   2   {values[1]:.6f}   {values[1]:.6f}\n"
            f" tot   {sum(values):.6f}   {sum(values):.6f}\n"
        )

    text = (
        "PROCAR lm decomposed\n"
        f"# of k-points: {nkpoints} # of bands: 2 # of ions: 2\n"
        " k-point 1 : 0 0 0 weight = 1.0\n"
        + table(10, band10)
        + table(11, band11)
    )
    path.write_text(text)


def _shprop(path, start, rows, nsw=4):
    header = (
        "# BMIN = 10\n"
        "# BMAX = 11\n"
        f"# NSW = {nsw}\n"
        f"# NAMDTINI = {start}\n"
        "# POTIM = 1.0\n"
    )
    body = "\n".join(" ".join(str(x) for x in row) for row in rows) + "\n"
    path.write_text(header + body)


class ProcarTests(unittest.TestCase):
    def test_reads_per_ion_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "PROCAR"
            _procar(path, (0.8, 0.2), (0.1, 0.9))
            result = read_procar_ion_totals(path)
            self.assertEqual(result.nions, 2)
            np.testing.assert_array_equal(result.bands, [10, 11])
            np.testing.assert_allclose(result.ion_totals, [[0.8, 0.2], [0.1, 0.9]])

    def test_rejects_multiple_kpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "PROCAR"
            _procar(path, (0.8, 0.2), (0.1, 0.9), nkpoints=2)
            with self.assertRaises(ProcarFormatError):
                read_procar_ion_totals(path)


class MetadataTests(unittest.TestCase):
    def test_shprop_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "SHPROP.1"
            _shprop(path, 2, [[1, 0, 1, 0], [2, 0, 0.5, 0.5]])
            record = read_shprop_with_metadata(path)
            self.assertEqual(record.metadata["NAMDTINI"], 2)
            self.assertEqual(record.metadata["BMIN"], 10)
            self.assertEqual(record.table.shape, (2, 4))


class CharacterPopulationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.procars = self.root / "procars"
        self.procars.mkdir()
        _procar(self.procars / "p1", (1.0, 0.0), (0.0, 1.0))
        _procar(self.procars / "p2", (0.0, 1.0), (1.0, 0.0))
        _procar(self.procars / "p3", (1.0, 0.0), (0.0, 1.0))
        manifest = {
            "frames": [
                {"frame": 1, "procar": "procars/p1"},
                {"frame": 2, "procar": "procars/p2"},
                {"frame": 3, "procar": "procars/p3"},
            ]
        }
        self.manifest = self.root / "projection_manifest.json"
        self.manifest.write_text(json.dumps(manifest))
        self.atom_groups_path = self.root / "atoms.json"
        self.atom_groups_path.write_text(
            json.dumps(
                {
                    "groups": {"A": [1], "B": [2]},
                    "complete_atoms": True,
                    "min_projection_weight": 0.5,
                }
            )
        )
        self.map = StateMap.from_dict(
            {
                "name": "two-band",
                "time_column": 0,
                "time_unit": "fs",
                "population_columns": [2, 3],
                "groups": {"fixed_A": [2], "fixed_B": [3]},
                "complete_population": True,
            }
        )
        self.s1 = self.root / "SHPROP.1"
        self.s2 = self.root / "SHPROP.2"
        rows = [[1, 0, 1, 0], [2, 0, 1, 0]]
        _shprop(self.s1, 1, rows)
        _shprop(self.s2, 2, rows)

    def tearDown(self):
        self.tmp.cleanup()

    def test_projection_series_normalizes_subsystem_character(self):
        groups = AtomGroupMap.from_json(self.atom_groups_path)
        projection = load_projection_series(self.manifest, groups, [10, 11])
        np.testing.assert_allclose(projection.weights.sum(axis=2), 1.0)
        self.assertEqual(projection.quality_summary()["samples_below_threshold"], 0)

    def test_character_is_applied_per_file_before_averaging(self):
        groups = AtomGroupMap.from_json(self.atom_groups_path)
        result = character_populations(
            [self.s1, self.s2], self.map, self.manifest, groups, "dish-cyclic"
        )
        # Both SHPROP files place all population in adiabatic band 10, but their
        # NAMDTINI values select different physical-character phases.
        np.testing.assert_allclose(result.per_file[0, :, 0], [1.0, 0.0])
        np.testing.assert_allclose(result.per_file[1, :, 0], [0.0, 1.0])
        np.testing.assert_allclose(result.mean[:, 0], [0.5, 0.5])
        np.testing.assert_allclose(result.mean.sum(axis=1), 1.0)
        self.assertEqual(result.file_alignment[0]["first_projection_frame"], 1)
        self.assertEqual(result.file_alignment[1]["first_projection_frame"], 2)

    def test_manifest_cycle_length_overrides_inconsistent_header_period(self):
        manifest = self.root / "explicit_cycle.json"
        manifest.write_text(
            json.dumps(
                {
                    "frames": [
                        {"frame": 1, "procar": "procars/p1"},
                        {"frame": 2, "procar": "procars/p2"},
                        {"frame": 3, "procar": "procars/p3"},
                    ],
                    "cycle_length": 3,
                }
            )
        )
        odd = self.root / "SHPROP.odd"
        # Header-derived period would be NSW-1 = 4, but the actual electronic
        # projection cycle is explicitly declared as 3.
        _shprop(odd, 3, [[1, 0, 1, 0], [2, 0, 1, 0]], nsw=5)
        groups = AtomGroupMap.from_json(self.atom_groups_path)
        result = character_populations([odd], self.map, manifest, groups, "dish-cyclic")
        np.testing.assert_allclose(result.per_file[0, :, 0], [1.0, 1.0])
        alignment = result.file_alignment[0]
        self.assertEqual(alignment["header_cycle_length"], 4)
        self.assertEqual(alignment["cycle_length_used"], 3)
        self.assertEqual(alignment["cycle_length_source"], "projection_manifest")
        self.assertTrue(alignment["header_cycle_mismatch"])

    def test_character_swaps_are_reported(self):
        groups = AtomGroupMap.from_json(self.atom_groups_path)
        projection = load_projection_series(self.manifest, groups, [10, 11])
        rows, summary = character_swap_rows(projection, dominance_threshold=0.6)
        self.assertEqual(summary["dominant_character_swaps"], 4)
        self.assertEqual(summary["mixed_frame_band_samples"], 0)
        self.assertEqual(len(rows), 4)

    def test_missing_projection_frame_is_fatal(self):
        manifest = self.root / "short.json"
        manifest.write_text(
            json.dumps(
                {
                    "frames": [
                        {"frame": 1, "procar": "procars/p1"},
                        {"frame": 2, "procar": "procars/p2"},
                    ]
                }
            )
        )
        groups = AtomGroupMap.from_json(self.atom_groups_path)
        with self.assertRaises(CharacterError):
            character_populations([self.s2], self.map, manifest, groups, "dish-cyclic")

    def test_incomplete_atom_map_rejected_when_declared_complete(self):
        bad = self.root / "bad_atoms.json"
        bad.write_text(json.dumps({"groups": {"A": [1]}, "complete_atoms": True}))
        groups = AtomGroupMap.from_json(bad)
        with self.assertRaises(CharacterError):
            load_projection_series(self.manifest, groups, [10, 11])


if __name__ == "__main__":
    unittest.main()
