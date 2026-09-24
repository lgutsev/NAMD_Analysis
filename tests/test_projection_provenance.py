"""What the PROCAR weights are (LORBIT) and how precisely they were printed."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from character_fixtures import write_procar

from namd_analysis.io.procar import read_procar_ion_totals
from namd_analysis.io.vasp_settings import (
    VaspSettingsError,
    outcar_settings,
    parse_incar_text,
    read_incar,
    vasprun_lorbit,
)
from namd_analysis.projection_provenance import (
    CONFLICTING,
    PAW_PROJECTORS,
    RWIGS_SPHERES,
    UNDETERMINED,
    procar_resolution_check,
    projection_method,
)

OUTCAR_SNIPPET = """ POTCAR:    PAW_PBE Pb_d 06Sep2000
   RWIGS  =    1.566;   RCUT  =    2.900  wigner-seitz radius (au A)
 Dimension of arrays:
   k-points           NKPTS =      1   k-points in BZ     NKDIM =      1
   LORBIT =     11    0 simple, 1 ext, 2 COOP (PROOUT), +10 PAW based schemes
   RWIGS  =  -1.00 -1.00 -1.00  wigner-seitz radius for each atom type
"""


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class ResolutionTests(TmpCase):
    def test_thin_band_loses_weight_to_three_decimal_printing(self):
        thin = [0.0004] * 100  # every ion prints as 0.000
        compact = [0.05] * 10 + [0.0] * 90
        path = write_procar(self.root / "PROCAR", {1: thin, 2: compact},
                            orbitals="spd", decimals=3)
        groups = {"a": list(range(1, 51)), "b": list(range(51, 101))}
        check = procar_resolution_check(path, [1, 2], groups, frame=7)
        self.assertEqual(check["per_ion_decimals"], 3)
        self.assertAlmostEqual(check["print_half_step"], 0.0005)
        thin_band, compact_band = check["bands"]
        self.assertEqual(thin_band["scopes"]["all_ions"]["sum_of_printed_values"], 0.0)
        self.assertEqual(thin_band["scopes"]["all_ions"]["n_ions_printed_zero"], 100)
        self.assertAlmostEqual(thin_band["printed_band_total"], 0.040)
        self.assertAlmostEqual(thin_band["printed_total_minus_sum"], 0.040)
        self.assertIn("summed before rounding", thin_band["reading"])
        # The true sum lies inside the rounding bounds.
        self.assertLessEqual(thin_band["scopes"]["all_ions"]["rounding_lower_bound"], 0.04)
        self.assertGreaterEqual(thin_band["scopes"]["all_ions"]["rounding_upper_bound"], 0.04)
        self.assertAlmostEqual(compact_band["printed_total_minus_sum"], 0.0)
        self.assertIn("cannot", compact_band["reading"])
        self.assertEqual(len(check["rows"]), 2 * 3)

    def test_reader_values_are_unchanged_by_the_new_fields(self):
        values = list(np.linspace(0.01, 0.06, 6))
        path = write_procar(self.root / "PROCAR", {5: values}, orbitals="s")
        projection = read_procar_ion_totals(path, bands=[5])
        np.testing.assert_allclose(projection.ion_totals[0], values, atol=1e-6)
        self.assertAlmostEqual(projection.tot_rows[5], sum(values), places=5)
        self.assertEqual(projection.tot_decimals, 6)

    def test_missing_tot_row_is_reported_not_invented(self):
        path = write_procar(self.root / "PROCAR", {1: [0.1, 0.2]}, drop_tot_row=True)
        check = procar_resolution_check(path, [1], {"a": [1], "b": [2]})
        self.assertIsNone(check["bands"][0]["printed_band_total"])
        self.assertIn("no tot row", check["bands"][0]["reading"])

    def test_scientific_notation_has_no_print_step(self):
        path = write_procar(self.root / "PROCAR", {1: [0.1, 0.2]}, scientific=True)
        self.assertIsNone(read_procar_ion_totals(path).tot_decimals)


class IncarTests(TmpCase):
    def test_comments_and_semicolons(self):
        pairs = parse_incar_text("ENCUT = 400 ! cutoff\n# a comment\nISMEAR = 0; sigma = 0.01\n\n")
        self.assertEqual(pairs, [("ENCUT", "400"), ("ISMEAR", "0"), ("SIGMA", "0.01")])

    def test_malformed_line_is_an_error(self):
        with self.assertRaises(VaspSettingsError):
            parse_incar_text("ENCUT 400\n")

    def test_conflicting_duplicate_is_an_error(self):
        path = self.root / "INCAR"
        path.write_text("LORBIT = 11\nLORBIT = 10\n")
        with self.assertRaises(VaspSettingsError):
            read_incar(path)

    def test_outcar_reads_run_rwigs_not_potcar_default(self):
        path = self.root / "OUTCAR"
        path.write_text(OUTCAR_SNIPPET)
        settings = outcar_settings(path)
        self.assertEqual(settings["LORBIT"], 11)
        self.assertEqual(settings["RWIGS"], [-1.0, -1.0, -1.0])

    def test_vasprun(self):
        path = self.root / "vasprun.xml"
        path.write_text('<modeling>\n <i type="int" name="LORBIT">    10</i>\n</modeling>\n')
        self.assertEqual(vasprun_lorbit(path), 10)


class MethodTests(TmpCase):
    def _procar(self):
        return write_procar(self.root / "PROCAR", {1: [0.1, 0.2]}, orbitals="spd")

    def test_paw_projectors_make_an_rwigs_sweep_a_no_op(self):
        incar = self.root / "INCAR"
        incar.write_text("LORBIT = 11\n")
        record = projection_method(procar=self._procar(), incar=incar)
        self.assertEqual(record["method"], PAW_PROJECTORS)
        self.assertFalse(record["rwigs_sweep_tests_these_weights"])
        self.assertTrue(record["procar_layout_check"]["consistent"])
        self.assertIn("ignores RWIGS", record["meaning"])

    def test_sphere_integration(self):
        incar = self.root / "INCAR"
        incar.write_text("LORBIT = 1\nRWIGS = 1.6 0.8 0.4\n")
        record = projection_method(procar=self._procar(), incar=incar)
        self.assertEqual(record["method"], RWIGS_SPHERES)
        self.assertTrue(record["rwigs_sweep_tests_these_weights"])
        self.assertEqual(record["RWIGS_by_source"]["INCAR"], [1.6, 0.8, 0.4])

    def test_procar_alone_is_undetermined(self):
        record = projection_method(procar=self._procar())
        self.assertEqual(record["method"], UNDETERMINED)
        self.assertIsNone(record["rwigs_sweep_tests_these_weights"])
        self.assertEqual(record["procar_layout"]["orbital_layout"], "lm-decomposed")

    def test_disagreeing_sources_are_conflicting(self):
        incar = self.root / "INCAR"
        incar.write_text("LORBIT = 10\n")
        outcar = self.root / "OUTCAR"
        outcar.write_text(OUTCAR_SNIPPET)
        record = projection_method(incar=incar, outcar=outcar)
        self.assertEqual(record["method"], CONFLICTING)
        self.assertIsNone(record["LORBIT"])

    def test_layout_contradiction_is_flagged(self):
        incar = self.root / "INCAR"
        incar.write_text("LORBIT = 10\n")
        record = projection_method(procar=self._procar(), incar=incar)
        self.assertFalse(record["procar_layout_check"]["consistent"])


if __name__ == "__main__":
    unittest.main()
