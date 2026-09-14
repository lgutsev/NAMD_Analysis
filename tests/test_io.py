import bootstrap  # noqa: F401  (puts src on sys.path)

import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.io.hefei import read_legacy_fit, read_natxt, load_run_directory
from namd_analysis.io.namelist import basis_size, read_namelist
from namd_analysis.io.tables import (
    TableFormatError,
    read_numeric_table,
    read_xy_with_header,
)
from namd_analysis.io.xdatcar import XdatcarFormatError, read_xdatcar
from synthetic import write_legacy_fit, write_run_directory, write_xdatcar


class TableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fortran_d_exponents_and_comments(self):
        path = self.root / "table.txt"
        path.write_text(
            "# a comment\n1.0D-3 2.5d+02\n\n3.0 4.0  ! trailing\n", encoding="utf-8"
        )
        table = read_numeric_table(path)
        np.testing.assert_allclose(table, [[1e-3, 250.0], [3.0, 4.0]])

    def test_ragged_rows_are_an_error(self):
        path = self.root / "ragged.txt"
        path.write_text("1.0 2.0\n3.0\n", encoding="utf-8")
        with self.assertRaises(TableFormatError) as ctx:
            read_numeric_table(path)
        self.assertIn("expected 2", str(ctx.exception))

    def test_non_numeric_field_is_an_error(self):
        path = self.root / "complex.txt"
        path.write_text("(1.0,2.0) (3.0,4.0)\n", encoding="utf-8")
        with self.assertRaises(TableFormatError):
            read_numeric_table(path)

    def test_non_finite_is_an_error(self):
        path = self.root / "nan.txt"
        path.write_text("1.0 nan\n", encoding="utf-8")
        with self.assertRaises(TableFormatError):
            read_numeric_table(path)

    def test_expected_column_count_is_enforced(self):
        path = self.root / "two.txt"
        path.write_text("1.0 2.0\n", encoding="utf-8")
        with self.assertRaises(TableFormatError):
            read_numeric_table(path, expect_columns=3)

    def test_header_line_is_separated_from_data(self):
        path = self.root / "spectrum.txt"
        path.write_text("Frequency\tDensity\n1.0 2.0\n2.0 3.0\n", encoding="utf-8")
        table, header = read_xy_with_header(path)
        self.assertEqual(header, "Frequency\tDensity")
        np.testing.assert_allclose(table, [[1.0, 2.0], [2.0, 3.0]])

    def test_headerless_two_column_file_is_accepted(self):
        path = self.root / "bare.txt"
        path.write_text("1.0 2.0\n2.0 3.0\n", encoding="utf-8")
        table, header = read_xy_with_header(path)
        self.assertIsNone(header)
        self.assertEqual(table.shape, (2, 2))


class NamelistTests(unittest.TestCase):
    def test_types_and_basis_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inp"
            path.write_text(
                "&NAMDPARA\n  BMIN = 976\n  BMAX = 981\n  POTIM = 1.0\n"
                '  ALGO = "DISH"\n  LHOLE = .FALSE.\n  LSHP = .TRUE.\n/\n',
                encoding="utf-8",
            )
            params = read_namelist(path)
        self.assertEqual(params["BMIN"], 976)
        self.assertEqual(params["ALGO"], "DISH")
        self.assertIs(params["LHOLE"], False)
        self.assertIs(params["LSHP"], True)
        self.assertEqual(params["POTIM"], 1.0)
        self.assertEqual(basis_size(params), 6)

    def test_basis_size_is_none_without_band_window(self):
        self.assertIsNone(basis_size({"POTIM": 1.0}))


class RunDirectoryTests(unittest.TestCase):
    def test_presence_params_and_legacy_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_run_directory(Path(tmp) / "run", nstates=3, bmin=100)
            write_legacy_fit(root, tau_ns=0.0001, r_squared=-17.4289)
            run = load_run_directory(root)
        self.assertIn("EIGTXT", run.present)
        self.assertEqual(run.declared_nstates, 3)
        self.assertEqual(run.dt_fs_from_inp, 1.0)
        self.assertAlmostEqual(run.legacy_fit["r_squared"], -17.4289)

    def test_legacy_fit_parser_handles_the_unicode_r_squared(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_legacy_fit(Path(tmp), tau_ns=0.6722, r_squared=0.9035)
            parsed = read_legacy_fit(path)
        self.assertAlmostEqual(parsed["tau_ns"], 0.6722)
        self.assertAlmostEqual(parsed["r_squared"], 0.9035)

    def test_non_square_nac_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "NATXT"
            np.savetxt(path, np.zeros((5, 6)))
            with self.assertRaises(Exception) as ctx:
                read_natxt(path)
        self.assertIn("perfect square", str(ctx.exception))

    def test_nac_state_count_must_match_eigtxt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "NATXT"
            np.savetxt(path, np.zeros((5, 9)))
            with self.assertRaises(TableFormatError) as ctx:
                read_natxt(path, nstates=4)
        self.assertIn("EIGTXT has 4", str(ctx.exception))


class XdatcarTests(unittest.TestCase):
    def test_reads_cell_species_and_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_xdatcar(
                Path(tmp) / "XDATCAR", nframes=50, species=("Pb", "I"), counts=(1, 3)
            )
            traj = read_xdatcar(path)
        self.assertEqual(traj.nframes, 50)
        self.assertEqual(traj.natoms, 4)
        self.assertEqual(traj.species, ["Pb", "I"])
        self.assertEqual(traj.atom_symbols(), ["Pb", "I", "I", "I"])
        self.assertTrue(traj.direct)
        self.assertFalse(traj.variable_cell)
        np.testing.assert_allclose(traj.lattices[0], np.eye(3) * 10.0)

    def test_negative_scale_is_a_target_volume(self):
        # VASP reads a negative value on line 2 as the target cell volume in
        # cubic Angstrom, not as a multiplier. Multiplying by it instead flips
        # the cell and rescales every velocity derived from it.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "XDATCAR"
            lines = [
                "target volume", "-64.0",
                "  2.0 0.0 0.0", "  0.0 2.0 0.0", "  0.0 0.0 2.0",
                "  C", "  2",
            ]
            for frame in range(3):
                lines += [f"Direct configuration=  {frame + 1}",
                          "  0.1 0.1 0.1", "  0.6 0.6 0.6"]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            traj = read_xdatcar(path)
        np.testing.assert_allclose(traj.lattices[0], np.eye(3) * 4.0, atol=1e-9)
        self.assertAlmostEqual(abs(np.linalg.det(traj.lattices[0])), 64.0, places=6)

    def test_negative_scale_on_a_triclinic_cell(self):
        # A diagonal cell leaves det() and the cube root under-constrained;
        # a sheared cell exercises them.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "XDATCAR"
            cell = np.array([[3.0, 0.0, 0.0], [1.0, 4.0, 0.0], [0.5, 0.5, 5.0]])
            target = 120.0
            lines = ["triclinic", f"{-target}"]
            lines += ["  " + "  ".join(f"{v:.8f}" for v in row) for row in cell]
            lines += ["  C", "  1"]
            for frame in range(3):
                lines += [f"Direct configuration=  {frame + 1}", "  0.1 0.2 0.3"]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            traj = read_xdatcar(path)
        self.assertAlmostEqual(abs(np.linalg.det(traj.lattices[0])), target, places=6)
        # The shape is preserved: only a uniform scale is applied.
        ratio = traj.lattices[0] / cell
        np.testing.assert_allclose(ratio[cell != 0], ratio[cell != 0][0], rtol=1e-12)

    def test_positive_scale_multiplies_the_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_xdatcar(Path(tmp) / "XDATCAR", nframes=10, cell_ang=10.0)
            traj = read_xdatcar(path)
        np.testing.assert_allclose(traj.lattices[0], np.eye(3) * 10.0)

    def test_degenerate_cell_with_negative_scale_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "XDATCAR"
            lines = [
                "flat", "-64.0",
                "  2.0 0.0 0.0", "  0.0 2.0 0.0", "  0.0 0.0 0.0",
                "  C", "  1",
            ]
            for frame in range(3):
                lines += [f"Direct configuration=  {frame + 1}", "  0.1 0.1 0.1"]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(XdatcarFormatError):
                read_xdatcar(path)

    def test_short_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "XDATCAR"
            path.write_text("only\na few\nlines\n", encoding="utf-8")
            with self.assertRaises(XdatcarFormatError):
                read_xdatcar(path)

    def test_truncated_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_xdatcar(Path(tmp) / "XDATCAR", nframes=10, counts=(2,))
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaises(XdatcarFormatError):
                read_xdatcar(path)


if __name__ == "__main__":
    unittest.main()
