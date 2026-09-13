import bootstrap  # noqa: F401

import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.audit import AuditError, audit_run
from namd_analysis.inventory import scan, summarize
from namd_analysis.units import HBAR_EV_FS, hbar_over_dt_mev, nac_to_mev
from synthetic import write_legacy_fit, write_run_directory


class UnitTests(unittest.TestCase):
    def test_energy_units_convert_directly(self):
        values = np.array([0.5])
        np.testing.assert_allclose(nac_to_mev(values, "eV", 1.0), [500.0])
        np.testing.assert_allclose(nac_to_mev(values, "meV", 1.0), [0.5])

    def test_derivative_coupling_uses_hbar(self):
        values = np.array([1.0])
        np.testing.assert_allclose(
            nac_to_mev(values, "fs^-1", 1.0), [HBAR_EV_FS * 1000.0]
        )

    def test_unknown_unit_is_rejected(self):
        with self.assertRaises(ValueError):
            nac_to_mev(np.array([1.0]), "hartree", 1.0)

    def test_hbar_over_dt_scales_with_the_timestep(self):
        self.assertAlmostEqual(hbar_over_dt_mev(1.0), 658.2119569509066, places=6)
        self.assertAlmostEqual(hbar_over_dt_mev(2.0), 329.1059784754533, places=6)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dimensions_and_coupling_statistics(self):
        run = write_run_directory(self.root / "run", nframes=100, nstates=3,
                                  coupling_ev=0.01)
        result = audit_run(run, nac_unit="eV")
        self.assertEqual(result.nframes, 100)
        self.assertEqual(result.nstates, 3)
        self.assertEqual(result.dt_fs, 1.0)
        self.assertEqual(result.dt_source, "inp POTIM")
        # The synthetic coupling is a sine sampled on a grid, so its largest
        # sample approaches but does not reach the 10 meV amplitude.
        self.assertAlmostEqual(result.couplings["max_abs_offdiagonal"], 10.0, delta=1e-3)
        self.assertEqual(len(result.pairs), 3)

    def test_declared_unit_changes_the_reported_magnitude(self):
        run = write_run_directory(self.root / "run", nframes=60, nstates=2,
                                  coupling_ev=0.01)
        in_ev = audit_run(run, nac_unit="eV").couplings["max_abs_offdiagonal"]
        in_mev = audit_run(run, nac_unit="meV").couplings["max_abs_offdiagonal"]
        self.assertAlmostEqual(in_ev / in_mev, 1000.0, places=6)

    def test_explicit_timestep_overrides_potim(self):
        run = write_run_directory(self.root / "run", nframes=60, nstates=2)
        result = audit_run(run, nac_unit="eV", dt_fs=0.5)
        self.assertEqual(result.dt_fs, 0.5)
        self.assertEqual(result.dt_source, "--dt-fs")
        self.assertAlmostEqual(
            result.couplings["hbar_over_dt_meV"], 2 * 658.2119569509066, places=6
        )

    def test_clipped_coupling_file_is_flagged(self):
        run = write_run_directory(self.root / "capped", nframes=200, nstates=2,
                                  coupling_ev=0.01, cap_ev=0.005)
        result = audit_run(run, nac_unit="eV")
        clipping = next(c for c in result.checks if c["check"] == "nac_clipping")
        self.assertEqual(clipping["status"], "capped")
        self.assertTrue(result.couplings["global_max_is_a_cap"])
        self.assertGreater(result.pairs[0].samples_at_global_max, 2)

    def test_unclipped_coupling_file_is_not_flagged(self):
        run = write_run_directory(self.root / "clean", nframes=200, nstates=2,
                                  coupling_ev=0.01)
        result = audit_run(run, nac_unit="eV")
        clipping = next(c for c in result.checks if c["check"] == "nac_clipping")
        self.assertEqual(clipping["status"], "ok")

    def test_antisymmetry_and_zero_diagonal_are_checked(self):
        run = write_run_directory(self.root / "run", nframes=50, nstates=3)
        result = audit_run(run, nac_unit="eV")
        statuses = {c["check"]: c["status"] for c in result.checks}
        self.assertEqual(statuses["nac_diagonal"], "ok")
        self.assertEqual(statuses["nac_antisymmetry"], "ok")

    def test_band_window_mismatch_is_reported_not_raised(self):
        run = write_run_directory(self.root / "run", nframes=50, nstates=3)
        (run / "inp").write_text(
            "&NAMDPARA\n BMIN = 1\n BMAX = 9\n POTIM = 1.0\n/\n", encoding="utf-8"
        )
        result = audit_run(run, nac_unit="eV")
        band = next(c for c in result.checks if c["check"] == "band_window")
        self.assertEqual(band["status"], "mismatch")

    def test_frame_count_mismatch_between_files_is_an_error(self):
        run = write_run_directory(self.root / "run", nframes=50, nstates=2)
        table = np.loadtxt(run / "NATXT")
        np.savetxt(run / "NATXT", table[:-1])
        with self.assertRaises(AuditError) as ctx:
            audit_run(run, nac_unit="eV")
        self.assertIn("frames", str(ctx.exception))

    def test_missing_timestep_is_an_error(self):
        run = write_run_directory(self.root / "run", nframes=40, nstates=2)
        (run / "inp").unlink()
        with self.assertRaises(AuditError) as ctx:
            audit_run(run, nac_unit="eV")
        self.assertIn("--dt-fs", str(ctx.exception))

    def test_unknown_nac_unit_is_an_error(self):
        run = write_run_directory(self.root / "run", nframes=40, nstates=2)
        with self.assertRaises(AuditError):
            audit_run(run, nac_unit="hartree")

    def test_pair_time_integral_scales_with_the_timestep(self):
        run = write_run_directory(self.root / "run", nframes=80, nstates=2)
        one = audit_run(run, nac_unit="eV", dt_fs=1.0).pairs[0]
        half = audit_run(run, nac_unit="eV", dt_fs=0.5).pairs[0]
        self.assertAlmostEqual(one.time_integral_mev_fs, 2 * half.time_integral_mev_fs)
        self.assertAlmostEqual(one.sample_sum_mev, half.sample_sum_mev)


class InventoryTests(unittest.TestCase):
    def test_scan_reports_runs_and_failed_legacy_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = write_run_directory(root / "campaign" / "good", nframes=30, nstates=2)
            bad = write_run_directory(root / "campaign" / "bad", nframes=30, nstates=2)
            write_legacy_fit(good, tau_ns=1.2, r_squared=0.999)
            write_legacy_fit(bad, tau_ns=0.0001, r_squared=-17.4289)
            entries = scan(root)
            summary = summarize(entries)
        self.assertEqual(summary["n_run_directories"], 2)
        self.assertEqual(summary["n_with_legacy_fit"], 2)
        self.assertEqual(summary["n_failed_legacy_fits"], 1)
        self.assertEqual(summary["failed_legacy_fits"][0]["name"], "bad")

    def test_scan_rejects_a_file_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not_a_dir"
            path.write_text("x", encoding="utf-8")
            with self.assertRaises(NotADirectoryError):
                scan(path)


if __name__ == "__main__":
    unittest.main()
