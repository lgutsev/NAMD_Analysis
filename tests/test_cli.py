import bootstrap  # noqa: F401

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.cli import main
from namd_analysis.report import OutputExistsError, prepare_output
from synthetic import (
    write_legacy_fit,
    write_run_directory,
    write_shprop_set,
    write_spectrum_file,
    write_xdatcar,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _report(out: Path) -> dict:
    return json.loads((out / "report.json").read_text(encoding="utf-8"))


class OutputDirectoryTests(unittest.TestCase):
    def test_non_empty_directory_is_refused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            out.mkdir()
            (out / "old.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(OutputExistsError):
                prepare_output(out)
            prepare_output(out, overwrite=True)

    def test_empty_directory_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            out.mkdir()
            self.assertEqual(prepare_output(out), out)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_inventory_writes_a_report_and_table(self):
        run = write_run_directory(self.root / "campaign" / "run_a", nframes=40, nstates=2)
        write_legacy_fit(run, tau_ns=0.0001, r_squared=-17.0)
        out = self.root / "out_inventory"
        self.assertEqual(main(["inventory", str(self.root / "campaign"), "--out", str(out)]), 0)
        report = _report(out)
        self.assertEqual(report["command"], "inventory")
        self.assertEqual(report["summary"]["n_failed_legacy_fits"], 1)
        self.assertTrue((out / "inventory.csv").is_file())

    def test_audit_writes_pairs_and_fingerprints(self):
        run = write_run_directory(self.root / "run", nframes=60, nstates=3)
        out = self.root / "out_audit"
        code = main(
            ["audit", str(run), "--nac-unit", "eV", "--threshold-mev", "0.5",
             "--out", str(out)]
        )
        self.assertEqual(code, 0)
        report = _report(out)
        self.assertEqual(report["audit"]["nstates"], 3)
        self.assertEqual(len(report["inputs"]), 5)
        self.assertEqual(len(report["inputs"][0]["sha256"]), 64)
        rows = (out / "pairs.csv").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(rows), 4)  # header plus three pairs

    def test_audit_requires_an_explicit_nac_unit(self):
        run = write_run_directory(self.root / "run", nframes=40, nstates=2)
        with self.assertRaises(SystemExit):
            main(["audit", str(run), "--out", str(self.root / "x")])

    def test_audit_reports_a_missing_directory_as_an_error(self):
        code = main(
            ["audit", str(self.root / "nope"), "--nac-unit", "eV",
             "--out", str(self.root / "out")]
        )
        self.assertEqual(code, 2)

    def test_populations_with_fit_writes_every_output(self):
        write_shprop_set(self.root / "run", n_files=3, nsteps=200,
                         dt_fs=1000.0, tau_fs=50000.0)
        out = self.root / "out_pop"
        code = main(
            [
                "populations",
                "--files", str(self.root / "run" / "SHPROP.*"),
                "--config", str(EXAMPLES / "two_state.json"),
                "--fit-group", "CBM",
                "--fit-start-ns", "0",
                "--fit-end-ns", "0.2",
                "--out", str(out),
            ]
        )
        self.assertEqual(code, 0)
        for name in ("report.json", "populations.csv", "fit.csv",
                     "populations.png", "populations.pdf"):
            self.assertTrue((out / name).is_file(), name)
        report = _report(out)
        self.assertEqual(report["averaging"]["n_files"], 3)
        self.assertAlmostEqual(report["fit"]["tau"], 0.05, places=3)
        self.assertLess(report["conservation"]["max_abs_deviation_from_one"], 1e-9)
        self.assertIsNotNone(report["survival"])
        header = (out / "populations.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("CBM_sem", header)
        self.assertIn("survival", header)

    def test_populations_reports_a_rejected_fit_without_failing(self):
        root = self.root / "flat"
        root.mkdir(parents=True)
        time = np.arange(50) * 1000.0
        table = np.column_stack(
            [time, np.full(50, -1.0), np.full(50, 0.5), np.full(50, 0.5)]
        )
        np.savetxt(root / "SHPROP.1", table)
        out = self.root / "out_flat"
        code = main(
            [
                "populations",
                "--files", str(root / "SHPROP.*"),
                "--config", str(EXAMPLES / "two_state.json"),
                "--fit-group", "CBM",
                "--out", str(out),
            ]
        )
        self.assertEqual(code, 0)
        report = _report(out)
        self.assertIsNone(report["fit"])
        self.assertIn("does not decay", report["fit_error"])

    def test_populations_rejects_an_unknown_fit_group(self):
        write_shprop_set(self.root / "run", n_files=1, nsteps=30)
        with self.assertRaises(SystemExit):
            main(
                [
                    "populations",
                    "--files", str(self.root / "run" / "SHPROP.*"),
                    "--config", str(EXAMPLES / "two_state.json"),
                    "--fit-group", "PCBM",
                    "--out", str(self.root / "out"),
                ]
            )

    def test_missing_input_pattern_exits(self):
        with self.assertRaises(SystemExit):
            main(
                [
                    "populations",
                    "--files", str(self.root / "absent" / "SHPROP.*"),
                    "--config", str(EXAMPLES / "two_state.json"),
                    "--out", str(self.root / "out"),
                ]
            )

    def test_vacf_spectra_compares_files(self):
        write_spectrum_file(self.root / "spectral_density_A.txt", (60.0, 300.0))
        write_spectrum_file(self.root / "spectral_density_B.txt", (40.0, 300.0))
        out = self.root / "out_spectra"
        code = main(
            [
                "vacf-spectra",
                "--files", str(self.root / "spectral_density_*.txt"),
                "--range", "0:800",
                "--reference", "A",
                "--xlim", "0:400",
                "--out", str(out),
            ]
        )
        self.assertEqual(code, 0)
        for name in ("report.json", "systems.csv", "bands.csv", "peaks.csv",
                     "spectra.png", "spectra.pdf"):
            self.assertTrue((out / name).is_file(), name)
        report = _report(out)
        self.assertTrue(report["shared_grid"])
        self.assertEqual(report["reference"], "A")
        self.assertEqual(len(report["systems"]), 2)

    def test_vacf_spectra_accepts_label_overrides(self):
        write_spectrum_file(self.root / "spectral_density_A.txt", (60.0,))
        out = self.root / "out_labels"
        code = main(
            [
                "vacf-spectra",
                "--files", str(self.root / "spectral_density_A.txt"),
                "--label", f"{self.root / 'spectral_density_A.txt'}=pristine",
                "--out", str(out),
            ]
        )
        self.assertEqual(code, 0)
        self.assertEqual(_report(out)["systems"][0]["label"], "pristine")

    def test_vacf_trajectory_end_to_end(self):
        path = write_xdatcar(
            self.root / "sys" / "XDATCAR",
            frequencies_cm1=(120.0,),
            nframes=2000,
            counts=(2,),
        )
        out = self.root / "out_traj"
        code = main(
            [
                "vacf-trajectory",
                "--xdatcar", str(path),
                "--label", f"{path}=model",
                "--dt-fs", "1.0",
                "--max-lag", "1000",
                "--out", str(out),
            ]
        )
        self.assertEqual(code, 0)
        for name in ("report.json", "vacf_model.csv", "spectral_density_model.txt",
                     "vacf_model.png", "spectra.png"):
            self.assertTrue((out / name).is_file(), name)
        report = _report(out)
        self.assertEqual(report["systems"][0]["atoms"], 2)
        self.assertTrue(report["settings"]["minimum_image_unwrap"])
        self.assertTrue(report["settings"]["center_of_mass_motion_removed"])
        written = np.loadtxt(out / "spectral_density_model.txt", skiprows=1)
        peak = written[int(np.argmax(written[:, 1])), 0]
        self.assertLess(abs(peak - 120.0), 40.0)

    def test_vacf_trajectory_rejects_a_bad_range(self):
        path = write_xdatcar(self.root / "sys" / "XDATCAR", nframes=200, counts=(2,))
        with self.assertRaises(SystemExit):
            main(
                [
                    "vacf-trajectory",
                    "--xdatcar", str(path),
                    "--dt-fs", "1.0",
                    "--max-lag", "100",
                    "--xlim", "800:0",
                    "--out", str(self.root / "out"),
                ]
            )

    def test_existing_output_directory_is_refused(self):
        run = write_run_directory(self.root / "run", nframes=40, nstates=2)
        out = self.root / "out"
        self.assertEqual(main(["audit", str(run), "--nac-unit", "eV", "--out", str(out)]), 0)
        self.assertEqual(main(["audit", str(run), "--nac-unit", "eV", "--out", str(out)]), 2)
        self.assertEqual(
            main(["audit", str(run), "--nac-unit", "eV", "--out", str(out), "--overwrite"]),
            0,
        )


if __name__ == "__main__":
    unittest.main()
