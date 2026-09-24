"""Full-space validation: inputs for the run, readers for its outputs, comparison."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from robustness_fixtures import GROUPS, POSCAR, write_campaign

from namd_analysis.dispatch import main as dispatch_main
from namd_analysis.fullspace import (
    FullspaceError,
    compare_sample,
    derive_incars,
    fragment_fractions,
    nearest_atom_integrals,
    render_job,
)
from namd_analysis.io.vasp_settings import read_incar
from namd_analysis.io.volumetric import (
    VolumetricFormatError,
    read_bader_acf,
    read_volumetric,
)


def write_volumetric(path, grid_values, header=POSCAR, trailing=""):
    """A CHGCAR-format file; ``grid_values[i, j, k]``, written x fastest, 10 per line."""
    nx, ny, nz = grid_values.shape
    flat = grid_values.transpose(2, 1, 0).reshape(-1)
    lines = [header.rstrip("\n"), "", f"   {nx}   {ny}   {nz}"]
    for start in range(0, flat.size, 10):
        lines.append(" ".join(f"{v:.11E}" for v in flat[start:start + 10]))
    Path(path).write_text("\n".join(lines) + "\n" + trailing)
    return Path(path)


def write_acf(path, charges, vacuum=0.0):
    lines = ["    #         X           Y           Z       CHARGE      MIN DIST   ATOMIC VOL",
             " " + "-" * 80]
    for i, q in enumerate(charges, start=1):
        lines.append(f"    {i}    0.0000    0.0000    0.0000    {q:.6f}    1.0000    10.0000")
    lines += [" " + "-" * 80,
              f"    VACUUM CHARGE:               {vacuum:.4f}",
              "    VACUUM VOLUME:               0.0000",
              f"    NUMBER OF ELECTRONS:         {sum(charges) + vacuum:.4f}"]
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def _run(argv):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = dispatch_main(argv)
    return code, stream.getvalue()


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class IncarDerivationTests(unittest.TestCase):
    BASE = {"ENCUT": "400", "LORBIT": "11", "NSW": "2000", "IBRION": "0",
            "LWAVE": ".FALSE.", "NBANDS": "1000", "LPARD": ".TRUE."}

    def test_production_settings_kept_and_every_change_recorded(self):
        scf, parchg, changes = derive_incars(self.BASE, [981, 977])
        self.assertEqual(scf["ENCUT"], "400")
        self.assertEqual(scf["LORBIT"], "11")  # the rerun reproduces the production PROCAR
        self.assertEqual(scf["NSW"], "0")
        self.assertEqual(scf["LAECHG"], ".TRUE.")
        self.assertNotIn("LPARD", scf)
        self.assertEqual(parchg["LPARD"], ".TRUE.")
        self.assertEqual(parchg["IBAND"], "981 977")
        self.assertEqual(parchg["ISTART"], "1")
        self.assertNotIn("LORBIT", parchg)
        self.assertEqual(parchg["ENCUT"], "400")
        changed = {(c["step"], c["tag"]) for c in changes}
        for expected in (("scf", "NSW"), ("scf", "LAECHG"), ("parchg", "IBAND"),
                         ("both", "LPARD"), ("parchg", "LORBIT")):
            self.assertIn(expected, changed)
        for change in changes:
            self.assertTrue(change["why"])

    def test_refusals(self):
        with self.assertRaises(FullspaceError):
            derive_incars({**self.BASE, "ISPIN": "2"}, [981])
        with self.assertRaises(FullspaceError):
            derive_incars({**self.BASE, "NBANDS": "900"}, [981])

    def test_job_carries_placeholders_until_told(self):
        job = render_job(981, [981, 977], 5, None, None, 1, 48, "24:00:00")
        self.assertIn("REQUIRED_EDIT_account", job)
        self.assertIn("#SBATCH --array=1-5", job)
        self.assertIn("BANDS=(981 977)", job)
        self.assertIn("-vac off", job)
        self.assertIn('VASP_CMD:?', job)
        job = render_job(981, [981], 2, "acct", "workq", 2, 64, "12:00:00",
                         campaign="/work/my campaign")
        self.assertNotIn("REQUIRED_EDIT", job)
        self.assertIn("cd '/work/my campaign'", job)


class ReaderTests(TmpCase):
    def test_volumetric_layout_is_x_fastest(self):
        values = np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2)
        path = write_volumetric(self.root / "CHGCAR", values,
                                trailing="augmentation occupancies 1 2\n 0.1 0.2\n")
        volume = read_volumetric(path)
        np.testing.assert_array_equal(volume.data, values)
        self.assertEqual(volume.counts, [2, 2, 2])
        self.assertEqual(volume.species, ["Pb", "C", "H"])
        self.assertAlmostEqual(volume.integral(), values.mean())
        np.testing.assert_allclose(volume.lattice, np.eye(3) * 10.0)

    def test_truncated_grid_is_refused(self):
        path = write_volumetric(self.root / "PARCHG", np.ones((4, 4, 4)))
        text = path.read_text().splitlines()
        path.write_text("\n".join(text[:-2]) + "\n")
        with self.assertRaises(VolumetricFormatError):
            read_volumetric(path)

    def test_bader_acf(self):
        path = write_acf(self.root / "ACF.dat", [0.1, 0.2, 0.3], vacuum=0.05)
        charges = read_bader_acf(path, natoms=3)
        np.testing.assert_allclose(charges.charges, [0.1, 0.2, 0.3])
        self.assertAlmostEqual(charges.vacuum_charge, 0.05)
        self.assertAlmostEqual(charges.total, 0.65)
        with self.assertRaises(VolumetricFormatError):
            read_bader_acf(path, natoms=4)
        broken = self.root / "broken.dat"
        broken.write_text("\n".join(path.read_text().splitlines()[:-3]) + "\n")
        with self.assertRaises(VolumetricFormatError):
            read_bader_acf(broken)


class PartitionTests(TmpCase):
    def test_density_on_atom_sites_goes_to_those_atoms(self):
        grid = np.zeros((10, 10, 10))
        # POSCAR atoms sit on grid points (multiples of 0.1).
        sites = [(1, 1, 1), (3, 1, 1), (1, 5, 5), (3, 5, 5), (6, 8, 8), (8, 8, 8)]
        charges = [0.30, 0.10, 0.05, 0.15, 0.25, 0.15]
        for site, q in zip(sites, charges):
            grid[site] = q * grid.size
        volume = read_volumetric(write_volumetric(self.root / "PARCHG", grid))
        np.testing.assert_allclose(nearest_atom_integrals(volume, chunk=97), charges)

    def test_assignment_wraps_across_the_cell_boundary(self):
        header = "two atoms\n1.0\n10 0 0\n0 10 0\n0 0 10\nX\n2\nDirect\n0.05 0.5 0.5\n0.45 0.5 0.5\n"
        grid = np.zeros((20, 1, 1))
        grid[19, 0, 0] = 20.0  # x = 0.95: 0.10 from atom 1 through the boundary, 0.50 from atom 2
        volume = read_volumetric(write_volumetric(self.root / "P", grid, header=header))
        np.testing.assert_allclose(nearest_atom_integrals(volume), [1.0, 0.0])

    def test_fragment_fractions_keep_the_vacuum_in_the_denominator(self):
        record = fragment_fractions(np.array([0.2, 0.2, 0.1, 0.1, 0.1, 0.1]), GROUPS,
                                    unassigned=0.2)
        self.assertAlmostEqual(record["integral"], 1.0)
        self.assertAlmostEqual(record["fractions"]["perovskite"], 0.4)
        self.assertAlmostEqual(record["unassigned_fraction"], 0.2)
        self.assertAlmostEqual(record["undeclared_atoms_fraction"], 0.0)


class CompareSampleTests(unittest.TestCase):
    REFERENCE = {
        "captured_projection": 0.5,
        "raw": {"perovskite": 0.40, "BCF": 0.05, "PCBM": 0.05},
        "normalized": {"perovskite": 0.8, "BCF": 0.1, "PCBM": 0.1},
    }

    def test_proportional_allocation_reproduces_normalization(self):
        full = {"fractions": {"perovskite": 0.8, "BCF": 0.1, "PCBM": 0.1}}
        sample = compare_sample(self.REFERENCE, full, ("BCF", "PCBM"))
        self.assertAlmostEqual(sample["fullspace_over_normalized_pair_min"], 1.0)
        self.assertTrue(sample["fullspace_within_allocation_bracket"])
        for g in ("perovskite", "BCF", "PCBM"):
            self.assertAlmostEqual(sample["uncaptured_allocated"][g],
                                   sample["uncaptured_proportional_share"][g])

    def test_uncaptured_weight_all_on_the_perovskite(self):
        full = {"fractions": {"perovskite": 0.9, "BCF": 0.05, "PCBM": 0.05}}
        sample = compare_sample(self.REFERENCE, full, ("BCF", "PCBM"))
        self.assertAlmostEqual(sample["fullspace_pair_min"], sample["procar_pair_min_worst_case"])
        self.assertAlmostEqual(sample["fullspace_over_normalized_pair_min"], 0.5)
        self.assertAlmostEqual(sample["uncaptured_allocated"]["perovskite"], 1.0)

    def test_outside_the_bracket_is_flagged(self):
        full = {"fractions": {"perovskite": 0.98, "BCF": 0.01, "PCBM": 0.01}}
        sample = compare_sample(self.REFERENCE, full, ("BCF", "PCBM"))
        self.assertFalse(sample["fullspace_within_allocation_bracket"])


class EndToEndTests(TmpCase):
    """prepare -> (a fabricated run) -> compare, through the command line."""

    def setUp(self):
        super().setUp()
        raw = {}
        for frame in range(1, 13):
            captured = 0.44 + 0.01 * frame
            pair = 0.05 if frame in (3, 4) else 0.01
            raw[frame] = {977: (0.0, 0.52, 0.0), 981: (captured - 2 * pair, pair, pair)}
        self.campaign = write_campaign(self.root, raw)
        self.raw = raw
        profile = {"configurations": {"A": {"episodes": {"crossing_A": "8:10"},
                                            "control_episodes": {}}}}
        self.profile = self.root / "profile.json"
        self.profile.write_text(json.dumps(profile))

    def _prepare(self, out):
        return _run([
            "character-fullspace-prepare",
            "--projection-character", str(self.campaign["projection_character"]),
            "--projection-manifest", str(self.campaign["manifest"]),
            "--atom-groups", str(self.campaign["atom_groups"]),
            "--band", "981", "--bands", "981,977",
            "--include", "issue4_low_capture=1,2",
            "--profile", str(self.profile), "--configuration", "A",
            "--account", "acct", "--partition", "workq",
            "--out", str(out),
        ])

    def _fabricate(self, campaign, allocation):
        """Write what the batch job would have written, with a chosen allocation.

        ``allocation(frame, band)`` returns the full-space fractions per group.
        """
        manifest = json.loads((campaign / "validation_manifest.json").read_text())
        for entry in manifest["frames"]:
            frame = entry["frame"]
            work = campaign / f"frame_{frame}"
            shutil.copyfile(entry["production_procar"], work / "PROCAR.rerun")
            for band in manifest["bands"]:
                fractions = allocation(frame, band)
                per_atom = []
                for name, atoms in GROUPS.items():
                    per_atom += [fractions[name] / len(atoms)] * len(atoms)
                write_acf(work / f"ACF_{band}.dat", per_atom)
                grid = np.zeros((10, 10, 10))
                sites = [(1, 1, 1), (3, 1, 1), (1, 5, 5), (3, 5, 5), (6, 8, 8), (8, 8, 8)]
                for site, q in zip(sites, per_atom):
                    grid[site] = q * grid.size
                write_volumetric(work / f"PARCHG_{band}", grid)
        return manifest

    def test_prepare_writes_a_complete_campaign(self):
        out = self.root / "campaign"
        code, text = self._prepare(out)
        self.assertEqual(code, 0, text)
        manifest = json.loads((out / "validation_manifest.json").read_text())
        reasons = {f["frame"]: f["reasons"] for f in manifest["frames"]}
        self.assertIn("worst_capture", reasons[1])
        self.assertIn("issue4_low_capture", reasons[1])
        self.assertIn("high_capture_control", reasons[12])
        self.assertTrue(any("procar_mixed" in r for r in reasons.values()))
        self.assertTrue(any("window:crossing_A" in r for r in reasons.values()))
        self.assertEqual(manifest["projection_method"]["method"], "paw_projectors")
        for entry in manifest["frames"]:
            work = out / f"frame_{entry['frame']}"
            scf = read_incar(work / "INCAR.scf")
            parchg = read_incar(work / "INCAR.parchg")
            self.assertEqual(scf["LORBIT"], "11")
            self.assertEqual(parchg["IBAND"], "981 977")
            self.assertTrue(Path(entry["source_dir"], "POSCAR").is_file())
        tsv = (out / "frames.tsv").read_text().splitlines()
        self.assertEqual(len(tsv), len(manifest["frames"]) + 1)
        job = (out / "run_fullspace_validation.sbatch").read_text()
        self.assertIn(f"#SBATCH --array=1-{len(manifest['frames'])}", job)
        self.assertTrue((out / "atom_groups.json").is_file())

    def test_compare_recovers_a_known_allocation(self):
        out = self.root / "campaign"
        self.assertEqual(self._prepare(out)[0], 0)

        def allocation(frame, band):
            # Everything PROCAR missed goes to the perovskite: the pair keeps
            # exactly its raw weight.
            w_p, w_b, w_c = self.raw[frame][band]
            return {"perovskite": 1.0 - w_b - w_c, "BCF": w_b, "PCBM": w_c}

        manifest = self._fabricate(out, allocation)
        result = self.root / "compare"
        code, text = _run(["character-fullspace-compare", "--campaign", str(out),
                           "--voronoi", "--out", str(result)])
        self.assertEqual(code, 0, text)
        summary = json.loads((result / "fullspace_summary.json").read_text())
        self.assertEqual(summary["methods"], ["bader", "voronoi"])
        self.assertEqual(summary["missing_outputs"], [])
        records = [r for r in summary["records"] if r["band"] == 981]
        self.assertEqual(len(records), 2 * len(manifest["frames"]))
        for r in records:
            self.assertAlmostEqual(r["fullspace_pair_min"], r["procar_pair_min_worst_case"], places=4)
            self.assertAlmostEqual(r["uncaptured_allocated"]["perovskite"], 1.0, places=3)
        for check in summary["rerun_checks"]:
            self.assertLess(check["max_abs_raw_difference"], 1e-5)
        reading = [x for x in summary["readings"] if x["band"] == 981 and x["method"] == "bader"][0]
        at_010 = [t for t in reading["by_threshold"] if t["threshold"] == 0.10][0]
        self.assertGreater(at_010["n_frames_procar_mixed"], 0)
        self.assertEqual(at_010["n_procar_mixed_also_fullspace_mixed"], 0)
        self.assertLess(at_010["ratio_fullspace_to_normalized_median"], 1.0)
        self.assertIn("deliberately not given a number", reading["statements"]["note"])

    def test_compare_reports_missing_outputs_and_a_rerun_mismatch(self):
        out = self.root / "campaign"
        self.assertEqual(self._prepare(out)[0], 0)

        def proportional(frame, band):
            w = self.raw[frame][band]
            total = sum(w)
            return {name: value / total for name, value in zip(GROUPS, w)}

        manifest = self._fabricate(out, proportional)
        first = manifest["frames"][0]["frame"]
        (out / f"frame_{first}" / "ACF_977.dat").unlink()
        rerun = out / f"frame_{first}" / "PROCAR.rerun"
        other = manifest["frames"][-1]["production_procar"]
        shutil.copyfile(other, rerun)
        second = manifest["frames"][1]["frame"]
        (out / f"frame_{second}" / "PROCAR.rerun").unlink()
        result = self.root / "compare"
        code, text = _run(["character-fullspace-compare", "--campaign", str(out),
                           "--out", str(result)])
        self.assertEqual(code, 0, text)
        summary = json.loads((result / "fullspace_summary.json").read_text())
        self.assertEqual(summary["missing_outputs"][0]["frame"], first)
        no_rerun = [m for m in summary["missing_outputs"] if m["frame"] == second]
        self.assertTrue(no_rerun)
        self.assertTrue(all("not possible" in m["consequence"] for m in no_rerun))
        # Band 977 is the same in every frame; band 981 is not, so a PROCAR
        # from another frame must show up as a mismatch there.
        mismatch = [c for c in summary["rerun_checks"] if c["frame"] == first and c["band"] == 981]
        self.assertEqual(len(mismatch), 1)
        self.assertGreater(mismatch[0]["max_abs_raw_difference"], 0.005)
        for r in summary["records"]:
            if r["band"] == 981:
                self.assertAlmostEqual(r["fullspace_over_normalized_pair_min"], 1.0, places=4)

    def test_compare_refuses_an_unrun_campaign(self):
        out = self.root / "campaign"
        self.assertEqual(self._prepare(out)[0], 0)
        code, text = _run(["character-fullspace-compare", "--campaign", str(out),
                           "--out", str(self.root / "compare")])
        self.assertEqual(code, 2)
        self.assertIn("Has the batch job run?", text)

    def test_prepare_refuses_inconsistent_frame_incars(self):
        path = Path(self.root / "production" / "0012" / "INCAR")
        path.write_text(path.read_text().replace("ENCUT = 400", "ENCUT = 500"))
        code, text = self._prepare(self.root / "campaign")
        self.assertEqual(code, 2)
        self.assertIn("ENCUT", text)

    def test_prepare_requires_the_focus_band_in_iband(self):
        code, text = _run([
            "character-fullspace-prepare",
            "--projection-character", str(self.campaign["projection_character"]),
            "--projection-manifest", str(self.campaign["manifest"]),
            "--atom-groups", str(self.campaign["atom_groups"]),
            "--band", "981", "--bands", "977",
            "--out", str(self.root / "campaign"),
        ])
        self.assertEqual(code, 2)
        self.assertIn("must include the focus band", text)


if __name__ == "__main__":
    unittest.main()
