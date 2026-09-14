import bootstrap  # noqa: F401

import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.audit import AuditError, audit_run
from namd_analysis.nac_policy import NacPolicy, NacPolicyError
from namd_analysis.inventory import scan, summarize
from namd_analysis.units import (
    HBAR_EV_FS,
    hbar_over_dt_ev,
    hbar_over_dt_mev,
    nac_to_mev,
)
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
            result.couplings["timestep_limit"]["hbar_over_dt_meV"],
            2 * 658.2119569509066,
            places=6,
        )

    def test_repeated_ceiling_is_not_called_accidental_clipping(self):
        # The archived 0.6 eV ceiling was engineered on purpose. Reporting it as
        # an accidental cap, or asserting that it makes the statistics a lower
        # bound, is the specific misreading this check exists to avoid.
        run = write_run_directory(self.root / "ceiling", nframes=200, nstates=2,
                                  coupling_ev=0.8, cap_ev=0.6)
        result = audit_run(run, nac_unit="eV")
        names = {c["check"] for c in result.checks}
        self.assertNotIn("nac_clipping", names)
        check = next(c for c in result.checks if c["check"] == "nac_repeated_ceiling")
        self.assertEqual(check["status"], "engineered_ceiling")
        self.assertIn("intentionally imposed upstream NAC safety ceiling", check["detail"])
        for forbidden in ("accidental", "capped before it was written"):
            self.assertNotIn(forbidden, check["detail"].lower())
        ceiling = result.couplings["engineered_ceiling"]
        self.assertTrue(ceiling["detected"])
        self.assertAlmostEqual(ceiling["ceiling_eV"], 0.6)
        self.assertGreater(ceiling["samples_at_ceiling"], 2)
        # No declared policy, so no claim that the mean is a lower bound.
        self.assertFalse(ceiling["statistics_are_a_lower_bound"])
        self.assertFalse(result.couplings["nac_policy"]["declared"])
        self.assertGreater(result.pairs[0].samples_at_engineered_ceiling, 2)

    def test_file_without_a_repeated_ceiling_is_not_flagged(self):
        run = write_run_directory(self.root / "clean", nframes=200, nstates=2,
                                  coupling_ev=0.01)
        result = audit_run(run, nac_unit="eV")
        check = next(c for c in result.checks if c["check"] == "nac_repeated_ceiling")
        self.assertEqual(check["status"], "ok")
        self.assertFalse(result.couplings["engineered_ceiling"]["detected"])
        self.assertEqual(result.pairs[0].samples_at_engineered_ceiling, 0)

    def test_audit_connects_coupling_magnitudes_to_hbar_over_dt(self):
        run = write_run_directory(self.root / "big", nframes=200, nstates=2,
                                  coupling_ev=0.8)
        result = audit_run(run, nac_unit="eV", dt_fs=1.0)
        limit = result.couplings["timestep_limit"]
        self.assertAlmostEqual(limit["hbar_over_dt_eV"], hbar_over_dt_ev(1.0))
        self.assertAlmostEqual(limit["hbar_over_dt_meV"], hbar_over_dt_mev(1.0))
        self.assertAlmostEqual(
            limit["global_max_over_hbar_dt"],
            limit["global_max_nac_meV"] / limit["hbar_over_dt_meV"],
        )
        self.assertGreater(limit["fraction_above_0p8_hbar_dt"], 0.0)
        self.assertGreaterEqual(
            limit["fraction_above_0p8_hbar_dt"], limit["fraction_above_0p9_hbar_dt"]
        )
        check = next(c for c in result.checks if c["check"] == "nac_timestep_limit")
        self.assertEqual(check["status"], "near_timestep_limit")
        self.assertIn("numerically pathological", check["detail"])
        pair = result.pairs[0]
        self.assertGreater(pair.max_abs_nac_over_hbar_dt, 0.8)
        self.assertGreater(pair.fraction_above_0p8_hbar_dt, 0.0)

    def test_timestep_limit_is_ok_for_a_small_coupling(self):
        run = write_run_directory(self.root / "small", nframes=120, nstates=2,
                                  coupling_ev=0.01)
        result = audit_run(run, nac_unit="eV")
        check = next(c for c in result.checks if c["check"] == "nac_timestep_limit")
        self.assertEqual(check["status"], "ok")
        self.assertEqual(
            result.couplings["timestep_limit"]["samples_above_0p8_hbar_dt"], 0
        )

    def test_the_timestep_limit_scales_with_dt(self):
        # hbar/dt halves when the step doubles, so the same file sits twice as
        # close to the limit.
        run = write_run_directory(self.root / "scale", nframes=120, nstates=2,
                                  coupling_ev=0.3)
        one = audit_run(run, nac_unit="eV", dt_fs=1.0).couplings["timestep_limit"]
        two = audit_run(run, nac_unit="eV", dt_fs=2.0).couplings["timestep_limit"]
        self.assertAlmostEqual(one["hbar_over_dt_eV"], 2 * two["hbar_over_dt_eV"])
        self.assertAlmostEqual(
            two["global_max_over_hbar_dt"], 2 * one["global_max_over_hbar_dt"]
        )

    def test_warning_fraction_suppresses_a_rare_excursion(self):
        run = write_run_directory(self.root / "rare", nframes=200, nstates=2,
                                  coupling_ev=0.8)
        loud = audit_run(run, nac_unit="eV")
        quiet = audit_run(run, nac_unit="eV", warn_fraction=1.0)
        self.assertEqual(
            next(c for c in loud.checks if c["check"] == "nac_timestep_limit")["status"],
            "near_timestep_limit",
        )
        self.assertEqual(
            next(c for c in quiet.checks if c["check"] == "nac_timestep_limit")["status"],
            "ok",
        )

    def test_a_declared_zeroing_policy_does_not_make_statistics_a_lower_bound(self):
        run = write_run_directory(self.root / "declared", nframes=200, nstates=2,
                                  coupling_ev=0.8, cap_ev=0.6)
        policy = NacPolicy.from_dict(
            {
                "nac_policy": {
                    "warning_threshold_eV": 0.6,
                    "numerical_limit": "hbar_over_dt",
                    "reject_above_eV": 0.66,
                    "action_above_limit": "zero",
                }
            }
        )
        result = audit_run(run, nac_unit="eV", policy=policy)
        check = next(c for c in result.checks if c["check"] == "nac_repeated_ceiling")
        self.assertIn("declared upstream warning threshold of 0.6 eV", check["detail"])
        self.assertIn("hbar/dt", check["detail"])
        self.assertFalse(
            result.couplings["engineered_ceiling"]["statistics_are_a_lower_bound"]
        )
        self.assertTrue(result.couplings["nac_policy"]["declared"])
        self.assertEqual(result.couplings["nac_policy"]["action_above_limit"], "zero")

    def test_a_declared_clipping_policy_does_make_statistics_a_lower_bound(self):
        # Only truncation of otherwise valid values supports that claim.
        run = write_run_directory(self.root / "clipped", nframes=200, nstates=2,
                                  coupling_ev=0.8, cap_ev=0.6)
        policy = NacPolicy.from_dict(
            {"warning_threshold_eV": 0.6, "action_above_limit": "clip"}
        )
        result = audit_run(run, nac_unit="eV", policy=policy)
        self.assertTrue(
            result.couplings["engineered_ceiling"]["statistics_are_a_lower_bound"]
        )
        check = next(c for c in result.checks if c["check"] == "nac_repeated_ceiling")
        self.assertIn("lower bound", check["detail"])


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


class NacPolicyTests(unittest.TestCase):
    def test_roundtrip_from_a_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            path.write_text(
                '{"nac_policy": {"warning_threshold_eV": 0.6, '
                '"numerical_limit": "hbar_over_dt", "reject_above_eV": 0.66, '
                '"action_above_limit": "zero"}}',
                encoding="utf-8",
            )
            policy = NacPolicy.from_json(path)
        self.assertEqual(policy.warning_threshold_ev, 0.6)
        self.assertEqual(policy.reject_above_ev, 0.66)
        self.assertEqual(policy.action_above_limit, "zero")
        self.assertAlmostEqual(policy.limit_ev(1.0), hbar_over_dt_ev(1.0))
        self.assertAlmostEqual(policy.limit_ev(2.0), hbar_over_dt_ev(2.0))
        self.assertFalse(policy.truncates_valid_values)
        self.assertTrue(policy.as_dict()["declared"])

    def test_the_shipped_campaign_policy_parses(self):
        root = Path(__file__).resolve().parents[1]
        policy = NacPolicy.from_json(root / "examples" / "nac_policy_bcf_campaign.json")
        self.assertEqual(policy.warning_threshold_ev, 0.6)
        self.assertEqual(policy.numerical_limit, "hbar_over_dt")
        self.assertEqual(policy.action_above_limit, "zero")

    def test_unknown_action_is_rejected(self):
        with self.assertRaises(NacPolicyError):
            NacPolicy.from_dict({"action_above_limit": "renormalize"})

    def test_numeric_numerical_limit_is_rejected(self):
        with self.assertRaises(NacPolicyError):
            NacPolicy.from_dict({"numerical_limit": 0.658})

    def test_rejection_limit_below_the_warning_threshold_is_rejected(self):
        with self.assertRaises(NacPolicyError):
            NacPolicy.from_dict(
                {"warning_threshold_eV": 0.66, "reject_above_eV": 0.6}
            )

    def test_negative_threshold_is_rejected(self):
        with self.assertRaises(NacPolicyError):
            NacPolicy.from_dict({"warning_threshold_eV": -1.0})

    def test_no_policy_is_assumed_for_an_undeclared_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run_directory(Path(tmp) / "run", nframes=80, nstates=2,
                                      coupling_ev=0.8, cap_ev=0.6)
            result = audit_run(run, nac_unit="eV")
        payload = result.as_dict()["nac_policy"]
        self.assertFalse(payload["declared"])
        self.assertIn("none is assumed", payload["note"])

if __name__ == "__main__":
    unittest.main()
