import bootstrap  # noqa: F401

import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.io.xdatcar import read_xdatcar
from namd_analysis.spectra import (
    SpectrumError,
    band_integral,
    compare,
    describe,
    load_spectrum,
    peak_table,
    shared_grid,
)
from namd_analysis.units import CM1_PER_INV_FS
from namd_analysis.vacf import (
    VacfError,
    cartesian_velocities,
    compute_vacf,
    compute_vacf_segmented,
    gaussian_smooth,
    masses_for,
    remove_center_of_mass_motion,
    spectral_density,
    trajectory_spectrum,
)
from synthetic import write_spectrum_file, write_xdatcar


def _peak_frequency(spectrum):
    return float(spectrum.frequency_cm1[int(np.argmax(spectrum.intensity))])


class VelocityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_minimum_image_unwrapping_removes_boundary_jumps(self):
        path = write_xdatcar(
            self.root / "XDATCAR", nframes=500, drift_ang_per_fs=0.05, counts=(2,)
        )
        trajectory = read_xdatcar(path)
        unwrapped = cartesian_velocities(trajectory, 1.0, unwrap=True)
        raw = cartesian_velocities(trajectory, 1.0, unwrap=False)
        # The drift is 0.05 A/fs in a 10 A cell, so an atom crosses the
        # boundary regularly; the raw difference records that as a 10 A jump.
        self.assertLess(np.abs(unwrapped).max(), 0.2)
        self.assertGreater(np.abs(raw).max(), 5.0)

    def test_velocity_scales_with_the_timestep(self):
        path = write_xdatcar(self.root / "XDATCAR", nframes=100)
        trajectory = read_xdatcar(path)
        one = cartesian_velocities(trajectory, 1.0)
        half = cartesian_velocities(trajectory, 0.5)
        np.testing.assert_allclose(half, 2.0 * one)

    def test_non_positive_timestep_is_rejected(self):
        path = write_xdatcar(self.root / "XDATCAR", nframes=50)
        trajectory = read_xdatcar(path)
        with self.assertRaises(VacfError):
            cartesian_velocities(trajectory, 0.0)

    def test_centre_of_mass_removal_zeroes_the_drift(self):
        velocities = np.zeros((10, 3, 3))
        velocities[:, :, 0] = 1.0  # rigid translation only
        cleaned, stats = remove_center_of_mass_motion(velocities)
        np.testing.assert_allclose(cleaned, 0.0, atol=1e-15)
        self.assertAlmostEqual(stats["com_speed_mean_ang_per_fs"], 1.0)

    def test_mass_lookup_rejects_unknown_elements(self):
        with self.assertRaises(VacfError) as ctx:
            masses_for(["Pb", "Xx"])
        self.assertIn("Xx", str(ctx.exception))
        np.testing.assert_allclose(masses_for(["Pb"]), [207.2])


class VacfTests(unittest.TestCase):
    def test_matches_the_direct_double_loop(self):
        rng = np.random.default_rng(3)
        velocities = rng.standard_normal((200, 4, 3))
        result = compute_vacf(velocities, max_lag=25, dt_fs=1.0, atom_reduction="sum")
        nsteps = velocities.shape[0]
        expected = np.array(
            [
                np.mean(
                    np.sum(velocities[: nsteps - lag] * velocities[lag:], axis=(1, 2))
                )
                for lag in range(25)
            ]
        )
        np.testing.assert_allclose(result.values, expected, rtol=1e-10, atol=1e-12)

    def test_mean_and_sum_reductions_differ_by_the_atom_count(self):
        rng = np.random.default_rng(4)
        velocities = rng.standard_normal((100, 5, 3))
        mean = compute_vacf(velocities, 10, 1.0, atom_reduction="mean").values
        total = compute_vacf(velocities, 10, 1.0, atom_reduction="sum").values
        np.testing.assert_allclose(total, 5 * mean, rtol=1e-10)

    def test_normalized_vacf_starts_at_one(self):
        rng = np.random.default_rng(5)
        result = compute_vacf(rng.standard_normal((100, 2, 3)), 10, 1.0)
        self.assertAlmostEqual(result.normalized[0], 1.0)

    def test_max_lag_bounds_are_enforced(self):
        rng = np.random.default_rng(6)
        velocities = rng.standard_normal((50, 2, 3))
        with self.assertRaises(VacfError):
            compute_vacf(velocities, max_lag=80, dt_fs=1.0)
        with self.assertRaises(VacfError):
            compute_vacf(velocities, max_lag=1, dt_fs=1.0)

    def test_wrong_dimensionality_is_rejected(self):
        with self.assertRaises(VacfError):
            compute_vacf(np.zeros((10, 3)), 5, 1.0)

    def test_segmented_average_matches_a_single_segment_when_there_is_one(self):
        rng = np.random.default_rng(7)
        velocities = rng.standard_normal((100, 3, 3))
        whole = compute_vacf(velocities, 20, 1.0).values
        segmented = compute_vacf_segmented(velocities, 20, 1.0, segment_length=100)
        np.testing.assert_allclose(segmented.values, whole, rtol=1e-10)
        self.assertEqual(segmented.n_segments, 1)

    def test_segmented_keeps_every_segment(self):
        rng = np.random.default_rng(8)
        velocities = rng.standard_normal((300, 2, 3))
        segmented = compute_vacf_segmented(velocities, 20, 1.0, segment_length=100)
        self.assertEqual(segmented.n_segments, 3)
        self.assertEqual(segmented.segment_values.shape, (3, 20))

    def test_segment_length_must_exceed_max_lag(self):
        rng = np.random.default_rng(9)
        velocities = rng.standard_normal((100, 2, 3))
        with self.assertRaises(VacfError):
            compute_vacf_segmented(velocities, 50, 1.0, segment_length=50)

    def test_full_lag_range_has_no_circular_wraparound(self):
        # max_lag == nsteps is the worst case: the final lag has a single time
        # origin, so any wrapped contribution would dominate it.
        rng = np.random.default_rng(10)
        for nsteps in (64, 100, 129):
            velocities = rng.standard_normal((nsteps, 3, 3))
            got = compute_vacf(velocities, nsteps, 1.0, atom_reduction="sum").values
            want = np.array(
                [
                    np.mean(
                        np.sum(
                            velocities[: nsteps - lag] * velocities[lag:], axis=(1, 2)
                        )
                    )
                    for lag in range(nsteps)
                ]
            )
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


class SmoothingTests(unittest.TestCase):
    def test_matches_scipy_including_the_edges(self):
        # numpy's "symmetric" padding is what scipy calls "reflect" (its
        # default). Using numpy's own "reflect" drops the edge sample and left
        # the lowest-frequency bins wrong by several percent.
        from scipy.ndimage import gaussian_filter1d

        rng = np.random.default_rng(21)
        values = rng.standard_normal(60)
        for sigma in (0.5, 1.0, 2.0, 3.0):
            np.testing.assert_allclose(
                gaussian_smooth(values, sigma),
                gaussian_filter1d(values, sigma),
                atol=1e-12,
                err_msg=f"sigma={sigma}",
            )

    def test_zero_sigma_is_a_no_op(self):
        values = np.arange(10.0)
        np.testing.assert_array_equal(gaussian_smooth(values, 0.0), values)

    def test_smoothing_preserves_length_and_total(self):
        values = np.exp(-((np.arange(200) - 100.0) ** 2) / 50.0)
        smoothed = gaussian_smooth(values, 2.0)
        self.assertEqual(smoothed.shape, values.shape)
        self.assertAlmostEqual(smoothed.sum(), values.sum(), places=6)


class SpectrumTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_known_oscillator_appears_at_its_own_frequency(self):
        path = write_xdatcar(
            self.root / "XDATCAR", frequencies_cm1=(120.0,), nframes=4000, counts=(2,)
        )
        trajectory = read_xdatcar(path)
        _, spectrum = trajectory_spectrum(trajectory, dt_fs=1.0, max_lag=3000)
        resolution = spectrum.meta["resolution_cm1"]
        self.assertLess(abs(_peak_frequency(spectrum) - 120.0), resolution)

    def test_two_oscillators_give_two_bands(self):
        path = write_xdatcar(
            self.root / "XDATCAR",
            frequencies_cm1=(100.0, 250.0),
            nframes=4000,
            counts=(2,),
        )
        trajectory = read_xdatcar(path)
        _, spectrum = trajectory_spectrum(trajectory, dt_fs=1.0, max_lag=2000)
        # Removing the centre-of-mass velocity of a two-atom cell leaves both
        # atoms carrying both modes, weighted by (amplitude x frequency)^2, so
        # the slower mode is the weaker of the two lines.
        peaks = peak_table(spectrum, 0.0, 500.0, rel_height=0.05)
        found = sorted(peak["frequency_cm1"] for peak in peaks)
        resolution = spectrum.meta["resolution_cm1"]
        self.assertTrue(any(abs(f - 100.0) < resolution for f in found), found)
        self.assertTrue(any(abs(f - 250.0) < resolution for f in found), found)

    def test_centre_of_mass_drift_adds_low_frequency_intensity(self):
        path = write_xdatcar(
            self.root / "XDATCAR",
            frequencies_cm1=(150.0,),
            nframes=4000,
            drift_ang_per_fs=0.002,
            counts=(2,),
        )
        trajectory = read_xdatcar(path)
        # The power convention is non-negative, so the two low-frequency
        # integrals can be compared without the ringing of a cosine transform.
        _, kept = trajectory_spectrum(
            trajectory, dt_fs=1.0, max_lag=2000, remove_com=False, convention="power"
        )
        _, removed = trajectory_spectrum(
            trajectory, dt_fs=1.0, max_lag=2000, remove_com=True, convention="power"
        )
        low_kept = band_integral(kept, 0.0, 50.0)
        low_removed = band_integral(removed, 0.0, 50.0)
        self.assertGreater(low_kept, 10 * low_removed)

    def test_resolution_follows_the_correlation_length(self):
        path = write_xdatcar(self.root / "XDATCAR", nframes=3000, counts=(2,))
        trajectory = read_xdatcar(path)
        _, short = trajectory_spectrum(trajectory, dt_fs=1.0, max_lag=500)
        _, long = trajectory_spectrum(trajectory, dt_fs=1.0, max_lag=2000)
        self.assertAlmostEqual(
            short.meta["resolution_cm1"], 4 * long.meta["resolution_cm1"], places=6
        )
        self.assertAlmostEqual(
            long.meta["resolution_cm1"], CM1_PER_INV_FS / 2000.0, places=6
        )

    def test_power_convention_is_the_squared_modulus(self):
        rng = np.random.default_rng(11)
        result = compute_vacf(rng.standard_normal((400, 2, 3)), 128, 1.0)
        cosine = spectral_density(result, convention="cosine")
        power = spectral_density(result, convention="power")
        self.assertTrue(np.all(power.intensity >= 0.0))
        self.assertFalse(np.array_equal(cosine.intensity, power.intensity))

    def test_unknown_convention_and_window_are_rejected(self):
        rng = np.random.default_rng(12)
        result = compute_vacf(rng.standard_normal((100, 2, 3)), 32, 1.0)
        with self.assertRaises(VacfError):
            spectral_density(result, convention="fourier")
        with self.assertRaises(VacfError):
            spectral_density(result, window="blackman")


class SpectraFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_and_describe_a_two_column_file(self):
        path = write_spectrum_file(self.root / "spectral_density_A.txt", (60.0, 300.0))
        spectrum = load_spectrum(path)
        self.assertEqual(spectrum.label, "A")
        self.assertEqual(spectrum.convention, "unknown")
        summary = describe(spectrum, analysis_range=(0.0, 800.0))
        peaks = sorted(p["frequency_cm1"] for p in summary["peaks"])
        self.assertEqual(len(peaks), 2)
        self.assertAlmostEqual(peaks[0], 60.0, delta=2.0)
        self.assertAlmostEqual(peaks[1], 300.0, delta=2.0)
        self.assertGreater(summary["centroid_cm1"], 60.0)
        self.assertLess(summary["centroid_cm1"], 300.0)

    def test_band_fractions_sum_to_the_range_integral(self):
        path = write_spectrum_file(self.root / "spectral_density_B.txt", (60.0, 300.0))
        summary = describe(load_spectrum(path), analysis_range=(0.0, 800.0))
        total = sum(band["fraction_of_range"] for band in summary["bands"])
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_three_column_file_is_rejected(self):
        path = self.root / "wide.txt"
        np.savetxt(path, np.zeros((10, 3)))
        with self.assertRaises(SpectrumError):
            load_spectrum(path)

    def test_unsorted_frequency_is_rejected(self):
        path = self.root / "unsorted.txt"
        np.savetxt(path, np.column_stack([[3.0, 1.0, 2.0], [1.0, 1.0, 1.0]]))
        with self.assertRaises(SpectrumError):
            load_spectrum(path)

    def test_comparison_needs_a_shared_grid(self):
        first = load_spectrum(write_spectrum_file(self.root / "spectral_density_a.txt", (50.0,)))
        second = load_spectrum(write_spectrum_file(self.root / "spectral_density_b.txt", (150.0,)))
        self.assertTrue(shared_grid([first, second]))
        payload = compare([first, second], reference_label="a")
        self.assertTrue(payload["shared_grid"])
        delta = payload["differences"][0]
        self.assertEqual(delta["label"], "b")
        self.assertGreater(delta["delta_centroid_cm1"], 50.0)

    def test_mismatched_grids_skip_the_pointwise_comparison(self):
        first = load_spectrum(write_spectrum_file(self.root / "spectral_density_a.txt", (50.0,)))
        path = self.root / "spectral_density_c.txt"
        np.savetxt(
            path,
            np.column_stack([np.arange(1, 500) * 2.0, np.ones(499)]),
            header="Frequency(cm^-1)\tSpectral_Density",
            comments="",
        )
        second = load_spectrum(path)
        payload = compare([first, second], reference_label="a")
        self.assertFalse(payload["shared_grid"])
        self.assertNotIn("differences", payload)

    def test_unknown_reference_is_rejected(self):
        first = load_spectrum(write_spectrum_file(self.root / "spectral_density_a.txt", (50.0,)))
        second = load_spectrum(write_spectrum_file(self.root / "spectral_density_b.txt", (80.0,)))
        with self.assertRaises(SpectrumError):
            compare([first, second], reference_label="missing")

    def test_empty_comparison_is_rejected(self):
        with self.assertRaises(SpectrumError):
            compare([])


if __name__ == "__main__":
    unittest.main()
