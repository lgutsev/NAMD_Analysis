import bootstrap  # noqa: F401

import unittest

import numpy as np

from namd_analysis.fitting import FitError, fit_single_exponential


class ExponentialFitTests(unittest.TestCase):
    def test_recovers_an_analytic_lifetime(self):
        time = np.linspace(0.0, 5.0, 500)
        values = 0.87 * np.exp(-time / 1.3)
        fit = fit_single_exponential(time, values, group="CBM")
        self.assertAlmostEqual(fit.tau, 1.3, places=6)
        self.assertAlmostEqual(fit.amplitude, 0.87, places=10)
        self.assertGreater(fit.r_squared, 0.999999)
        self.assertEqual(fit.warnings, [])

    def test_prefactor_is_the_first_sample_in_the_window(self):
        time = np.linspace(0.0, 5.0, 500)
        values = 1.0 * np.exp(-time / 2.0)
        fit = fit_single_exponential(time, values, t_start=1.0, t_end=4.0)
        first_in_window = float(values[time >= 1.0][0])
        self.assertAlmostEqual(fit.amplitude, first_in_window, places=12)
        self.assertAlmostEqual(fit.tau, 2.0, places=6)
        self.assertAlmostEqual(fit.window[0], 1.0, places=2)

    def test_window_restricts_the_samples_used(self):
        time = np.linspace(0.0, 10.0, 1001)
        values = np.exp(-time / 3.0)
        fit = fit_single_exponential(time, values, t_start=2.0, t_end=5.0)
        self.assertEqual(fit.n_points, 301)

    def test_flat_population_is_rejected(self):
        time = np.linspace(0.0, 5.0, 100)
        with self.assertRaises(FitError) as ctx:
            fit_single_exponential(time, np.full(100, 0.25))
        self.assertIn("does not decay", str(ctx.exception))

    def test_rising_population_is_rejected(self):
        time = np.linspace(0.0, 5.0, 100)
        with self.assertRaises(FitError):
            fit_single_exponential(time, 0.1 + 0.05 * time)

    def test_non_positive_start_is_rejected(self):
        time = np.linspace(0.0, 5.0, 100)
        values = np.linspace(0.0, -1.0, 100)
        with self.assertRaises(FitError) as ctx:
            fit_single_exponential(time, values)
        self.assertIn("not positive", str(ctx.exception))

    def test_too_few_points_is_rejected(self):
        time = np.linspace(0.0, 1.0, 100)
        values = np.exp(-time)
        with self.assertRaises(FitError) as ctx:
            fit_single_exponential(time, values, t_start=0.0, t_end=0.01)
        self.assertIn("fewer than", str(ctx.exception))

    def test_empty_window_is_rejected(self):
        time = np.linspace(0.0, 1.0, 50)
        with self.assertRaises(FitError):
            fit_single_exponential(time, np.exp(-time), t_start=0.8, t_end=0.2)

    def test_long_extrapolation_is_flagged(self):
        # A 10 ns window on a 650 ns decay: the archived campaigns fitted
        # exactly this kind of trace and reported R^2 close to one.
        time = np.linspace(0.0, 10.0, 1000)
        values = np.exp(-time / 650.0)
        fit = fit_single_exponential(time, values)
        self.assertGreater(fit.r_squared, 0.99)
        self.assertLess(fit.windows_per_tau, 1.0)
        self.assertTrue(any("extrapolation" in w for w in fit.warnings))
        self.assertTrue(any("falls by only" in w for w in fit.warnings))

    def test_poor_fit_is_flagged(self):
        time = np.linspace(0.0, 5.0, 400)
        values = 0.5 * np.exp(-time / 0.2) + 0.4 * np.exp(-time / 8.0)
        fit = fit_single_exponential(time, values)
        self.assertTrue(any("R^2" in w for w in fit.warnings))

    def test_legacy_style_fit_reproduces_a_negative_r_squared(self):
        # A trace that grows then falls: the legacy script fitted exp(-t/A)
        # against the normalized trace and reported R^2 far below zero.
        time = np.linspace(0.0, 5.0, 400)
        values = np.exp(-time / 0.05) + 0.3
        values[:20] = np.linspace(0.05, values[20], 20)
        fit = fit_single_exponential(time, values, t_start=float(time[20]))
        self.assertLess(fit.r_squared, 0.9)
        self.assertTrue(fit.warnings)

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaises(FitError):
            fit_single_exponential(np.arange(10.0), np.arange(5.0))

    def test_residual_diagnostics_are_reported(self):
        time = np.linspace(0.0, 5.0, 200)
        values = np.exp(-time / 1.0)
        fit = fit_single_exponential(time, values)
        self.assertLess(fit.rms_residual, 1e-9)
        self.assertLess(fit.max_abs_residual, 1e-8)
        self.assertGreater(fit.decay_fraction_in_window, 0.99)
        payload = fit.as_dict()
        self.assertEqual(payload["tau_unit"], "ns")
        self.assertEqual(len(payload["window"]), 2)


if __name__ == "__main__":
    unittest.main()
