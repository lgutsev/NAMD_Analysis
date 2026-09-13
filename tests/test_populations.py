import bootstrap  # noqa: F401

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.populations import (
    ConfigError,
    InputMismatchError,
    StateMap,
    group_series,
    load_population_set,
    survival,
)
from synthetic import write_shprop_set

TWO_STATE = {
    "name": "two_state",
    "time_column": 0,
    "time_unit": "fs",
    "population_columns": [2, 3],
    "groups": {"VBM": [2], "CBM": [3]},
    "complete_population": True,
    "recombined_group": "VBM",
}


class StateMapTests(unittest.TestCase):
    def test_overlapping_groups_are_rejected(self):
        payload = dict(TWO_STATE, groups={"A": [2, 3], "B": [3]})
        with self.assertRaises(ConfigError) as ctx:
            StateMap.from_dict(payload)
        self.assertIn("disjoint", str(ctx.exception))

    def test_group_column_must_be_declared(self):
        payload = dict(TWO_STATE, groups={"A": [2], "B": [9]})
        with self.assertRaises(ConfigError) as ctx:
            StateMap.from_dict(payload)
        self.assertIn("population_columns", str(ctx.exception))

    def test_time_column_cannot_also_be_a_population(self):
        payload = dict(TWO_STATE, population_columns=[0, 2, 3])
        with self.assertRaises(ConfigError):
            StateMap.from_dict(payload)

    def test_survival_requires_a_complete_map(self):
        payload = dict(TWO_STATE, complete_population=False)
        with self.assertRaises(ConfigError) as ctx:
            StateMap.from_dict(payload)
        self.assertIn("complete_population", str(ctx.exception))

    def test_partial_map_without_survival_is_allowed(self):
        payload = {
            "name": "partial",
            "time_column": 0,
            "time_unit": "fs",
            "population_columns": [2, 3, 4],
            "groups": {"CBM": [4]},
        }
        state_map = StateMap.from_dict(payload)
        self.assertEqual(state_map.ungrouped_columns(), [2, 3])

    def test_unknown_time_unit_is_rejected(self):
        with self.assertRaises(ConfigError):
            StateMap.from_dict(dict(TWO_STATE, time_unit="microseconds"))

    def test_missing_key_names_the_key(self):
        with self.assertRaises(ConfigError) as ctx:
            StateMap.from_dict({"time_column": 0, "groups": {"A": [1]}})
        self.assertIn("population_columns", str(ctx.exception))


class AveragingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_map = StateMap.from_dict(TWO_STATE)

    def tearDown(self):
        self.tmp.cleanup()

    def test_population_is_conserved_and_time_is_converted(self):
        paths = write_shprop_set(self.root, n_files=3, nsteps=50, dt_fs=1000.0)
        population = load_population_set(paths, self.state_map)
        self.assertLess(population.conservation["max_abs_deviation_from_one"], 1e-12)
        self.assertAlmostEqual(population.time_ns[0], 0.0)
        self.assertAlmostEqual(population.time_ns[1], 1e-3)

    def test_every_column_is_averaged_not_only_one(self):
        # Two files that differ in both population columns; a script that
        # averaged one column would leave the other at the first file's value.
        time = np.arange(20) * 1000.0
        first = np.column_stack([time, np.full(20, -1.0), np.full(20, 0.2), np.full(20, 0.8)])
        second = np.column_stack([time, np.full(20, -1.0), np.full(20, 0.6), np.full(20, 0.4)])
        paths = []
        for index, table in enumerate((first, second)):
            path = self.root / f"SHPROP.{index}"
            np.savetxt(path, table)
            paths.append(path)
        population = load_population_set(paths, self.state_map)
        series = {s.name: s for s in group_series(population, self.state_map)}
        np.testing.assert_allclose(series["VBM"].values, 0.4)
        np.testing.assert_allclose(series["CBM"].values, 0.6)

    def test_sem_is_the_between_file_standard_error(self):
        time = np.arange(10) * 1000.0
        values = (0.2, 0.4, 0.6)
        paths = []
        for index, cbm in enumerate(values):
            table = np.column_stack(
                [time, np.full(10, -1.0), np.full(10, 1.0 - cbm), np.full(10, cbm)]
            )
            path = self.root / f"SHPROP.{index}"
            np.savetxt(path, table)
            paths.append(path)
        population = load_population_set(paths, self.state_map)
        expected = np.std(values, ddof=1) / np.sqrt(len(values))
        np.testing.assert_allclose(population.sem[:, 3], expected)

    def test_sem_is_omitted_for_a_single_file(self):
        paths = write_shprop_set(self.root, n_files=1, nsteps=20)
        population = load_population_set(paths, self.state_map)
        self.assertIsNone(population.sem)
        self.assertIsNone(group_series(population, self.state_map)[0].sem)

    def test_different_shapes_are_rejected(self):
        long_paths = write_shprop_set(self.root / "a", n_files=1, nsteps=30)
        short_paths = write_shprop_set(self.root / "b", n_files=1, nsteps=20)
        with self.assertRaises(InputMismatchError) as ctx:
            load_population_set(long_paths + short_paths, self.state_map)
        self.assertIn("different shapes", str(ctx.exception))

    def test_different_time_grids_are_rejected(self):
        paths = write_shprop_set(self.root, n_files=2, nsteps=20, dt_fs=1000.0)
        table = np.loadtxt(paths[1])
        table[:, 0] += 0.5
        np.savetxt(paths[1], table)
        with self.assertRaises(InputMismatchError) as ctx:
            load_population_set(paths, self.state_map)
        self.assertIn("identical time grid", str(ctx.exception))

    def test_configuration_beyond_the_file_width_is_rejected(self):
        paths = write_shprop_set(self.root, n_files=1, nsteps=20)
        wide = StateMap.from_dict(
            dict(TWO_STATE, population_columns=[2, 3, 4], groups={"A": [2], "B": [4]},
                 complete_population=False, recombined_group=None)
        )
        with self.assertRaises(InputMismatchError) as ctx:
            load_population_set(paths, wide)
        self.assertIn("column 4", str(ctx.exception))

    def test_empty_input_is_rejected(self):
        with self.assertRaises(InputMismatchError):
            load_population_set([], self.state_map)

    def test_survival_is_one_minus_the_recombined_group(self):
        paths = write_shprop_set(self.root, n_files=2, nsteps=30)
        population = load_population_set(paths, self.state_map)
        series = {s.name: s for s in group_series(population, self.state_map)}
        np.testing.assert_allclose(
            survival(population, self.state_map), series["CBM"].values, atol=1e-12
        )

    def test_survival_is_none_for_a_partial_map(self):
        paths = write_shprop_set(self.root, n_files=1, nsteps=20)
        partial = StateMap.from_dict(
            {
                "name": "partial",
                "time_column": 0,
                "time_unit": "fs",
                "population_columns": [2, 3],
                "groups": {"CBM": [3]},
            }
        )
        population = load_population_set(paths, partial)
        self.assertIsNone(survival(population, partial))

    def test_group_summary_reports_net_change_and_integral(self):
        paths = write_shprop_set(self.root, n_files=1, nsteps=100, dt_fs=1000.0,
                                 tau_fs=50000.0)
        population = load_population_set(paths, self.state_map)
        series = {s.name: s for s in group_series(population, self.state_map)}
        summary = series["CBM"].summary(population.time_ns)
        self.assertLess(summary["net_change"], 0.0)
        self.assertAlmostEqual(summary["initial"], 1.0)
        self.assertGreater(summary["window_integral_ns"], 0.0)


class ExampleConfigTests(unittest.TestCase):
    def test_shipped_examples_validate(self):
        root = Path(__file__).resolve().parents[1] / "examples"
        for name in ("two_state.json", "interface_six_state.json"):
            payload = json.loads((root / name).read_text(encoding="utf-8"))
            state_map = StateMap.from_dict(payload)
            self.assertEqual(state_map.ungrouped_columns(), [])


if __name__ == "__main__":
    unittest.main()
