import bootstrap  # noqa: F401

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.cli import main
from namd_analysis.io.hefei import read_shprop
from namd_analysis.master import (
    AVERAGED,
    COPIED,
    MasterError,
    build_master,
    write_master,
)
from namd_analysis.populations import StateMap

CONFIG = {
    "name": "master_two_state",
    "time_column": 0,
    "time_unit": "fs",
    "population_columns": [2, 3],
    "groups": {"VBM": [2], "CBM": [3]},
    "complete_population": True,
    "recombined_group": "VBM",
}


def _state_map(**overrides):
    payload = dict(CONFIG)
    payload.update(overrides)
    return StateMap.from_dict(payload)


def write_files(root, populations, energies=None, time=None):
    """One SHPROP per entry of ``populations``, shape (nfiles, nrows, nstates)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    populations = np.asarray(populations, dtype=float)
    n_files, n_rows, _ = populations.shape
    if time is None:
        time = np.arange(n_rows, dtype=float) * 1000.0
    time = np.asarray(time, dtype=float)
    if energies is None:
        energies = np.full((n_files, n_rows), -1.5)
    energies = np.asarray(energies, dtype=float)
    paths = []
    for index in range(n_files):
        grid = time if time.ndim == 1 else time[index]
        table = np.column_stack([grid, energies[index], populations[index]])
        path = root / f"SHPROP.{index + 1}"
        np.savetxt(path, table, fmt="%.17g")
        paths.append(path)
    return paths


def two_state(n_files=4, n_rows=50, seed=0):
    """Populations that sum to one exactly in every file, but differ per file."""
    rng = np.random.default_rng(seed)
    time = np.arange(n_rows, dtype=float)
    out = np.empty((n_files, n_rows, 2))
    for index in range(n_files):
        cbm = np.exp(-time / (10.0 + index)) * (0.9 + 0.1 * rng.random())
        out[index, :, 1] = cbm
        out[index, :, 0] = 1.0 - cbm
    return out


class AveragingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_population_column_is_averaged(self):
        # The legacy SHPROP_avg.sh averaged column 4 alone and copied the rest
        # from the first file. Reproducing that is the failure this guards.
        populations = two_state()
        paths = write_files(self.root / "run", populations)
        master = build_master(paths, _state_map())
        expected = populations.mean(axis=0)
        np.testing.assert_allclose(master.table[:, 2], expected[:, 0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(master.table[:, 3], expected[:, 1], rtol=0, atol=1e-15)
        for index, column in enumerate((2, 3)):
            self.assertFalse(
                np.allclose(master.table[:, column], populations[0][:, index]),
                f"column {column} matches the first file, so it was copied",
            )

    def test_the_time_column_is_copied_exactly(self):
        time = np.arange(40, dtype=float) * 1234.5
        paths = write_files(self.root / "run", two_state(n_rows=40), time=time)
        master = build_master(paths, _state_map())
        np.testing.assert_array_equal(master.time_values, time)

    def test_a_mismatched_time_grid_is_rejected(self):
        populations = two_state(n_files=3, n_rows=30)
        grids = np.stack(
            [
                np.arange(30, dtype=float),
                np.arange(30, dtype=float),
                np.arange(30, dtype=float) + 0.5,
            ]
        )
        paths = write_files(self.root / "run", populations, time=grids)
        with self.assertRaises(MasterError) as ctx:
            build_master(paths, _state_map())
        message = str(ctx.exception)
        self.assertIn("identical time grid", message)
        self.assertIn("No interpolation", message)

    def test_mismatched_lengths_are_not_truncated(self):
        long = write_files(self.root / "a", two_state(n_files=1, n_rows=40))
        short = write_files(self.root / "b", two_state(n_files=1, n_rows=30, seed=1))
        with self.assertRaises(MasterError) as ctx:
            build_master(long + short, _state_map())
        self.assertIn("different shapes", str(ctx.exception))

    def test_duplicate_paths_are_rejected(self):
        paths = write_files(self.root / "run", two_state(n_files=2))
        with self.assertRaises(MasterError) as ctx:
            build_master([paths[0], paths[1], paths[0]], _state_map())
        self.assertIn("more than once", str(ctx.exception))

    def test_byte_identical_copies_are_rejected(self):
        paths = write_files(self.root / "run", two_state(n_files=2))
        copy = self.root / "run" / "SHPROP.copy"
        copy.write_bytes(paths[0].read_bytes())
        with self.assertRaises(MasterError) as ctx:
            build_master(paths + [copy], _state_map())
        self.assertIn("byte-identical", str(ctx.exception))

    def test_a_single_bad_file_blocks_the_master(self):
        populations = two_state(n_files=3)
        populations[1, 10, 1] += 0.2  # this file no longer conserves population
        paths = write_files(self.root / "run", populations)
        with self.assertRaises(MasterError) as ctx:
            build_master(paths, _state_map())
        message = str(ctx.exception)
        self.assertIn("SHPROP.2", message)
        self.assertIn("complete population sums to", message)

    def test_master_population_is_conserved(self):
        paths = write_files(self.root / "run", two_state(n_files=5))
        master = build_master(paths, _state_map())
        totals = master.table[:, master.population_columns].sum(axis=1)
        np.testing.assert_allclose(totals, 1.0, atol=1e-12)
        self.assertLess(
            master.master_conservation["max_abs_deviation_from_one"], 1e-12
        )
        self.assertEqual(len(master.per_file_conservation), 5)

    def test_an_identical_extra_column_is_copied(self):
        populations = two_state(n_files=3)
        energies = np.tile(np.linspace(-2.0, -1.0, populations.shape[1]), (3, 1))
        paths = write_files(self.root / "run", populations, energies=energies)
        master = build_master(paths, _state_map())
        self.assertEqual(master.extra_columns, [1])
        self.assertEqual(master.extra_column_policy[1], COPIED)
        np.testing.assert_array_equal(master.table[:, 1], energies[0])

    def test_a_differing_extra_column_is_refused_without_a_policy(self):
        populations = two_state(n_files=3)
        energies = np.stack(
            [np.full(populations.shape[1], -1.5 - 0.1 * index) for index in range(3)]
        )
        paths = write_files(self.root / "run", populations, energies=energies)
        with self.assertRaises(MasterError) as ctx:
            build_master(paths, _state_map())
        message = str(ctx.exception)
        self.assertIn("column 1", message)
        self.assertIn("--average-extra-columns", message)

    def test_a_differing_extra_column_is_averaged_under_an_explicit_policy(self):
        populations = two_state(n_files=3)
        energies = np.stack(
            [np.full(populations.shape[1], -1.5 - 0.1 * index) for index in range(3)]
        )
        paths = write_files(self.root / "run", populations, energies=energies)
        master = build_master(paths, _state_map(), average_extra_columns=True)
        self.assertEqual(master.extra_column_policy[1], AVERAGED)
        np.testing.assert_allclose(master.table[:, 1], energies.mean(axis=0))

    def test_group_sem_sums_within_a_file_before_taking_the_spread(self):
        # Two states in one group: the group SEM is the spread of the summed
        # per-file series, never a combination of the marginal state SEMs.
        rng = np.random.default_rng(3)
        n_files, n_rows = 4, 25
        populations = np.empty((n_files, n_rows, 3))
        for index in range(n_files):
            a = 0.3 + 0.05 * rng.standard_normal(n_rows)
            b = 0.2 + 0.05 * rng.standard_normal(n_rows)
            populations[index, :, 0] = a
            populations[index, :, 1] = b
            populations[index, :, 2] = 1.0 - a - b
        paths = write_files(self.root / "run", populations)
        state_map = _state_map(
            population_columns=[2, 3, 4],
            groups={"donor": [2, 3], "acceptor": [4]},
            recombined_group="acceptor",
        )
        master = build_master(paths, state_map)
        per_file_sum = populations[:, :, 0] + populations[:, :, 1]
        expected = per_file_sum.std(axis=0, ddof=1) / np.sqrt(n_files)
        np.testing.assert_allclose(master.group_sem["donor"], expected)
        marginal = np.sqrt(
            master.population_sem[:, 0] ** 2 + master.population_sem[:, 1] ** 2
        )
        self.assertFalse(np.allclose(master.group_sem["donor"], marginal))

    def test_a_single_file_has_no_sem(self):
        paths = write_files(self.root / "run", two_state(n_files=1))
        master = build_master(paths, _state_map())
        self.assertIsNone(master.population_sem)
        self.assertEqual(master.group_sem, {})

    def test_a_partial_map_refuses_an_undeclared_population_column(self):
        # A partial map leaves the sibling population columns undeclared, and
        # those certainly differ between files. Copying the first file's copy
        # beside an averaged column is exactly the legacy defect.
        populations = two_state(n_files=3)
        paths = write_files(self.root / "run", populations)
        partial = _state_map(
            population_columns=[3],
            groups={"CBM": [3]},
            complete_population=False,
            recombined_group=None,
        )
        with self.assertRaises(MasterError) as ctx:
            build_master(paths, partial)
        self.assertIn("column 2", str(ctx.exception))

    def test_a_partial_map_works_when_the_other_columns_agree(self):
        populations = two_state(n_files=3)
        populations[:, :, 0] = 0.25  # the undeclared column now agrees
        paths = write_files(self.root / "run", populations)
        partial = _state_map(
            population_columns=[3],
            groups={"CBM": [3]},
            complete_population=False,
            recombined_group=None,
        )
        master = build_master(paths, partial)
        self.assertEqual(master.population_columns, [3])
        self.assertIn(2, master.extra_columns)
        self.assertEqual(master.extra_column_policy[2], COPIED)
        self.assertEqual(
            master.as_dict()["conservation"]["checked"],
            "range only; a partial map makes no conservation claim",
        )

    def test_no_files_is_an_error(self):
        with self.assertRaises(MasterError):
            build_master([], _state_map())


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_written_master_reads_back_bit_for_bit(self):
        paths = write_files(self.root / "run", two_state(n_files=4))
        master = build_master(paths, _state_map())
        written = write_master(master, self.root / "SHPROP.master")
        readback = read_shprop(written)
        np.testing.assert_array_equal(readback, master.table)

    def test_the_master_can_be_read_by_the_existing_population_reader(self):
        from namd_analysis.populations import load_population_set

        paths = write_files(self.root / "run", two_state(n_files=4))
        master = build_master(paths, _state_map())
        written = write_master(master, self.root / "SHPROP.master")
        population = load_population_set([written], _state_map())
        self.assertEqual(population.n_files, 1)
        np.testing.assert_allclose(
            population.mean[:, [2, 3]].sum(axis=1), 1.0, atol=1e-12
        )


class AverageShpropCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = self.root / "map.json"
        self.config.write_text(json.dumps(CONFIG), encoding="utf-8")
        self.paths = write_files(self.root / "run", two_state(n_files=4))

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, out, *extra):
        return main(
            [
                "average-shprop",
                "--files", str(self.root / "run" / "SHPROP.*"),
                "--config", str(self.config),
                "--out", str(out),
                *extra,
            ]
        )

    def test_end_to_end_writes_every_artifact(self):
        out = self.root / "master"
        self.assertEqual(self._run(out), 0)
        for name in ("SHPROP.master", "report.json", "population_sem.csv",
                     "input_files.csv"):
            self.assertTrue((out / name).is_file(), name)

    def test_provenance_fingerprints_every_source_file(self):
        out = self.root / "master"
        self._run(out)
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        inputs = report["inputs"]
        self.assertEqual(len(inputs), len(self.paths))
        recorded = {Path(item["path"]).name for item in inputs}
        self.assertEqual(recorded, {p.name for p in self.paths})
        for item in inputs:
            self.assertEqual(len(item["sha256"]), 64)
            self.assertGreater(item["bytes"], 0)
            self.assertIn("modified_utc", item)
            self.assertEqual(item["n_rows"], 50)
            self.assertEqual(item["n_columns"], 4)

    def test_report_records_the_rule_the_map_and_the_environment(self):
        out = self.root / "master"
        self._run(out)
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        master = report["master"]
        self.assertEqual(master["n_source_files"], 4)
        self.assertIn("(1/N) sum_r", master["averaging_rule"]["population_columns"])
        self.assertIn("copied through", master["averaging_rule"]["time_column"])
        self.assertEqual(master["averaging_rule"]["other_columns"]["1"], COPIED)
        self.assertEqual(
            master["state_map_fingerprint"], StateMap.from_dict(CONFIG).fingerprint()
        )
        self.assertEqual(len(master["conservation"]["per_source_file"]), 4)
        self.assertEqual(report["environment"]["tool"], "namd-analysis")
        self.assertIn("argv", report["environment"])
        self.assertEqual(report["options"]["average_extra_columns"], False)
        self.assertIn("launcher_manifests", report)
        self.assertTrue(master["readback_verified"])

    def test_the_sem_table_carries_columns_and_groups(self):
        out = self.root / "master"
        self._run(out)
        header = (out / "population_sem.csv").read_text(encoding="utf-8").splitlines()[0]
        for field in ("time", "column_2_sem", "column_3_sem", "group_CBM_mean",
                      "group_CBM_sem", "group_VBM_sem"):
            self.assertIn(field, header)
        # SEM must not leak into the SHPROP-compatible master.
        self.assertEqual(read_shprop(out / "SHPROP.master").shape[1], 4)

    def test_the_input_table_lists_each_file_once(self):
        out = self.root / "master"
        self._run(out)
        lines = (out / "input_files.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1 + len(self.paths))
        self.assertIn("sha256", lines[0])

    def test_a_differing_extra_column_fails_the_command(self):
        populations = two_state(n_files=3, seed=9)
        energies = np.stack(
            [np.full(populations.shape[1], -1.5 - index) for index in range(3)]
        )
        write_files(self.root / "other", populations, energies=energies)
        code = main(
            [
                "average-shprop",
                "--files", str(self.root / "other" / "SHPROP.*"),
                "--config", str(self.config),
                "--out", str(self.root / "fails"),
            ]
        )
        self.assertEqual(code, 2)

    def test_the_master_feeds_the_populations_command(self):
        out = self.root / "master"
        self._run(out)
        code = main(
            [
                "populations",
                "--files", str(out / "SHPROP.master"),
                "--config", str(self.config),
                "--out", str(self.root / "pop"),
            ]
        )
        self.assertEqual(code, 0)
        report = json.loads(
            (self.root / "pop" / "report.json").read_text(encoding="utf-8")
        )
        self.assertLess(report["conservation"]["max_abs_deviation_from_one"], 1e-12)


if __name__ == "__main__":
    unittest.main()
