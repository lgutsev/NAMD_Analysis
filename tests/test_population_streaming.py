"""Streaming the population set must change residency and nothing else.

The whole-file loader holds every history at once, which a campaign of
hundreds cannot do.  The streaming loader is only worth having if it is
numerically identical to it, and refuses everything it refuses.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.populations import (
    InputMismatchError,
    StateMap,
    load_population_set,
    load_population_set_streaming,
)

NCOL = 8
POP_COLUMNS = [2, 3, 4, 5, 6, 7]


def write_state_map(root, complete=True):
    path = Path(root) / "state_map.json"
    path.write_text(
        json.dumps(
            {
                "name": "stream_test",
                "time_column": 0,
                "time_unit": "fs",
                "population_columns": POP_COLUMNS,
                "groups": {"VBM": [2], "BCF": [3], "PCBM": [4, 5, 6], "CBM": [7]},
                "complete_population": complete,
                "recombined_group": "VBM",
            }
        ),
        encoding="utf-8",
    )
    return StateMap.from_json(path)


def write_history(path, nrows, seed, normalize=True, times=None):
    rng = np.random.default_rng(seed)
    pops = rng.random((nrows, len(POP_COLUMNS)))
    if normalize:
        pops /= pops.sum(axis=1, keepdims=True)
    t = np.arange(1, nrows + 1) * 1000.0 if times is None else times
    with Path(path).open("w", encoding="utf-8") as fh:
        for i in range(nrows):
            fh.write(" ".join(f"{v:.12E}" for v in [t[i], -0.8, *pops[i]]) + "\n")
    return Path(path)


class EquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_map = write_state_map(self.root)
        self.paths = [
            write_history(self.root / f"SHPROP.{k}", 500, seed=k) for k in (37, 171, 425)
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def test_mean_sem_and_time_are_identical(self):
        whole = load_population_set(self.paths, self.state_map)
        for chunk_rows in (1, 7, 128, 499, 500, 5000):
            stream = load_population_set_streaming(
                self.paths, self.state_map, chunk_rows=chunk_rows
            )
            np.testing.assert_allclose(
                stream.mean, whole.mean, rtol=0, atol=1e-12,
                err_msg=f"mean differs at chunk_rows={chunk_rows}",
            )
            np.testing.assert_allclose(
                stream.sem, whole.sem, rtol=0, atol=1e-12,
                err_msg=f"sem differs at chunk_rows={chunk_rows}",
            )
            np.testing.assert_allclose(stream.time_ns, whole.time_ns, rtol=0, atol=1e-12)
            np.testing.assert_allclose(
                stream.total_population, whole.total_population, rtol=0, atol=1e-12
            )

    def test_the_conservation_block_matches(self):
        whole = load_population_set(self.paths, self.state_map)
        stream = load_population_set_streaming(self.paths, self.state_map, chunk_rows=64)
        for key in whole.conservation:
            self.assertAlmostEqual(
                stream.conservation[key], whole.conservation[key], places=12, msg=key
            )

    def test_a_single_history_has_no_sem(self):
        stream = load_population_set_streaming(self.paths[:1], self.state_map)
        self.assertIsNone(stream.sem)
        self.assertEqual(stream.n_files, 1)

    def test_the_stack_is_dropped_unless_asked_for(self):
        # It is O(nfiles x nrows x ncols) and cannot scale; it must be declared.
        self.assertIsNone(
            load_population_set_streaming(self.paths, self.state_map).stack
        )
        kept = load_population_set_streaming(
            self.paths, self.state_map, retain_stack=True
        )
        self.assertEqual(kept.stack.shape, (3, 500, NCOL))
        whole = load_population_set(self.paths, self.state_map)
        np.testing.assert_allclose(kept.stack, whole.stack, rtol=0, atol=1e-12)


class RefusalTests(unittest.TestCase):
    """Every check the whole-file path makes, preserved."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_map = write_state_map(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _both_refuse(self, paths, fragment):
        for loader in (load_population_set, load_population_set_streaming):
            with self.assertRaises(InputMismatchError, msg=loader.__name__) as ctx:
                loader(paths, self.state_map)
            self.assertIn(fragment, str(ctx.exception).lower())

    def test_differing_lengths_are_refused_not_truncated(self):
        paths = [
            write_history(self.root / "SHPROP.1", 500, 1),
            write_history(self.root / "SHPROP.2", 400, 2),
        ]
        self._both_refuse(paths, "different shapes")

    def test_a_differing_time_grid_is_refused_not_interpolated(self):
        a = write_history(self.root / "SHPROP.1", 300, 1)
        b = write_history(
            self.root / "SHPROP.2", 300, 2,
            times=np.arange(1, 301) * 1000.0 + 0.5,
        )
        self._both_refuse([a, b], "time grid")

    def test_populations_outside_the_unit_range_are_refused(self):
        a = write_history(self.root / "SHPROP.1", 200, 1)
        b = write_history(self.root / "SHPROP.2", 200, 2, normalize=False)
        rows = np.loadtxt(b)
        rows[10, 3] = 1.9
        np.savetxt(b, rows, fmt="%.12E")
        self._both_refuse([a, b], "outside [0,1]")

    def test_unconserved_populations_are_refused_not_renormalized(self):
        a = write_history(self.root / "SHPROP.1", 200, 1)
        b = write_history(self.root / "SHPROP.2", 200, 2, normalize=False)
        self._both_refuse([a, b], "sum to one")

    def test_duplicate_files_are_refused(self):
        a = write_history(self.root / "SHPROP.1", 200, 1)
        self._both_refuse([a, a], "duplicate")

    def test_no_files_is_refused(self):
        self._both_refuse([], "no shprop files")

    def test_a_column_beyond_the_table_is_refused(self):
        a = write_history(self.root / "SHPROP.1", 200, 1)
        b = write_history(self.root / "SHPROP.2", 200, 2)
        # Consistent map, but naming a column the files do not have.
        wide = write_state_map(self.root)
        wide.population_columns = [2, 3, 4, 5, 6, 99]
        wide.groups = {"VBM": [2], "BCF": [3], "PCBM": [4, 5, 6], "CBM": [99]}
        for loader in (load_population_set, load_population_set_streaming):
            with self.assertRaises(InputMismatchError, msg=loader.__name__) as ctx:
                loader([a, b], wide)
            self.assertIn("99", str(ctx.exception))


class ScaleTests(unittest.TestCase):
    """Residency must not grow with the number of histories."""

    def test_the_accumulator_does_not_grow_with_file_count(self):
        import tracemalloc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_map = write_state_map(root)
            peaks = {}
            for nfiles in (2, 8):
                paths = [
                    write_history(root / f"SHPROP.{k}", 400, seed=k)
                    for k in range(nfiles)
                ]
                tracemalloc.start()
                load_population_set_streaming(paths, state_map, chunk_rows=100)
                peaks[nfiles] = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                for p in paths:
                    p.unlink()
            # Four times the histories must not cost four times the memory.
            self.assertLess(
                peaks[8], peaks[2] * 2.0,
                f"residency grew with file count: {peaks}",
            )


if __name__ == "__main__":
    unittest.main()
