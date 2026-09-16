"""v0.5.2: chunked SHPROP I/O must be cheap, and must change no number.

The first real campaign OOM-killed during preflight on five ~889 MB histories.
These tests pin both halves of the fix: preflight never materializes a table,
and the streamed analysis is numerically identical to reading each table whole.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from character_fixtures import swap_campaign, write_manifest, write_procar, write_shprop
from namd_analysis.character import (
    AUTO_SINGLE_CHUNK_BYTES,
    DEFAULT_CHUNK_ROWS,
    AtomGroupMap,
    CharacterError,
    character_populations,
    character_swap_rows,
    plan_analysis,
    preflight_report,
    resolve_chunk_rows,
)
from namd_analysis.io import hefei, tables
from namd_analysis.io.hefei import iter_shprop_chunks, read_shprop, shprop_structure
from namd_analysis.io.tables import TableFormatError
from namd_analysis.populations import StateMap
from namd_analysis.streaming import OnlineEnsemble, StreamingError, chunk_plan
from prepare_fixtures import six_state_populations, write_campaign_shprop


class _Campaign(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.campaign = swap_campaign(self.root)
        self.state_map = StateMap.from_json(self.campaign["state_map"])
        self.atom_groups = AtomGroupMap.from_json(self.campaign["atom_groups"])

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, **kwargs):
        return character_populations(
            self.campaign["shprop"],
            self.state_map,
            self.campaign["manifest"],
            self.atom_groups,
            "dish-cyclic",
            **kwargs,
        )


class StructureScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = write_campaign_shprop(
            self.root / "SHPROP.1", 3, six_state_populations(137, seed=5)
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_structure_reports_shape_and_endpoints_without_the_table(self):
        structure = shprop_structure(self.path)
        whole = read_shprop(self.path)
        self.assertEqual(structure.n_rows, whole.shape[0])
        self.assertEqual(structure.n_columns, whole.shape[1])
        np.testing.assert_array_equal(structure.first_row, whole[0])
        np.testing.assert_array_equal(structure.last_row, whole[-1])
        self.assertEqual(structure.metadata["NAMDTINI"], 3)
        self.assertGreater(structure.bytes_on_disk, 0)

    def test_chunks_reassemble_into_the_whole_table_at_every_size(self):
        whole = read_shprop(self.path)
        for size in (1, 2, 7, 136, 137, 138, 10_000):
            parts = list(iter_shprop_chunks(self.path, size))
            rebuilt = np.concatenate([chunk for _, chunk in parts], axis=0)
            np.testing.assert_array_equal(rebuilt, whole)
            offsets = [offset for offset, _ in parts]
            self.assertEqual(offsets, list(range(0, 137, size))[: len(parts)])

    def test_chunking_still_rejects_a_ragged_row(self):
        bad = self.root / "SHPROP.ragged"
        lines = self.path.read_text(encoding="utf-8").splitlines()
        lines[10] = lines[10] + " 1.0"
        bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(TableFormatError):
            list(iter_shprop_chunks(bad, 4))
        with self.assertRaises(TableFormatError):
            shprop_structure(bad)

    def test_chunking_still_rejects_a_non_finite_value(self):
        bad = self.root / "SHPROP.nan"
        lines = self.path.read_text(encoding="utf-8").splitlines()
        fields = lines[8].split()
        fields[3] = "NaN"
        lines[8] = " ".join(fields)
        bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(TableFormatError) as ctx:
            list(iter_shprop_chunks(bad, 4))
        self.assertIn("non-finite", str(ctx.exception))

    def test_fortran_d_exponents_survive_chunking(self):
        path = self.root / "SHPROP.d"
        path.write_text(
            "# NAMDTINI = 1\n# NSW = 5\n# BMIN = 1\n# BMAX = 2\n"
            "1.0D+00 0.5D+00 0.5D+00\n2.0D+00 0.25D+00 0.75D+00\n",
            encoding="utf-8",
        )
        chunks = [chunk for _, chunk in iter_shprop_chunks(path, 1)]
        np.testing.assert_allclose(
            np.concatenate(chunks, axis=0),
            np.array([[1.0, 0.5, 0.5], [2.0, 0.25, 0.75]]),
        )


class PreflightCostTests(_Campaign):
    def test_preflight_never_calls_the_whole_table_reader(self):
        saved = tables.read_numeric_table

        def forbidden(*args, **kwargs):
            raise AssertionError("preflight materialized a whole SHPROP table")

        tables.read_numeric_table = forbidden
        hefei_saved = hefei.read_numeric_table
        hefei.read_numeric_table = forbidden
        try:
            report = preflight_report(
                self.campaign["shprop"],
                self.state_map,
                self.campaign["manifest"],
                self.atom_groups,
                "dish-cyclic",
            )
        finally:
            tables.read_numeric_table = saved
            hefei.read_numeric_table = hefei_saved
        self.assertTrue(report["ok"])

    def test_preflight_reports_what_the_analysis_would_read(self):
        report = preflight_report(
            self.campaign["shprop"],
            self.state_map,
            self.campaign["manifest"],
            self.atom_groups,
            "dish-cyclic",
        )
        block = report["shprop_io"]
        self.assertEqual(len(block["per_file"]), 2)
        self.assertEqual(block["rows_per_history"], 4)
        self.assertGreater(block["total_bytes"], 0)
        self.assertIn("no SHPROP table was materialized", block["preflight_note"])

    def test_planning_scales_with_headers_not_with_table_size(self):
        # A history an order of magnitude longer must not cost an order of
        # magnitude more memory to plan; the scan keeps one line at a time.
        import tracemalloc

        long_path = write_campaign_shprop(
            self.root / "SHPROP.long", 1, six_state_populations(20_000, seed=4)
        )
        tracemalloc.start()
        structure = shprop_structure(long_path)
        _, scan_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.start()
        whole = read_shprop(long_path)
        _, whole_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertEqual(structure.n_rows, whole.shape[0])
        self.assertLess(scan_peak * 20, whole_peak)


class ChunkEquivalenceTests(_Campaign):
    def test_every_chunk_size_gives_bit_identical_results(self):
        reference = self._run(chunk_rows=10_000)
        for size in (1, 2, 3, 4, 7):
            result = self._run(chunk_rows=size)
            np.testing.assert_array_equal(result.mean, reference.mean)
            np.testing.assert_array_equal(result.sem, reference.sem)
            np.testing.assert_array_equal(result.time_ns, reference.time_ns)
            for name, values in reference.fixed_mean.items():
                np.testing.assert_array_equal(result.fixed_mean[name], values)
            self.assertEqual(result.conservation, reference.conservation)

    def test_memory_mode_and_stream_mode_agree(self):
        plan = plan_analysis(
            self.campaign["shprop"], self.state_map, self.campaign["manifest"], "dish-cyclic"
        )
        whole, _ = resolve_chunk_rows(plan.shprop_structures, "memory", None)
        streamed, _ = resolve_chunk_rows(plan.shprop_structures, "stream", 1)
        a = self._run(chunk_rows=whole)
        b = self._run(chunk_rows=streamed)
        np.testing.assert_array_equal(a.mean, b.mean)
        self.assertEqual(a.io["chunks_per_history"], 1)
        self.assertGreater(b.io["chunks_per_history"], 1)

    def test_welford_sem_matches_the_stack_formula(self):
        result = self._run(chunk_rows=1, keep_per_file=True)
        stack = result.per_file
        expected = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
        np.testing.assert_allclose(result.sem, expected, rtol=0, atol=1e-15)

    def test_the_mean_matches_the_mean_of_the_retained_per_file_results(self):
        result = self._run(chunk_rows=2, keep_per_file=True)
        np.testing.assert_allclose(
            result.mean, result.per_file.mean(axis=0), rtol=0, atol=1e-15
        )

    def test_swap_diagnostics_are_unchanged_by_chunking(self):
        a = character_swap_rows(self._run(chunk_rows=1).projection)
        b = character_swap_rows(self._run(chunk_rows=10_000).projection)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])


class ChunkBoundaryAlignmentTests(unittest.TestCase):
    """The cyclic wrap must map identically inside and across a chunk join."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        frames = {}
        for frame in range(1, 6):
            weights = np.zeros(4)
            weights[: 2 if frame <= 2 else 0] = 1.0
            bcf = [1.0, 1.0, 0.0, 0.0]
            pcbm = [0.0, 0.0, 1.0, 1.0]
            swapped = frame >= 3
            frames[frame] = write_procar(
                self.root / f"PROCAR.{frame}",
                {10: pcbm if swapped else bcf, 11: bcf if swapped else pcbm},
            )
        self.manifest = write_manifest(self.root / "m.json", frames, cycle_length=5)
        from character_fixtures import write_atom_groups, write_state_map

        self.state_map = StateMap.from_json(write_state_map(self.root / "sm.json"))
        self.atom_groups = AtomGroupMap.from_json(
            write_atom_groups(self.root / "ag.json", {"BCF": [1, 2], "PCBM": [3, 4]})
        )
        pops = np.zeros((10, 2))
        pops[:, 0] = np.linspace(0.9, 0.1, 10)
        pops[:, 1] = 1.0 - pops[:, 0]
        # NAMDTINI 4 with period 5: frames run 4,5,1,2,3,4,5,1,2,3 so the wrap
        # falls at row index 1->2 and again at 6->7.
        self.history = write_shprop(self.root / "SHPROP.4", 4, pops, nsw=6)

    def tearDown(self):
        self.tmp.cleanup()

    def test_frames_are_identical_however_the_rows_are_split(self):
        plan = plan_analysis(
            [self.history], self.state_map, self.manifest, "dish-cyclic"
        )
        self.assertEqual(list(plan.frames_by_file[0]), [4, 5, 1, 2, 3, 4, 5, 1, 2, 3])
        reference = None
        for size in (1, 2, 3, 5, 7, 10, 100):
            result = character_populations(
                [self.history], self.state_map, self.manifest, self.atom_groups,
                "dish-cyclic", chunk_rows=size,
            )
            if reference is None:
                reference = result.mean
            else:
                np.testing.assert_array_equal(result.mean, reference)

    def test_a_chunk_boundary_on_the_wrap_changes_nothing(self):
        # size 2 puts the 5 -> 1 step exactly on a chunk join; size 3 puts it
        # inside a chunk. Both must give the same populations.
        a = character_populations(
            [self.history], self.state_map, self.manifest, self.atom_groups,
            "dish-cyclic", chunk_rows=2,
        )
        b = character_populations(
            [self.history], self.state_map, self.manifest, self.atom_groups,
            "dish-cyclic", chunk_rows=3,
        )
        np.testing.assert_array_equal(a.mean, b.mean)


class ValidationAcrossChunksTests(_Campaign):
    def test_conservation_is_checked_in_every_chunk(self):
        rows = np.array([[0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.2, 0.2]])
        broken = write_shprop(self.root / "SHPROP.broken", 1, rows, nsw=5)
        for size in (1, 2, 4):
            with self.assertRaises(CharacterError) as ctx:
                character_populations(
                    [broken], self.state_map, self.campaign["manifest"],
                    self.atom_groups, "dish-cyclic", chunk_rows=size,
                )
            self.assertIn("sum to one", str(ctx.exception))

    def test_an_out_of_range_population_is_caught_in_a_late_chunk(self):
        rows = np.array([[0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [1.4, -0.4]])
        broken = write_shprop(self.root / "SHPROP.range", 1, rows, nsw=5)
        with self.assertRaises(CharacterError) as ctx:
            character_populations(
                [broken], self.state_map, self.campaign["manifest"],
                self.atom_groups, "dish-cyclic", chunk_rows=1,
            )
        self.assertIn("outside [0,1]", str(ctx.exception))

    def test_a_differing_time_grid_is_caught_without_storing_both_tables(self):
        pops = np.array([[0.5, 0.5]] * 4)
        shifted = write_shprop(self.root / "SHPROP.shift", 1, pops, nsw=5, dt_fs=2000.0)
        with self.assertRaises(CharacterError) as ctx:
            character_populations(
                [self.campaign["shprop"][0], shifted], self.state_map,
                self.campaign["manifest"], self.atom_groups, "dish-cyclic", chunk_rows=1,
            )
        self.assertIn("time grid differs", str(ctx.exception))

    def test_duplicate_files_are_still_refused(self):
        with self.assertRaises(CharacterError) as ctx:
            character_populations(
                [self.campaign["shprop"][0], self.campaign["shprop"][0]],
                self.state_map, self.campaign["manifest"], self.atom_groups,
                "dish-cyclic",
            )
        self.assertIn("duplicate", str(ctx.exception))


class PerFileRetentionTests(_Campaign):
    def test_small_campaigns_keep_per_file_results(self):
        self.assertIsNotNone(self._run().per_file)
        self.assertTrue(self._run().io["per_file_retained"])

    def test_per_file_can_be_dropped_without_changing_the_mean(self):
        kept = self._run(keep_per_file=True)
        dropped = self._run(keep_per_file=False)
        self.assertIsNone(dropped.per_file)
        np.testing.assert_array_equal(kept.mean, dropped.mean)
        np.testing.assert_array_equal(kept.sem, dropped.sem)


class AccumulatorTests(unittest.TestCase):
    def test_online_ensemble_matches_numpy_over_whole_arrays(self):
        rng = np.random.default_rng(0)
        files = [rng.normal(size=(17, 3)) for _ in range(5)]
        acc = OnlineEnsemble(17, 3)
        for values in files:
            acc.begin_file()
            for start in range(0, 17, 4):
                acc.update(start, values[start : start + 4])
            acc.finish_file()
        stack = np.stack(files, axis=0)
        np.testing.assert_allclose(acc.mean_array(), stack.mean(axis=0), atol=1e-13)
        np.testing.assert_allclose(
            acc.sem(), stack.std(axis=0, ddof=1) / np.sqrt(5), atol=1e-13
        )

    def test_a_single_file_has_no_standard_error(self):
        acc = OnlineEnsemble(4, 2)
        acc.begin_file()
        acc.update(0, np.ones((4, 2)))
        acc.finish_file()
        self.assertIsNone(acc.sem())

    def test_a_partial_file_is_an_error_not_a_silent_bias(self):
        acc = OnlineEnsemble(4, 2)
        acc.begin_file()
        acc.update(0, np.ones((2, 2)))
        with self.assertRaises(StreamingError) as ctx:
            acc.finish_file()
        self.assertIn("every row of every history exactly once", str(ctx.exception))

    def test_overrunning_the_grid_is_refused(self):
        acc = OnlineEnsemble(4, 2)
        acc.begin_file()
        with self.assertRaises(StreamingError):
            acc.update(3, np.ones((3, 2)))

    def test_memmap_backing_is_used_above_the_threshold(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            acc = OnlineEnsemble(
                64, 4, memmap_dir=Path(tmp.name), threshold_bytes=16, name="spill"
            )
            self.assertIsInstance(acc.mean, np.memmap)
            acc.begin_file()
            acc.update(0, np.full((64, 4), 2.0))
            acc.finish_file()
            np.testing.assert_allclose(acc.mean_array(), 2.0)
            self.assertTrue(acc.spill_files)
            self.assertTrue(any(Path(tmp.name).iterdir()))
            # Releasing must close the handles and remove the spill files:
            # a campaign large enough to need them would otherwise leave
            # gigabytes behind in scratch.
            acc.release()
            self.assertFalse(any(Path(tmp.name).iterdir()))
        finally:
            tmp.cleanup()

    def test_chunk_plan_covers_every_row_once(self):
        spans = chunk_plan(10, 3)
        self.assertEqual(spans, [(0, 3), (3, 6), (6, 9), (9, 10)])
        self.assertEqual(sum(stop - start for start, stop in spans), 10)


class ChunkSelectionTests(_Campaign):
    def test_auto_reads_a_small_history_in_one_chunk(self):
        plan = plan_analysis(
            self.campaign["shprop"], self.state_map, self.campaign["manifest"], "dish-cyclic"
        )
        chosen, record = resolve_chunk_rows(plan.shprop_structures, "auto", None)
        self.assertEqual(chosen, 4)
        self.assertIn("single-chunk threshold", record["reason"])
        self.assertLess(record["largest_history_bytes"], AUTO_SINGLE_CHUNK_BYTES)

    def test_stream_always_chunks_even_when_small(self):
        plan = plan_analysis(
            self.campaign["shprop"], self.state_map, self.campaign["manifest"], "dish-cyclic"
        )
        chosen, record = resolve_chunk_rows(plan.shprop_structures, "stream", None)
        self.assertEqual(chosen, DEFAULT_CHUNK_ROWS)
        self.assertEqual(record["reason"], "default streaming chunk")

    def test_an_explicit_chunk_size_wins(self):
        plan = plan_analysis(
            self.campaign["shprop"], self.state_map, self.campaign["manifest"], "dish-cyclic"
        )
        chosen, record = resolve_chunk_rows(plan.shprop_structures, "auto", 3)
        self.assertEqual(chosen, 3)
        self.assertEqual(record["reason"], "explicit --shprop-chunk-rows")

    def test_a_bad_mode_or_size_is_refused(self):
        plan = plan_analysis(
            self.campaign["shprop"], self.state_map, self.campaign["manifest"], "dish-cyclic"
        )
        with self.assertRaises(CharacterError):
            resolve_chunk_rows(plan.shprop_structures, "nonsense", None)
        with self.assertRaises(CharacterError):
            resolve_chunk_rows(plan.shprop_structures, "auto", 0)


class CliStreamingTests(_Campaign):
    def test_the_chunk_flags_reach_the_report(self):
        from namd_analysis.dispatch import main as dispatch_main

        out = self.root / "out"
        code = dispatch_main([
            "character-populations",
            "--files", *[str(p) for p in self.campaign["shprop"]],
            "--config", str(self.campaign["state_map"]),
            "--projection-manifest", str(self.campaign["manifest"]),
            "--atom-groups", str(self.campaign["atom_groups"]),
            "--frame-mode", "dish-cyclic",
            "--shprop-chunk-rows", "2",
            "--shprop-io-mode", "stream",
            "--out", str(out),
        ])
        self.assertEqual(code, 0)
        import json

        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        block = report["shprop_io"]
        self.assertEqual(block["chunk_rows"], 2)
        self.assertEqual(block["requested_mode"], "stream")
        self.assertEqual(block["chunks_per_history"], 2)
        self.assertEqual(block["mode"], "streaming")


if __name__ == "__main__":
    unittest.main()
