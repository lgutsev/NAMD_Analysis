from unittest import TestCase

from namd_analysis.prepare_cli import build_parser


class PrepareMemoryOptionTests(TestCase):
    def _base(self):
        return [
            "--shprop-dir",
            "/tmp/shprop",
            "--projection-dir",
            "/tmp/projection",
            "--frame-mode",
            "dish-cyclic",
        ]

    def test_streaming_controls_have_safe_defaults(self):
        args = build_parser().parse_args(self._base())
        self.assertEqual(args.shprop_io_mode, "auto")
        self.assertIsNone(args.shprop_chunk_rows)
        self.assertIsNone(args.accumulator_memmap_dir)

    def test_streaming_controls_accept_explicit_values(self):
        args = build_parser().parse_args(
            self._base()
            + [
                "--shprop-io-mode",
                "stream",
                "--shprop-chunk-rows",
                "25000",
                "--accumulator-memmap-dir",
                "/scratch/job 123",
            ]
        )
        self.assertEqual(args.shprop_io_mode, "stream")
        self.assertEqual(args.shprop_chunk_rows, 25000)
        self.assertEqual(args.accumulator_memmap_dir, "/scratch/job 123")

    def test_invalid_io_mode_is_refused_by_argparse(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(self._base() + ["--shprop-io-mode", "unbounded"])
