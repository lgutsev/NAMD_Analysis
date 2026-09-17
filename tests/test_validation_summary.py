"""The referee-facing summary must not misstate its own numbers.

The memory figures here are the ones the real Campaign A run reported
(5 x 868 MB SHPROP, 1999 PROCARs): 1.5 GiB of arrays, 512 MiB of assumed
overhead, 2.0 GiB estimated total.  The row labelled "assumed overhead" once
printed the *total*, which is how a reader would have come away with 2.0 GiB
of overhead on top of 1.5 GiB of arrays.
"""

import tempfile
import unittest
from pathlib import Path

from namd_analysis.validation_summary import render, write

REAL_MEMORY = {
    "budget_bytes": None,
    "budget_human": None,
    "estimated_resident_bytes": 1640287856,
    "estimated_resident_human": "1.5 GiB",
    "assumed_overhead_bytes": 536870912,
    "estimated_total_human": "2.0 GiB",
    "fits_budget": None,
    "retain_per_file": False,
    "spill_accumulators_to_memmap": False,
    "shprop_chunk_rows": 100000,
}


def payload(**overrides):
    base = {
        "command": "character-populations",
        "environment": {"version": "0.6.0", "python": "3.12.12"},
        "frame_mode": "dish-cyclic",
        "memory": dict(REAL_MEMORY),
        "shprop_io": {"chunk_rows": 100000, "chunks_per_history": 100},
    }
    base.update(overrides)
    return base


def row(text, label):
    for line in text.splitlines():
        if line.startswith(f"| {label} |"):
            return [cell.strip() for cell in line.strip("|").split("|")]
    raise AssertionError(f"no row {label!r} in:\n{text}")


class MemoryTableTests(unittest.TestCase):
    def test_the_overhead_row_is_the_overhead_not_the_total(self):
        text = render(payload())
        self.assertEqual(row(text, "assumed overhead")[1], "512.0 MiB")

    def test_the_total_is_reported_on_its_own_row(self):
        text = render(payload())
        self.assertEqual(row(text, "estimated total")[1], "2.0 GiB")

    def test_the_resident_arrays_row_is_unchanged(self):
        text = render(payload())
        self.assertEqual(row(text, "estimated resident arrays")[1], "1.5 GiB")

    def test_the_three_figures_are_consistent(self):
        # arrays + overhead = total, which is the whole point of the table.
        text = render(payload())
        self.assertEqual(REAL_MEMORY["assumed_overhead_bytes"], 512 * 1024**2)
        self.assertAlmostEqual(
            (REAL_MEMORY["estimated_resident_bytes"]
             + REAL_MEMORY["assumed_overhead_bytes"]) / 1024**3,
            2.0,
            places=1,
        )
        self.assertIn("estimated total", text)

    def test_a_missing_overhead_is_left_blank_rather_than_guessed(self):
        memory = dict(REAL_MEMORY)
        memory.pop("assumed_overhead_bytes")
        text = render(payload(memory=memory))
        self.assertEqual(row(text, "assumed overhead")[1], "")

    def test_a_budget_that_was_given_is_shown(self):
        memory = dict(REAL_MEMORY)
        memory["budget_human"] = "4.0 GiB"
        text = render(payload(memory=memory))
        self.assertEqual(row(text, "budget")[1], "4.0 GiB")

    def test_the_summary_writes_and_reads_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp) / "validation_summary.md", payload())
            self.assertIsNotNone(path)
            text = Path(path).read_text(encoding="utf-8")
            self.assertEqual(row(text, "assumed overhead")[1], "512.0 MiB")


if __name__ == "__main__":
    unittest.main()
