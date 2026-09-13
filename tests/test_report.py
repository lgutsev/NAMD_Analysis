import bootstrap  # noqa: F401

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.report import _plain, write_csv, write_json


def _strict_loads(text):
    """Parse JSON, refusing the NaN/Infinity tokens the JSON spec does not allow."""

    def reject(token):
        raise ValueError(f"non-standard JSON token {token!r}")

    return json.loads(text, parse_constant=reject)


class JsonSerializationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_numpy_non_finite_becomes_null(self):
        # A numpy NaN used to pass through unwrapped and reach json.dumps as a
        # bare NaN token, which strict parsers reject.
        payload = {
            "numpy_nan": np.float64("nan"),
            "numpy_inf": np.float64(np.inf),
            "numpy_neg_inf": np.float32(-np.inf),
            "python_nan": float("nan"),
        }
        self.assertEqual(
            _plain(payload),
            {"numpy_nan": None, "numpy_inf": None,
             "numpy_neg_inf": None, "python_nan": None},
        )

    def test_report_with_a_nan_round_trips_through_a_strict_parser(self):
        path = write_json(
            self.root / "report.json",
            {"centroid_cm1": np.float64("nan"), "peaks": [np.float64(1.5)]},
        )
        payload = _strict_loads(path.read_text(encoding="utf-8"))
        self.assertIsNone(payload["centroid_cm1"])
        self.assertEqual(payload["peaks"], [1.5])

    def test_numpy_scalar_types_are_preserved(self):
        payload = _plain(
            {"i": np.int64(7), "b": np.bool_(True), "f": np.float64(2.5)}
        )
        self.assertEqual(payload, {"i": 7, "b": True, "f": 2.5})
        self.assertIsInstance(payload["i"], int)
        self.assertIsInstance(payload["b"], bool)

    def test_nested_array_non_finite_becomes_null(self):
        payload = _plain({"rows": np.array([[1.0, np.nan], [np.inf, 4.0]])})
        self.assertEqual(payload["rows"], [[1.0, None], [None, 4.0]])

    def test_written_report_is_always_strict_json(self):
        path = write_json(
            self.root / "nested.json",
            {"a": {"b": [np.float64("nan"), {"c": float("inf")}]}},
        )
        parsed = _strict_loads(path.read_text(encoding="utf-8"))
        self.assertEqual(parsed, {"a": {"b": [None, {"c": None}]}})

    def test_non_string_keys_are_stringified(self):
        self.assertEqual(_plain({1: "a", 2.5: "b"}), {"1": "a", "2.5": "b"})


class CsvTests(unittest.TestCase):
    def test_csv_writes_numpy_and_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(
                Path(tmp) / "t.csv",
                ["a", "b", "c"],
                [[np.float64(1.5), None, np.int64(3)]],
            )
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "a,b,c")
        self.assertEqual(lines[1], "1.5,,3")

    def test_csv_round_trips_through_numpy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(
                Path(tmp) / "t.csv", ["x", "y"], [[1.0, 2.0], [3.0, 4.0]]
            )
            table = np.loadtxt(path, delimiter=",", skiprows=1)
        np.testing.assert_allclose(table, [[1.0, 2.0], [3.0, 4.0]])

    def test_non_finite_becomes_an_empty_cell(self):
        # NaN and infinity are written as empty cells, the same as None, so a
        # CSV reader sees a missing value rather than a locale-dependent token.
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(
                Path(tmp) / "t.csv",
                ["x", "y", "z"],
                [[float("nan"), np.float64(np.inf), 1.25]],
            )
            row = path.read_text(encoding="utf-8").splitlines()[1]
        self.assertEqual(row, ",,1.25")
        self.assertFalse(math.isnan(1.25))


if __name__ == "__main__":
    unittest.main()
