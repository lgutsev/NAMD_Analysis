import bootstrap  # noqa: F401

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.cli import main
from namd_analysis.observables import COUNTERFACTUAL, MODEL_INFERRED, OBSERVED
from namd_analysis.reviewer import (
    REVIEWER_CLASSES,
    REVIEWER_HEADER,
    ReviewerError,
    build_summary,
)
from namd_analysis.kinetics import build_rate_matrix, parse_edges, propagate
from synthetic import kinetic_config

GROUPS = ["CBM", "BCF", "PCBM", "VBM"]


def _write_campaign(root, rates, spec, n_files=4, nsteps=300, span_ns=4.0, noise=0.002,
                    seed=0):
    root.mkdir(parents=True, exist_ok=True)
    edges = parse_edges(spec, GROUPS)
    K = build_rate_matrix(rates, edges, len(GROUPS))
    time = np.linspace(0.0, span_ns, nsteps)
    exact = propagate(K, np.array([1.0, 0.0, 0.0, 0.0]), time)
    rng = np.random.default_rng(seed)
    for index in range(n_files):
        values = np.clip(exact + noise * rng.standard_normal(exact.shape), 0.0, None)
        values = values / values.sum(axis=1, keepdims=True)
        table = np.column_stack([time * 1e6, np.full(time.size, -1.5), values])
        np.savetxt(root / f"SHPROP.{index + 1}", table)
    return time


class ReviewerTableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _write_campaign(
            self.root / "run", [6.0, 3.0, 1.0], "CBM->BCF,BCF->PCBM,BCF->VBM"
        )
        (self.root / "map.json").write_text(
            json.dumps(kinetic_config(GROUPS, recombined="VBM")), encoding="utf-8"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _manifest(self, **overrides):
        entry = {
            "name": "A",
            "files": ["run/SHPROP.*"],
            "state_map": "map.json",
            "transient_window": "0:0.5",
            "late_window": "0.5:4",
            "scheme": "CBM->BCF,BCF->PCBM,BCF->VBM",
            "fit_start_ns": 0.0,
            "escape_rates": "0.01:100:12",
        }
        entry.update(overrides)
        path = self.root / "manifest.json"
        path.write_text(
            json.dumps({"criterion": "aicc", "configurations": [entry]}), encoding="utf-8"
        )
        return path

    def test_header_matches_the_declared_columns(self):
        payload = build_summary(self._manifest())
        self.assertEqual(payload["header"], REVIEWER_HEADER)
        self.assertEqual(len(payload["rows"][0]), len(REVIEWER_HEADER))

    def test_every_column_is_classified(self):
        for column in REVIEWER_HEADER:
            if column in ("configuration", "transient_window_ns", "late_window_ns"):
                continue
            self.assertIn(column, REVIEWER_CLASSES, column)
            self.assertIn(
                REVIEWER_CLASSES[column], (OBSERVED, MODEL_INFERRED, COUNTERFACTUAL)
            )

    def test_observed_and_inferred_columns_are_filled_for_a_clean_campaign(self):
        payload = build_summary(self._manifest())
        row = dict(zip(REVIEWER_HEADER, payload["rows"][0]))
        self.assertEqual(row["configuration"], "A")
        self.assertEqual(row["n_shprop_files"], 4)
        self.assertEqual(row["transient_window_ns"], "0:0.5")
        self.assertIsNotNone(row["pcbm_peak"])
        self.assertIsNotNone(row["pcbm_integral_ns"])
        self.assertIsNotNone(row["bcf_integral_ns"])
        # BCF -> PCBM at 3 against BCF -> VBM at 1 gives 3/4.
        self.assertAlmostEqual(row["first_passage_pcbm_before_vbm"], 0.75, delta=0.05)
        self.assertAlmostEqual(row["local_branch_pcbm"], 0.75, delta=0.05)
        self.assertEqual(row["branching_status"], "identified")
        # With no exit from PCBM, anything that reaches it is eventually
        # extracted at any escape rate, so extraction wins everywhere and
        # there is no crossover to report.
        self.assertIsNone(row["escape_crossover_rate_per_ns"])
        competition = payload["configurations"][0]["extraction_competition"]
        self.assertIn("already outcompetes", competition["crossover_note"])
        yields = [p["extracted_yield"] for p in competition["points"]]
        self.assertTrue(all(y > 0.5 for y in yields))

    def test_back_transfer_produces_a_crossover(self):
        # PCBM can return to BCF, so slow escape loses to recombination and a
        # required escape rate exists.
        _write_campaign(
            self.root / "back", [20.0, 10.0, 6.0, 0.5],
            "CBM->BCF,BCF->PCBM,PCBM->BCF,BCF->VBM", span_ns=4.0, nsteps=400,
        )
        payload = build_summary(
            self._manifest(
                name="back", files=["back/SHPROP.*"],
                scheme="CBM->BCF,BCF->PCBM,PCBM->BCF,BCF->VBM",
                escape_rates="0.001:1000:30",
            )
        )
        row = dict(zip(REVIEWER_HEADER, payload["rows"][0]))
        self.assertIsNotNone(row["escape_crossover_rate_per_ns"])
        self.assertAlmostEqual(
            row["escape_crossover_time_ns"],
            1.0 / row["escape_crossover_rate_per_ns"],
            places=8,
        )

    def test_candidate_schemes_are_scored_and_one_is_selected(self):
        payload = build_summary(
            self._manifest(
                scheme=None,
                candidate_schemes={
                    "minimal": "CBM->BCF,BCF->PCBM,BCF->VBM",
                    "back_transfer": "CBM->BCF,BCF->PCBM,PCBM->BCF,BCF->VBM",
                },
            )
        )
        row = dict(zip(REVIEWER_HEADER, payload["rows"][0]))
        self.assertIn(row["selected_scheme"], ("minimal", "back_transfer"))
        self.assertIsNotNone(row["scheme_score"])
        detail = payload["configurations"][0]
        self.assertIn("scheme_comparison", detail)
        self.assertIn("not evidence that the mechanism is right",
                      detail["scheme_selection_note"])

    def test_an_unidentifiable_branch_leaves_an_empty_cell(self):
        # Fitting only after the fast pair has equilibrated makes BCF->PCBM
        # unidentifiable; the probability must not be quoted.
        _write_campaign(
            self.root / "fast", [40.0, 20.0, 12.0, 0.4],
            "CBM->BCF,BCF->PCBM,PCBM->BCF,BCF->VBM", span_ns=10.0, nsteps=400,
        )
        path = self._manifest(
            name="fast", files=["fast/SHPROP.*"],
            scheme="CBM->BCF,BCF->PCBM,PCBM->BCF,BCF->VBM",
            fit_start_ns=1.0, transient_window="0:1", late_window="1:10",
        )
        payload = build_summary(path)
        row = dict(zip(REVIEWER_HEADER, payload["rows"][0]))
        self.assertEqual(row["branching_status"], "not_identifiable")
        self.assertIsNone(row["first_passage_pcbm_before_vbm"])
        self.assertIsNone(row["local_branch_pcbm"])
        self.assertIsNone(row["k_bcf_to_pcbm_per_ns"])
        # The observed columns are still filled: they do not depend on a model.
        self.assertIsNotNone(row["pcbm_peak"])
        notes = " ".join(payload["configurations"][0]["notes"])
        self.assertIn("left empty", notes)

    def test_a_missing_role_leaves_its_columns_empty(self):
        groups = ["CBM", "BCF", "VBM"]
        _write_campaign_groups = kinetic_config(groups, recombined="VBM")
        (self.root / "map3.json").write_text(
            json.dumps(_write_campaign_groups), encoding="utf-8"
        )
        edges = parse_edges("CBM->BCF,BCF->VBM", groups)
        K = build_rate_matrix([5.0, 1.0], edges, 3)
        time = np.linspace(0.0, 4.0, 200)
        exact = propagate(K, np.array([1.0, 0.0, 0.0]), time)
        target = self.root / "three"
        target.mkdir(parents=True, exist_ok=True)
        for index in range(3):
            table = np.column_stack([time * 1e6, np.full(time.size, -1.5), exact])
            np.savetxt(target / f"SHPROP.{index + 1}", table)
        path = self._manifest(
            name="three", files=["three/SHPROP.*"], state_map="map3.json",
            scheme="CBM->BCF,BCF->VBM",
        )
        payload = build_summary(path)
        row = dict(zip(REVIEWER_HEADER, payload["rows"][0]))
        self.assertIsNone(row["pcbm_peak"])
        self.assertIsNotNone(row["bcf_integral_ns"])
        notes = " ".join(payload["configurations"][0]["notes"])
        self.assertIn("acceptor role", notes)

    def test_a_failing_configuration_yields_a_blank_row_not_a_crash(self):
        path = self.root / "bad.json"
        path.write_text(
            json.dumps({
                "configurations": [
                    {"name": "missing", "files": ["nope/SHPROP.*"],
                     "state_map": "map.json"}
                ]
            }),
            encoding="utf-8",
        )
        payload = build_summary(path)
        self.assertEqual(len(payload["failures"]), 1)
        row = dict(zip(REVIEWER_HEADER, payload["rows"][0]))
        self.assertEqual(row["configuration"], "missing")
        self.assertIsNone(row["pcbm_peak"])

    def test_manifest_without_configurations_is_rejected(self):
        path = self.root / "empty.json"
        path.write_text(json.dumps({"configurations": []}), encoding="utf-8")
        with self.assertRaises(ReviewerError):
            build_summary(path)

    def test_cli_writes_the_table_and_report(self):
        out = self.root / "out"
        code = main(["reviewer-summary", str(self._manifest()), "--out", str(out)])
        self.assertEqual(code, 0)
        self.assertTrue((out / "reviewer_branching.csv").is_file())
        header = (out / "reviewer_branching.csv").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        self.assertEqual(header.split(","), REVIEWER_HEADER)
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["command"], "reviewer-summary")
        self.assertIn("observable_class", report)
        self.assertIn("observable_class_legend", report)


class ExampleTemplateTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1] / "examples" / "bcf_pcbm"

    def test_templates_exist(self):
        for name in (
            "README.md",
            "candidate_schemes.json",
            "comparison_manifest.template.json",
            "state_map_A.template.json",
            "state_map_B.template.json",
            "state_map_C.template.json",
        ):
            self.assertTrue((self.ROOT / name).is_file(), name)

    def test_state_map_templates_do_not_guess_column_numbers(self):
        for cfg in ("A", "B", "C"):
            payload = json.loads(
                (self.ROOT / f"state_map_{cfg}.template.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["population_columns"], [])
            for columns in payload["groups"].values():
                self.assertEqual(columns, [])

    def test_unedited_template_fails_loudly(self):
        from namd_analysis.populations import ConfigError, StateMap

        with self.assertRaises(ConfigError):
            StateMap.from_json(self.ROOT / "state_map_A.template.json")

    def test_candidate_schemes_parse_against_the_declared_groups(self):
        from namd_analysis.kinetics import parse_edges as parse

        payload = json.loads(
            (self.ROOT / "candidate_schemes.json").read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(len(payload["schemes"]), 2)
        for name, spec in payload["schemes"].items():
            edges = parse(spec, GROUPS)
            self.assertTrue(edges, name)

    def test_manifest_template_declares_the_three_configurations(self):
        payload = json.loads(
            (self.ROOT / "comparison_manifest.template.json").read_text(
                encoding="utf-8"
            )
        )
        names = [c["name"] for c in payload["configurations"]]
        self.assertEqual(names, ["A", "B", "C"])
        for config in payload["configurations"]:
            self.assertIn("REPLACE", config["files"][0])


if __name__ == "__main__":
    unittest.main()
