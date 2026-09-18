"""Figures, and the labels they are not allowed to drop.

The plotting layer exists to turn machine-readable output into figures without
changing any number on the way. Two things are therefore pinned hard: a quoted
number must come from the authoritative summary rather than from the curve
drawn on screen, and a shaded band must never be described as a confidence
interval.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.ensemble_plots import (
    BAND_LABEL,
    BOOKKEEPING_CAPTION,
    PlotError,
    load_curves,
    plot_decomposition_distributions,
    plot_delta_distributions,
    plot_fixed_vs_dynamic,
    plot_run_comparison,
    plot_run_populations,
    write_figure_manifest,
)

GROUPS = ["perovskite", "BCF", "PCBM"]


def curves(n=200, with_fixed=True, n_histories=100):
    t = np.linspace(0.0, 10.0, n)
    out = {
        "time_ns": t.tolist(),
        "dynamic_groups": list(GROUPS),
        "fixed_groups": ["fixed:BCF", "fixed:PCBM"] if with_fixed else [],
        "groups": {},
    }
    for g, base in (("perovskite", 0.1), ("BCF", 0.7), ("PCBM", 0.2)):
        series = base + 0.05 * np.sin(t)
        out["groups"][g] = {
            "n_histories": n_histories,
            "median": series.tolist(),
            "q25": (series - 0.03).tolist(),
            "q75": (series + 0.03).tolist(),
            "min": (series - 0.08).tolist(),
            "max": (series + 0.08).tolist(),
        }
    if with_fixed:
        for g in ("BCF", "PCBM"):
            series = np.asarray(out["groups"][g]["median"]) * 0.9
            out["groups"][f"fixed:{g}"] = {
                "n_histories": n_histories,
                "median": series.tolist(),
                "q25": (series - 0.02).tolist(),
                "q75": (series + 0.02).tolist(),
            }
    return out


def summary(run="A", n=100):
    def block(median):
        return {g: {"n": n, "median": median, "n_gain": n, "n_loss": 0}
                for g in ("BCF", "PCBM")}

    return {
        "run": run,
        "n_histories": n,
        "quantities": {
            "net_full": block(0.7800),
            "net_full_fixed": block(0.1540),
            "net_late": block(-0.0570),
            "net_late_fixed": block(-0.0570),
            "range_late": block(0.8461),
            "range_late_fixed": block(0.1760),
        },
    }


def rows(run="A", n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        b, p = rng.normal(0.4, 0.3), rng.normal(-0.2, 0.3)
        out.append({
            "run": run, "history": f"SHPROP.{k}", "NAMDTINI": str(k),
            "net_full_BCF": str(b), "net_full_PCBM": str(p),
            "net_late_BCF": str(b / 10), "net_late_PCBM": str(p / 10),
            "occupation_redistribution_BCF": str(b * 0.8),
            "occupation_redistribution_PCBM": str(p * 0.1),
            "character_evolution_BCF": str(b * 0.2),
            "character_evolution_PCBM": str(p * 0.9),
        })
    return out


class _Figures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.manifest = []

    def tearDown(self):
        self.tmp.cleanup()

    def both_formats(self, written, stem):
        self.assertEqual(sorted(written), [f"{stem}.pdf", f"{stem}.png"])
        for name in written:
            path = self.out / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 1000, f"{name} looks empty")


class PopulationFigureTests(_Figures):
    def test_png_and_pdf_are_both_written(self):
        written = plot_run_populations(
            curves(), "A", self.out, self.manifest, "ensemble_curves.json")
        self.both_formats(written, "run_A_populations")

    def test_the_band_is_never_called_a_confidence_interval(self):
        plot_run_populations(
            curves(), "A", self.out, self.manifest, "ensemble_curves.json")
        entry = self.manifest[0]
        self.assertIn("not a confidence interval", entry["caption"].lower())
        self.assertIn("not a standard error", entry["caption"].lower())
        self.assertIn("NOT a CI or SEM", BAND_LABEL)

    def test_the_caption_names_the_quantity_and_the_omitted_coherence(self):
        plot_run_populations(
            curves(), "A", self.out, self.manifest, "ensemble_curves.json")
        caption = self.manifest[0]["caption"]
        self.assertIn("DIAGONAL", caption)
        self.assertIn("omitted, not estimated", caption)

    def test_a_file_that_is_not_a_curves_file_is_refused(self):
        path = self.out / "nope.json"
        path.write_text(json.dumps({"hello": 1}), encoding="utf-8")
        with self.assertRaises(PlotError):
            load_curves(path)


class FixedVsDynamicTests(_Figures):
    def test_quoted_numbers_come_from_the_summary_not_the_thinned_curve(self):
        # The regression this pins: curves are thinned by --curve-stride, so
        # recomputing net/range from them moves the endpoints and hides the
        # extrema. The figure must quote the authoritative values instead.
        thinned = curves(n=25)          # a deliberately coarse curve
        stats = summary()
        plot_fixed_vs_dynamic(
            thinned, "A", self.out, self.manifest, "ensemble_curves.json",
            stats=stats, stats_source="run_summary.json",
        )
        entry = self.manifest[0]
        self.assertIn("run_summary.json", entry["sources"])
        self.assertIn("not from the displayed curves", entry["caption"])

        from namd_analysis.ensemble_plots import _quoted_stats

        late = _quoted_stats(stats, "PCBM", True)
        self.assertIn("-0.0570", late)      # net_late median
        self.assertIn("0.8461", late)       # range_late median
        self.assertIn("0.1760", late)       # range_late_fixed median
        # And the thinned curve would have given something else entirely.
        series = np.asarray(thinned["groups"]["PCBM"]["median"])
        self.assertNotAlmostEqual(
            float(series[-1] - series[0]), -0.0570, places=3,
            msg="the fixture must actually differ from the authoritative value",
        )

    def test_no_stats_means_no_numbers_rather_than_wrong_numbers(self):
        from namd_analysis.ensemble_plots import _quoted_stats

        self.assertEqual(_quoted_stats(None, "PCBM", True), "")
        written = plot_fixed_vs_dynamic(
            curves(), "A", self.out, self.manifest, "ensemble_curves.json")
        self.both_formats(written, "run_A_fixed_vs_dynamic")

    def test_a_missing_fixed_curve_is_refused_with_the_remedy(self):
        with self.assertRaises(PlotError) as ctx:
            plot_fixed_vs_dynamic(
                curves(with_fixed=False), "A", self.out, self.manifest, "c.json")
        self.assertIn("--fixed-state-map", str(ctx.exception))


class RunComparisonTests(_Figures):
    def test_each_run_is_drawn_separately(self):
        written = plot_run_comparison(
            {"A": curves(), "B": curves(), "C": curves()},
            self.out, self.manifest, ["A/c.json", "B/c.json", "C/c.json"],
        )
        self.both_formats(written, "run_comparison_populations")
        caption = self.manifest[0]["caption"]
        self.assertIn("NOT averaged together", caption)
        self.assertIn("grand mean", caption)

    def test_no_runs_is_refused(self):
        with self.assertRaises(PlotError):
            plot_run_comparison({}, self.out, self.manifest, [])


class DistributionTests(_Figures):
    def test_run_identity_is_preserved_in_the_labels(self):
        written = plot_delta_distributions(
            {"A": rows("A", seed=1), "B": rows("B", seed=2)},
            self.out, self.manifest, ["A/per_history.csv", "B/per_history.csv"],
        )
        self.both_formats(written, "delta_distributions")
        self.assertIn("heterogeneity", self.manifest[0]["caption"])

    def test_a_missing_column_does_not_crash_the_figure(self):
        stripped = [{k: v for k, v in r.items() if "net_late" not in k}
                    for r in rows()]
        written = plot_delta_distributions(
            {"A": stripped}, self.out, self.manifest, ["per_history.csv"])
        self.both_formats(written, "delta_distributions")

    def test_the_decomposition_figure_carries_the_bookkeeping_caveat(self):
        plot_decomposition_distributions(
            {"A": rows()}, self.out, self.manifest, ["per_history.csv"])
        caption = self.manifest[0]["caption"]
        self.assertIn("NOT", caption)
        self.assertIn("branching fractions", caption)
        self.assertEqual(caption[:len(BOOKKEEPING_CAPTION)], BOOKKEEPING_CAPTION)

    def test_no_runs_is_refused(self):
        with self.assertRaises(PlotError):
            plot_decomposition_distributions({}, self.out, self.manifest, [])


class ManifestTests(_Figures):
    def test_every_figure_records_its_sources_and_caption(self):
        plot_run_populations(curves(), "A", self.out, self.manifest, "c.json")
        plot_decomposition_distributions(
            {"A": rows()}, self.out, self.manifest, ["per_history.csv"])
        path = write_figure_manifest(self.out, self.manifest)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["figures"]), 2)
        for entry in data["figures"]:
            self.assertTrue(entry["sources"])
            self.assertTrue(entry["caption"])
            self.assertEqual(len(entry["files"]), 2)
        self.assertIn("nothing is recomputed", data["note"])
        self.assertIn("NOT physical branching fractions", data["bookkeeping_caveat"])


if __name__ == "__main__":
    unittest.main()
