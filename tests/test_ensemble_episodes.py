"""Several named crossing windows in one ``character-ensemble`` pass.

Configuration B has four distinct BCF/PCBM mixing regions on one recycled
nuclear trajectory. Rereading a ~900 MB SHPROP once per region is wasteful, and
reducing the four to one number is wrong: they are four regions of the *same*
trajectory, not four samples of one region. So the requirement has two halves,
and both are pinned here.

**One pass.** Every window is a mask over the single symmetric-midpoint
decomposition the streaming read already produced. The test for that is
behavioural rather than instrumental: running with four windows must give
exactly what running four times with one window each gives, so no window can
perturb another and none is computed from a different read.

**No pooling.** Nothing in the outputs averages, sums or otherwise combines two
windows, and each number carries the name of the region it came from.

The legacy single ``--episode-window`` keeps its own unprefixed columns and its
own run-level verdict throughout, so an existing recipe is unaffected.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from namd_analysis.dispatch import main as dispatch_main
from namd_analysis.ensemble import (
    EPISODE_QUANTITIES,
    EPISODES_NOTE,
    EnsembleError,
    EpisodeResult,
    HistoryResult,
    compare_episodes,
    episode_long_header,
    episode_long_rows,
    history_header,
    parse_episode_window,
    parse_episode_windows,
    summarize_episodes,
    summarize_run,
)

GROUPS = ["BCF", "PCBM"]

#: The recycled nuclear trajectory of the fixture, and the two regions in it.
CYCLE = 12
#: Occupation moves here and the character does not.
WINDOW_OCCUPATION = (3, 4)
#: The character is exchanged here and the occupation does not move. The
#: window covers the entry into the swap only: over 9:11 the exchange and its
#: reversal cancel, which would make the fixture agree with a pooled reading by
#: accident rather than by construction.
WINDOW_CHARACTER = (9, 10)


def write_character(path, frames, weights_by_band, groups=GROUPS):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "band", "group", "normalized_weight"])
        for index, frame in enumerate(frames):
            for band, table in weights_by_band.items():
                for gi, group in enumerate(groups):
                    writer.writerow(
                        [int(frame), int(band), group, float(table[index, gi])]
                    )
    return Path(path)


def write_state_map(path):
    Path(path).write_text(
        json.dumps({
            "name": "synthetic",
            "time_column": 0,
            "time_unit": "fs",
            "population_columns": [2, 3],
            "groups": {"BCF": [2], "PCBM": [3]},
            "complete_population": True,
        }),
        encoding="utf-8",
    )
    return Path(path)


class _Episodes(unittest.TestCase):
    """One history on a 12-frame cycle, with the two regions made different.

    Band 976 is pure BCF and 977 pure PCBM everywhere except frames 9 and 10,
    where they exchange -- a character swap at fixed occupation. The SHPROP
    occupation, meanwhile, moves only at the row that lands on frame 4, inside
    the *other* window. So the two regions have genuinely different
    decompositions, and any code that pooled them would show it.
    """

    n_passes = 3

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = CYCLE * self.n_passes
        frames = np.arange(1, CYCLE + 1)

        swap = np.isin(frames, [9, 10])
        band_976 = np.where(swap[:, None], [0.0, 1.0], [1.0, 0.0]).astype(float)
        band_977 = np.where(swap[:, None], [1.0, 0.0], [0.0, 1.0]).astype(float)
        self.character_path = write_character(
            self.root / "projection_character.csv", frames,
            {976: band_976, 977: band_977},
        )
        self.state_map_path = write_state_map(self.root / "state_map.json")

        # NAMDTINI 1, dish-cyclic, period 12: row i occupies frame (i+1) mod 12
        # (with 0 -> 12). Occupation steps only at the rows landing on frame 4.
        pop = np.full(self.rows, 0.9)
        for row in range(self.rows):
            if (row + 1) % CYCLE == 4:
                pop[row:] -= 0.1
        self.history = self.root / "SHPROP.1"
        with self.history.open("w", encoding="utf-8") as handle:
            print(f"# NSW = {CYCLE + 1}", file=handle)
            for index in range(self.rows):
                values = [(index + 1) * 10000.0, -0.8, pop[index], 1.0 - pop[index]]
                print(" ".join(f"{v:.10E}" for v in values), file=handle)
        self.runs = 0

    def tearDown(self):
        self.tmp.cleanup()

    def run_ensemble(self, *episode_args, label="B"):
        """One ``character-ensemble`` run; returns its output directory."""
        self.runs += 1
        out = self.root / f"out{self.runs}"
        code = dispatch_main([
            "character-ensemble",
            "--run-label", label,
            "--shprop", str(self.history),
            "--projection-character", str(self.character_path),
            "--state-map", str(self.state_map_path),
            "--frame-mode", "dish-cyclic",
            "--cycle-length", str(CYCLE),
            "--curve-stride", "1",
            "--no-figures",
            "--out", str(out),
            *episode_args,
        ])
        self.assertEqual(code, 0)
        return out

    @staticmethod
    def summary(out):
        return json.loads((out / "run_summary.json").read_text(encoding="utf-8"))

    @staticmethod
    def per_history(out):
        with (out / "per_history.csv").open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def long_form(out):
        with (out / "episode_per_history.csv").open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


class WindowSpecificationTests(unittest.TestCase):
    """The name is what makes a number traceable, so it is validated."""

    def test_a_named_window_is_parsed(self):
        window = parse_episode_window("crossing_B1=1488:1492")
        self.assertEqual(
            (window.name, window.first, window.last, window.role),
            ("crossing_B1", 1488, 1492, "crossing"),
        )

    def test_a_bare_window_keeps_the_legacy_name(self):
        window = parse_episode_window("1320:1331")
        self.assertEqual(window.name, "episode")
        self.assertEqual((window.first, window.last), (1320, 1331))

    def test_the_legacy_flag_and_named_windows_coexist(self):
        windows = parse_episode_windows(
            "1320:1331",
            ["crossing_B1=1488:1492", "crossing_B2=1687:1696"],
            ["closest_approach_C=1628:1632"],
        )
        self.assertEqual(
            [w.name for w in windows],
            ["episode", "crossing_B1", "crossing_B2", "closest_approach_C"],
        )
        self.assertTrue(windows[0].legacy)
        self.assertFalse(any(w.legacy for w in windows[1:]))
        self.assertEqual(windows[-1].role, "control")

    def test_a_reversed_window_is_refused_rather_than_reordered(self):
        with self.assertRaises(EnsembleError) as ctx:
            parse_episode_window("crossing_B1=1492:1488")
        self.assertIn("after LAST", str(ctx.exception))

    def test_two_windows_may_not_share_a_name(self):
        with self.assertRaises(EnsembleError) as ctx:
            parse_episode_windows(None, ["a=1:2", "a=3:4"])
        self.assertIn("both named", str(ctx.exception))

    def test_a_name_that_would_break_a_column_is_refused(self):
        # ``__`` separates the episode from the group in a column name.
        with self.assertRaises(EnsembleError):
            parse_episode_window("a__b=1:2")
        with self.assertRaises(EnsembleError):
            parse_episode_window("crossing B1=1:2")

    def test_a_window_without_a_range_is_refused(self):
        with self.assertRaises(EnsembleError):
            parse_episode_window("crossing_B1=1488")
        with self.assertRaises(EnsembleError):
            parse_episode_window("crossing_B1=x:y")


class MultipleWindowsInOnePassTests(_Episodes):
    def test_four_named_windows_are_all_reported(self):
        out = self.run_ensemble(
            "--episode", "crossing_B1=1:2",
            "--episode", "crossing_B2=3:4",
            "--episode", "crossing_B3=6:7",
            "--episode", "crossing_B4=9:11",
        )
        episodes = self.summary(out)["episodes"]
        self.assertEqual(
            sorted(episodes),
            ["crossing_B1", "crossing_B2", "crossing_B3", "crossing_B4"],
        )
        for name, block in episodes.items():
            self.assertEqual(block["n_histories"], 1)
            self.assertEqual(block["role"], "crossing")
            for quantity in EPISODE_QUANTITIES:
                self.assertEqual(sorted(block["quantities"][quantity]), sorted(GROUPS))
            self.assertFalse(block["pooled_with_other_episodes"], name)

    def test_several_windows_give_exactly_what_one_window_at_a_time_gives(self):
        """The one-pass claim, tested through its only observable consequence.

        If a window were evaluated against a different read of the file, or if
        one window's mask leaked into another's bins, these numbers would
        differ. They must not.
        """
        together = self.run_ensemble(
            "--episode", f"occ={WINDOW_OCCUPATION[0]}:{WINDOW_OCCUPATION[1]}",
            "--episode", f"chr={WINDOW_CHARACTER[0]}:{WINDOW_CHARACTER[1]}",
        )
        alone_occ = self.run_ensemble(
            "--episode", f"occ={WINDOW_OCCUPATION[0]}:{WINDOW_OCCUPATION[1]}"
        )
        alone_chr = self.run_ensemble(
            "--episode", f"chr={WINDOW_CHARACTER[0]}:{WINDOW_CHARACTER[1]}"
        )
        joint = self.summary(together)["episodes"]
        for name, separate in (("occ", alone_occ), ("chr", alone_chr)):
            single = self.summary(separate)["episodes"][name]
            self.assertEqual(joint[name]["window"], single["window"])
            for quantity in EPISODE_QUANTITIES:
                for group in GROUPS:
                    self.assertAlmostEqual(
                        joint[name]["quantities"][quantity][group]["median"],
                        single["quantities"][quantity][group]["median"],
                        places=12,
                        msg=f"{name}/{quantity}/{group} depends on which other "
                            "windows were requested",
                    )

    def test_the_windows_carry_genuinely_different_numbers(self):
        # The fixture is built so that pooling would be detectable; if the two
        # windows agreed, the no-pooling tests below would prove nothing.
        out = self.run_ensemble(
            "--episode", f"occ={WINDOW_OCCUPATION[0]}:{WINDOW_OCCUPATION[1]}",
            "--episode", f"chr={WINDOW_CHARACTER[0]}:{WINDOW_CHARACTER[1]}",
        )
        episodes = self.summary(out)["episodes"]
        occupation = episodes["occ"]["quantities"]["occupation_per_pass"]["BCF"]["median"]
        character = episodes["chr"]["quantities"]["character_per_pass"]["BCF"]["median"]
        self.assertGreater(abs(occupation), 1e-6)
        self.assertGreater(abs(character), 1e-6)
        self.assertNotAlmostEqual(
            episodes["occ"]["quantities"]["net_per_pass"]["BCF"]["median"],
            episodes["chr"]["quantities"]["net_per_pass"]["BCF"]["median"],
            places=6,
        )

    def test_each_episode_closes_on_its_own_decomposition(self):
        # net = occupation + character, per window, within machine precision.
        out = self.run_ensemble(
            "--episode", "e1=3:4", "--episode", "e2=9:11", "--episode", "e3=1:12",
        )
        for row in self.long_form(out):
            self.assertAlmostEqual(
                float(row["net_per_pass"]),
                float(row["occupation_per_pass"]) + float(row["character_per_pass"]),
                places=12,
                msg=f"{row['episode']}/{row['group']} does not close",
            )
        residual = float(self.per_history(out)[0]["decomposition_residual"])
        self.assertLess(residual, 1e-9)

    def test_a_window_the_trajectory_never_visits_is_absent_not_zero(self):
        out = self.run_ensemble("--episode", "nowhere=900:999")
        self.assertEqual(self.summary(out)["episodes"], {})
        self.assertEqual(self.long_form(out), [])


class NoPoolingAcrossWindowsTests(_Episodes):
    """B1..B4 are regions of one trajectory. Nothing may combine them."""

    def setUp(self):
        super().setUp()
        self.out = self.run_ensemble(
            "--episode", "occ=3:4", "--episode", "chr=9:11",
        )
        self.episodes = self.summary(self.out)["episodes"]

    def test_no_combined_or_averaged_episode_appears_anywhere(self):
        text = json.dumps(self.summary(self.out))
        for forbidden in (
            "episodes_combined", "episodes_pooled", "mean_over_episodes",
            "all_episodes", "episode_average",
        ):
            self.assertNotIn(forbidden, text)

    def test_no_reported_value_is_the_mean_of_two_windows(self):
        values = [
            self.episodes[name]["quantities"]["net_per_pass"]["BCF"]["median"]
            for name in ("occ", "chr")
        ]
        pooled = sum(values) / len(values)
        for block in self.episodes.values():
            for quantity in EPISODE_QUANTITIES:
                for group in GROUPS:
                    self.assertNotAlmostEqual(
                        block["quantities"][quantity][group]["median"],
                        pooled, places=9,
                    )

    def test_the_summary_states_the_windows_are_not_samples(self):
        summary = self.summary(self.out)
        self.assertEqual(summary["episodes_note"], EPISODES_NOTE)
        self.assertIn("NOT independent samples", summary["episodes_note"])
        self.assertIn("nothing here averages, pools or combines them",
                      summary["episodes_note"])
        self.assertIn("SAME streamed pass", summary["episodes_single_pass_note"])

    def test_every_number_is_keyed_by_the_window_it_came_from(self):
        header = self.per_history(self.out)[0]
        for name in ("occ", "chr"):
            for group in GROUPS:
                for quantity in EPISODE_QUANTITIES:
                    self.assertIn(f"{name}__{group}_{quantity}", header)
            self.assertIn(f"{name}__verdict", header)

    def test_the_long_form_names_its_episode_on_every_row(self):
        rows = self.long_form(self.out)
        self.assertEqual(sorted(episode_long_header()), sorted(rows[0]))
        self.assertEqual(
            {(r["episode"], r["group"]) for r in rows},
            {(name, g) for name in ("occ", "chr") for g in GROUPS},
        )
        for row in rows:
            self.assertEqual(row["role"], "crossing")
            self.assertTrue(row["first_frame"])
            self.assertTrue(row["verdict"])

    def test_the_csv_and_the_json_agree(self):
        row = self.per_history(self.out)[0]
        long_rows = {(r["episode"], r["group"]): r for r in self.long_form(self.out)}
        for name in ("occ", "chr"):
            for group in GROUPS:
                for quantity in EPISODE_QUANTITIES:
                    wide = float(row[f"{name}__{group}_{quantity}"])
                    long = float(long_rows[(name, group)][quantity])
                    self.assertAlmostEqual(wide, long, places=12)
                    self.assertAlmostEqual(
                        wide,
                        self.episodes[name]["quantities"][quantity][group]["median"],
                        places=12,
                    )

    def test_cross_configuration_comparison_lists_and_never_merges(self):
        other = self.run_ensemble("--episode", "crossing_A=3:4", label="A")
        comparison = compare_episodes(
            [self.summary(other), self.summary(self.out)], GROUPS
        )
        self.assertEqual(sorted(comparison["per_configuration"]), ["A", "B"])
        self.assertEqual(
            sorted(comparison["per_configuration"]["A"]), ["crossing_A"])
        self.assertEqual(
            sorted(comparison["per_configuration"]["B"]), ["chr", "occ"])
        self.assertIn("never merged", comparison["note"])


class BackwardCompatibilityTests(_Episodes):
    """The single-window interface keeps working, and keeps its own columns."""

    def test_the_legacy_flag_still_fills_the_original_columns(self):
        out = self.run_ensemble("--episode-window", "3:4")
        row = self.per_history(out)[0]
        for group in GROUPS:
            self.assertNotEqual(row[f"episode_net_per_pass_{group}"], "")
            self.assertNotEqual(row[f"episode_occupation_per_pass_{group}"], "")
        self.assertNotEqual(row["episode_verdict"], "not_classified")
        summary = self.summary(out)
        self.assertEqual(summary["episode_window"], [3, 4])
        self.assertEqual(sum(summary["episode_verdicts"].values()), 1)

    def test_the_legacy_flag_and_an_equivalent_named_window_agree(self):
        legacy = self.run_ensemble("--episode-window", "3:4")
        named = self.run_ensemble("--episode", "same=3:4")
        legacy_row = self.per_history(legacy)[0]
        named_summary = self.summary(named)["episodes"]["same"]
        for group in GROUPS:
            self.assertAlmostEqual(
                float(legacy_row[f"episode_net_per_pass_{group}"]),
                named_summary["quantities"]["net_per_pass"][group]["median"],
                places=12,
            )
            self.assertAlmostEqual(
                float(legacy_row[f"episode_occupation_per_pass_{group}"]),
                named_summary["quantities"]["occupation_per_pass"][group]["median"],
                places=12,
            )
        self.assertEqual(legacy_row["episode_verdict"], named_summary["verdicts"].popitem()[0])

    def test_the_legacy_window_gets_no_duplicate_named_columns(self):
        out = self.run_ensemble("--episode-window", "3:4")
        header = self.per_history(out)[0]
        self.assertNotIn("episode__BCF_net_per_pass", header)
        # But it is still a first-class episode in the nested outputs.
        self.assertIn("episode", self.summary(out)["episodes"])
        self.assertTrue(self.summary(out)["episodes"]["episode"]["legacy_single_window"])

    def test_named_windows_do_not_hijack_the_run_level_verdict(self):
        # With four regions there is no basis for calling one of them "the"
        # episode, so the legacy verdict stays unclassified rather than
        # silently reporting whichever window came first.
        out = self.run_ensemble("--episode", "a=3:4", "--episode", "b=9:11")
        row = self.per_history(out)[0]
        self.assertEqual(row["episode_verdict"], "not_classified")
        self.assertEqual(row["episode_net_per_pass_BCF"], "")
        self.assertIsNone(self.summary(out)["episode_window"])

    def test_a_run_with_no_window_at_all_still_produces_everything_else(self):
        out = self.run_ensemble()
        summary = self.summary(out)
        self.assertEqual(summary["episodes"], {})
        self.assertIsNone(summary["episode_window"])
        self.assertEqual(summary["episode_verdicts"], {"not_classified": 1})
        self.assertIn("net_full", summary["quantities"])
        self.assertLess(summary["max_decomposition_residual"], 1e-9)

    def test_the_header_and_the_row_line_up_with_and_without_episodes(self):
        windows = parse_episode_windows(None, ["a=1:2", "b=3:4"])
        result = HistoryResult(run="B", history="SHPROP.1", namdtini=1,
                               n_rows=36, n_passes=3)
        result.episodes["a"] = EpisodeResult(name="a", first=1, last=2)
        self.assertEqual(
            len(history_header(GROUPS, windows)),
            len(result.as_row(GROUPS, windows)),
        )
        self.assertEqual(len(history_header(GROUPS)), len(result.as_row(GROUPS)))


class ControlWindowTests(_Episodes):
    """A control window is labelled a control everywhere it appears."""

    def test_a_control_window_is_recorded_with_its_role(self):
        out = self.run_ensemble("--control-episode", "closest_approach_C=9:11")
        block = self.summary(out)["episodes"]["closest_approach_C"]
        self.assertEqual(block["role"], "control")
        self.assertIn("NOT an avoided crossing", block["role_meaning"])
        self.assertIn("NOT a BCF/PCBM transfer event", block["role_meaning"])
        self.assertEqual(
            [r["role"] for r in self.long_form(out)], ["control"] * len(GROUPS)
        )

    def test_a_control_is_never_described_as_a_crossing(self):
        out = self.run_ensemble("--control-episode", "closest_approach_C=9:11")
        window = self.summary(out)["episode_windows"][0]
        self.assertEqual(window["role"], "control")
        self.assertEqual(window["name"], "closest_approach_C")
        self.assertFalse(window["from_legacy_episode_window_flag"])

    def test_controls_and_crossings_stay_distinguishable_in_one_run(self):
        out = self.run_ensemble(
            "--episode", "crossing_X=3:4",
            "--control-episode", "control_Y=9:11",
        )
        episodes = self.summary(out)["episodes"]
        self.assertEqual(episodes["crossing_X"]["role"], "crossing")
        self.assertEqual(episodes["control_Y"]["role"], "control")
        roles = {r["episode"]: r["role"] for r in self.long_form(out)}
        self.assertEqual(roles, {"crossing_X": "crossing", "control_Y": "control"})


class EpisodeAggregationTests(unittest.TestCase):
    """Aggregation runs over histories, and never over episodes."""

    def _history(self, k, values):
        result = HistoryResult(run="B", history=f"SHPROP.{k}", namdtini=k,
                               n_rows=36, n_passes=3, net_full={"BCF": 0.1})
        for name, (net, occ, chr_) in values.items():
            result.episodes[name] = EpisodeResult(
                name=name, first=1, last=2,
                net_per_pass={"BCF": net, "PCBM": -net},
                occupation_per_pass={"BCF": occ, "PCBM": -occ},
                character_per_pass={"BCF": chr_, "PCBM": -chr_},
                verdict="persistent_donor_gain" if net > 0 else "essentially_reversible",
                n_passes_covering=3, n_steps=6,
            )
        return result

    def test_each_episode_is_summarized_over_the_histories_that_produced_it(self):
        histories = [
            self._history(1, {"B1": (0.2, 0.1, 0.1), "B2": (-0.4, -0.2, -0.2)}),
            self._history(2, {"B1": (0.4, 0.2, 0.2), "B2": (-0.8, -0.4, -0.4)}),
        ]
        out = summarize_episodes(histories, GROUPS)
        self.assertAlmostEqual(
            out["B1"]["quantities"]["net_per_pass"]["BCF"]["median"], 0.3)
        self.assertAlmostEqual(
            out["B2"]["quantities"]["net_per_pass"]["BCF"]["median"], -0.6)
        self.assertEqual(out["B1"]["n_histories"], 2)

    def test_the_verdicts_are_tallied_per_episode(self):
        histories = [
            self._history(1, {"B1": (0.2, 0.1, 0.1), "B2": (-0.4, -0.2, -0.2)}),
            self._history(2, {"B1": (0.4, 0.2, 0.2), "B2": (-0.8, -0.4, -0.4)}),
        ]
        out = summarize_episodes(histories, GROUPS)
        self.assertEqual(out["B1"]["verdicts"], {"persistent_donor_gain": 2})
        self.assertEqual(out["B2"]["verdicts"], {"essentially_reversible": 2})

    def test_a_history_missing_one_episode_shrinks_only_that_denominator(self):
        histories = [
            self._history(1, {"B1": (0.2, 0.1, 0.1), "B2": (-0.4, -0.2, -0.2)}),
            self._history(2, {"B1": (0.4, 0.2, 0.2)}),
        ]
        out = summarize_episodes(histories, GROUPS)
        self.assertEqual(out["B1"]["n_histories"], 2)
        self.assertEqual(out["B2"]["n_histories"], 1)

    def test_the_run_summary_carries_the_episodes_and_the_caveat(self):
        summary = summarize_run(
            "B", [self._history(1, {"B1": (0.2, 0.1, 0.1)})], GROUPS
        )
        self.assertIn("B1", summary["episodes"])
        self.assertEqual(summary["episodes_note"], EPISODES_NOTE)

    def test_a_window_that_moved_between_histories_is_flagged_not_averaged(self):
        first = self._history(1, {"B1": (0.2, 0.1, 0.1)})
        second = self._history(2, {"B1": (0.4, 0.2, 0.2)})
        second.episodes["B1"].last = 99
        out = summarize_episodes([first, second], GROUPS)
        self.assertIn("inconsistent_windows", out["B1"])
        self.assertEqual(out["B1"]["inconsistent_windows"], [[1, 2], [1, 99]])

    def test_the_long_form_rows_carry_every_quantity(self):
        result = self._history(1, {"B1": (0.2, 0.1, 0.1)})
        rows = episode_long_rows(result, GROUPS)
        self.assertEqual(len(rows), len(GROUPS))
        header = episode_long_header()
        for row in rows:
            self.assertEqual(len(row), len(header))
            record = dict(zip(header, row))
            self.assertEqual(record["episode"], "B1")
            self.assertAlmostEqual(
                float(record["net_per_pass"]),
                float(record["occupation_per_pass"])
                + float(record["character_per_pass"]),
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
