"""The production contract: what a configuration needs, and what it does not.

These pin three corrections that are easy to undo by accident:

* the production script must pass the fixed state map, or the central
  fixed-vs-dynamic comparison silently vanishes from the output;
* a crossing window is **optional** -- a configuration whose manifold has not
  been located yet must still produce populations and the decomposition;
* A, B and C are distinct interface configurations, so no shipped text may
  call them replicates or make one a prerequisite for another.

and the study layout the production profile records: where the shared
reference/control workspace is, where each configuration's archive is, where
its analysis output goes, and that the three configurations are described by
the same fields with none of A's values leaking into B or C.
"""

import json
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SBATCH = REPO / "examples" / "bcf_pcbm" / "run_ensemble_production.sbatch"
PROFILE = REPO / "examples" / "bcf_pcbm" / "production_profile.json"
SHIPPED_TEXT = [
    REPO / "src" / "namd_analysis" / "ensemble.py",
    REPO / "src" / "namd_analysis" / "ensemble_cli.py",
    REPO / "src" / "namd_analysis" / "ensemble_plots.py",
    REPO / "docs" / "ensemble_hierarchy.md",
    SBATCH,
]


class ProductionScriptTests(unittest.TestCase):
    def setUp(self):
        self.text = SBATCH.read_text(encoding="utf-8")

    def test_the_fixed_state_map_is_passed(self):
        # Without it the fixed-vs-dynamic comparison is skipped and the
        # central R3.2 figure never appears.
        self.assertIn("--fixed-state-map", self.text)
        self.assertRegex(self.text, r"RUN_FIXED_MAP=\(")

    def test_every_configuration_declares_a_fixed_map(self):
        block = re.search(r"RUN_FIXED_MAP=\((.*?)\)", self.text, re.S).group(1)
        for key in ("[1]", "[2]", "[3]"):
            self.assertIn(key, block, f"no fixed map declared for {key}")

    def test_cycle_length_is_required_provenance(self):
        self.assertIn('if [[ "${CYCLE}" == "0" ]]', self.text)

    def test_the_episode_window_is_not_a_prerequisite(self):
        # The guard must test the cycle length alone. An episode window that
        # gates the whole run would block a configuration whose manifold has
        # not been located yet.
        guard = re.search(r'if \[\[ "\$\{CYCLE\}" == "0".*?\]\]; then', self.text, re.S)
        self.assertIsNotNone(guard)
        self.assertNotIn("EPISODE", guard.group(0))
        self.assertIn("EPISODE_ARGS=()", self.text)
        self.assertIn("Episode classification is skipped", self.text)

    def test_the_required_inputs_are_checked_and_named(self):
        self.assertIn('for required in "${CHAR}" "${STATE_MAP}" "${FIXED_MAP}"', self.text)


class ConfigurationLanguageTests(unittest.TestCase):
    """A/B/C are configurations, not replicates, and not prerequisites."""

    FORBIDDEN = (
        "reproducibility unit",
        "licenses a Campaign-level statement",
        "only level that licenses",
        "across-run",
        "per-run",
        "n = 3 replicates",
    )

    def test_no_shipped_text_calls_them_replicates_or_a_reproducibility_unit(self):
        for path in SHIPPED_TEXT:
            text = path.read_text(encoding="utf-8")
            for phrase in self.FORBIDDEN:
                self.assertNotIn(
                    phrase, text,
                    f"{path.name} still says {phrase!r}; A/B/C are distinct "
                    "interface configurations",
                )

    def test_the_hierarchy_note_says_they_are_distinct_configurations(self):
        from namd_analysis.ensemble import HIERARCHY_NOTE

        self.assertIn("DISTINCT INTERFACE CONFIGURATIONS", HIERARCHY_NOTE)
        self.assertIn("not statistical replicates", HIERARCHY_NOTE)

    def test_one_configuration_is_interpretable_alone(self):
        doc = (REPO / "docs" / "ensemble_hierarchy.md").read_text(encoding="utf-8")
        self.assertIn("license Campaign A statements on their own", doc)
        self.assertIn("not a prerequisite", doc)

    def test_the_comparison_is_named_a_comparison_not_a_level(self):
        doc = (REPO / "docs" / "ensemble_hierarchy.md").read_text(encoding="utf-8")
        self.assertIn("comparison, not a statistical level", doc)


def _bash_array(text, name):
    """Read a ``declare -A NAME=( [k]=v ... )`` block into a dict."""
    block = re.search(r"declare -A %s=\((.*?)\n\)" % name, text, re.S)
    if block is None:
        block = re.search(r"declare -A %s=\((.*?)\)" % name, text, re.S)
    entries = re.findall(r"\[(\d+)\]=(\S*)", block.group(1))
    return {key: value.strip('"') for key, value in entries}


class ProductionProfileTests(unittest.TestCase):
    """The recorded study layout, and the symmetry of A, B and C inside it."""

    def setUp(self):
        self.profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        self.configs = self.profile["configurations"]

    def test_the_reference_root_is_the_shared_control_workspace(self):
        study = self.profile["study"]
        self.assertEqual(study["root"], "/ddnB/work/lgutsev/MD/FAPI_MD_NAMD_2026")
        self.assertEqual(study["reference_root"], "NuTest")
        # Stated, not implied: NuTest belongs to the study, not to A.
        self.assertIn("not configuration a", study["reference_root_role"].lower())

    def test_each_configuration_has_its_own_production_archive(self):
        for label in ("A", "B", "C"):
            self.assertEqual(
                self.configs[label]["data_dir"], f"FAPI_001_BCF_PCBM_{label}"
            )

    def test_analysis_output_goes_under_the_configuration_it_belongs_to(self):
        self.assertEqual(self.profile["study"]["analysis_subdir"], "analysis")
        for label in ("A", "B", "C"):
            entry = self.configs[label]
            self.assertEqual(entry["analysis_dir"], entry["data_dir"] + "/analysis")

    def test_every_configuration_is_described_by_the_same_fields(self):
        declared = set(self.profile["configuration_fields"])
        for label in ("A", "B", "C"):
            entry = self.configs[label]
            self.assertTrue(
                declared.issubset(set(entry)),
                f"{label} is missing {sorted(declared - set(entry))}",
            )
        # And the same fields as each other, notes included.
        shapes = {label: sorted(self.configs[label]) for label in ("A", "B", "C")}
        self.assertEqual(shapes["A"], shapes["B"])
        self.assertEqual(shapes["A"], shapes["C"])

    def test_provenance_is_tracked_by_the_same_items_for_all_three(self):
        items = set(self.profile["provenance_fields"])
        states = set(self.profile["provenance_states"])
        for label in ("A", "B", "C"):
            provenance = self.configs[label]["provenance"]
            self.assertEqual(set(provenance), items, f"{label} tracks different items")
            for item, state in provenance.items():
                self.assertIn(state, states, f"{label}.{item} has state {state!r}")

    def test_no_campaign_a_value_is_carried_into_b_or_c(self):
        a = self.configs["A"]
        for label in ("B", "C"):
            other = self.configs[label]
            for field in (
                "data_dir", "reference_dir", "analysis_dir", "projection_character",
                "state_map", "fixed_state_map", "atom_groups", "projection_manifest",
            ):
                self.assertNotEqual(
                    a[field], other[field],
                    f"{label} points at A's {field}; reference products do not transfer",
                )
            self.assertIsNone(other["episode_window"], f"{label} carries a crossing window")
            self.assertIsNone(
                other["history_count_observed"],
                f"{label} carries a history count it has not been counted for",
            )
            self.assertFalse(other["preset_registered"])

    def test_the_shared_trajectory_is_shared_and_says_why(self):
        shared = self.profile["shared"]
        self.assertEqual(shared["cycle_length"], 1999)
        for label in ("A", "B", "C"):
            self.assertEqual(self.configs[label]["cycle_length"], 1999)
        # Shared is not unchecked, and nothing scientific rides along with it.
        self.assertIn("confirm", shared["cycle_length_basis"].lower())
        for item in ("VASP band numbers", "atom partition", "fixed nominal state map"):
            self.assertIn(item, shared["not_shared"])

    def test_the_real_archive_denominator_is_recorded_as_it_stands(self):
        self.assertEqual(self.configs["A"]["history_count_observed"], 98)


class ProfileMatchesTheProductionScriptTests(unittest.TestCase):
    """The script and the profile describe one layout, or the run is elsewhere."""

    def setUp(self):
        self.text = SBATCH.read_text(encoding="utf-8")
        self.profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        self.labels = _bash_array(self.text, "RUN_LABEL")

    def test_the_roots_agree(self):
        study = self.profile["study"]
        self.assertIn(f"ROOT={study['root']}", self.text)
        self.assertIn(
            'REFERENCE_ROOT="${ROOT}/%s"' % study["reference_root"], self.text
        )

    def test_the_archive_directories_agree(self):
        dirs = _bash_array(self.text, "RUN_DIR")
        for index, label in self.labels.items():
            self.assertEqual(
                dirs[index],
                "${ROOT}/" + self.profile["configurations"][label]["data_dir"],
            )

    def test_the_output_directory_agrees(self):
        prefix = self.profile["study"]["analysis_run_prefix"]
        self.assertIn(
            'OUT="${DIR}/%s/%s_${JOBID}"' % (self.profile["study"]["analysis_subdir"], prefix),
            self.text,
        )

    def test_the_cycle_lengths_and_windows_agree(self):
        cycles = _bash_array(self.text, "RUN_CYCLE")
        episodes = _bash_array(self.text, "RUN_EPISODE")
        for index, label in self.labels.items():
            entry = self.profile["configurations"][label]
            self.assertEqual(int(cycles[index]), entry["cycle_length"])
            self.assertEqual(episodes[index] or None, entry["episode_window"])


if __name__ == "__main__":
    unittest.main()
