"""The production contract: what a configuration needs, and what it does not.

These pin three corrections that are easy to undo by accident:

* the production script must pass the fixed state map, or the central
  fixed-vs-dynamic comparison silently vanishes from the output;
* a crossing window is **optional** -- a configuration whose manifold has not
  been located yet must still produce populations and the decomposition;
* A, B and C are distinct interface configurations, so no shipped text may
  call them replicates or make one a prerequisite for another.
"""

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SBATCH = REPO / "examples" / "bcf_pcbm" / "run_ensemble_production.sbatch"
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


if __name__ == "__main__":
    unittest.main()
