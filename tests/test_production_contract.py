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


class ScientificWordingTests(unittest.TestCase):
    """The distinctions the shipped text must keep, checked at the source.

    Each of these is a sentence somebody will be tempted to shorten. They are
    pinned as strings because the whole point is the wording: a summary that
    says "no transfer" where it means "not evaluated", or "replicates" where it
    means "different physical systems", is wrong in a way no numeric test
    catches.
    """

    def test_a_character_swap_is_not_a_surface_hop(self):
        from namd_analysis import crossings

        # Wrapped across lines in the source, so compare on normalized text.
        prose = " ".join(crossings.__doc__.split())
        self.assertIn("a character swap is not a surface hop", prose)
        self.assertIn("a surface hop is not charge transfer", prose)

    def test_a_character_swap_is_not_automatically_charge_transfer(self):
        from namd_analysis.crossings import NO_PROJECTED_CHANGE, _classify

        label, note = _classify(True, True, True, 0.0, 1.0e-3)
        self.assertEqual(label, NO_PROJECTED_CHANGE)
        self.assertIn("did not move beyond tolerance", note)

    def test_a_character_driven_change_is_never_called_no_charge_moved(self):
        from namd_analysis.crossings import PROJECTED_CHANGE, _classify

        # Occupation flat, character carrying everything: the label must still
        # record that P_g moved, and the note must say so in words.
        label, note = _classify(
            True, True, True, 0.5, 1.0e-3, None, "character_dominated"
        )
        self.assertEqual(label, PROJECTED_CHANGE)
        self.assertIn("MUST NOT be reported as 'no charge moved'", note)
        self.assertIn("adiabatic passage", note)
        self.assertNotIn("no charge moved between fragments", note)

    #: Phrasings that assert a character-driven change moved no charge. Each
    #: was shipped at some point, and each is wrong: P_g sums over the whole
    #: basis, so a relabelling cannot move it, and a state whose character
    #: turns can correspond to a spatial redistribution of its density.
    FORBIDDEN_MECHANISM_CLAIMS = (
        "no charge going anywhere",
        "no charge moved between fragments",
        "with no charge going anywhere",
        "classified as state relabelling",
        "only the former corresponds to charge motion",
        "moves no charge",
        "most of it is relabelling",
        "character_swap_without_fragment_transfer",
    )

    def test_no_shipped_text_says_a_character_driven_change_moved_no_charge(self):
        roots = [REPO / "src" / "namd_analysis", REPO / "docs"]
        checked = 0
        for root in roots:
            for path in sorted(root.rglob("*")):
                if path.suffix not in (".py", ".md") or "__pycache__" in path.parts:
                    continue
                checked += 1
                text = path.read_text(encoding="utf-8")
                for phrase in self.FORBIDDEN_MECHANISM_CLAIMS:
                    self.assertNotIn(
                        phrase, text,
                        f"{path.name} says {phrase!r}. A change in P_g carried "
                        "by character evolution is not a relabelling; it can "
                        "correspond to a spatial redistribution of density",
                    )
        self.assertGreater(checked, 10, "the sweep found almost no files")
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        for phrase in self.FORBIDDEN_MECHANISM_CLAIMS:
            self.assertNotIn(phrase, readme)

    def test_no_shipped_text_quotes_the_endpoint_biased_decomposition(self):
        """Prose must not claim a split the code does not compute.

        ``history_fragment_population`` uses the symmetric midpoint form. The
        endpoint-biased form is equally exact *in the sum*, so nothing numeric
        catches the difference -- but it apportions up to half a step's
        movement differently between the two terms, which is exactly what
        ``occupation_redistribution`` and ``character_evolution`` report. A
        document quoting the endpoint form would misdescribe every one of
        those numbers.
        """
        endpoint = [
            # dP_g^pop written with the weight at one endpoint, not the mean.
            re.compile(r"P_i\(t\s*[-−]\s*1\)\s*\]\s*[·*]?\s*\{?\s*w_ig"),
            # dP_g^char written with P_i at one endpoint, not the mean.
            re.compile(r"P_i\(t\s*[-−]\s*1\)\s*[·*]?\s*[\[{(]\s*w_ig"),
            re.compile(r"P_i\^?\{?t\}?\s*[·*]\s*[ΔdD]w_i"),
        ]
        roots = [REPO / "src" / "namd_analysis", REPO / "docs"]
        paths = [
            path
            for root in roots
            for path in sorted(root.rglob("*"))
            if path.suffix in (".py", ".md") and "__pycache__" not in path.parts
        ] + [REPO / "README.md"]
        self.assertGreater(len(paths), 10, "the sweep found almost no files")
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for pattern in endpoint:
                match = pattern.search(text)
                if match is not None:
                    self.fail(
                        f"{path.name} quotes an endpoint-biased decomposition "
                        f"({match.group(0)!r} at offset {match.start()}). The "
                        "code computes the symmetric midpoint form; see "
                        "crossings.DECOMPOSITION_IDENTITY"
                    )

    def test_the_identity_constant_is_the_midpoint_form(self):
        from namd_analysis.crossings import DECOMPOSITION_IDENTITY

        for half in (
            "0.5*(w_ig[f(t)] + w_ig[f(t-1)])",
            "0.5*(P_i(t) + P_i(t-1))",
        ):
            self.assertIn(half, DECOMPOSITION_IDENTITY)

    def test_the_report_quotes_the_constant_rather_than_retyping_it(self):
        # A retyped formula is one that can drift. The CLI payload must take
        # the identity from the module, so there is one place to change.
        source = (
            REPO / "src" / "namd_analysis" / "crossings_cli.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"identity": DECOMPOSITION_IDENTITY', source)

    def test_the_split_is_described_never_used_as_a_mechanism(self):
        from namd_analysis.crossings import PROJECTION_NOTE

        # The correction that must survive: character_evolution is not a
        # relabelling and can be spatial redistribution.
        self.assertIn("NOT mere relabelling", PROJECTION_NOTE)
        self.assertIn("spatial redistribution of the occupied density", PROJECTION_NOTE)
        self.assertIn("'no charge moved' is wrong", PROJECTION_NOTE)
        # And the limit that must travel with it: neither term is charge, and
        # P_g is a diagonal quantity rather than exact fragment charge.
        self.assertIn(
            "Neither term of the split may be equated with charge motion",
            PROJECTION_NOTE,
        )
        self.assertIn("NOT exact fragment charge", PROJECTION_NOTE)
        self.assertIn("no coherences", PROJECTION_NOTE)
        self.assertIn("unassigned", PROJECTION_NOTE)
        self.assertIn(
            "bookkeeping convention rather than a branching fraction",
            PROJECTION_NOTE,
        )
        self.assertIn("no surface-hopping record is an input", PROJECTION_NOTE)

    def test_p_g_is_named_a_projection_weighted_diagonal_population(self):
        from namd_analysis.crossings import PROJECTION_NOTE

        self.assertIn("projection-weighted DIAGONAL fragment population", PROJECTION_NOTE)
        # And nothing in the crossings family leaves the qualifier off.
        for name in ("crossings.py", "crossings_cli.py", "crossing_summary.py"):
            text = (REPO / "src" / "namd_analysis" / name).read_text(encoding="utf-8")
            self.assertNotIn(
                "projection-weighted fragment population", text,
                f"{name} calls P_g a projection-weighted fragment population "
                "without saying it is diagonal",
            )

    def test_no_shipped_text_asserts_a_split_term_moves_real_charge(self):
        forbidden = (
            "Both terms move real charge",
            "both terms move real charge",
            "Both move the occupied density",
            "displace the occupied density",
            "moves its density between the fragments in real space",
        )
        roots = [REPO / "src" / "namd_analysis", REPO / "docs"]
        for root in roots:
            for path in sorted(root.rglob("*")):
                if path.suffix not in (".py", ".md") or "__pycache__" in path.parts:
                    continue
                text = path.read_text(encoding="utf-8")
                for phrase in forbidden:
                    self.assertNotIn(
                        phrase, text,
                        f"{path.name} says {phrase!r}. P_g is a diagonal "
                        "quantity; a term of the split is not a quantity of "
                        "charge that moved",
                    )

    def test_a_configuration_only_scan_cannot_determine_population_transfer(self):
        from namd_analysis.crossings import (
            NO_PROJECTED_CHANGE,
            NOT_EVALUATED,
            _classify,
        )

        label, note = _classify(True, True, True, None, 1.0e-3)
        self.assertEqual(label, NOT_EVALUATED)
        self.assertNotEqual(label, NO_PROJECTED_CHANGE)
        self.assertIn("NOT EVALUATED", note)
        self.assertIn("nothing here rules it out", note)

    def test_the_decomposition_is_called_bookkeeping_not_a_mechanism(self):
        from namd_analysis.ensemble import BOOKKEEPING_NOTE

        self.assertIn("exact", BOOKKEEPING_NOTE)
        self.assertIn("one of infinitely many exact splits", BOOKKEEPING_NOTE)
        self.assertIn("bookkeeping convention", BOOKKEEPING_NOTE)
        self.assertIn("NOT physical branching fractions", BOOKKEEPING_NOTE)

    def test_repeated_passes_are_not_independent(self):
        from namd_analysis.ensemble import HIERARCHY_NOTE

        self.assertIn(
            "re-traversals of one recycled nuclear trajectory", HIERARCHY_NOTE)
        self.assertIn("not independent samples", HIERARCHY_NOTE)
        self.assertIn("No standard error is quoted over passes", HIERARCHY_NOTE)

    def test_configurations_are_not_statistical_replicates(self):
        from namd_analysis.ensemble import HIERARCHY_NOTE

        self.assertIn("DISTINCT INTERFACE CONFIGURATIONS", HIERARCHY_NOTE)
        self.assertIn("not statistical replicates", HIERARCHY_NOTE)
        self.assertIn("never averaged or pooled", HIERARCHY_NOTE)

    def test_several_windows_in_one_configuration_are_not_samples_either(self):
        from namd_analysis.ensemble import EPISODES_NOTE

        self.assertIn("NOT independent samples", EPISODES_NOTE)
        self.assertIn("NOT replicates of one another", EPISODES_NOTE)
        self.assertIn("a mean over B1..B4", EPISODES_NOTE)

    def test_a_control_window_is_never_called_a_crossing(self):
        from namd_analysis.ensemble import EPISODE_ROLES

        control = EPISODE_ROLES["control"]
        self.assertIn("NOT an avoided crossing", control)
        self.assertIn("NOT a BCF/PCBM transfer event", control)


def _bash_array(text, name):
    """Read a ``declare -A NAME=( [k]=v ... )`` block into a dict.

    Values may be quoted and may contain spaces -- one episode array entry
    holds several ``NAME=FIRST:LAST`` specs -- so a quoted value is matched
    whole rather than up to the first space.
    """
    # Single-line form first. A multi-line search would otherwise run past the
    # closing paren of a one-line array and swallow the next declaration.
    block = re.search(r"declare -A %s=\(([^()\n]*)\)" % name, text)
    if block is None:
        block = re.search(r"declare -A %s=\((.*?)\n\)" % name, text, re.S)
    entries = re.findall(r'\[(\d+)\]=(?:"([^"]*)"|(\S*))', block.group(1))
    return {key: (quoted if quoted else bare) for key, quoted, bare in entries}


def _episode_specs(value):
    """``"a=1:2 b=3:4"`` into ``{"a": "1:2", "b": "3:4"}``."""
    out = {}
    for item in value.split():
        name, _, window = item.partition("=")
        out[name] = window
    return out


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
            self.assertFalse(
                set(a["episodes"]) & set(other["episodes"]),
                f"{label} shares a named crossing window with A; windows are "
                "located per configuration and do not transfer",
            )
            self.assertFalse(
                set(a["episodes"].values()) & set(other["episodes"].values()),
                f"{label} reuses one of A's frame ranges",
            )
            self.assertIsNone(
                other["history_count_observed"],
                f"{label} carries a history count it has not been counted for",
            )
            self.assertFalse(other["preset_registered"])

    def test_configuration_b_carries_its_four_distinct_mixing_regions(self):
        episodes = self.configs["B"]["episodes"]
        self.assertEqual(
            episodes,
            {
                "crossing_B1": "1488:1492",
                "crossing_B2": "1687:1696",
                "crossing_B3": "1801:1811",
                "crossing_B4": "1846:1855",
            },
        )
        # None of the four is promoted to "the" window: they are four regions
        # of one recycled trajectory, and picking one would have no basis.
        self.assertIsNone(self.configs["B"]["episode_window"])
        windows = sorted(tuple(int(x) for x in w.split(":")) for w in episodes.values())
        for earlier, later in zip(windows, windows[1:]):
            self.assertLess(earlier[1], later[0], "B's regions must be disjoint")

    def test_configuration_c_has_no_crossing_episode(self):
        self.assertEqual(self.configs["C"]["episodes"], {})
        self.assertIsNone(self.configs["C"]["episode_window"])

    def test_the_c_closest_approach_window_is_recorded_only_as_a_control(self):
        entry = self.configs["C"]
        self.assertEqual(
            entry["control_episodes"], {"closest_approach_C": "1628:1632"}
        )
        self.assertNotIn("closest_approach_C", entry["episodes"])
        notes = " ".join(entry["notes"]).lower()
        self.assertIn("control window", notes)
        self.assertIn("not an avoided crossing", notes)
        self.assertIn("not a bcf/pcbm transfer event", notes)

    def test_the_profile_states_that_windows_are_never_pooled(self):
        conventions = self.profile["episode_conventions"]
        self.assertIn("never averaged, pooled or combined", conventions["episodes"])
        self.assertIn("NOT an avoided crossing", conventions["control_episodes"])
        self.assertIn("same streamed pass", conventions["one_pass"])
        self.assertIn("no level at which averaging", conventions["not_pooled"])

    def test_b_and_c_are_not_described_as_replicates_of_each_other(self):
        for label in ("B", "C"):
            notes = " ".join(self.configs[label]["notes"]).lower()
            self.assertNotIn("replicate of", notes)
        b_notes = " ".join(self.configs["B"]["notes"]).lower()
        self.assertIn("not four replicates", b_notes)
        self.assertIn("same recycled nuclear trajectory", b_notes)

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

    def test_the_cycle_lengths_agree(self):
        cycles = _bash_array(self.text, "RUN_CYCLE")
        for index, label in self.labels.items():
            entry = self.profile["configurations"][label]
            self.assertEqual(int(cycles[index]), entry["cycle_length"])

    def test_the_named_crossing_windows_agree(self):
        episodes = _bash_array(self.text, "RUN_EPISODES")
        for index, label in self.labels.items():
            entry = self.profile["configurations"][label]
            self.assertEqual(
                _episode_specs(episodes[index]), entry["episodes"],
                f"configuration {label}: the script and the profile disagree "
                "about which crossing regions it has",
            )

    def test_the_control_windows_agree_and_stay_separate_from_crossings(self):
        controls = _bash_array(self.text, "RUN_CONTROL_EPISODES")
        episodes = _bash_array(self.text, "RUN_EPISODES")
        for index, label in self.labels.items():
            entry = self.profile["configurations"][label]
            self.assertEqual(
                _episode_specs(controls[index]), entry["control_episodes"],
                f"configuration {label}: the script and the profile disagree "
                "about its control windows",
            )
            # A control must never be smuggled in as a crossing.
            self.assertFalse(
                set(_episode_specs(controls[index]))
                & set(_episode_specs(episodes[index])),
                f"configuration {label} declares one window as both a crossing "
                "and a control",
            )

    def test_the_legacy_single_window_flag_is_still_wired_up(self):
        # Backward compatibility: the array and the flag it feeds stay, so an
        # existing single-window recipe keeps working unchanged.
        legacy = _bash_array(self.text, "RUN_EPISODE")
        self.assertEqual(sorted(legacy), sorted(self.labels))
        self.assertIn("--episode-window", self.text)


if __name__ == "__main__":
    unittest.main()
