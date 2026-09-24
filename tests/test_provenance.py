"""Code provenance: which implementation produced an output.

A version string names a release. It does not name the commit that release was
cut from, and it certainly does not name an editable checkout that has moved on
since. The A/B/C comparison is only valid if the three configurations were
analysed by the *same* implementation, so every report records the commit when
one can be had, and says plainly when one cannot.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

from namd_analysis import __version__
from namd_analysis.provenance import (
    _code_provenance_uncached,
    _porcelain_paths,
    code_provenance,
    environment,
)

REPO = Path(__file__).resolve().parents[1]


class PorcelainParsingTests(unittest.TestCase):
    """``git status --porcelain`` keeps its status flags in two fixed columns."""

    def test_an_unstaged_change_keeps_its_whole_path(self):
        # The first column is blank for an unstaged change. Stripping it would
        # shift every path by one character -- "docs/x.md" becoming "ocs/x.md".
        self.assertEqual(
            _porcelain_paths(" M docs/adiabatic_vs_diabatic.md"),
            ["docs/adiabatic_vs_diabatic.md"],
        )

    def test_staged_untracked_and_renamed_entries(self):
        self.assertEqual(
            _porcelain_paths("MM src/b.py\n?? new.txt\nR  old.py -> new.py\n"),
            ["src/b.py", "new.txt", "new.py"],
        )

    def test_nothing_and_none_are_empty(self):
        self.assertEqual(_porcelain_paths(None), [])
        self.assertEqual(_porcelain_paths(""), [])
        self.assertEqual(_porcelain_paths("\n\n"), [])


class CodeProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.record = code_provenance()

    def test_the_package_version_is_always_recorded(self):
        self.assertEqual(self.record["version"], __version__)
        self.assertIsNotNone(self.record["package_path"])

    def test_it_never_raises_and_is_json_serializable(self):
        json.dumps(self.record)

    def test_a_git_checkout_yields_a_commit_that_matches_git(self):
        try:
            expected = subprocess.run(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            self.skipTest("git is not available here")
        if expected.returncode != 0:
            self.skipTest("not a git checkout")

        git = self.record["git"]
        self.assertTrue(git["available"], git.get("reason"))
        self.assertEqual(git["commit"], expected.stdout.strip())
        self.assertEqual(git["commit_short"], git["commit"][:12])
        self.assertEqual(len(git["commit"]), 40)
        self.assertTrue(self.record["editable_checkout"])
        self.assertIsInstance(git["dirty"], bool)
        self.assertIn("dirty=true means", git["note"])

    def test_a_dirty_tree_is_reported_as_such(self):
        git = self.record["git"]
        if not git.get("available"):
            self.skipTest("no git commit available here")
        # The flag and the path list must agree: a claim of clean with listed
        # paths, or dirty with none, would misdescribe the run.
        self.assertEqual(git["dirty"], bool(git["uncommitted_paths"]) or git["dirty"])
        if git["uncommitted_paths"]:
            self.assertTrue(git["dirty"])

    def test_it_degrades_without_git_rather_than_failing(self):
        # Provenance that could abort an analysis is worse than provenance
        # that says it does not know.
        import namd_analysis.provenance as provenance

        real = subprocess.run

        def boom(*args, **kwargs):
            raise OSError("git not found")

        _code_provenance_uncached.cache_clear()
        subprocess.run = boom
        try:
            record = provenance.code_provenance()
        finally:
            subprocess.run = real
            _code_provenance_uncached.cache_clear()
        self.assertEqual(record["version"], __version__)
        self.assertFalse(record["git"]["available"])
        self.assertIn("git not found", record["git"]["reason"])
        self.assertFalse(record["editable_checkout"])
        json.dumps(record)

    def test_the_lookup_is_cached_but_hands_back_a_private_copy(self):
        # Every report asks for this, and a production run must not pay for
        # several git subprocesses per output file. A cached dict shared by
        # reference would let one report's edits leak into the next.
        first, second = code_provenance(), code_provenance()
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        first["git"]["commit"] = "MUTATED"
        self.assertNotEqual(code_provenance()["git"].get("commit"), "MUTATED")


class EnvironmentTests(unittest.TestCase):
    def test_every_report_carries_the_code_record(self):
        env = environment(argv=["namd-analysis", "character-ensemble"])
        self.assertIn("code", env)
        self.assertEqual(env["code"]["version"], __version__)
        self.assertEqual(env["version"], __version__)
        json.dumps(env)


class VersionTests(unittest.TestCase):
    """The declared version, the profile and the output prefix are one value."""

    def test_the_package_and_pyproject_agree(self):
        text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(f'version = "{__version__}"', text)

    def test_the_production_profile_names_this_version(self):
        profile = json.loads(
            (REPO / "examples" / "bcf_pcbm" / "production_profile.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(profile["analysis_version"], __version__)
        self.assertEqual(
            profile["study"]["analysis_run_prefix"], f"ensemble_v{__version__}"
        )
        self.assertIn(
            f"ensemble_v{__version__}_<jobid>",
            profile["study"]["analysis_run_pattern"],
        )

    def test_the_production_script_writes_into_this_versions_directory(self):
        text = (
            REPO / "examples" / "bcf_pcbm" / "run_ensemble_production.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn(f"ensemble_v{__version__}_${{JOBID}}", text)
        # And the combine example it prints must point at the same runs.
        self.assertIn(f"ensemble_v{__version__}_*/run_summary.json", text)

    def test_no_output_path_still_names_the_frozen_predecessor(self):
        # v0.7.1-frozen predates the multi-episode windows and the corrected
        # projected-population semantics; an output directory carrying that
        # prefix would misattribute these results. Prose *explaining* why the
        # version moved is fine and wanted -- it is the paths that must not.
        for name in ("production_profile.json", "run_ensemble_production.sbatch"):
            text = (REPO / "examples" / "bcf_pcbm" / name).read_text(encoding="utf-8")
            self.assertNotIn(
                "ensemble_v0.7.1", text,
                f"{name} still writes into an ensemble_v0.7.1 directory",
            )

    def test_the_profile_says_why_it_left_the_frozen_predecessor(self):
        profile = json.loads(
            (REPO / "examples" / "bcf_pcbm" / "production_profile.json")
            .read_text(encoding="utf-8")
        )
        basis = profile["analysis_version_basis"]
        self.assertIn("v0.7.1-frozen", basis)
        self.assertIn("not comparable", basis)
        self.assertIn("environment.code", basis)

    def test_the_script_reports_the_code_before_it_runs(self):
        text = (
            REPO / "examples" / "bcf_pcbm" / "run_ensemble_production.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("code_provenance()", text)
        self.assertIn("UNCOMMITTED CHANGES", text)


if __name__ == "__main__":
    sys.exit(unittest.main())
