"""Aggregation within a campaign, and comparison between campaigns.

    passes  subset of  SHPROP histories  subset of  campaign

**Passes** are re-traversals of one recycled nuclear trajectory. They are not
independent samples of anything, so nothing here computes a standard error over
passes.

**Histories** are the electronic-history ensemble of one campaign: 100 SHPROP
files differing in electronic initial condition (``NAMDTINI``) and in the
stochastic surface-hopping realization, on that campaign's nuclear trajectory.
They *are* the statistical unit within a campaign. They are **not** independent
nuclear configurations, so a spread across them is a spread over electronic
initial conditions at fixed nuclei.

**Campaigns are not a statistical level at all.** A, B and C are *distinct
interface configurations* -- different physical systems, each with its own band
map, atom partition, cycle length and crossing manifold. They are **not**
replicate runs of one system, and nothing here averages them, pools them, or
treats their agreement as reproducibility. Comparing them is a comparison of
physical cases, and a difference between them is a result about the systems
rather than scatter about a common truth.

So: per-history quantities are reported individually; per-campaign statistics
summarize that campaign's 100 histories; and the across-campaign table sets the
three cases side by side. A grand 300-history average is never the headline --
it would mix three physically different configurations into one meaningless
number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "EnsembleError",
    "SIGN_TOLERANCE",
    "LEGACY_EPISODE_NAME",
    "EPISODE_QUANTITIES",
    "EPISODE_ROLES",
    "EpisodeWindow",
    "EpisodeResult",
    "parse_episode_window",
    "parse_episode_windows",
    "episode_header",
    "episode_long_header",
    "episode_long_rows",
    "HistoryResult",
    "history_header",
    "summarize_run",
    "summarize_episodes",
    "compare_episodes",
    "compare_runs",
    "reviewer_table",
    "ensemble_curve",
]

#: A net change smaller than this counts as neither gain nor loss.  Reported
#: alongside every sign fraction so the reader can see what was called flat.
SIGN_TOLERANCE = 1.0e-3

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)

HIERARCHY_NOTE = (
    "passes are re-traversals of one recycled nuclear trajectory and are not "
    "independent samples; the 100 histories are the electronic-history ensemble "
    "of one campaign, differing in electronic initial condition at fixed nuclei, "
    "and are NOT independent nuclear configurations; campaigns A, B and C are "
    "DISTINCT INTERFACE CONFIGURATIONS, not statistical replicates of one "
    "system, so they are never averaged or pooled and their agreement is not a "
    "reproducibility check. No standard error is quoted over passes, and no "
    "grand 300-history average is reported as a result"
)

BOOKKEEPING_NOTE = (
    "occupation_redistribution and character_evolution come from an exact, "
    "endpoint-unbiased split of the population change. It is one of infinitely "
    "many exact splits: their relative sizes are a bookkeeping convention and "
    "are NOT physical branching fractions of the Hamiltonian dynamics"
)

#: The name given to the window supplied through the legacy single
#: ``--episode-window``. It is an ordinary episode in every output; the only
#: thing special about it is that it also fills the original unprefixed
#: ``episode_*`` columns, so a downstream reader written against the
#: single-window interface keeps working unchanged.
LEGACY_EPISODE_NAME = "episode"

#: Everything retained per history and per named episode.
EPISODE_QUANTITIES = ("net_per_pass", "occupation_per_pass", "character_per_pass")

#: What a window is being used for. A window is a range of frames; what it
#: *means* is a scientific claim, and the two are recorded separately so that a
#: control can never be read back as a crossing.
EPISODE_ROLES = {
    "crossing": (
        "a located BCF/PCBM character-exchange region of this configuration"
    ),
    "control": (
        "an explicitly labelled CONTROL window. It is NOT an avoided crossing "
        "and NOT a BCF/PCBM transfer event: it is a range of frames chosen for "
        "comparison, and no character-exchange or transfer interpretation "
        "attaches to it"
    ),
}

EPISODES_NOTE = (
    "each named episode is a distinct region of the SAME recycled nuclear "
    "trajectory. The episodes are NOT independent samples, NOT replicates of "
    "one another and NOT repeat measurements of one quantity, so nothing here "
    "averages, pools or combines them: a mean over B1..B4 would be a mean over "
    "four different parts of one trajectory and would measure nothing. Each is "
    "reported on its own, per history and per pass. A window whose role is "
    "'control' is a labelled comparison window, not an avoided crossing"
)

ONE_PASS_NOTE = (
    "every named episode was evaluated during the SAME streamed pass over each "
    "SHPROP history: the file is read once, the symmetric midpoint "
    "decomposition is formed once, and each episode's frame mask is applied to "
    "that one result. Adding an episode does not reread the archive, and cannot "
    "change another episode's numbers"
)


class EnsembleError(ValueError):
    """Raised when an aggregation cannot be formed from the inputs given."""


# --------------------------------------------------------------------------
# Episode windows: several per run, named, never pooled
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeWindow:
    """One named, inclusive band of MD frames to evaluate an episode over.

    A configuration may have several. Configuration B has four distinct
    BCF/PCBM mixing regions, and they are four separate regions of one
    recycled nuclear trajectory -- not four samples of one region. They
    therefore travel by *name* from the command line through to every output,
    and are never reduced to a single "the episode".
    """

    name: str
    first: int
    last: int
    #: ``crossing`` or ``control``; see :data:`EPISODE_ROLES`.
    role: str = "crossing"
    #: True only for the window given through the legacy ``--episode-window``.
    legacy: bool = False

    def as_spec(self) -> str:
        return f"{self.name}={self.first}:{self.last}"


@dataclass
class EpisodeResult:
    """One history's response inside one named episode.

    Everything is **per pass**: a 10M-step history re-traverses a 1999-frame
    cycle ~5003 times, so a sum over the window across the whole history adds
    thousands of re-encounters of one geometry and is not a population.
    """

    name: str
    first: int
    last: int
    role: str = "crossing"
    #: Mean per-pass net fragment change inside the window.
    net_per_pass: Dict[str, float] = field(default_factory=dict)
    #: The occupation-redistribution contribution to that net, per pass.
    occupation_per_pass: Dict[str, float] = field(default_factory=dict)
    #: The character-evolution contribution to that net, per pass.
    character_per_pass: Dict[str, float] = field(default_factory=dict)
    verdict: str = "not_classified"
    #: Passes with at least one step inside the window, and steps in total.
    n_passes_covering: int = 0
    n_steps: int = 0

    def quantity(self, which: str) -> Dict[str, float]:
        if which not in EPISODE_QUANTITIES:
            raise EnsembleError(f"unknown episode quantity {which!r}")
        return getattr(self, which)


def parse_episode_window(
    spec: str, default_name: str = LEGACY_EPISODE_NAME, role: str = "crossing"
) -> EpisodeWindow:
    """``NAME=FIRST:LAST``, or a bare ``FIRST:LAST`` taking *default_name*.

    The name is what makes an episode unambiguous downstream, so it is
    validated here rather than left to produce a surprising CSV column: it must
    be a plain identifier, and it must not contain the ``__`` that separates an
    episode from its group in a column name.
    """
    if role not in EPISODE_ROLES:
        raise EnsembleError(
            f"unknown episode role {role!r}; expected one of {sorted(EPISODE_ROLES)}"
        )
    text = str(spec).strip()
    if not text:
        raise EnsembleError("empty episode specification")
    name, sep, window = text.partition("=")
    if not sep:
        name, window = default_name, text
    name, window = name.strip(), window.strip()
    if not name:
        raise EnsembleError(f"episode {spec!r} has an empty name")
    if "__" in name or not all(c.isalnum() or c in "_-." for c in name):
        raise EnsembleError(
            f"episode name {name!r} must be alphanumeric with _ - . and must not "
            "contain '__', which separates the episode from the group in a column name"
        )
    low, colon, high = window.partition(":")
    if not colon:
        raise EnsembleError(
            f"episode {spec!r} needs a FIRST:LAST frame range, e.g. "
            f"{name}=1488:1492"
        )
    try:
        first, last = int(low.strip()), int(high.strip())
    except ValueError:
        raise EnsembleError(
            f"episode {spec!r} has a non-integer frame range {window!r}"
        ) from None
    if first > last:
        raise EnsembleError(
            f"episode {name!r} has FIRST {first} after LAST {last}; the window is "
            "inclusive and is not reordered silently"
        )
    return EpisodeWindow(name=name, first=first, last=last, role=role)


def parse_episode_windows(
    legacy: Optional[str] = None,
    named: Sequence[str] = (),
    controls: Sequence[str] = (),
) -> List[EpisodeWindow]:
    """Resolve the whole episode set of one run, legacy window included.

    The legacy single window keeps its own name and its own unprefixed columns,
    so the existing interface is unchanged; the named ones are added beside it.
    Two episodes may not share a name -- the name is the only thing that tells
    B1 from B2 in an output file.
    """
    windows: List[EpisodeWindow] = []
    if legacy:
        parsed = parse_episode_window(legacy, default_name=LEGACY_EPISODE_NAME)
        windows.append(
            EpisodeWindow(parsed.name, parsed.first, parsed.last, parsed.role, legacy=True)
        )
    for spec in named or ():
        windows.append(parse_episode_window(spec))
    for spec in controls or ():
        windows.append(parse_episode_window(spec, role="control"))
    seen: Dict[str, EpisodeWindow] = {}
    for window in windows:
        if window.name in seen:
            raise EnsembleError(
                f"two episodes are both named {window.name!r} "
                f"({seen[window.name].as_spec()} and {window.as_spec()}); names "
                "must be distinct, or an output cannot say which region a number "
                "belongs to"
            )
        seen[window.name] = window
    return windows


def episode_header(groups: Sequence[str], episodes: Sequence[EpisodeWindow]) -> List[str]:
    """Named per-episode columns for ``per_history.csv``.

    ``<episode>__<group>_<quantity>``, plus ``<episode>__verdict``. The legacy
    window is skipped: it already has the unprefixed ``episode_*`` columns, and
    writing it twice would invite a reader to treat the two as different
    numbers.
    """
    header: List[str] = []
    for window in episodes:
        if window.legacy:
            continue
        for quantity in EPISODE_QUANTITIES:
            header += [f"{window.name}__{g}_{quantity}" for g in groups]
        header.append(f"{window.name}__verdict")
    return header


def episode_long_header() -> List[str]:
    """Header of the long-form ``episode_per_history.csv``.

    One row per (history, episode, fragment). Long form because a wide table
    over four episodes and three fragments is unreadable, and because a row
    that carries its own episode name cannot be misfiled.
    """
    return [
        "run", "history", "NAMDTINI", "n_rows", "n_passes",
        "episode", "role", "first_frame", "last_frame",
        "n_passes_covering_episode", "n_steps_in_episode",
        "group", "net_per_pass", "occupation_per_pass", "character_per_pass",
        "verdict",
    ]


def episode_long_rows(
    result: "HistoryResult", groups: Sequence[str]
) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for name, episode in result.episodes.items():
        for group in groups:
            rows.append([
                result.run, result.history, result.namdtini,
                result.n_rows, result.n_passes,
                name, episode.role, episode.first, episode.last,
                episode.n_passes_covering, episode.n_steps,
                group,
                episode.net_per_pass.get(group, ""),
                episode.occupation_per_pass.get(group, ""),
                episode.character_per_pass.get(group, ""),
                episode.verdict,
            ])
    return rows


@dataclass
class HistoryResult:
    """One SHPROP history's outcome, as the per-history level reports it."""

    run: str
    history: str
    namdtini: int
    n_rows: int
    n_passes: int
    #: Net change over the full trajectory, per fragment.
    net_full: Dict[str, float] = field(default_factory=dict)
    #: Net change over the late window (t >= start), per fragment.
    net_late: Dict[str, float] = field(default_factory=dict)
    #: Excursion range over the late window, per fragment.
    range_late: Dict[str, float] = field(default_factory=dict)
    #: Summed over the full trajectory, per fragment.
    occupation_redistribution: Dict[str, float] = field(default_factory=dict)
    character_evolution: Dict[str, float] = field(default_factory=dict)
    #: Mean per-pass response inside the crossing episode, per fragment. These
    #: three carry the **legacy single window** only, so a reader written
    #: against ``--episode-window`` is unaffected by named episodes. When only
    #: named episodes were given they stay empty and the verdict stays
    #: ``not_classified``: promoting an arbitrary one of B1..B4 to "the"
    #: episode would be a choice the analysis has no basis to make.
    episode_net_per_pass: Dict[str, float] = field(default_factory=dict)
    episode_occupation_per_pass: Dict[str, float] = field(default_factory=dict)
    #: How the episode response is classified for this history.
    episode_verdict: str = "not_classified"
    #: Every episode of this run by name, the legacy window included. Never
    #: reduced to a single number across episodes: see :data:`EPISODES_NOTE`.
    episodes: Dict[str, EpisodeResult] = field(default_factory=dict)
    decomposition_residual: float = float("nan")
    #: The fixed-column reading of the same history, when a nominal state
    #: map was supplied.  Kept beside the dynamic one rather than replacing
    #: it: the comparison between the two is the point.
    net_full_fixed: Dict[str, float] = field(default_factory=dict)
    net_late_fixed: Dict[str, float] = field(default_factory=dict)
    range_late_fixed: Dict[str, float] = field(default_factory=dict)

    def as_row(
        self, groups: Sequence[str], episodes: Sequence[EpisodeWindow] = ()
    ) -> List[Any]:
        row: List[Any] = [
            self.run, self.history, self.namdtini, self.n_rows, self.n_passes,
        ]
        for mapping in (
            self.net_full, self.net_late, self.range_late,
            self.occupation_redistribution, self.character_evolution,
            self.episode_net_per_pass, self.episode_occupation_per_pass,
            self.net_full_fixed, self.net_late_fixed, self.range_late_fixed,
        ):
            row += [mapping.get(g, "") for g in groups]
        row += [self.episode_verdict, self.decomposition_residual]
        for window in episodes:
            if window.legacy:
                continue
            episode = self.episodes.get(window.name)
            for quantity in EPISODE_QUANTITIES:
                values = episode.quantity(quantity) if episode else {}
                row += [values.get(g, "") for g in groups]
            row.append(episode.verdict if episode else "not_classified")
        return row


def history_header(
    groups: Sequence[str], episodes: Sequence[EpisodeWindow] = ()
) -> List[str]:
    header = ["run", "history", "NAMDTINI", "n_rows", "n_passes"]
    for prefix in (
        "net_full", "net_late", "range_late",
        "occupation_redistribution", "character_evolution",
        "episode_net_per_pass", "episode_occupation_per_pass",
        "net_full_fixed", "net_late_fixed", "range_late_fixed",
    ):
        header += [f"{prefix}_{g}" for g in groups]
    header += ["episode_verdict", "decomposition_residual"]
    header += episode_header(groups, episodes)
    return header


def _stats(values: np.ndarray, tolerance: float) -> Dict[str, Any]:
    """Distribution summary plus sign fractions, for one fragment."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0, "note": "no finite values"}
    gain = int((finite > tolerance).sum())
    loss = int((finite < -tolerance).sum())
    flat = int(finite.size - gain - loss)
    out: Dict[str, Any] = {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std_between_histories": float(finite.std(ddof=1)) if finite.size > 1 else None,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "n_gain": gain,
        "n_loss": loss,
        "n_flat": flat,
        "fraction_gain": gain / finite.size,
        "fraction_loss": loss / finite.size,
        "sign_tolerance": float(tolerance),
    }
    for q in QUANTILES:
        out[f"q{int(q*100):02d}"] = float(np.quantile(finite, q))
    return out


def summarize_run(
    run: str,
    histories: Sequence[HistoryResult],
    groups: Sequence[str],
    sign_tolerance: float = SIGN_TOLERANCE,
) -> Dict[str, Any]:
    """Aggregate the histories of ONE run.

    The spread reported is between histories at fixed nuclear trajectory. It is
    labelled that way rather than as an uncertainty on the system, because the
    nuclei did not vary.
    """
    if not histories:
        raise EnsembleError(f"run {run!r} has no histories")
    wrong = sorted({h.run for h in histories} - {run})
    if wrong:
        raise EnsembleError(f"histories from runs {wrong} passed to run {run!r}")

    fields = {
        "net_full": lambda h: h.net_full,
        "net_late": lambda h: h.net_late,
        "range_late": lambda h: h.range_late,
        "occupation_redistribution": lambda h: h.occupation_redistribution,
        "character_evolution": lambda h: h.character_evolution,
        "episode_net_per_pass": lambda h: h.episode_net_per_pass,
        "episode_occupation_per_pass": lambda h: h.episode_occupation_per_pass,
        "net_full_fixed": lambda h: h.net_full_fixed,
        "net_late_fixed": lambda h: h.net_late_fixed,
        "range_late_fixed": lambda h: h.range_late_fixed,
    }
    summary: Dict[str, Any] = {
        "run": run,
        "n_histories": len(histories),
        "namdtini": sorted({h.namdtini for h in histories}),
        "total_passes": int(sum(h.n_passes for h in histories)),
        "max_decomposition_residual": float(
            max((h.decomposition_residual for h in histories
                 if np.isfinite(h.decomposition_residual)), default=float("nan"))
        ),
        "quantities": {},
        "episode_verdicts": {},
        "spread_means": (
            "between histories of one run, i.e. over electronic initial "
            "conditions at fixed nuclear trajectory -- not an uncertainty on "
            "the system"
        ),
        "hierarchy_note": HIERARCHY_NOTE,
        "bookkeeping_note": BOOKKEEPING_NOTE,
    }
    for name, getter in fields.items():
        summary["quantities"][name] = {
            g: _stats(np.array([getter(h).get(g, np.nan) for h in histories]),
                      sign_tolerance)
            for g in groups
        }
    verdicts: Dict[str, int] = {}
    for h in histories:
        verdicts[h.episode_verdict] = verdicts.get(h.episode_verdict, 0) + 1
    summary["episode_verdicts"] = verdicts
    summary["episodes"] = summarize_episodes(histories, groups, sign_tolerance)
    summary["episodes_note"] = EPISODES_NOTE
    summary["episodes_single_pass_note"] = ONE_PASS_NOTE
    return summary


def summarize_episodes(
    histories: Sequence[HistoryResult],
    groups: Sequence[str],
    sign_tolerance: float = SIGN_TOLERANCE,
) -> Dict[str, Any]:
    """Aggregate each named episode over the histories -- and only over them.

    An episode is summarized across the histories that produced it, which is
    the one legitimate direction: the histories differ in electronic initial
    condition at fixed nuclei. Nothing is aggregated **across** episodes. B1,
    B2, B3 and B4 are four regions of one recycled trajectory, so a number
    combining them would describe no region at all, and none is produced here.
    """
    names: List[str] = []
    for h in histories:
        for name in h.episodes:
            if name not in names:
                names.append(name)

    out: Dict[str, Any] = {}
    for name in names:
        present = [h for h in histories if name in h.episodes]
        first = present[0].episodes[name]
        block: Dict[str, Any] = {
            "window": [first.first, first.last],
            "role": first.role,
            "role_meaning": EPISODE_ROLES.get(first.role, ""),
            "legacy_single_window": name == LEGACY_EPISODE_NAME,
            "n_histories": len(present),
            "quantities": {},
            "verdicts": {},
            "pooled_with_other_episodes": False,
        }
        # A window that moved between histories is a bug upstream, not
        # something to average over: say so rather than quietly using the first.
        spans = sorted({(h.episodes[name].first, h.episodes[name].last) for h in present})
        if len(spans) > 1:
            block["inconsistent_windows"] = [list(s) for s in spans]
        for quantity in EPISODE_QUANTITIES:
            block["quantities"][quantity] = {
                g: _stats(
                    np.array([
                        h.episodes[name].quantity(quantity).get(g, np.nan)
                        for h in present
                    ]),
                    sign_tolerance,
                )
                for g in groups
            }
        tally: Dict[str, int] = {}
        for h in present:
            verdict = h.episodes[name].verdict
            tally[verdict] = tally.get(verdict, 0) + 1
        block["verdicts"] = tally
        out[name] = block
    return out


def compare_runs(
    runs: Sequence[Dict[str, Any]],
    groups: Sequence[str],
    quantity: str = "net_full",
) -> Dict[str, Any]:
    """Set the campaigns side by side. This is a comparison of physical cases.

    A, B and C are distinct interface configurations, so this is **not** a
    reproducibility check, and the numbers are not repeat measurements of one
    quantity. Nothing is averaged across campaigns: a "mean over configurations"
    would be a mean over different systems. What is reported is each campaign's
    own value, and whether the campaigns differ -- which, if they do, is a
    result about the interfaces rather than scatter.
    """
    if not runs:
        raise EnsembleError("no run summaries to compare")
    names = [r["run"] for r in runs]
    out: Dict[str, Any] = {
        "runs": names,
        "n_runs": len(runs),
        "quantity": quantity,
        "groups": {},
        "note": (
            "campaigns are distinct interface configurations, not replicates. "
            "No value is averaged across them, because a mean over different "
            "physical systems is not a measurement of anything. A difference "
            "between campaigns is a result about the interfaces"
        ),
        "hierarchy_note": HIERARCHY_NOTE,
    }
    for g in groups:
        medians, fractions, ns = [], [], []
        for r in runs:
            block = r["quantities"].get(quantity, {}).get(g, {})
            medians.append(block.get("median", float("nan")))
            fractions.append(block.get("fraction_gain", float("nan")))
            ns.append(block.get("n", 0))
        medians_a = np.array(medians, dtype=float)
        finite = medians_a[np.isfinite(medians_a)]
        out["groups"][g] = {
            "per_campaign_median": {n: m for n, m in zip(names, medians)},
            "per_campaign_fraction_gain": {n: f for n, f in zip(names, fractions)},
            "per_campaign_n_histories": {n: c for n, c in zip(names, ns)},
            # Deliberately NOT a mean: these are different systems. The spread
            # says how far apart the configurations are, not how uncertain one
            # value is.
            "spread_between_campaigns": (
                float(finite.max() - finite.min()) if finite.size else float("nan")
            ),
            "campaigns_agree_in_sign": (
                bool(np.all(finite > 0) or np.all(finite < 0))
                if finite.size else False
            ),
            "interpretation": (
                "same sign in every campaign means the three interfaces behave "
                "alike in direction; differing signs mean they do not, which is "
                "a statement about the interfaces and not about precision"
            ),
        }
    return out


def compare_episodes(
    runs: Sequence[Dict[str, Any]], groups: Sequence[str]
) -> Dict[str, Any]:
    """List every configuration's episodes side by side, and combine none.

    The episodes of different configurations are not the same region under a
    different name: A's ``crossing_A`` and B's ``crossing_B2`` are windows in
    different interface configurations, located separately. There is no
    correspondence to align them on, so this only tabulates what each
    configuration reported, keyed by configuration and then by episode name.
    """
    if not runs:
        raise EnsembleError("no run summaries to compare")
    per_run: Dict[str, Any] = {}
    for record in runs:
        episodes = record.get("episodes") or {}
        per_run[record["run"]] = {
            name: {
                "window": block.get("window"),
                "role": block.get("role"),
                "n_histories": block.get("n_histories"),
                "verdicts": block.get("verdicts", {}),
                "median_net_per_pass": {
                    g: block.get("quantities", {})
                       .get("net_per_pass", {}).get(g, {}).get("median")
                    for g in groups
                },
                "median_occupation_per_pass": {
                    g: block.get("quantities", {})
                       .get("occupation_per_pass", {}).get(g, {}).get("median")
                    for g in groups
                },
                "median_character_per_pass": {
                    g: block.get("quantities", {})
                       .get("character_per_pass", {}).get(g, {}).get("median")
                    for g in groups
                },
            }
            for name, block in episodes.items()
        }
    return {
        "runs": [r["run"] for r in runs],
        "per_configuration": per_run,
        "note": (
            "episodes are listed per configuration and never merged. Windows "
            "located in different interface configurations are different "
            "regions of different systems, and within one configuration the "
            "windows are different regions of one recycled trajectory. There is "
            "no level at which averaging them is defined, and none is computed"
        ),
        "episodes_note": EPISODES_NOTE,
        "hierarchy_note": HIERARCHY_NOTE,
    }


def _dominant_term(summary, group):
    """Which bookkeeping term carries the larger share, or neither clearly."""
    q = summary.get("quantities", {})
    occ = q.get("occupation_redistribution", {}).get(group, {}).get("median")
    chr_ = q.get("character_evolution", {}).get(group, {}).get("median")
    if occ is None or chr_ is None:
        return ""
    total = abs(occ) + abs(chr_)
    if total <= 0.0:
        return "neither (no movement)"
    share = abs(occ) / total
    if share >= 0.7:
        return f"occupation ({100*share:.0f}%)"
    if share <= 0.3:
        return f"character ({100*(1-share):.0f}%)"
    return f"mixed (occupation {100*share:.0f}%)"


def reviewer_table(
    runs: Sequence[Dict[str, Any]],
    donor: str = "BCF",
    acceptor: str = "PCBM",
    quantity: str = "net_full",
) -> Dict[str, Any]:
    """The compact per-campaign table a referee should see.

    One column per campaign, and nothing pooled across them: A, B and C are
    distinct interface configurations, so there is no column for a combined
    value and no row that averages them.

    Counts are given as ``x/N`` rather than as percentages, so the denominator
    travels with them.
    """
    if not runs:
        raise EnsembleError("no campaign summaries")
    names = [r["run"] for r in runs]
    rows: List[Dict[str, Any]] = []

    def block(r, group, which=None):
        return r["quantities"].get(which or quantity, {}).get(group, {})

    def add(label, fn):
        rows.append({"row": label, "values": {n: fn(r) for n, r in zip(names, runs)}})

    for label, group, key in (
        (f"histories with net {acceptor} gain", acceptor, "n_gain"),
        (f"histories with net {donor} gain", donor, "n_gain"),
        (f"histories with net {acceptor} loss", acceptor, "n_loss"),
        (f"histories with net {donor} loss", donor, "n_loss"),
    ):
        add(label, lambda r, g=group, k=key:
            f"{block(r, g).get(k, 0)}/{block(r, g).get('n', 0)}")

    for group in (acceptor, donor):
        add(f"median dP_{group}",
            lambda r, g=group: block(r, g).get("median", float("nan")))

    # The late window, which is the manuscript's own fit region.
    for group in (acceptor, donor):
        add(f"late-window net dP_{group}",
            lambda r, g=group: block(r, g, "net_late").get("median", float("nan")))

    # Fixed against dynamic: a net difference means the fixed labelling was
    # wrong somewhere, and a range difference means the population is not
    # static even where the nets agree.
    for group in (acceptor, donor):
        def discrepancy(r, g=group):
            dyn = block(r, g, "net_late").get("median")
            fix = block(r, g, "net_late_fixed").get("median")
            rng_d = block(r, g, "range_late").get("median")
            rng_f = block(r, g, "range_late_fixed").get("median")
            if dyn is None or fix is None:
                return "no fixed reading"
            parts = [f"net {dyn - fix:+.4f}"]
            if rng_d is not None and rng_f is not None and rng_f > 0:
                parts.append(f"range x{rng_d / rng_f:.1f}")
            return ", ".join(parts)

        add(f"fixed vs dynamic discrepancy, {group} (late)", discrepancy)

    for group in (acceptor, donor):
        add(f"dominant decomposition term, {group}",
            lambda r, g=group: _dominant_term(r, g))

    add("histories analysed", lambda r: str(r.get("n_histories", "")))
    add("max decomposition residual",
        lambda r: r.get("max_decomposition_residual", float("nan")))

    return {
        "runs": names,
        "campaigns": names,
        "quantity": quantity,
        "rows": rows,
        "note": (
            "one column per campaign; nothing is pooled across them, because A, "
            "B and C are distinct interface configurations rather than "
            "replicates. Counts are out of that campaign's own history count, "
            "and a fraction is a fraction of electronic initial conditions, not "
            "of nuclear configurations"
        ),
        "decomposition_caveat": BOOKKEEPING_NOTE,
    }


def ensemble_curve(
    series: Sequence[np.ndarray],
    quantiles: Sequence[float] = (0.25, 0.50, 0.75),
) -> Dict[str, np.ndarray]:
    """Median and quantile band across histories on a shared time grid.

    Quantiles across histories, not a standard error: with a shared nuclear
    trajectory the histories are not independent draws, so a band that looks
    like a confidence interval would be misread as one.
    """
    if not series:
        raise EnsembleError("no series to combine")
    lengths = {np.asarray(s).size for s in series}
    if len(lengths) != 1:
        raise EnsembleError(
            f"series have different lengths {sorted(lengths)}; no interpolation "
            "or truncation is performed"
        )
    stack = np.vstack([np.asarray(s, dtype=float) for s in series])
    out = {"n_histories": stack.shape[0], "median": np.median(stack, axis=0)}
    for q in quantiles:
        out[f"q{int(q*100):02d}"] = np.quantile(stack, q, axis=0)
    out["min"] = stack.min(axis=0)
    out["max"] = stack.max(axis=0)
    return out
