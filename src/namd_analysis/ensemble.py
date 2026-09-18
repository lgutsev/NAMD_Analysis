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
from typing import Any, Dict, List, Sequence

import numpy as np

__all__ = [
    "EnsembleError",
    "SIGN_TOLERANCE",
    "HistoryResult",
    "history_header",
    "summarize_run",
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


class EnsembleError(ValueError):
    """Raised when an aggregation cannot be formed from the inputs given."""


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
    #: Mean per-pass response inside the crossing episode, per fragment.
    episode_net_per_pass: Dict[str, float] = field(default_factory=dict)
    episode_occupation_per_pass: Dict[str, float] = field(default_factory=dict)
    #: How the episode response is classified for this history.
    episode_verdict: str = "not_classified"
    decomposition_residual: float = float("nan")
    #: The fixed-column reading of the same history, when a nominal state
    #: map was supplied.  Kept beside the dynamic one rather than replacing
    #: it: the comparison between the two is the point.
    net_full_fixed: Dict[str, float] = field(default_factory=dict)
    net_late_fixed: Dict[str, float] = field(default_factory=dict)
    range_late_fixed: Dict[str, float] = field(default_factory=dict)

    def as_row(self, groups: Sequence[str]) -> List[Any]:
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
        return row


def history_header(groups: Sequence[str]) -> List[str]:
    header = ["run", "history", "NAMDTINI", "n_rows", "n_passes"]
    for prefix in (
        "net_full", "net_late", "range_late",
        "occupation_redistribution", "character_evolution",
        "episode_net_per_pass", "episode_occupation_per_pass",
        "net_full_fixed", "net_late_fixed", "range_late_fixed",
    ):
        header += [f"{prefix}_{g}" for g in groups]
    header += ["episode_verdict", "decomposition_residual"]
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
    return summary


def compare_runs(
    runs: Sequence[Dict[str, Any]],
    groups: Sequence[str],
    quantity: str = "net_full",
) -> Dict[str, Any]:
    """Set the campaigns side by side. This is a comparison of physical cases.

    A, B and C are distinct interface configurations, so this is **not** a
    reproducibility check and the numbers are not repeat measurements of one
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
