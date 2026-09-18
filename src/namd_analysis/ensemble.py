"""Three-level aggregation: passes inside histories inside runs.

The nesting is not decoration; it is what keeps the statistics honest.

    passes  subset of  SHPROP histories  subset of  runs

**Passes** are re-traversals of one recycled nuclear trajectory. They are not
independent samples of anything, so nothing here computes a standard error over
passes.

**Histories** within a run share that run's nuclear trajectory. They differ in
electronic initial condition (``NAMDTINI``) and in the stochastic surface-hopping
realization, so they *are* distinct realizations of the electronic dynamics --
but they are **not** 100 independent nuclear configurations. A spread across
histories is a spread over electronic initial conditions at fixed nuclei.

**Runs** are the top-level reproducibility unit. Three runs means n = 3 for any
claim about the system rather than about one trajectory, and 300 histories do
not become 300 nuclear realizations by being counted together.

So: per-history quantities are reported individually; per-run statistics
summarize the 100 histories of that run; and the across-run comparison is the
only place a Campaign-level statement may be made. A grand mean over all 300 is
available but deliberately labelled as *not* the headline, because it hides the
between-run variation that is the actual reproducibility check.
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
    "independent samples; histories within a run share that run's nuclear "
    "trajectory and differ in electronic initial condition, so they are "
    "realizations of the electronic dynamics but NOT independent nuclear "
    "configurations; runs are the top-level reproducibility unit. No standard "
    "error is quoted over passes, and the 300 histories are never treated as "
    "300 nuclear realizations"
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

    def as_row(self, groups: Sequence[str]) -> List[Any]:
        row: List[Any] = [
            self.run, self.history, self.namdtini, self.n_rows, self.n_passes,
        ]
        for mapping in (
            self.net_full, self.net_late, self.range_late,
            self.occupation_redistribution, self.character_evolution,
            self.episode_net_per_pass, self.episode_occupation_per_pass,
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
    """Compare run-level results. This is the reproducibility check.

    With three runs there are three numbers per quantity. A mean of three is
    reported, and so is the full spread, because with n = 3 the spread is the
    informative part and a mean alone would imply more precision than exists.
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
            "runs are the top-level reproducibility unit. With three of them, "
            "read the spread and not the mean: three values cannot support a "
            "standard error, and the between-run difference is the result"
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
        out["groups"][g] = {
            "per_run_median": {n: m for n, m in zip(names, medians)},
            "per_run_fraction_gain": {n: f for n, f in zip(names, fractions)},
            "per_run_n_histories": {n: c for n, c in zip(names, ns)},
            "median_of_run_medians": float(np.nanmedian(medians_a)),
            "range_of_run_medians": (
                float(np.nanmax(medians_a) - np.nanmin(medians_a))
                if np.isfinite(medians_a).any() else float("nan")
            ),
            "runs_agree_in_sign": bool(
                np.all(medians_a[np.isfinite(medians_a)] > 0)
                or np.all(medians_a[np.isfinite(medians_a)] < 0)
            ) if np.isfinite(medians_a).any() else False,
        }
    return out


def reviewer_table(
    runs: Sequence[Dict[str, Any]],
    donor: str = "BCF",
    acceptor: str = "PCBM",
    quantity: str = "net_full",
) -> Dict[str, Any]:
    """The compact per-run table a referee should see.

    Rows are counts and medians per run; columns are runs.  Counts are given as
    ``x/N`` rather than as percentages, so the denominator travels with them.
    """
    if not runs:
        raise EnsembleError("no run summaries")
    names = [r["run"] for r in runs]
    rows: List[Dict[str, Any]] = []

    def block(r, group):
        return r["quantities"].get(quantity, {}).get(group, {})

    for label, group, key in (
        (f"histories with net {acceptor} gain", acceptor, "n_gain"),
        (f"histories with net {donor} gain", donor, "n_gain"),
        (f"histories with net {acceptor} loss", acceptor, "n_loss"),
        (f"histories with net {donor} loss", donor, "n_loss"),
    ):
        rows.append({
            "row": label,
            "values": {
                n: f"{block(r, group).get(key, 0)}/{block(r, group).get('n', 0)}"
                for n, r in zip(names, runs)
            },
        })
    for label, group in ((f"median dP_{acceptor}", acceptor),
                         (f"median dP_{donor}", donor)):
        rows.append({
            "row": label,
            "values": {
                n: block(r, group).get("median", float("nan"))
                for n, r in zip(names, runs)
            },
        })
    return {
        "runs": names,
        "quantity": quantity,
        "rows": rows,
        "note": (
            "counts are per run out of that run's history count. Histories "
            "within a run share a nuclear trajectory, so a fraction here is a "
            "fraction of electronic initial conditions, not of nuclear "
            "configurations"
        ),
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
