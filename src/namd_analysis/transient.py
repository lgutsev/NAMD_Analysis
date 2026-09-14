"""Direct population observables, window by window.

Everything here is read off the reconstructed SHPROP populations.  No kinetic
model is fitted and no rate is inferred, so nothing in this module depends on
a declared kinetic graph or on the Markovian assumption.

The reason it exists: a trajectory whose late-time behaviour is dominated by
one process can carry a large, entirely real transient in another, and a fit
that starts after that transient will not see it.  A group can rise to a
substantial occupation and fall back before the fit window opens, and the
late-window fit will then report it as flat.  Reporting the peak, the time of
the peak and the time-integrated population alongside the endpoints makes that
impossible to state as "the population does not change".

A time-integrated population is not a flux and not a transferred amount.  It
is the area under an occupation curve, in population x ns.  Population that
arrives, leaves and arrives again is counted every time it is present, and
population that never moves contributes just as much as population that
turns over continuously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .observables import OBSERVED, uniform
from .populations import trapezoid


class TransientError(ValueError):
    """Raised when a window cannot be analysed as given."""


@dataclass(frozen=True)
class Window:
    """A named, half-open-ended analysis window in nanoseconds."""

    name: str
    start_ns: Optional[float] = None
    end_ns: Optional[float] = None

    def resolve(self, time_ns: np.ndarray) -> Tuple[float, float]:
        low = float(time_ns[0]) if self.start_ns is None else float(self.start_ns)
        high = float(time_ns[-1]) if self.end_ns is None else float(self.end_ns)
        if high <= low:
            raise TransientError(
                f"window {self.name!r} spans [{low}, {high}] ns, which is empty"
            )
        return low, high

    def mask(self, time_ns: np.ndarray) -> np.ndarray:
        low, high = self.resolve(time_ns)
        selected = (time_ns >= low) & (time_ns <= high)
        if np.count_nonzero(selected) < 2:
            raise TransientError(
                f"window {self.name!r} = [{low}, {high}] ns contains "
                f"{int(np.count_nonzero(selected))} samples; at least two are needed"
            )
        return selected

    def label(self, time_ns: Optional[np.ndarray] = None) -> str:
        if time_ns is not None:
            low, high = self.resolve(time_ns)
            return f"{low:g}:{high:g}"
        low = "" if self.start_ns is None else f"{self.start_ns:g}"
        high = "" if self.end_ns is None else f"{self.end_ns:g}"
        return f"{low}:{high}"


def parse_window(text: str, name: Optional[str] = None) -> Window:
    """Parse ``NAME=START:END``, ``START:END``, ``START:`` or ``:END``."""
    raw = text.strip()
    label = name
    if "=" in raw:
        label, _, raw = raw.partition("=")
        label = label.strip()
        raw = raw.strip()
    if ":" not in raw:
        raise TransientError(
            f"window {text!r} must be written START:END in ns, with either end "
            "left blank to run to the edge of the data"
        )
    low_text, _, high_text = raw.partition(":")

    def _value(piece: str, which: str) -> Optional[float]:
        piece = piece.strip()
        if not piece:
            return None
        try:
            return float(piece)
        except ValueError as exc:
            raise TransientError(
                f"window {text!r} has a non-numeric {which} bound {piece!r}"
            ) from exc

    start = _value(low_text, "start")
    end = _value(high_text, "end")
    if start is not None and end is not None and end <= start:
        raise TransientError(f"window {text!r} does not increase")
    return Window(name=label or raw, start_ns=start, end_ns=end)


#: Every field produced per group and window is read from the data.
METRIC_FIELDS = (
    "initial_population",
    "final_population",
    "peak_population",
    "peak_time_ns",
    "minimum_population",
    "minimum_time_ns",
    "net_change",
    "integrated_population_ns",
    "mean_population",
)

METRIC_CLASSES = uniform(METRIC_FIELDS, OBSERVED)


@dataclass
class GroupWindowMetrics:
    """Population observables for one group inside one window."""

    group: str
    window: str
    window_start_ns: float
    window_end_ns: float
    n_points: int
    initial_population: float
    final_population: float
    peak_population: float
    peak_time_ns: float
    minimum_population: float
    minimum_time_ns: float
    net_change: float
    integrated_population_ns: float
    mean_population: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "group": self.group,
            "window": self.window,
            "window_start_ns": self.window_start_ns,
            "window_end_ns": self.window_end_ns,
            "n_points": self.n_points,
            **{name: getattr(self, name) for name in METRIC_FIELDS},
        }

    def as_row(self) -> List[Any]:
        payload = self.as_dict()
        return [payload[key] for key in METRIC_HEADER]


METRIC_HEADER = [
    "group",
    "window",
    "window_start_ns",
    "window_end_ns",
    "n_points",
    *METRIC_FIELDS,
]


def window_metrics(
    time_ns: np.ndarray, values: np.ndarray, group: str, window: Window
) -> GroupWindowMetrics:
    """Population observables for one group over one window."""
    time_ns = np.asarray(time_ns, dtype=float)
    values = np.asarray(values, dtype=float)
    if time_ns.shape != values.shape:
        raise TransientError(
            f"group {group!r}: {values.shape} populations for {time_ns.shape} times"
        )
    selected = window.mask(time_ns)
    low, high = window.resolve(time_ns)
    t = time_ns[selected]
    y = values[selected]
    peak = int(np.argmax(y))
    trough = int(np.argmin(y))
    span = float(t[-1] - t[0])
    integral = trapezoid(y, t)
    return GroupWindowMetrics(
        group=group,
        window=window.name,
        window_start_ns=low,
        window_end_ns=high,
        n_points=int(t.size),
        initial_population=float(y[0]),
        final_population=float(y[-1]),
        peak_population=float(y[peak]),
        peak_time_ns=float(t[peak]),
        minimum_population=float(y[trough]),
        minimum_time_ns=float(t[trough]),
        net_change=float(y[-1] - y[0]),
        integrated_population_ns=float(integral),
        mean_population=float(integral / span) if span > 0 else float("nan"),
    )


def analyze(
    time_ns: np.ndarray,
    series: Sequence[Tuple[str, np.ndarray]],
    windows: Sequence[Window],
) -> List[GroupWindowMetrics]:
    """Population observables for every (group, window) pair."""
    if not series:
        raise TransientError("no groups to analyse")
    if not windows:
        raise TransientError("no analysis windows were given")
    seen = set()
    for window in windows:
        if window.name in seen:
            raise TransientError(f"window name {window.name!r} is used twice")
        seen.add(window.name)
    return [
        window_metrics(time_ns, values, group, window)
        for window in windows
        for group, values in series
    ]


def flat_group_fields(
    metrics: Sequence[GroupWindowMetrics], group: str, window: str
) -> Dict[str, Any]:
    """The ``pcbm_peak``-style flattened view for one group and window."""
    match = next(
        (m for m in metrics if m.group == group and m.window == window), None
    )
    if match is None:
        return {}
    prefix = group.lower().replace("-", "_").replace(" ", "_")
    return {
        f"{prefix}_initial": match.initial_population,
        f"{prefix}_peak": match.peak_population,
        f"{prefix}_peak_time_ns": match.peak_time_ns,
        f"{prefix}_final": match.final_population,
        f"{prefix}_integrated_population_ns": match.integrated_population_ns,
        f"{prefix}_net_change": match.net_change,
    }


REGIME_HEADER = [
    "group",
    "transient_window",
    "late_window",
    "transient_initial_population",
    "transient_final_population",
    "transient_peak_population",
    "transient_peak_time_ns",
    "transient_integrated_population_ns",
    "transient_net_change",
    "late_initial_population",
    "late_final_population",
    "late_peak_population",
    "late_peak_time_ns",
    "late_integrated_population_ns",
    "late_net_change",
    "peak_ratio_transient_over_late",
]


def compare_regimes(
    time_ns: np.ndarray,
    series: Sequence[Tuple[str, np.ndarray]],
    transient_window: Window,
    late_window: Window,
) -> Dict[str, Any]:
    """Side-by-side observables for an early and a late window.

    The windows are supplied by the caller.  There is no universal transient
    window: where one regime ends and the next begins is a property of the
    system being analysed, not of this package.
    """
    early = {m.group: m for m in analyze(time_ns, series, [transient_window])}
    late = {m.group: m for m in analyze(time_ns, series, [late_window])}

    rows: List[List[Any]] = []
    records: List[Dict[str, Any]] = []
    for group, _ in series:
        a, b = early[group], late[group]
        ratio = (
            float(a.peak_population / b.peak_population)
            if b.peak_population > 0
            else float("nan")
        )
        record = {
            "group": group,
            "transient_window": f"{a.window_start_ns:g}:{a.window_end_ns:g}",
            "late_window": f"{b.window_start_ns:g}:{b.window_end_ns:g}",
            "transient_initial_population": a.initial_population,
            "transient_final_population": a.final_population,
            "transient_peak_population": a.peak_population,
            "transient_peak_time_ns": a.peak_time_ns,
            "transient_integrated_population_ns": a.integrated_population_ns,
            "transient_net_change": a.net_change,
            "late_initial_population": b.initial_population,
            "late_final_population": b.final_population,
            "late_peak_population": b.peak_population,
            "late_peak_time_ns": b.peak_time_ns,
            "late_integrated_population_ns": b.integrated_population_ns,
            "late_net_change": b.net_change,
            "peak_ratio_transient_over_late": ratio,
        }
        records.append(record)
        rows.append([record[key] for key in REGIME_HEADER])

    notes: List[str] = []
    for record in records:
        if (
            np.isfinite(record["peak_ratio_transient_over_late"])
            and record["peak_ratio_transient_over_late"] > 1.5
            and abs(record["late_net_change"]) < 0.02
        ):
            notes.append(
                f"{record['group']}: the early window reaches "
                f"{record['transient_peak_population']:.4g} at "
                f"{record['transient_peak_time_ns']:.4g} ns, well above anything in "
                "the late window, where the population is nearly flat. A fit that "
                "starts after the early window does not see this occupation and "
                "must not be described as showing the group does not change"
            )
    return {
        "records": records,
        "rows": rows,
        "notes": notes,
        "observable_class": METRIC_CLASSES,
    }
