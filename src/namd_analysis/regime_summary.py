"""``early_late_summary.md`` -- the two regimes, stated for a referee.

Every sentence carries its observable class, because the difference between a
population that was read and a rate that was fitted is the difference between
a measurement and a model, and a reviewer response that blurs them is the one
that gets caught.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

OBSERVED = "observed"
MODEL = "model-inferred"
INTERPRETIVE = "interpretive"


def _fmt(value: Any, places: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{places}g}"
    return str(value)


def _table(header: List[str], rows: List[List[Any]]) -> List[str]:
    if not rows:
        return ["_none_", ""]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    lines.append("")
    return lines


def _regime_block(regime: Dict[str, Any], title: str) -> List[str]:
    low, high = regime["window_ns"]
    out = [
        f"## {title} regime  [{low:g}, {high:g}] ns",
        "",
        f"{regime['n_time_points']} time points.",
        "",
        f"**Populations** ({OBSERVED})",
        "",
    ]
    out += _table(
        ["group", "initial", "final", "net change", "peak", "peak time (ns)",
         "minimum", "integral (ns)"],
        [
            [g["group"], g["initial_population"], g["final_population"],
             g["net_change"], g["peak_population"], g["peak_time_ns"],
             g["minimum_population"], g["integrated_population_ns"]]
            for g in regime["observables"]["per_group"]
        ],
    )

    kinetics = regime.get("kinetics", {})
    identifiability = regime.get("identifiability", {})
    out += [f"**Kinetics** ({MODEL})", ""]
    if kinetics.get("status") == "not_fitted":
        out += [
            f"No rates are reported for this window: {kinetics.get('reason')}",
            "",
            "A window that cannot constrain a rate is reported as such. It is not "
            "widened, and no edge is dropped to make a number appear.",
            "",
        ]
        return out

    rates = kinetics.get("rates", [])
    out += _table(
        ["rate", "value (1/ns)", "timescale (ns)", "rel. s.e.", "identified"],
        [
            [r.get("name"), r.get("value_per_ns"), r.get("timescale_ns"),
             r.get("relative_standard_error"), r.get("identified")]
            for r in rates
        ],
    )
    timescales = kinetics.get("eigen_timescales_ns") or []
    if timescales:
        out += [
            "Relaxation timescales from the eigenvalues of K: "
            + ", ".join(f"{t:.4g} ns" for t in timescales)
            + ". These are what the population curves actually constrain; "
            "individual rates may not be determined even when these are.",
            "",
        ]
    quality = kinetics.get("fit_quality", {})
    if quality:
        out += [
            f"Fit quality: R² total {_fmt(quality.get('r_squared_total'))}; "
            "per-group RMS residual "
            + ", ".join(
                f"{g} {_fmt(v)}" for g, v in (quality.get("rms_residual_per_group") or {}).items()
            )
            + ".",
            "",
        ]
    out += [
        f"Identifiability: **{identifiability.get('status')}** "
        f"({identifiability.get('identified_rates')} of "
        f"{identifiability.get('total_rates')} rates). "
        + (
            "Unidentified: " + ", ".join(identifiability.get("unidentified") or []) + ". "
            if identifiability.get("unidentified")
            else ""
        )
        + "A rate that is not identified is not a small rate — the data do not "
        "constrain it.",
        "",
    ]
    bootstrap = regime.get("bootstrap")
    if bootstrap and bootstrap.get("intervals"):
        out += [
            f"Bootstrap ({MODEL}): {bootstrap.get('n_resamples')} whole-file "
            "resamples; intervals are in `report.json`.",
            "",
        ]
    elif bootstrap:
        out += [f"Bootstrap unavailable: {bootstrap.get('reason')}", ""]
    return out


def render(payload: Dict[str, Any]) -> str:
    out: List[str] = [
        "# Early and late regimes",
        "",
        "Two windows analysed separately, then compared. Each statement is "
        f"labelled **{OBSERVED}** (read from the histories), **{MODEL}** "
        f"(conditional on the declared kinetic graph), or **{INTERPRETIVE}**.",
        "",
    ]

    windows = payload.get("windows", {})
    early_w, late_w = windows.get("early", {}), windows.get("late", {})
    out += [
        "## Windows",
        "",
        f"- early: `{early_w.get('requested')}` → {early_w.get('resolved_ns')} ns "
        f"(source: {early_w.get('source')})",
        f"- late: `{late_w.get('requested')}` → {late_w.get('resolved_ns')} ns "
        f"(source: {late_w.get('source')})",
        "",
        late_w.get("note", ""),
        "",
    ]

    regimes = payload.get("regimes", [])
    for regime, title in zip(regimes, ("Early", "Late")):
        out += _regime_block(regime, title)

    comparison = payload.get("model_comparison", {})
    out += [
        "## One model, or one per regime?",
        "",
        f"({MODEL})",
        "",
    ]
    if comparison.get("status") == "scored":
        out += _table(
            ["model", comparison.get("criterion", "score")],
            [
                ["global", (comparison.get("global") or {}).get("scores", {}).get(
                    comparison.get("criterion"))],
                ["early + late", (comparison.get("piecewise_combined") or {}).get(
                    comparison.get("criterion"))],
            ],
        )
        out += [
            f"**{comparison.get('criterion','aicc').upper()} prefers the "
            f"{comparison.get('preferred')} description** by "
            f"{abs(comparison.get('delta', 0.0)):.4g} (lower is better).",
            "",
        ]
    else:
        out += [f"Status: `{comparison.get('status')}`.", ""]
    out += [comparison.get("conclusion", ""), ""]
    out += [
        f"> **This is model comparison, not proof of a mechanistic transition** "
        f"({INTERPRETIVE} if read that way). A piecewise description scoring "
        "better says one constant-rate matrix does not describe both windows "
        "equally well. It does not say what changed, and it does not locate a "
        "transition: the boundary between the windows was supplied, not fitted.",
        "",
    ]

    narrative = payload.get("early_transient_questions", {})
    findings = narrative.get("findings", [])
    if findings:
        out += ["## The early transient, question by question", ""]
        for finding in findings:
            kind = {
                "observed_from_SHPROP": OBSERVED,
                "model_inferred": MODEL,
                "interpretive": INTERPRETIVE,
            }.get(finding.get("observable_class"), finding.get("observable_class"))
            out += [
                f"**{finding['question']}**",
                "",
                f"{finding['answer']}  _({kind})_",
                "",
            ]
        out += [narrative.get("note", ""), ""]

    out += ["## Limitations", ""]
    for item in payload.get("interpretation_limits", []):
        out.append(f"- {item}")
    out.append("")
    return "\n".join(out).rstrip() + "\n"


def write(path, payload: Dict[str, Any]) -> Optional[Path]:
    path = Path(path)
    path.write_text(render(payload), encoding="utf-8")
    return path
