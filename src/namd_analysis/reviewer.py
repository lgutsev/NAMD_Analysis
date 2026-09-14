"""One reviewer-facing row per interface configuration.

This runs the whole v0.4 chain for each configuration named in a manifest --
canonical populations, transient and late observables, candidate-scheme
comparison, the hardened kinetic fit, the first-passage branch and the
counterfactual extraction competition -- and writes a single table.

Every cell is one of three kinds of quantity and the report says which:
population observables are read from the data, rates and branch probabilities
are inferred from a fitted model, and extraction yields are counterfactual.
A cell whose quantity is unavailable, or whose underlying rates the data did
not determine, is left empty.  Nothing is filled in with a plausible number.
"""

from __future__ import annotations

import glob as globlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .branching import (
    NOT_IDENTIFIABLE,
    BranchingError,
    bootstrap_branch_probability,
    branch_probability,
    extraction_competition,
)
from .comparison import compare_schemes
from .kinetics import KineticsError, fit_master_equation, parse_edges
from .observables import COUNTERFACTUAL, MODEL_INFERRED, OBSERVED
from .populations import (
    StateMap,
    group_series,
    load_population_set,
    per_file_group_populations,
)
from .transient import Window, analyze, parse_window

#: Default physical roles.  The manifest may rename any of them; the column
#: names stay fixed so the table is comparable across campaigns.
DEFAULT_ROLES = {
    "acceptor": "PCBM",
    "interfacial": "BCF",
    "recombination": "VBM",
    "donor": "CBM",
}

ROLE_PREFIX = {"acceptor": "pcbm", "interfacial": "bcf", "recombination": "vbm"}

REVIEWER_HEADER = [
    "configuration",
    "n_shprop_files",
    "transient_window_ns",
    "late_window_ns",
    "pcbm_initial",
    "pcbm_peak",
    "pcbm_peak_time_ns",
    "pcbm_final",
    "pcbm_integral_ns",
    "pcbm_transient_integral_ns",
    "pcbm_late_integral_ns",
    "bcf_initial",
    "bcf_final",
    "bcf_integral_ns",
    "vbm_initial",
    "vbm_final",
    "selected_scheme",
    "scheme_score",
    "scheme_rank_deficient",
    "k_bcf_to_pcbm_per_ns",
    "k_bcf_to_vbm_per_ns",
    "local_branch_pcbm",
    "first_passage_pcbm_before_vbm",
    "branching_status",
    "branching_bootstrap_low",
    "branching_bootstrap_high",
    "escape_crossover_rate_per_ns",
    "escape_crossover_time_ns",
]

#: What kind of quantity each column carries.
REVIEWER_CLASSES = {
    **{
        name: OBSERVED
        for name in (
            "n_shprop_files", "pcbm_initial", "pcbm_peak", "pcbm_peak_time_ns",
            "pcbm_final", "pcbm_integral_ns", "pcbm_transient_integral_ns",
            "pcbm_late_integral_ns", "bcf_initial", "bcf_final",
            "bcf_integral_ns", "vbm_initial", "vbm_final",
        )
    },
    **{
        name: MODEL_INFERRED
        for name in (
            "selected_scheme", "scheme_score", "scheme_rank_deficient",
            "k_bcf_to_pcbm_per_ns", "k_bcf_to_vbm_per_ns", "local_branch_pcbm",
            "first_passage_pcbm_before_vbm", "branching_status",
            "branching_bootstrap_low", "branching_bootstrap_high",
        )
    },
    **{
        name: COUNTERFACTUAL
        for name in ("escape_crossover_rate_per_ns", "escape_crossover_time_ns")
    },
}


class ReviewerError(ValueError):
    """Raised when a configuration entry cannot be analysed as declared."""


def _expand(patterns: Sequence[str], base: Path) -> List[Path]:
    found: List[Path] = []
    for pattern in patterns:
        candidate = pattern if Path(pattern).is_absolute() else str(base / pattern)
        matches = sorted(globlib.glob(candidate))
        if not matches:
            raise ReviewerError(f"no file matches {pattern!r}")
        found.extend(Path(m) for m in matches)
    return found


def _role_group(roles: Dict[str, str], role: str, groups: Sequence[str]) -> Optional[str]:
    name = roles.get(role)
    return name if name in groups else None


def summarize_configuration(
    entry: Dict[str, Any],
    base: Path,
    roles: Dict[str, str],
    criterion: str = "aicc",
) -> Dict[str, Any]:
    """Run the whole chain for one configuration and return its row and detail."""
    name = str(entry.get("name") or "unnamed")
    if "files" not in entry or "state_map" not in entry:
        raise ReviewerError(f"{name}: entry needs 'files' and 'state_map'")

    paths = _expand(
        entry["files"] if isinstance(entry["files"], list) else [entry["files"]], base
    )
    map_path = entry["state_map"]
    map_path = Path(map_path) if Path(map_path).is_absolute() else base / map_path
    state_map = StateMap.from_json(map_path)
    population = load_population_set(paths, state_map)
    series = group_series(population, state_map)
    groups = [s.name for s in series]
    pairs = [(s.name, s.values) for s in series]
    time_ns = population.time_ns

    row: Dict[str, Any] = {key: None for key in REVIEWER_HEADER}
    row["configuration"] = name
    row["n_shprop_files"] = population.n_files
    detail: Dict[str, Any] = {
        "configuration": name,
        "inputs": [str(p) for p in paths],
        "state_map": str(map_path),
        "groups": groups,
        "n_files": population.n_files,
        "conservation": population.conservation,
        "notes": [],
    }

    # ---- observed population metrics -------------------------------------
    transient_window = (
        parse_window(entry["transient_window"], name="transient")
        if entry.get("transient_window")
        else None
    )
    late_window = (
        parse_window(entry["late_window"], name="late")
        if entry.get("late_window")
        else None
    )
    windows = [Window(name="full")]
    if transient_window:
        windows.append(transient_window)
    if late_window:
        windows.append(late_window)
    metrics = analyze(time_ns, pairs, windows)
    by_key = {(m.group, m.window): m for m in metrics}

    if transient_window:
        low, high = transient_window.resolve(time_ns)
        row["transient_window_ns"] = f"{low:g}:{high:g}"
    if late_window:
        low, high = late_window.resolve(time_ns)
        row["late_window_ns"] = f"{low:g}:{high:g}"

    acceptor = _role_group(roles, "acceptor", groups)
    interfacial = _role_group(roles, "interfacial", groups)
    recombination = _role_group(roles, "recombination", groups)
    for role, group in (
        ("acceptor", acceptor), ("interfacial", interfacial),
        ("recombination", recombination),
    ):
        if group is None:
            detail["notes"].append(
                f"no group plays the {role} role (looked for "
                f"{roles.get(role)!r} among {groups}); its columns are empty"
            )
    detail["roles"] = {
        "acceptor": acceptor, "interfacial": interfacial,
        "recombination": recombination,
    }

    if acceptor:
        full = by_key[(acceptor, "full")]
        row.update(
            pcbm_initial=full.initial_population,
            pcbm_peak=full.peak_population,
            pcbm_peak_time_ns=full.peak_time_ns,
            pcbm_final=full.final_population,
            pcbm_integral_ns=full.integrated_population_ns,
        )
        if transient_window:
            row["pcbm_transient_integral_ns"] = by_key[
                (acceptor, "transient")
            ].integrated_population_ns
        if late_window:
            row["pcbm_late_integral_ns"] = by_key[
                (acceptor, "late")
            ].integrated_population_ns
    if interfacial:
        full = by_key[(interfacial, "full")]
        row.update(
            bcf_initial=full.initial_population,
            bcf_final=full.final_population,
            bcf_integral_ns=full.integrated_population_ns,
        )
    if recombination:
        full = by_key[(recombination, "full")]
        row.update(vbm_initial=full.initial_population, vbm_final=full.final_population)
    detail["population_metrics"] = [m.as_dict() for m in metrics]

    # ---- fit window ------------------------------------------------------
    start = entry.get("fit_start_ns")
    end = entry.get("fit_end_ns")
    if start is None and late_window is not None:
        start = late_window.resolve(time_ns)[0]
    lo = time_ns[0] if start is None else float(start)
    hi = time_ns[-1] if end is None else float(end)
    mask = (time_ns >= lo) & (time_ns <= hi)
    if np.count_nonzero(mask) < 3:
        detail["notes"].append(
            f"the fit window [{lo}, {hi}] ns holds too few samples; no kinetic "
            "model was fitted and every model-inferred column is empty"
        )
        return {"row": row, "detail": detail}
    observed = np.column_stack([s.values for s in series])[mask]
    fit_time = time_ns[mask]
    detail["fit_window_ns"] = [float(lo), float(hi)]

    if not state_map.complete_population:
        detail["notes"].append(
            "the state map is not a complete population basis, so no master "
            "equation was fitted; model-inferred columns are empty"
        )
        return {"row": row, "detail": detail}

    # ---- scheme selection ------------------------------------------------
    candidates = entry.get("candidate_schemes")
    if candidates:
        try:
            payload, fits = compare_schemes(
                fit_time, observed, groups, dict(candidates), criterion=criterion
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            detail["notes"].append(f"scheme comparison failed: {exc}")
            return {"row": row, "detail": detail}
        detail["scheme_comparison"] = payload
        best = payload.get("lowest_score_candidate")
        if best is None:
            detail["notes"].append("no candidate scheme could be scored")
            return {"row": row, "detail": detail}
        best_row = next(r for r in payload["candidates"] if r["candidate"] == best)
        row["selected_scheme"] = best
        row["scheme_score"] = best_row.get("score")
        row["scheme_rank_deficient"] = best_row.get("rank_deficient")
        detail["scheme_selection_note"] = (
            f"{best} has the lowest {criterion}. A lower information criterion "
            "is a descriptive ranking on these observations, not evidence that "
            "the mechanism is right; check the identifiability columns"
        )
        fit = fits[best]
        scheme_spec = candidates[best]
    elif entry.get("scheme"):
        scheme_spec = entry["scheme"]
        row["selected_scheme"] = scheme_spec
        try:
            fit = fit_master_equation(
                fit_time, observed, groups, parse_edges(scheme_spec, groups)
            )
        except (KineticsError, ValueError, np.linalg.LinAlgError) as exc:
            detail["notes"].append(f"the declared scheme could not be fitted: {exc}")
            return {"row": row, "detail": detail}
        row["scheme_rank_deficient"] = fit.rank_deficient
    else:
        detail["notes"].append(
            "no 'scheme' and no 'candidate_schemes' were declared; only observed "
            "population columns are filled"
        )
        return {"row": row, "detail": detail}

    detail["fit"] = fit.as_dict()
    detail["selected_scheme_spec"] = scheme_spec

    # ---- rates out of the interfacial state ------------------------------
    def _rate(source: Optional[str], target: Optional[str]) -> Optional[float]:
        if not source or not target:
            return None
        match = next(
            (r for r in fit.rates if r.source == source and r.target == target), None
        )
        if match is None:
            return None
        if not match.identified:
            detail["notes"].append(
                f"{match.name} is not identified ({match.unidentified_reason}); "
                "its column is left empty rather than quoted"
            )
            return None
        return match.rate_per_ns

    row["k_bcf_to_pcbm_per_ns"] = _rate(interfacial, acceptor)
    row["k_bcf_to_vbm_per_ns"] = _rate(interfacial, recombination)

    # ---- branching -------------------------------------------------------
    if interfacial and acceptor and recombination:
        try:
            branch = branch_probability(
                fit, interfacial, [acceptor], [recombination]
            )
        except BranchingError as exc:
            detail["notes"].append(f"branching is not well posed: {exc}")
        else:
            detail["branching"] = branch.as_dict()
            row["branching_status"] = branch.status
            if branch.status == NOT_IDENTIFIABLE:
                # The value exists in report.json next to the reason it cannot
                # be trusted. It does not go in the reviewer table, where a
                # number would simply be read off and quoted.
                detail["notes"].append(
                    f"the first-passage probability evaluates to "
                    f"{branch.probability:.4g} but the model does not determine "
                    f"it ({branch.status_reason}); the table cell is left empty "
                    "and only the status is reported"
                )
            else:
                row["first_passage_pcbm_before_vbm"] = branch.probability
                if branch.local_ratio.get("available"):
                    row["local_branch_pcbm"] = branch.local_ratio["ratio"]
            n_boot = int(entry.get("bootstrap") or 0)
            per_file = per_file_group_populations(population, state_map)
            if n_boot and per_file is not None and per_file.shape[0] >= 2:
                boot = bootstrap_branch_probability(
                    fit_time, per_file[:, mask, :], groups,
                    parse_edges(scheme_spec, groups),
                    interfacial, [acceptor], [recombination],
                    n_resamples=n_boot, seed=int(entry.get("bootstrap_seed", 0)),
                )
                detail["branching_bootstrap"] = boot
                if (
                    boot["branch_ci_status"] == "reported"
                    and branch.status != NOT_IDENTIFIABLE
                ):
                    row["branching_bootstrap_low"] = boot["branch_ci_low"]
                    row["branching_bootstrap_high"] = boot["branch_ci_high"]
                else:
                    detail["notes"].append(
                        f"branching interval suppressed: {boot['branch_ci_status']}"
                    )

    # ---- counterfactual extraction ---------------------------------------
    if acceptor and recombination and entry.get("escape_rates"):
        try:
            competition = extraction_competition(
                fit, acceptor, [recombination],
                _escape_grid(entry["escape_rates"]),
                source=entry.get("competition_source"),
            )
        except BranchingError as exc:
            detail["notes"].append(f"extraction competition failed: {exc}")
        else:
            detail["extraction_competition"] = competition
            row["escape_crossover_rate_per_ns"] = competition[
                "required_escape_rate_per_ns"
            ]
            row["escape_crossover_time_ns"] = competition["required_escape_time_ns"]

    return {"row": row, "detail": detail}


def _escape_grid(spec: Any) -> List[float]:
    if isinstance(spec, (list, tuple)):
        return [float(v) for v in spec]
    text = str(spec)
    if ":" in text:
        low, high, count = text.split(":")
        return [
            float(v) for v in np.geomspace(float(low), float(high), int(count))
        ]
    return [float(v) for v in text.split(",") if v.strip()]


def build_summary(
    manifest_path, criterion: Optional[str] = None
) -> Dict[str, Any]:
    """Run every configuration in a manifest and assemble the reviewer table."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    entries = manifest.get("configurations")
    if not entries:
        raise ReviewerError("the manifest declares no 'configurations'")
    roles = {**DEFAULT_ROLES, **(manifest.get("roles") or {})}
    criterion = criterion or manifest.get("criterion", "aicc")

    rows: List[List[Any]] = []
    details: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for entry in entries:
        name = str(entry.get("name") or "unnamed")
        try:
            result = summarize_configuration(entry, base, roles, criterion=criterion)
        except (ReviewerError, ValueError, OSError, np.linalg.LinAlgError) as exc:
            failures.append({"configuration": name, "error": str(exc)})
            blank = {key: None for key in REVIEWER_HEADER}
            blank["configuration"] = name
            rows.append([blank[key] for key in REVIEWER_HEADER])
            continue
        rows.append([result["row"][key] for key in REVIEWER_HEADER])
        details.append(result["detail"])

    return {
        "criterion": criterion,
        "roles": roles,
        "header": REVIEWER_HEADER,
        "rows": rows,
        "configurations": details,
        "failures": failures,
        "observable_class": REVIEWER_CLASSES,
        "interpretation_limits": [
            "population columns are read from the data; rate, branch and scheme "
            "columns come from a fitted model; escape columns are counterfactual",
            "an empty cell means the quantity was unavailable or its underlying "
            "rates were not identifiable. It is never a zero",
            "the first-passage probability is not a device extraction efficiency "
            "and the crossover is not a measured extraction time",
            "a lower information criterion ranks candidates on these "
            "observations; it does not establish a microscopic mechanism",
        ],
    }
