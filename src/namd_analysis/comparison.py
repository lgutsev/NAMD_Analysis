"""Comparisons with common observations and explicit modeling assumptions."""
from __future__ import annotations
import glob
import json
from pathlib import Path
import numpy as np
from scipy.linalg import helmert
from .kinetics import fit_master_equation, parse_edges
from .populations import StateMap, load_population_set, group_series
from .provenance import fingerprint, launcher_manifests


def compare_schemes(time, observed, groups, schemes, *, criterion="aicc", n_starts=5):
    """Conditional Gaussian IC on G-1 orthonormal population contrasts.

    P0 is conditioned on, its zero residual is excluded. Time samples are
    treated as independent for the score only: autocorrelation invalidates a
    formal likelihood interpretation. Rank deficiency remains a separate flag.
    """
    time = np.asarray(time, float)
    observed = np.asarray(observed, float)
    if criterion not in {"aic", "aicc", "bic"}:
        raise ValueError("criterion must be aic, aicc or bic")
    if not isinstance(schemes, dict) or len(schemes) < 2:
        raise ValueError("Supply at least two named candidate schemes")
    if len(groups) < 2 or observed.shape != (len(time), len(groups)) or len(time) < 3:
        raise ValueError("Need >=3 times and >=2 groups with matching observations")
    if not np.isfinite(observed).all() or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Finite observations and increasing times required")
    if observed.min() < -1e-5 or not np.allclose(observed.sum(axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("Scheme comparison requires conserved complete populations")
    graphs = {name: parse_edges(spec, groups) for name, spec in schemes.items()}
    canonical = [tuple(sorted(edges)) for edges in graphs.values()]
    if len(set(canonical)) != len(canonical):
        raise ValueError("Candidate schemes contain duplicate graphs")
    contrast = helmert(len(groups))
    n = (len(time) - 1) * (len(groups) - 1)
    fits, rows = {}, []
    for name, edges in graphs.items():
        try:
            fit = fit_master_equation(time, observed, groups, edges, n_starts=n_starts)
            residual = (fit.model - observed)[1:] @ contrast.T
            rss = float(np.sum(residual**2))
            k = len(edges) + 1  # estimated rates plus shared residual variance
            variance = max(rss / n, np.finfo(float).eps**2)
            likelihood_term = n * np.log(variance)
            aic = float(likelihood_term + 2 * k)
            scores = {"aic": aic, "bic": float(likelihood_term + k*np.log(n)),
                      "aicc": aic + 2*k*(k+1)/(n-k-1) if n > k+1 else None}
            score = scores[criterion]
            row = {"candidate": name, "scheme": schemes[name], "status": "scored" if score is not None else "insufficient_observations",
                   "rss_contrasts": rss, "n_observations": n, "k_parameters": k,
                   **scores, "score": score, "delta": None,
                   "unidentified_rates": len(fit.rates) - len(fit.identified_rates),
                   "rank_deficient": fit.rank_deficient,
                   "numerically_exact": rss / n <= np.finfo(float).eps**2}
            fits[name] = fit
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            row = {"candidate": name, "scheme": schemes[name], "status": "failed", "error": str(exc), "score": None, "delta": None}
        rows.append(row)
    valid = [row for row in rows if row["score"] is not None and np.isfinite(row["score"])]
    valid.sort(key=lambda row: row["score"])
    for rank, row in enumerate(valid, 1):
        row["rank"] = rank
        row["delta"] = row["score"] - valid[0]["score"]
    return {"criterion": criterion, "candidates": rows,
            "lowest_score_candidate": valid[0]["candidate"] if valid else None,
            "observation_count": {"n": n, "formula": "(n_times - 1) * (n_groups - 1)",
                                  "matches_kinetics_covariance_dimension": True,
                                  "note": ("the same conservation subspace the kinetic covariance "
                                           "is scaled by; Helmert contrasts are orthonormal, so the "
                                           "sum of squares is unchanged and only the count differs")},
            "interpretation_limits": [
                "Descriptive conditional-Gaussian scores; equal unweighted observations for every candidate.",
                "G-1 orthonormal contrasts avoid double counting population conservation; P0 row excluded.",
                "Temporal correlation and noisy fixed P0 invalidate naive independent-error evidence; no model probabilities reported.",
                "Lowest score does not establish a mechanism or identify individual rates. Boundary/rank-deficient fits violate regular IC assumptions.",
                "Numerically exact scores use a machine-precision variance floor and are not statistical evidence."
            ]}, fits


def compare_runs(manifest, *, start_ns=None, end_ns=None):
    """Compare separately averaged runs on exact common saved time points."""
    manifest = Path(manifest).resolve()
    config = json.loads(manifest.read_text())
    entries = config.get("runs", [])
    if len(entries) < 2:
        raise ValueError("Comparison manifest needs at least two runs")
    labels = [entry["label"] for entry in entries]
    if any(not isinstance(label, str) or not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Run labels must be unique nonempty strings")
    results = []
    inputs = [manifest]
    explicit_manifests = []
    for entry in entries:
        mapping = (manifest.parent / entry["config"]).resolve()
        state_map = StateMap.from_json(mapping)
        paths = []
        for pattern in entry["files"]:
            matched = sorted(glob.glob(str(manifest.parent / pattern)))
            if not matched:
                raise ValueError(f"No files matched for {entry['label']}: {pattern}")
            paths.extend(Path(p).resolve() for p in matched)
        if len(set(paths)) != len(paths):
            raise ValueError(f"Duplicate files for {entry['label']}")
        population = load_population_set(paths, state_map)
        series = {g.name: g for g in group_series(population, state_map)}
        results.append((entry, state_map, population, series))
        inputs.extend([mapping, *paths])
        explicit_manifests.extend((manifest.parent / p).resolve() for p in entry.get("launcher_manifests", []))
    common_time = results[0][2].time_ns.copy()
    common_groups = set(results[0][3])
    for _, _, population, series in results[1:]:
        common_time = np.intersect1d(common_time, population.time_ns)
        common_groups &= set(series)
    if start_ns is not None:
        common_time = common_time[common_time >= start_ns]
    if end_ns is not None:
        common_time = common_time[common_time <= end_ns]
    if len(common_time) < 2 or not common_groups:
        raise ValueError("Need at least two exact shared time samples and a shared group; no interpolation")
    names = [name for name in results[0][3] if name in common_groups]
    reference = config.get("reference", labels[0])
    if reference not in labels:
        raise ValueError("Reference label is not in runs")
    curves, run_reports = {}, []
    for entry, mapping, pop, series in results:
        indices = np.searchsorted(pop.time_ns, common_time)
        curves[entry["label"]] = {name: series[name].values[indices] for name in names}
        run_reports.append({"label": entry["label"], "initial_state_description": entry.get("initial_state", "not supplied"),
                            "initial_population_at_first_saved_sample": {name: float(g.values[0]) for name, g in series.items()},
                            "n_files": pop.n_files, "original_time_points": len(pop.time_ns),
                            "excluded_time_points": len(pop.time_ns)-len(common_time),
                            "nonshared_groups": sorted(set(series)-common_groups),
                            "recombined_group": mapping.recombined_group,
                            "groups": [g.summary(pop.time_ns) for g in series.values()]})
    from .populations import trapezoid
    differences = []
    for label in labels:
        if label == reference:
            continue
        for name in names:
            values = curves[label][name] - curves[reference][name]
            differences.append({"run": label, "reference": reference, "group": name,
                                "initial_difference": float(values[0]), "final_difference": float(values[-1]),
                                "max_abs_difference": float(np.abs(values).max()),
                                "difference_integral_ns": trapezoid(values, common_time)})
    return {"runs": run_reports, "reference": reference, "shared_groups": names,
            "common_time_points": len(common_time), "window_ns": [float(common_time[0]),float(common_time[-1])],
            "differences": differences, "inputs": fingerprint(inputs),
            "launcher_manifests": launcher_manifests(inputs, explicit_manifests),
            "interpretation_limits": ["Exact intersection of saved absolute times; no interpolation, time shifting or zero-filling absent groups.",
                                      "Same group names must mean the same physical observable; semantic equivalence is user-declared.",
                                      "Different initial populations answer different conditional questions; differences do not establish improved extraction."]}, common_time, curves
