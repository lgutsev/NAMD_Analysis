"""Descriptive analysis and comparison of phonon spectral densities.

These are measurements of a curve, not assignments.  A band integral changes
when the modes in that band change *or* when the overall normalization
changes, so both the raw and the in-range-normalized numbers are reported and
comparisons across systems are only made on a shared frequency grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks, peak_widths

from .io.tables import read_xy_with_header
from .vacf import Spectrum

#: Default analysis bands in cm^-1.  The 0-200 window is the range usually
#: discussed as low-frequency dynamic disorder in halide perovskites.
DEFAULT_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.0, 50.0),
    (50.0, 100.0),
    (100.0, 200.0),
    (200.0, 400.0),
    (400.0, 800.0),
)


class SpectrumError(ValueError):
    """Raised when spectra cannot be analyzed or compared as given."""


def load_spectrum(path, label: Optional[str] = None) -> Spectrum:
    """Read a two-column ``spectral_density_*.txt`` file."""
    path = Path(path)
    table, header = read_xy_with_header(path)
    if table.shape[1] < 2:
        raise SpectrumError(f"{path}: expected two columns, found {table.shape[1]}")
    if table.shape[1] > 2:
        raise SpectrumError(
            f"{path}: {table.shape[1]} columns; this reader takes frequency and "
            "intensity only"
        )
    frequency = table[:, 0]
    if np.any(np.diff(frequency) <= 0):
        raise SpectrumError(f"{path}: frequency column is not strictly increasing")
    name = label or path.stem.replace("spectral_density_", "")
    return Spectrum(
        frequency_cm1=frequency,
        intensity=table[:, 1],
        convention="unknown",
        window="unknown",
        smooth_sigma=float("nan"),
        label=name,
        meta={
            "source": str(path),
            "header": header,
            "resolution_cm1": float(frequency[1] - frequency[0]),
            "note": (
                "convention, window and smoothing are not recorded in this file "
                "format and are reported as unknown"
            ),
        },
    )


def _integrate(y: np.ndarray, x: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(y, x=x))


def band_integral(spectrum: Spectrum, low: float, high: float) -> float:
    mask = (spectrum.frequency_cm1 >= low) & (spectrum.frequency_cm1 <= high)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return _integrate(spectrum.intensity[mask], spectrum.frequency_cm1[mask])


def centroid(spectrum: Spectrum, low: float, high: float) -> float:
    """Intensity-weighted mean frequency in a window.

    Undefined when the windowed intensity is not everywhere non-negative,
    which the cosine convention can produce near a truncated tail.
    """
    mask = (spectrum.frequency_cm1 >= low) & (spectrum.frequency_cm1 <= high)
    freq = spectrum.frequency_cm1[mask]
    weight = spectrum.intensity[mask]
    if freq.size < 2 or np.any(weight < 0):
        return float("nan")
    total = _integrate(weight, freq)
    if total <= 0:
        return float("nan")
    return _integrate(weight * freq, freq) / total


def peak_table(
    spectrum: Spectrum,
    low: float,
    high: float,
    rel_height: float = 0.1,
    max_peaks: int = 20,
) -> List[Dict[str, float]]:
    """Local maxima above ``rel_height`` of the in-window maximum, with FWHM."""
    mask = (spectrum.frequency_cm1 >= low) & (spectrum.frequency_cm1 <= high)
    freq = spectrum.frequency_cm1[mask]
    values = spectrum.intensity[mask]
    if freq.size < 5:
        return []
    ceiling = float(np.max(values))
    if not np.isfinite(ceiling) or ceiling <= 0:
        return []
    indices, _ = find_peaks(values, height=rel_height * ceiling)
    if indices.size == 0:
        return []
    widths = peak_widths(values, indices, rel_height=0.5)[0]
    step = float(freq[1] - freq[0])
    order = np.argsort(values[indices])[::-1][:max_peaks]
    peaks = []
    for position in sorted(indices[order]):
        local = int(np.flatnonzero(indices == position)[0])
        peaks.append(
            {
                "frequency_cm1": float(freq[position]),
                "intensity": float(values[position]),
                "relative_intensity": float(values[position] / ceiling),
                "fwhm_cm1": float(widths[local] * step),
            }
        )
    return peaks


def describe(
    spectrum: Spectrum,
    bands: Sequence[Tuple[float, float]] = DEFAULT_BANDS,
    analysis_range: Tuple[float, float] = (0.0, 800.0),
    peak_rel_height: float = 0.1,
) -> Dict[str, Any]:
    """Band integrals, in-range fractions, centroid and peaks for one spectrum."""
    low, high = analysis_range
    total = band_integral(spectrum, low, high)
    in_range = (spectrum.frequency_cm1 >= low) & (spectrum.frequency_cm1 <= high)
    negative = int(np.count_nonzero(spectrum.intensity[in_range] < 0))
    bands_out = []
    for band_low, band_high in bands:
        value = band_integral(spectrum, band_low, band_high)
        bands_out.append(
            {
                "low_cm1": band_low,
                "high_cm1": band_high,
                "integral": value,
                "fraction_of_range": (
                    value / total if total not in (0.0,) and np.isfinite(total) else float("nan")
                ),
            }
        )
    return {
        "label": spectrum.label,
        "convention": spectrum.convention,
        "window": spectrum.window,
        "smooth_sigma": spectrum.smooth_sigma,
        "n_points": int(spectrum.frequency_cm1.size),
        "frequency_min_cm1": float(spectrum.frequency_cm1[0]),
        "frequency_max_cm1": float(spectrum.frequency_cm1[-1]),
        "resolution_cm1": float(spectrum.frequency_cm1[1] - spectrum.frequency_cm1[0]),
        "analysis_range_cm1": [low, high],
        "range_integral": total,
        "negative_samples_in_range": negative,
        "centroid_cm1": centroid(spectrum, low, high),
        "bands": bands_out,
        "peaks": peak_table(spectrum, low, high, rel_height=peak_rel_height),
        "meta": spectrum.meta,
    }


def shared_grid(spectra: Sequence[Spectrum], atol: float = 1e-6) -> bool:
    """True when every spectrum is sampled on the same frequency grid."""
    reference = spectra[0].frequency_cm1
    for spectrum in spectra[1:]:
        if spectrum.frequency_cm1.shape != reference.shape:
            return False
        if not np.allclose(spectrum.frequency_cm1, reference, rtol=0, atol=atol):
            return False
    return True


def compare(
    spectra: Sequence[Spectrum],
    reference_label: Optional[str] = None,
    bands: Sequence[Tuple[float, float]] = DEFAULT_BANDS,
    analysis_range: Tuple[float, float] = (0.0, 800.0),
    peak_rel_height: float = 0.1,
) -> Dict[str, Any]:
    """Describe each spectrum and, on a shared grid, difference them."""
    if not spectra:
        raise SpectrumError("no spectra to compare")
    descriptions = [
        describe(s, bands=bands, analysis_range=analysis_range,
                 peak_rel_height=peak_rel_height)
        for s in spectra
    ]
    payload: Dict[str, Any] = {
        "shared_grid": shared_grid(spectra),
        "systems": descriptions,
    }
    if len(spectra) < 2:
        return payload

    labels = [s.label for s in spectra]
    ref = reference_label or labels[0]
    if ref not in labels:
        raise SpectrumError(f"reference {ref!r} is not among {labels}")
    if not payload["shared_grid"]:
        payload["comparison_note"] = (
            "spectra are on different frequency grids; only per-system integrals "
            "are reported and no pointwise difference is computed"
        )
        return payload

    base = descriptions[labels.index(ref)]
    deltas = []
    for description in descriptions:
        if description["label"] == ref:
            continue
        band_rows = []
        for band, base_band in zip(description["bands"], base["bands"]):
            band_rows.append(
                {
                    "low_cm1": band["low_cm1"],
                    "high_cm1": band["high_cm1"],
                    "delta_fraction_of_range": band["fraction_of_range"]
                    - base_band["fraction_of_range"],
                    "ratio_of_integral": (
                        band["integral"] / base_band["integral"]
                        if base_band["integral"] not in (0.0,)
                        else float("nan")
                    ),
                }
            )
        deltas.append(
            {
                "label": description["label"],
                "reference": ref,
                "delta_centroid_cm1": description["centroid_cm1"] - base["centroid_cm1"],
                "ratio_range_integral": (
                    description["range_integral"] / base["range_integral"]
                    if base["range_integral"] not in (0.0,)
                    else float("nan")
                ),
                "bands": band_rows,
            }
        )
    payload["reference"] = ref
    payload["differences"] = deltas
    payload["comparison_note"] = (
        "band fractions are normalized inside the analysis range, so they are "
        "comparable between systems; raw integrals are not, unless the spectra "
        "share normalization, trajectory length and atom count"
    )
    return payload


SYSTEM_HEADER = [
    "label",
    "n_points",
    "resolution_cm1",
    "range_integral",
    "centroid_cm1",
    "negative_samples_in_range",
]

PEAK_HEADER = ["label", "frequency_cm1", "fwhm_cm1", "intensity", "relative_intensity"]


def system_rows(payload: Dict[str, Any]) -> List[List[Any]]:
    rows = []
    for system in payload["systems"]:
        row = [system[key] for key in SYSTEM_HEADER]
        rows.append(row)
    return rows


def band_rows(payload: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    header = ["label", "low_cm1", "high_cm1", "integral", "fraction_of_range"]
    rows = []
    for system in payload["systems"]:
        for band in system["bands"]:
            rows.append(
                [
                    system["label"],
                    band["low_cm1"],
                    band["high_cm1"],
                    band["integral"],
                    band["fraction_of_range"],
                ]
            )
    return header, rows


def peak_rows(payload: Dict[str, Any]) -> List[List[Any]]:
    rows = []
    for system in payload["systems"]:
        for peak in system["peaks"]:
            rows.append(
                [
                    system["label"],
                    peak["frequency_cm1"],
                    peak["fwhm_cm1"],
                    peak["intensity"],
                    peak["relative_intensity"],
                ]
            )
    return rows
