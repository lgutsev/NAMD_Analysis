#!/bin/bash
#SBATCH --job-name=crossings_A
#SBATCH --account=loni_perovsk27
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=crossings_A_%j.out
#SBATCH --error=crossings_A_%j.err
#
# Occupation-driven BCF -> PCBM transfer for Campaign A.
#
# Runtime: the per-history detector walks time steps in Python at ~85 us/step,
# measured. Five histories of 10,000,000 rows is ~1.2 h of detection on top of
# the SHPROP streaming, hence the 6 h wall request. It does NOT reopen a single
# PROCAR -- it reuses projection_character.csv from the character run.
#
# Memory: each history retains its fragment population, resolved frames and
# time axis (~400 MiB per history at 10M rows), and they are held together for
# the aggregate, so expect ~2.0-2.5 GiB. There is no memory-budget control on
# this path yet; if it is tight, run fewer --shprop files per job and combine.

set -euo pipefail

CAMPAIGN_DIR=/ddnB/work/lgutsev/MD/FAPI_MD_NAMD_2026/NuTest
CHARACTER_DIR="${CAMPAIGN_DIR}/character_test_A_1030646"
OUT="crossings_A_${SLURM_JOB_ID:-manual}"

# The late window is NOT defaulted anywhere in this package, deliberately: no
# manuscript fit window is encoded in the repository. Set it, in ns, to the
# window the manuscript actually uses, or leave it empty to skip the
# early-vs-late comparison entirely. The early window defaults to 0:0.1 ns.
LATE_WINDOW="${LATE_WINDOW:-}"

echo "========================================"
echo "Campaign A adiabatic character events"
echo "Date:    $(date)"
echo "Host:    $(hostname)"
echo "Job ID:  ${SLURM_JOB_ID:-none}"
echo "========================================"

cd "${CAMPAIGN_DIR}"

# ---------------------------------------------------------------------------
# The installed build must be new enough. The 0.5.2 build that produced
# character_test_A_1030646 has no character-crossings subcommand at all: it
# predates it. Update the install before running.
# ---------------------------------------------------------------------------
echo "namd-analysis: $(command -v namd-analysis)"
namd-analysis --version || true

if ! namd-analysis character-crossings --help >/dev/null 2>&1; then
    echo "ERROR: this build has no 'character-crossings' subcommand."
    echo "       Update the install from the branch first, e.g."
    echo "         cd /path/to/NAMD_Analysis && git fetch && git checkout <sha>"
    echo "         pip install -e ."
    echo "       then resubmit."
    exit 1
fi

# ---------------------------------------------------------------------------
# Inputs. EIGTXT is required; NATXT is optional but wanted -- without it no
# coupling is attached and the |NAC| column is empty.
# ---------------------------------------------------------------------------
for required in \
    "${CHARACTER_DIR}/projection_character.csv" \
    "${CAMPAIGN_DIR}/state_map.json" \
    "${CAMPAIGN_DIR}/projection_manifest.json" \
    "${CAMPAIGN_DIR}/EIGTXT"
do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: missing required input: ${required}"
        [[ "${required}" == *EIGTXT ]] && echo "       (state energies, one row per MD frame)"
        exit 1
    fi
done

NAC_ARGS=()
if [[ -f "${CAMPAIGN_DIR}/NATXT" ]]; then
    NAC_ARGS=(--natxt "${CAMPAIGN_DIR}/NATXT" --nac-unit eV)
    echo "NATXT found: couplings will be attached."
else
    echo "NOTE: no NATXT; gap and character only, |NAC| will be absent."
fi

ALIGN_ARGS=()
if [[ -f "${CHARACTER_DIR}/shprop_alignment.csv" ]]; then
    ALIGN_ARGS=(--shprop-alignment "${CHARACTER_DIR}/shprop_alignment.csv")
fi

WINDOW_ARGS=()
if [[ -n "${LATE_WINDOW}" ]]; then
    WINDOW_ARGS=(--late-window "${LATE_WINDOW}")
    echo "Late window: ${LATE_WINDOW} ns"
else
    echo "NOTE: no LATE_WINDOW set; no early-vs-late comparison will be made."
fi

echo
echo "========== CHARACTER CROSSINGS =========="

/usr/bin/time -v namd-analysis character-crossings \
    --projection-character "${CHARACTER_DIR}/projection_character.csv" \
    --eigtxt "${CAMPAIGN_DIR}/EIGTXT" \
    "${NAC_ARGS[@]}" \
    --dt-fs 1 \
    --shprop SHPROP.37 SHPROP.171 SHPROP.425 SHPROP.848 SHPROP.1625 \
    --state-map "${CAMPAIGN_DIR}/state_map.json" \
    --projection-manifest "${CAMPAIGN_DIR}/projection_manifest.json" \
    --frame-mode dish-cyclic \
    "${ALIGN_ARGS[@]}" \
    "${WINDOW_ARGS[@]}" \
    --out "${OUT}"

echo
echo "========== FINISHED =========="
date
echo
echo "Read these, in this order:"
echo "  ${OUT}/crossing_summary.md      <- per-history counts before the totals"
echo "  ${OUT}/per_history_events.csv   <- population_driven_change is the"
echo "                                     column that decides transfer;"
echo "                                     character_driven_change is not"
echo "  ${OUT}/report.json              <- provenance and the decomposition note"
echo
du -sh "${OUT}"
