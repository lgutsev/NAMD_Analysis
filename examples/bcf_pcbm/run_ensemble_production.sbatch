#!/bin/bash
#SBATCH --job-name=ens_prod
#SBATCH --account=loni_perovsk27
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=1-3
#SBATCH --output=ens_prod_%A_%a.out
#SBATCH --error=ens_prod_%A_%a.err
#
# Production ensemble: 100 SHPROP histories per run, three runs.
#
#   passes  subset of  histories  subset of  runs
#
# One array task per RUN, not per history: histories stream one at a time
# inside a task, so peak memory is per-history (~1.5 GiB measured) regardless
# of how many there are. Measured 73.6 s per 10M-row history, so ~2.0 h for
# 100. The 4 h wall is that plus headroom.
#
# To split finer, set --array=1-300 and select a single history per task with
# SLURM_ARRAY_TASK_ID; the per_history.csv rows concatenate, then run the
# combine step. Rows are flushed per history either way, so a crash at
# history 87 keeps the first 86.
#
# After all three tasks finish, run the COMBINE step at the bottom of this
# file. Only the combine output licenses a Campaign-level statement.

set -euo pipefail

# ---------------------------------------------------------------------------
# EDIT THESE. Each run needs its OWN provenance: its own projection character,
# its own state map, its own cycle length. Campaign A values must not be
# reused for B or C -- the band mapping, SHPROP state order, atom partition and
# cyclic-frame provenance are all per-run facts.
# ---------------------------------------------------------------------------
declare -A RUN_DIR=(
  [1]=/ddnB/work/lgutsev/MD/FAPI_MD_NAMD_2026/NuTest
  [2]=/ddnB/work/lgutsev/MD/FAPI_MD_NAMD_2026/RUN_B_PLACEHOLDER
  [3]=/ddnB/work/lgutsev/MD/FAPI_MD_NAMD_2026/RUN_C_PLACEHOLDER
)
declare -A RUN_CHAR=(
  [1]=/ddnB/work/lgutsev/MD/FAPI_MD_NAMD_2026/NuTest/character_test_A_1030646/projection_character.csv
  [2]=RUN_B_projection_character.csv
  [3]=RUN_C_projection_character.csv
)
declare -A RUN_LABEL=( [1]=A [2]=B [3]=C )
# The recycled nuclear trajectory length. A is 1999; B and C must be
# established from their own manifests, not assumed equal.
declare -A RUN_CYCLE=( [1]=1999 [2]=0 [3]=0 )
# Crossing manifold window, per run. A is 1320:1331. B and C must be located
# from their own gap/character data before this is filled in.
declare -A RUN_EPISODE=( [1]=1320:1331 [2]= [3]= )

LATE_START_NS=0.1      # the Campaign A manuscript window: t >= 100 ps

T=${SLURM_ARRAY_TASK_ID:-1}
LABEL=${RUN_LABEL[$T]}
DIR=${RUN_DIR[$T]}
CHAR=${RUN_CHAR[$T]}
CYCLE=${RUN_CYCLE[$T]}
EPISODE=${RUN_EPISODE[$T]}
OUT=ens_run${LABEL}_${SLURM_ARRAY_JOB_ID:-manual}

echo "========================================"
echo "Ensemble production, run ${LABEL}"
echo "Date:  $(date)"
echo "Host:  $(hostname)"
echo "Dir:   ${DIR}"
echo "========================================"

if [[ "${CYCLE}" == "0" || -z "${EPISODE}" ]]; then
    echo "ERROR: run ${LABEL} has no cycle length and/or no episode window set."
    echo "       These are per-run provenance facts and must be established"
    echo "       from run ${LABEL}'s own manifest and gap/character data."
    echo "       Campaign A values must NOT be reused. Edit this script."
    exit 1
fi

cd "${DIR}"

if ! namd-analysis character-ensemble --help >/dev/null 2>&1; then
    echo "ERROR: this build has no 'character-ensemble' subcommand."
    echo "       Update the install from the branch: pip install -e ."
    exit 1
fi

N=$(ls -1 SHPROP.* 2>/dev/null | wc -l)
echo "SHPROP histories found: ${N}"
if [[ "${N}" -lt 1 ]]; then
    echo "ERROR: no SHPROP histories in ${DIR}"
    exit 1
fi
if [[ "${N}" -ne 100 ]]; then
    echo "NOTE: expected 100 histories for a production run, found ${N}."
    echo "      Proceeding, but the per-run denominators will say ${N}."
fi

echo
echo "========== PER HISTORY + PER RUN =========="

/usr/bin/time -v namd-analysis character-ensemble \
    --run-label "${LABEL}" \
    --shprop 'SHPROP.*' \
    --projection-character "${CHAR}" \
    --state-map state_map.json \
    --frame-mode dish-cyclic \
    --cycle-length "${CYCLE}" \
    --episode-window "${EPISODE}" \
    --late-window-start-ns "${LATE_START_NS}" \
    --out "${OUT}"

echo
echo "========== FINISHED run ${LABEL} =========="
date
echo
echo "  ${OUT}/per_history.csv       <- one row per SHPROP, level 1"
echo "  ${OUT}/run_summary.json      <- the 100 histories aggregated, level 2"
echo "  ${OUT}/ensemble_curves.json  <- median and quantile band across histories"
echo
echo "When all three runs are done, gather them (level 3):"
echo
echo "  namd-analysis character-ensemble --combine \\"
echo "      ens_runA_*/run_summary.json \\"
echo "      ens_runB_*/run_summary.json \\"
echo "      ens_runC_*/run_summary.json \\"
echo "      --out ens_across_runs"
echo
echo "Only that output licenses a Campaign-level statement. Read the spread"
echo "across the three runs, not a grand mean over 300 histories: histories"
echo "within a run share a nuclear trajectory and are not independent nuclear"
echo "configurations."
du -sh "${OUT}"
