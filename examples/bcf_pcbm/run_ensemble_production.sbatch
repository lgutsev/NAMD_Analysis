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
# Production ensemble: 100 SHPROP electronic histories per configuration.
#
#   passes  subset of  histories  subset of  configuration
#
# A, B and C are DISTINCT INTERFACE CONFIGURATIONS, not replicates. Each one's
# 100-history summary licenses statements about that configuration on its own;
# a Campaign A statement needs Campaign A, not B and C as well. Combining them
# afterwards is a cross-configuration comparison, not a statistical level.
#
# One array task per configuration, not per history: histories stream one at a
# time inside a task, so peak memory is per-history (~1.5 GiB measured)
# regardless of how many there are. Measured 73.6 s per 10M-row history, so
# ~2.0 h for 100. The 4 h wall is that plus headroom.
#
# To split finer, set --array=1-300 and select a single history per task with
# SLURM_ARRAY_TASK_ID; per_history.csv rows concatenate. Rows are flushed per
# history either way, so a crash at history 87 keeps the first 86.

set -euo pipefail

# ---------------------------------------------------------------------------
# EDIT THESE. Each configuration needs its OWN provenance. Campaign A values
# must not be reused for B or C -- the band mapping, SHPROP state order, atom
# partition and cyclic-frame provenance are per-configuration facts.
#
# REQUIRED per configuration: directory, projection character, state map,
#   fixed (nominal) state map, cycle length.
# OPTIONAL: the crossing/episode window. Leave it empty if that configuration's
#   manifold has not been located yet -- the population and decomposition
#   analysis runs without it, and episode classification can be added later
#   once the manifold is found.
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
# The projection state map: SHPROP population columns in basis order, used for
# the PROCAR-weighted dynamic reading.
declare -A RUN_STATE_MAP=( [1]=state_map.json [2]=state_map.json [3]=state_map.json )
# The NOMINAL fixed-column state map: each column assigned to a fragment for
# the whole run, with no PROCAR weighting. This is the reading the
# fixed-vs-dynamic comparison exists to test, so it is REQUIRED, not optional.
# For A this is the established nominal map (VBM / BCF / PCBM1-3 / CBM).
declare -A RUN_FIXED_MAP=(
  [1]=state_map.json
  [2]=RUN_B_fixed_state_map.json
  [3]=RUN_C_fixed_state_map.json
)
declare -A RUN_LABEL=( [1]=A [2]=B [3]=C )
# The recycled nuclear trajectory length. A is 1999; B and C must be
# established from their own manifests, not assumed equal.
declare -A RUN_CYCLE=( [1]=1999 [2]=0 [3]=0 )
# Crossing manifold window, per configuration. OPTIONAL. A is 1320:1331.
# Leave B and C empty until their manifolds are located.
declare -A RUN_EPISODE=( [1]=1320:1331 [2]= [3]= )

LATE_START_NS=0.1      # the Campaign A manuscript window: t >= 100 ps

T=${SLURM_ARRAY_TASK_ID:-1}
LABEL=${RUN_LABEL[$T]}
DIR=${RUN_DIR[$T]}
CHAR=${RUN_CHAR[$T]}
STATE_MAP=${RUN_STATE_MAP[$T]}
FIXED_MAP=${RUN_FIXED_MAP[$T]}
CYCLE=${RUN_CYCLE[$T]}
EPISODE=${RUN_EPISODE[$T]}
OUT=ens_${LABEL}_${SLURM_ARRAY_JOB_ID:-manual}

echo "========================================"
echo "Ensemble production, configuration ${LABEL}"
echo "Date:  $(date)"
echo "Host:  $(hostname)"
echo "Dir:   ${DIR}"
echo "========================================"

# Required provenance. The episode window is deliberately NOT in this list.
if [[ "${CYCLE}" == "0" ]]; then
    echo "ERROR: configuration ${LABEL} has no cycle length set."
    echo "       The recycled nuclear trajectory length is required provenance"
    echo "       and is per-configuration. Campaign A's 1999 must NOT be reused."
    exit 1
fi

cd "${DIR}"

for required in "${CHAR}" "${STATE_MAP}" "${FIXED_MAP}"; do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: configuration ${LABEL} is missing required input: ${required}"
        echo "       Required per configuration: projection character, state map,"
        echo "       fixed (nominal) state map, cycle length."
        exit 1
    fi
done

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
    echo "NOTE: expected 100 histories, found ${N}. Proceeding; the per-"
    echo "      configuration denominators will say ${N}."
fi

EPISODE_ARGS=()
if [[ -n "${EPISODE}" ]]; then
    EPISODE_ARGS=(--episode-window "${EPISODE}")
    echo "Crossing window: ${EPISODE}"
else
    echo "NOTE: no crossing window set for configuration ${LABEL}."
    echo "      Populations, the symmetric decomposition and the fixed-vs-"
    echo "      dynamic comparison all run. Episode classification is skipped"
    echo "      and every history reports episode_verdict=not_classified."
    echo "      Locate this configuration's manifold separately, then rerun"
    echo "      with --episode-window to add it."
fi

echo
echo "========== PER HISTORY + PER CONFIGURATION =========="

/usr/bin/time -v namd-analysis character-ensemble \
    --run-label "${LABEL}" \
    --shprop 'SHPROP.*' \
    --projection-character "${CHAR}" \
    --state-map "${STATE_MAP}" \
    --fixed-state-map "${FIXED_MAP}" \
    --frame-mode dish-cyclic \
    --cycle-length "${CYCLE}" \
    "${EPISODE_ARGS[@]}" \
    --late-window-start-ns "${LATE_START_NS}" \
    --out "${OUT}"

echo
echo "========== FINISHED configuration ${LABEL} =========="
date
echo
echo "  ${OUT}/per_history.csv       <- one row per SHPROP, including the"
echo "                                 fixed-column columns net_*_fixed and"
echo "                                 range_late_fixed"
echo "  ${OUT}/run_summary.json      <- this configuration's 100 histories"
echo "  ${OUT}/ensemble_curves.json  <- median and quantile band, dynamic and"
echo "                                 fixed: keys and fixed:<group>"
echo "  ${OUT}/*.png  ${OUT}/*.pdf   <- populations, fixed-vs-dynamic,"
echo "                                 distributions, decomposition"
echo
echo "This output on its own licenses statements about configuration ${LABEL}."
echo "It does not need the other configurations to be interpretable."
echo
echo "To compare configurations afterwards (a cross-configuration comparison,"
echo "not a statistical level):"
echo
echo "  namd-analysis character-ensemble --combine \\"
echo "      ens_A_*/run_summary.json \\"
echo "      ens_B_*/run_summary.json \\"
echo "      ens_C_*/run_summary.json \\"
echo "      --combine-curves A=ens_A_JOBID B=ens_B_JOBID C=ens_C_JOBID \\"
echo "      --out ens_across_configurations"
echo
echo "Nothing is pooled across configurations: they are different physical"
echo "systems, so a mean over them is not a measurement of anything."
du -sh "${OUT}"
