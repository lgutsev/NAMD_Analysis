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
# Production ensemble: every SHPROP electronic history a configuration has.
#
#   passes  subset of  histories  subset of  configuration
#
# A, B and C are DISTINCT INTERFACE CONFIGURATIONS, not replicates. Each one's
# whole-archive summary licenses statements about that configuration on its
# own; a Campaign A statement needs Campaign A, not B and C as well. Combining
# them afterwards is a cross-configuration comparison, not a statistical level.
#
# The denominator is whatever the archive holds, counted below and carried into
# the summary. A currently has 98 numbered histories, and 98 is what its
# results say: a missing number is a history that was never written, never one
# to be manufactured to round the count up.
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
# THE STUDY LAYOUT. Recorded in examples/bcf_pcbm/production_profile.json,
# which the tests hold to the values below; change both together.
#
#   common root:       ${ROOT}
#   reference/control: ${ROOT}/NuTest        shared by the whole A/B/C study,
#                                            NOT an A-specific directory
#   production data:   ${ROOT}/FAPI_001_BCF_PCBM_{A,B,C}   the SHPROP archives
#   analysis output:   <configuration>/analysis/ensemble_<version>_<jobid>/
#
# Configuration-specific reference files live under NuTest/<label>/, one
# directory per configuration on equal footing. A is not the parent of B and C;
# it is only the configuration whose reference analysis was established first.
#
# Every path below is absolute, so this can be submitted from anywhere,
# including the common root itself.
#
# EDIT THESE. Each configuration needs its OWN provenance. Campaign A values
# must not be reused for B or C -- the band mapping, SHPROP state order, atom
# partition and crossing window are per-configuration facts. What IS shared is
# study-level: the recycled nuclear trajectory below.
#
# REQUIRED per configuration: directory, projection character, state map,
#   fixed (nominal) state map, cycle length.
# OPTIONAL: the crossing/episode window. Leave it empty if that configuration's
#   manifold has not been located yet -- the population and decomposition
#   analysis runs without it, and episode classification can be added later
#   once the manifold is found.
# ---------------------------------------------------------------------------

ROOT=/ddnB/work/lgutsev/MD/FAPI_MD_NAMD_2026
REFERENCE_ROOT="${ROOT}/NuTest"

declare -A RUN_LABEL=(
  [1]=A
  [2]=B
  [3]=C
)

# Real production SHPROP folders
declare -A RUN_DIR=(
  [1]="${ROOT}/FAPI_001_BCF_PCBM_A"
  [2]="${ROOT}/FAPI_001_BCF_PCBM_B"
  [3]="${ROOT}/FAPI_001_BCF_PCBM_C"
)

# Each configuration's reference products, prepared into its own directory
# under the shared control workspace. A's were established by the character run
# in NuTest/character_test_A_1030646; place (or link) them as NuTest/A/ so all
# three configurations are addressed the same way.
declare -A RUN_CHAR=(
  [1]="${REFERENCE_ROOT}/A/projection_character.csv"
  [2]="${REFERENCE_ROOT}/B/projection_character.csv"
  [3]="${REFERENCE_ROOT}/C/projection_character.csv"
)

declare -A RUN_STATE_MAP=(
  [1]="${REFERENCE_ROOT}/A/state_map.json"
  [2]="${REFERENCE_ROOT}/B/state_map.json"
  [3]="${REFERENCE_ROOT}/C/state_map.json"
)

declare -A RUN_FIXED_MAP=(
  [1]="${REFERENCE_ROOT}/A/fixed_state_map.json"
  [2]="${REFERENCE_ROOT}/B/fixed_state_map.json"
  [3]="${REFERENCE_ROOT}/C/fixed_state_map.json"
)


# The recycled nuclear trajectory length. 1999 frames is a property of the
# shared nuclear trajectory the whole study recycles, not a value borrowed from
# A: confirm it against each configuration's own manifest before trusting the
# frame axis. Nothing else transfers -- state maps, fixed maps, band numbers,
# atom partitions and crossing windows stay configuration-specific.
declare -A RUN_CYCLE=( [1]=1999 [2]=1999 [3]=1999 )
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
JOBID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
# One dedicated directory per analysis run, under the configuration it belongs
# to. Nothing is written into the top level of the configuration directory:
# those already hold the archive itself.
OUT="${DIR}/analysis/ensemble_v0.7.1_${JOBID}"

echo "========================================"
echo "Ensemble production, configuration ${LABEL}"
echo "Date:  $(date)"
echo "Host:  $(hostname)"
echo "Dir:   ${DIR}"
echo "========================================"

# Required provenance. The episode window is deliberately NOT in this list.
if [[ "${CYCLE}" == "0" ]]; then
    echo "ERROR: configuration ${LABEL} has no cycle length set."
    echo "       The recycled nuclear trajectory length is required provenance."
    echo "       Confirm it against configuration ${LABEL}'s own manifest rather"
    echo "       than assuming another configuration's value."
    exit 1
fi

if [[ ! -d "${DIR}" ]]; then
    echo "ERROR: configuration ${LABEL} has no data directory: ${DIR}"
    echo "       Each configuration keeps its SHPROP archive in its own"
    echo "       FAPI_001_BCF_PCBM_<label> directory under the common root."
    exit 1
fi

cd "${DIR}"
mkdir -p "${DIR}/analysis"

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
echo "This configuration's denominator is therefore N = ${N}."
echo "Gaps in the numbering are not filled in: a history that was never"
echo "written is not one this analysis invents."

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
echo "  ${OUT}/run_summary.json      <- this configuration's ${N} histories"
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
echo "  cd ${ROOT}"
echo "  namd-analysis character-ensemble --combine \\"
echo "      FAPI_001_BCF_PCBM_A/analysis/ensemble_v0.7.1_*/run_summary.json \\"
echo "      FAPI_001_BCF_PCBM_B/analysis/ensemble_v0.7.1_*/run_summary.json \\"
echo "      FAPI_001_BCF_PCBM_C/analysis/ensemble_v0.7.1_*/run_summary.json \\"
echo "      --combine-curves \\"
echo "          A=FAPI_001_BCF_PCBM_A/analysis/ensemble_v0.7.1_JOBID \\"
echo "          B=FAPI_001_BCF_PCBM_B/analysis/ensemble_v0.7.1_JOBID \\"
echo "          C=FAPI_001_BCF_PCBM_C/analysis/ensemble_v0.7.1_JOBID \\"
echo "      --out analysis_across_configurations"
echo
echo "Nothing is pooled across configurations: they are different physical"
echo "systems, so a mean over them is not a measurement of anything."
du -sh "${OUT}"
