#!/bin/bash
#SBATCH --job-name=ens_prod
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
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
# No --mem is requested: one task is one core on a shared partition, and the
# partition's per-core default covers the ~1.5 GiB this needs. The figure is a
# measurement, not a budget -- it does not grow with history count, because
# histories are streamed rather than stacked. Re-measure before raising it.
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
# OPTIONAL: the crossing/episode windows. Leave them empty if that
#   configuration's manifold has not been located yet -- the population and
#   decomposition analysis runs without them, and episode classification can be
#   added later once the manifold is found.
#
# A configuration may have SEVERAL crossing windows. B has four distinct
# BCF/PCBM mixing regions; they are passed by name and every one of them is
# evaluated during the SAME streamed pass over each ~900 MB SHPROP history, so
# four regions cost one read of the archive rather than four. They are four
# regions of one recycled nuclear trajectory, NOT four replicates and NOT four
# independent samples, and nothing averages or pools them.
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

# Crossing manifold windows, per configuration. OPTIONAL, and a configuration
# may have more than one. Space-separated NAME=FIRST:LAST entries; the name
# travels through per_history.csv, episode_per_history.csv and run_summary.json
# so a number can always be traced back to the region it came from.
#
#   A  one located BCF/PCBM mixing region
#   B  FOUR distinct BCF/PCBM mixing regions, reported separately by name.
#      They are four regions of the SAME recycled nuclear trajectory. They are
#      not replicates, not independent samples and not repeat measurements, so
#      nothing averages or pools them and no combined B number is produced.
#   C  NO clear BCF/PCBM character-exchange event, so no crossing window. That
#      emptiness is the finding, not a gap to be filled.
declare -A RUN_EPISODES=(
  [1]="crossing_A=1320:1331"
  [2]="crossing_B1=1488:1492 crossing_B2=1687:1696 crossing_B3=1801:1811 crossing_B4=1846:1855"
  [3]=""
)

# CONTROL windows, per configuration. A control window is a range of frames
# chosen for comparison. It is NOT an avoided crossing, NOT a character-exchange
# region and NOT a BCF/PCBM transfer event, and no output may describe it as
# one. It is passed through --control-episode so that every file records it as
# role=control and it can never be read back as a crossing.
#
#   C  closest_approach_C is where the two fragments come closest. It is a
#      control for C's absence of a crossing, and nothing more.
declare -A RUN_CONTROL_EPISODES=(
  [1]=""
  [2]=""
  [3]="closest_approach_C=1628:1632"
)

# The legacy single-window flag, kept so an existing recipe keeps working. It
# is NOT used by the configurations above, which pass named windows instead.
# Set it only for a one-off run of a single unnamed window.
declare -A RUN_EPISODE=( [1]= [2]= [3]= )

LATE_START_NS=0.1      # the Campaign A manuscript window: t >= 100 ps

T=${SLURM_ARRAY_TASK_ID:-1}
LABEL=${RUN_LABEL[$T]}
DIR=${RUN_DIR[$T]}
CHAR=${RUN_CHAR[$T]}
STATE_MAP=${RUN_STATE_MAP[$T]}
FIXED_MAP=${RUN_FIXED_MAP[$T]}
CYCLE=${RUN_CYCLE[$T]}
EPISODE=${RUN_EPISODE[$T]}
EPISODES=${RUN_EPISODES[$T]}
CONTROL_EPISODES=${RUN_CONTROL_EPISODES[$T]}
JOBID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
# One dedicated directory per analysis run, under the configuration it belongs
# to. Nothing is written into the top level of the configuration directory:
# those already hold the archive itself.
OUT="${DIR}/analysis/ensemble_v0.7.2_${JOBID}"

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

# WHICH IMPLEMENTATION IS ABOUT TO RUN. The final A/B/C comparison is only
# valid if all three array tasks used the same code, and "same version" is a
# weaker claim than "same commit": an editable checkout can move between tasks.
# The same record is written into run_summary.json under environment.code, so
# a result can be traced back to its implementation long after the log is gone.
echo
echo "========== ANALYSIS CODE =========="
python -c "import json,namd_analysis.provenance as p; print(json.dumps(p.code_provenance(), indent=2))"
if python -c "import sys,namd_analysis.provenance as p; sys.exit(0 if p.code_provenance().get('git',{}).get('dirty') else 1)"; then
    echo
    echo "WARNING: the analysis package has UNCOMMITTED CHANGES."
    echo "         The commit above does not identify what ran, and two array"
    echo "         tasks may not have used the same code. Commit before a"
    echo "         production run, or A/B/C cannot be attributed to one"
    echo "         frozen analysis version."
fi

# The individual electronic histories are SHPROP.<integer> and nothing else.
# A bare SHPROP.* glob also matches SHPROP.master, SHPROP.avg, SHPROP.1.bak and
# whatever else happens to sit in the archive. An averaged or derived file
# ingested as though it were one history would inflate the denominator and
# contribute a row that is not an electronic initial condition at all. So
# select by the exact pattern, and pass the SAME list that was counted: the
# denominator and the analysed set then cannot differ.
HISTORIES=()
while IFS= read -r history; do
    HISTORIES+=("${history}")
done < <(ls -1 2>/dev/null | grep -E '^SHPROP\.[0-9]+$' | sort -t. -k2 -n)

N=${#HISTORIES[@]}
MATCHED=$(ls -1 2>/dev/null | grep -c '^SHPROP\.' || true)
echo "SHPROP.<integer> histories found: ${N}"
if [[ "${N}" -lt 1 ]]; then
    echo "ERROR: no SHPROP.<integer> histories in ${DIR}"
    echo "       Files named SHPROP.* that are not SHPROP.<integer> are not"
    echo "       individual electronic histories and are never analysed as one."
    exit 1
fi
if [[ "${MATCHED}" -ne "${N}" ]]; then
    echo "NOTE: $(( MATCHED - N )) file(s) match SHPROP.* but are NOT"
    echo "      SHPROP.<integer>, and are excluded. They are not individual"
    echo "      electronic histories; analysing an averaged or derived file as"
    echo "      one would inflate the denominator. Excluded:"
    ls -1 2>/dev/null | grep -E '^SHPROP\.' | grep -vE '^SHPROP\.[0-9]+$' \
        | sed 's/^/        /'
fi
echo "This configuration's denominator is therefore N = ${N}."
echo "Gaps in the numbering are not filled in: a history that was never"
echo "written is not one this analysis invents."

EPISODE_ARGS=()
if [[ -n "${EPISODE}" ]]; then
    EPISODE_ARGS+=(--episode-window "${EPISODE}")
    echo "Crossing window (legacy single-window flag): ${EPISODE}"
fi
for spec in ${EPISODES}; do
    EPISODE_ARGS+=(--episode "${spec}")
    echo "Crossing window: ${spec}"
done
for spec in ${CONTROL_EPISODES}; do
    EPISODE_ARGS+=(--control-episode "${spec}")
    echo "CONTROL window: ${spec}"
    echo "      A control window is a range of frames chosen for comparison."
    echo "      It is NOT an avoided crossing and NOT a BCF/PCBM transfer"
    echo "      event, and it is recorded as role=control everywhere."
done
if [[ -z "${EPISODE}" && -z "${EPISODES}" ]]; then
    echo "NOTE: no crossing window set for configuration ${LABEL}."
    echo "      Populations, the symmetric decomposition and the fixed-vs-"
    echo "      dynamic comparison all run. Episode classification is skipped"
    echo "      and every history reports episode_verdict=not_classified."
    echo "      Locate this configuration's manifold separately, then rerun"
    echo "      with --episode to add it."
fi
if [[ -n "${EPISODES}" ]]; then
    echo "All windows above are evaluated in ONE streamed pass per history."
    echo "They are distinct regions of one recycled nuclear trajectory:"
    echo "reported separately, never averaged, never pooled."
fi

echo
echo "========== PER HISTORY + PER CONFIGURATION =========="

/usr/bin/time -v namd-analysis character-ensemble \
    --run-label "${LABEL}" \
    --shprop "${HISTORIES[@]}" \
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
echo "                                 range_late_fixed, plus one named column"
echo "                                 block per episode, e.g."
echo "                                 crossing_B1__BCF_net_per_pass"
echo "  ${OUT}/episode_per_history.csv  <- long form: one row per history,"
echo "                                 episode and fragment, each row naming"
echo "                                 its own episode and role"
echo "  ${OUT}/run_summary.json      <- this configuration's ${N} histories,"
echo "                                 with environment.code recording the"
echo "                                 package version and the git commit"
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
echo "      FAPI_001_BCF_PCBM_A/analysis/ensemble_v0.7.2_*/run_summary.json \\"
echo "      FAPI_001_BCF_PCBM_B/analysis/ensemble_v0.7.2_*/run_summary.json \\"
echo "      FAPI_001_BCF_PCBM_C/analysis/ensemble_v0.7.2_*/run_summary.json \\"
echo "      --combine-curves \\"
echo "          A=FAPI_001_BCF_PCBM_A/analysis/ensemble_v0.7.2_JOBID \\"
echo "          B=FAPI_001_BCF_PCBM_B/analysis/ensemble_v0.7.2_JOBID \\"
echo "          C=FAPI_001_BCF_PCBM_C/analysis/ensemble_v0.7.2_JOBID \\"
echo "      --out analysis_across_configurations"
echo
echo "Nothing is pooled across configurations: they are different physical"
echo "systems, so a mean over them is not a measurement of anything."
du -sh "${OUT}"
