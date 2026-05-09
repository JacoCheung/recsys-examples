#!/bin/bash
# Round-4 sweep on cw-dfw: same 9-variant matrix as round3, but on the
# squashed + rebased junzhang/rework-mtms branch.
#
# Single-clone setup: HSTU_LA_DEPTH={3,6} env var picks pipeline depth
# (= max(la)+1) at runtime — depth=3 → plateau cascade (2/2/2/1/1/1),
# depth=6 → 6-la cascade (5/4/3/3/2/1).
#
# Variant naming: <thread_map>_d{depth}. la is encoded in the suffix
# rather than via an "OLD/NEW" preset shorthand.
#
# Variant matrix (9 jobs × N reps):
#   1. legacy_none                  (legacy backend, no la concept)
#   2. default_d3                   d3, default 2-thread map
#   3. by_stream_d3                 d3, by_stream 4-thread
#   4. per_task_d3                  d3, per_task 14-thread
#   5. io_prefetch_compute_d3       d3, io/prefetch/compute 3-thread
#   6. io_data_dist_compute_d3      d3, io/data_dist/compute 3-thread
#   7. full_split_d3                d3, full_split 4-thread
#   8. default_d6                   d6, default 2-thread map (= round3 la_cascade)
#   9. full_split_d6                d6, full_split 4-thread (= round3 la_cascade_full_split)
set -euo pipefail

ACCOUNT="${ACCOUNT:-coreai_devtech_all}"
CONTAINER="${CONTAINER:-gitlab-master.nvidia.com/devtech-compute/distributed-recommender:devel_latest}"
NODES="${NODES:-1}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
PARTITION="${PARTITION:-batch_short}"
JOBNAME_PREFIX="${JOBNAME_PREFIX:-r4}"

CLONE="${CLONE:-/lustre/fsw/portfolios/coreai/users/junzhang/workspace/recsys-rework-mtms}"

REPS="${REPS:-3}"
SWEEP_BASE="${SWEEP_BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/junzhang/benchmark_runs/round4_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "${SWEEP_BASE}"

TRACK="${SWEEP_BASE}/_track.txt"
{
  echo "# Round-4 sweep — 1 node x 8 H100 cw-dfw, nsys ON, ${REPS} reps × 9 variants"
  echo "# Submitted: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# CLONE:      ${CLONE}"
  echo "# SWEEP_BASE: ${SWEEP_BASE}"
  echo "#"
  echo "# rep | variant_label                            | jobid"
  echo "# ----|------------------------------------------|---------"
} > "${TRACK}"

submit_one() {
  local rep="$1"
  local backend="$2"
  local variant="$3"
  local depth="$4"
  local label="$5"
  local outdir="${SWEEP_BASE}/rep${rep}/${label}"
  local jobname="${JOBNAME_PREFIX}.rep${rep}_${label}"

  cd "${CLONE}/examples/hstu"
  local part_arg=()
  [ -n "${PARTITION}" ] && part_arg+=(--partition="${PARTITION}")

  RECSYS_PIPELINE_BACKEND="${backend}" \
  HSTU_THREAD_MAP_VARIANT="${variant}" \
  HSTU_LA_DEPTH="${depth}" \
    bash training/benchmark/scripts/submit_all_experiments_slurm.sh \
      --account="${ACCOUNT}" \
      --job-name="${jobname}" \
      --container-image="${CONTAINER}" \
      --nodes="${NODES}" \
      --time="${TIME_LIMIT}" \
      "${part_arg[@]}" \
      --nsys \
      --results-dir="${outdir}" \
      --exp-file=training/benchmark/experiments_pipeline_sweep.txt \
      --hstu-root="${CLONE}/examples/hstu" \
      -y 2>&1 | tee "/tmp/r4_${rep}_${label}.log"

  local jobid
  jobid=$(grep -oE "Job ID [0-9]+" "/tmp/r4_${rep}_${label}.log" | head -1 | awk '{print $3}' || true)
  [ -z "${jobid}" ] && jobid="UNKNOWN"
  printf "%4s | %-40s | %s\n" "${rep}" "${label}" "${jobid}" >> "${TRACK}"
}

for rep in $(seq 1 "${REPS}"); do
    # Legacy backend — no la concept.
    submit_one "${rep}" "legacy" ""                                 ""  "legacy_none"
    # 6 d3 (plateau cascade) new-pipeline variants — thread_map sweep.
    submit_one "${rep}" "new"    "default"                          "3" "default_d3"
    submit_one "${rep}" "new"    "by_stream"                        "3" "by_stream_d3"
    submit_one "${rep}" "new"    "per_task"                         "3" "per_task_d3"
    submit_one "${rep}" "new"    "io_prefetch_compute"              "3" "io_prefetch_compute_d3"
    submit_one "${rep}" "new"    "io_data_dist_compute"             "3" "io_data_dist_compute_d3"
    submit_one "${rep}" "new"    "io_data_dist_prefetch_compute"    "3" "full_split_d3"
    # 2 d6 (6-la cascade) variants.
    submit_one "${rep}" "new"    "default"                          "6" "default_d6"
    submit_one "${rep}" "new"    "io_data_dist_prefetch_compute"    "6" "full_split_d6"
done

echo ""
echo "================================ tracking ================================"
cat "${TRACK}"
echo ""
echo "Sweep root: ${SWEEP_BASE}"
echo "Total jobs: $((REPS * 9))"
