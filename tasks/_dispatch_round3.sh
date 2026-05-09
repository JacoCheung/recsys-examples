#!/bin/bash
# Round-3 sweep on cw-dfw: 7 round2 thread_map variants on OLD la
# (depth+1 = 2/2/2/1/1/1, 3 batches in-flight) + 1 la_cascade variant
# on NEW 6-la cascade (5/4/3/3/2/1, 6 batches in-flight).
#
# Output layout (2-level):
#   <SWEEP_ROOT>/
#     legacy_none/                          (12 files: gin + log + 8 nsys-rep + sbatch out + ...)
#     new_default/
#     new_by_stream/
#     new_per_task/
#     new_io_prefetch_compute/
#     new_io_data_dist_compute/
#     new_io_data_dist_prefetch_compute/
#     la_cascade/
#
# Files are post-flattened after jobs complete via _flatten_round3.sh.
set -euo pipefail

ACCOUNT="${ACCOUNT:-coreai_devtech_all}"
CONTAINER="${CONTAINER:-gitlab-master.nvidia.com/devtech-compute/distributed-recommender:devel_latest}"
NODES="${NODES:-1}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
PARTITION="${PARTITION:-batch_short}"

OLD_LA_CLONE="${OLD_LA_CLONE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/junzhang/benchmark_runs/mtms_sweep_round2_20260429_233012}"
NEW_LA_CLONE="${NEW_LA_CLONE:-/lustre/fsw/portfolios/coreai/users/junzhang/workspace/recsys-rework-mtms-la-cascade}"

SWEEP_ROOT="${SWEEP_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/junzhang/benchmark_runs/mtms_sweep_round3_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "${SWEEP_ROOT}"

TRACK="${SWEEP_ROOT}/_track.txt"
{
  echo "# Round 3 sweep — 1 node x 8 H100 cw-dfw, nsys ON"
  echo "# Submitted: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# OLD_LA_CLONE: ${OLD_LA_CLONE}"
  echo "# NEW_LA_CLONE: ${NEW_LA_CLONE}"
  echo "# SWEEP_ROOT:   ${SWEEP_ROOT}"
  echo "#"
  echo "# variant_label                             | jobid"
  echo "# -----------------------------------------|---------"
} > "${TRACK}"

submit_one() {
  local clone="$1"     # repo clone path
  local backend="$2"   # legacy | new
  local variant="$3"   # thread_map preset name (or "" for legacy)
  local label="$4"     # output dir name
  local outdir="${SWEEP_ROOT}/${label}"
  local jobname="r3_${label}"

  echo ""
  echo "=================================================================="
  echo "[combo] backend=${backend} variant='${variant}' label=${label}"
  echo "[clone]  ${clone}"
  echo "[outdir] ${outdir}"
  echo "=================================================================="

  cd "${clone}/examples/hstu"

  local part_arg=()
  [ -n "${PARTITION}" ] && part_arg+=(--partition="${PARTITION}")

  RECSYS_PIPELINE_BACKEND="${backend}" \
  HSTU_THREAD_MAP_VARIANT="${variant}" \
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
      --hstu-root="${clone}/examples/hstu" \
      -y 2>&1 | tee "/tmp/r3_${label}.log"

  local jobid
  jobid=$(grep -oE "Job ID [0-9]+" "/tmp/r3_${label}.log" | head -1 | awk '{print $3}' || true)
  [ -z "${jobid}" ] && jobid="UNKNOWN"
  printf "%-42s | %s\n" "${label}" "${jobid}" >> "${TRACK}"
}

# 7 round2 variants on OLD la clone
submit_one "${OLD_LA_CLONE}" "legacy" ""                           "legacy_none"
submit_one "${OLD_LA_CLONE}" "new"    "default"                    "new_default"
submit_one "${OLD_LA_CLONE}" "new"    "by_stream"                  "new_by_stream"
submit_one "${OLD_LA_CLONE}" "new"    "per_task"                   "new_per_task"
submit_one "${OLD_LA_CLONE}" "new"    "io_prefetch_compute"        "new_io_prefetch_compute"
submit_one "${OLD_LA_CLONE}" "new"    "io_data_dist_compute"       "new_io_data_dist_compute"
submit_one "${OLD_LA_CLONE}" "new"    "io_data_dist_prefetch_compute" "new_io_data_dist_prefetch_compute"

# 8th: la_cascade — NEW 6-la cascade clone, default thread_map
submit_one "${NEW_LA_CLONE}" "new"    "default"                    "la_cascade"

echo ""
echo "================================ tracking ================================"
cat "${TRACK}"
echo ""
echo "Sweep root: ${SWEEP_ROOT}"
echo "Run _flatten_round3.sh \"${SWEEP_ROOT}\" once all jobs are COMPLETED."
