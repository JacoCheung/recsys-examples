#!/bin/bash
# Re-submit 5 still-pending round3 variants to default `batch` partition
# (no QOSGrpNodeLimit) instead of batch_short (capped at 20 nodes shared
# across all batch_short users → was forcing serial execution).
#
# Reuses the existing SWEEP_ROOT so flatten still works with one path.
set -euo pipefail

ACCOUNT="${ACCOUNT:-coreai_devtech_all}"
CONTAINER="${CONTAINER:-gitlab-master.nvidia.com/devtech-compute/distributed-recommender:devel_latest}"
NODES="${NODES:-1}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"

OLD_LA_CLONE="${OLD_LA_CLONE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/junzhang/benchmark_runs/mtms_sweep_round2_20260429_233012}"
NEW_LA_CLONE="${NEW_LA_CLONE:-/lustre/fsw/portfolios/coreai/users/junzhang/workspace/recsys-rework-mtms-la-cascade}"

SWEEP_ROOT="${SWEEP_ROOT:?SWEEP_ROOT required}"
TRACK="${SWEEP_ROOT}/_track.txt"

submit_one() {
  local clone="$1"
  local backend="$2"
  local variant="$3"
  local label="$4"
  local outdir="${SWEEP_ROOT}/${label}"
  local jobname="r3b_${label}"

  echo ""
  echo "=================================================================="
  echo "[combo] backend=${backend} variant='${variant}' label=${label}"
  echo "=================================================================="

  cd "${clone}/examples/hstu"

  # NOTE: no --partition arg → defaults to `batch` (QOS p_batch, no node cap)
  RECSYS_PIPELINE_BACKEND="${backend}" \
  HSTU_THREAD_MAP_VARIANT="${variant}" \
    bash training/benchmark/scripts/submit_all_experiments_slurm.sh \
      --account="${ACCOUNT}" \
      --job-name="${jobname}" \
      --container-image="${CONTAINER}" \
      --nodes="${NODES}" \
      --time="${TIME_LIMIT}" \
      --nsys \
      --results-dir="${outdir}" \
      --exp-file=training/benchmark/experiments_pipeline_sweep.txt \
      --hstu-root="${clone}/examples/hstu" \
      -y 2>&1 | tee "/tmp/r3b_${label}.log"

  local jobid
  jobid=$(grep -oE "Job ID [0-9]+" "/tmp/r3b_${label}.log" | head -1 | awk '{print $3}' || true)
  [ -z "${jobid}" ] && jobid="UNKNOWN"
  printf "%-42s | %s (resubmitted to batch)\n" "${label}" "${jobid}" >> "${TRACK}"
}

# 5 variants that were cancelled from batch_short
submit_one "${OLD_LA_CLONE}" "new" "per_task"                        "new_per_task"
submit_one "${OLD_LA_CLONE}" "new" "io_prefetch_compute"             "new_io_prefetch_compute"
submit_one "${OLD_LA_CLONE}" "new" "io_data_dist_compute"            "new_io_data_dist_compute"
submit_one "${OLD_LA_CLONE}" "new" "io_data_dist_prefetch_compute"   "new_io_data_dist_prefetch_compute"
submit_one "${NEW_LA_CLONE}" "new" "default"                         "la_cascade"

echo ""
echo "================================ tracking ================================"
cat "${TRACK}"
