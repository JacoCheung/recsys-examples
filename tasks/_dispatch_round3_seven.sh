#!/bin/bash
# Re-submit the 7 round2 thread_map variants with the patched executor.py
# (now has [engine] orange NVTX prefix + inline_thread="compute" so step N
# projects to default stream). All to default `batch` partition for max
# concurrency.
set -euo pipefail

ACCOUNT="${ACCOUNT:-coreai_devtech_all}"
CONTAINER="${CONTAINER:-gitlab-master.nvidia.com/devtech-compute/distributed-recommender:devel_latest}"
NODES="${NODES:-1}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"

OLD_LA_CLONE="${OLD_LA_CLONE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/junzhang/benchmark_runs/mtms_sweep_round2_20260429_233012}"
SWEEP_ROOT="${SWEEP_ROOT:?SWEEP_ROOT required}"
TRACK="${SWEEP_ROOT}/_track.txt"

submit_one() {
  local backend="$1"
  local variant="$2"
  local label="$3"
  local outdir="${SWEEP_ROOT}/${label}"
  local jobname="r3c_${label}"

  echo ""
  echo "=================================================================="
  echo "[combo] backend=${backend} variant='${variant}' label=${label}"
  echo "=================================================================="

  cd "${OLD_LA_CLONE}/examples/hstu"

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
      --hstu-root="${OLD_LA_CLONE}/examples/hstu" \
      -y 2>&1 | tee "/tmp/r3c_${label}.log" | grep -E "Job ID|Submitted|✅"

  local jobid
  jobid=$(grep -oE "Job ID [0-9]+" "/tmp/r3c_${label}.log" | head -1 | awk '{print $3}' || true)
  [ -z "${jobid}" ] && jobid="UNKNOWN"
  printf "%-42s | %s (re-run with patched executor)\n" "${label}" "${jobid}" >> "${TRACK}"
}

submit_one "legacy" ""                                "legacy_none"
submit_one "new"    "default"                         "new_default"
submit_one "new"    "by_stream"                       "new_by_stream"
submit_one "new"    "per_task"                        "new_per_task"
submit_one "new"    "io_prefetch_compute"             "new_io_prefetch_compute"
submit_one "new"    "io_data_dist_compute"            "new_io_data_dist_compute"
submit_one "new"    "io_data_dist_prefetch_compute"   "new_io_data_dist_prefetch_compute"

echo ""
cat "${TRACK}"
