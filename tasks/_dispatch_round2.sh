#!/bin/bash
# Round-2 sweep on EOS: 7 sbatch jobs in parallel, each 1 node x 8 H100, nsys ON.
#
# Combos:
#   1. legacy  | (no thread_map)               -> legacy pipeline baseline
#   2. new     | default                       -> new + HSTU_DEFAULT_THREAD_MAP
#   3. new     | by_stream                     -> new + by_stream routing
#   4. new     | per_task                      -> new + per_task routing
#   5. new     | io_prefetch_compute           -> new + 3-thread split
#   6. new     | io_data_dist_compute          -> new + 3-thread split (data_dist)
#   7. new     | io_data_dist_prefetch_compute -> new + 4-thread split
set -euo pipefail

CLONE="${CLONE:?CLONE env required (path to rework-mtms clone on lustre)}"
ACCOUNT="${ACCOUNT:-coreai_devtech_all}"
CONTAINER="${CONTAINER:-gitlab-master.nvidia.com/devtech-compute/distributed-recommender:devel_latest}"
NODES="${NODES:-1}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
PARTITION="${PARTITION:-}"   # optional; cw-dfw needs batch_short for shortest queue
TRACK="${CLONE}/tasks/mtms_sweep_round2.txt"

cd "${CLONE}/examples/hstu"

mkdir -p "${CLONE}/tasks"
{
  echo "# Round 2 sweep — 1 node x 8 H100 EOS, nsys ON"
  echo "# Clone: ${CLONE}"
  echo "# Commit: $(cd ${CLONE} && git log -1 --oneline)"
  echo "# Submitted: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "#"
  echo "# combo                                                  | jobid"
  echo "# -------------------------------------------------------|---------"
} > "${TRACK}"

submit_one() {
  local backend="$1"
  local variant="$2"
  local label="${backend}_${variant:-none}"
  local outdir="${CLONE}/examples/hstu/training/benchmark/results/round2/${label}"
  local jobname="mtms_r2_${label}"

  echo ""
  echo "=================================================================="
  echo "[combo] backend=${backend} variant='${variant}' label=${label}"
  echo "[outdir] ${outdir}"
  echo "=================================================================="

  local part_arg=()
  if [ -n "${PARTITION}" ]; then
    part_arg+=(--partition="${PARTITION}")
  fi

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
      --hstu-root="${CLONE}/examples/hstu" \
      -y 2>&1 | tee "/tmp/r2_${label}.log"

  # Pull jobid out of the submit log (submit_all prints "✅ Submitted: Job ID 12345")
  local jobid
  jobid=$(grep -oE "Job ID [0-9]+" "/tmp/r2_${label}.log" | head -1 | awk '{print $3}' || true)
  if [ -z "${jobid}" ]; then
    jobid="UNKNOWN"
  fi
  printf "%-55s | %s\n" "${label}" "${jobid}" >> "${TRACK}"
}

submit_one "legacy" ""
submit_one "new" "default"
submit_one "new" "by_stream"
submit_one "new" "per_task"
submit_one "new" "io_prefetch_compute"
submit_one "new" "io_data_dist_compute"
submit_one "new" "io_data_dist_prefetch_compute"

echo ""
echo "================================ tracking ================================"
cat "${TRACK}"
