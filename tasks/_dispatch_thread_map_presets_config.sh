#!/bin/bash
# Submit the 9-variant HSTU thread-map preset matrix using JSON pipeline configs.
set -euo pipefail

ACCOUNT="${ACCOUNT:-coreai_devtech_all}"
CONTAINER="${CONTAINER:-gitlab-master.nvidia.com/devtech-compute/distributed-recommender:devel_latest}"
NODES="${NODES:-2}"
RANKS_PER_NODE="${RANKS_PER_NODE:-8}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
PARTITION="${PARTITION:-batch_short}"
JOBNAME_PREFIX="${JOBNAME_PREFIX:-tmcfg}"
CLONE="${CLONE:-/lustre/fsw/portfolios/coreai/users/junzhang/workspace/recsys-rework-mtms}"
GIT_REMOTE="${GIT_REMOTE:-forked}"
BRANCH="${BRANCH:-junzhang/rework-mtms}"
RESULTS_BASE="${RESULTS_BASE:-/lustre/fsw/portfolios/coreai/users/junzhang/benchmark_runs}"
BATCH_NAME="${BATCH_NAME:-thread_map_presets_config_2n_cwdfw_$(date -u +%Y%m%d_%H%M%S)}"
SCP_DEST="${SCP_DEST:-vnc:/home/recsys-example-dashboard/benchmark/}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
DRY_RUN="${DRY_RUN:-0}"

echo "Syncing clone from git branch..."
git -C "${CLONE}" fetch "${GIT_REMOTE}" "${BRANCH}:refs/remotes/${GIT_REMOTE}/${BRANCH}"
git -C "${CLONE}" switch --recurse-submodules -C "${BRANCH}" "${GIT_REMOTE}/${BRANCH}"
git -C "${CLONE}" submodule update --init --recursive
if [[ -n "$(git -C "${CLONE}" status --porcelain)" ]]; then
    echo "ERROR: ${CLONE} is dirty after checking out ${GIT_REMOTE}/${BRANCH}" >&2
    git -C "${CLONE}" status --short >&2
    exit 1
fi

echo "Batch:       ${BATCH_NAME}"
echo "Clone:       ${CLONE}"
echo "Branch:      ${GIT_REMOTE}/${BRANCH}"
echo "Commit:      $(git -C "${CLONE}" rev-parse --short HEAD)"
CONFIG_DIR="${CLONE}/examples/hstu/training/benchmark/pipeline_configs/thread_map_presets"
EXP_FILE="${CLONE}/examples/hstu/training/benchmark/experiments_thread_map_presets_config.txt"
echo "Config dir:  ${CONFIG_DIR}"
echo "Results:     ${RESULTS_BASE}/${BATCH_NAME}"
echo "Experiment:  ${EXP_FILE}"
echo "SCP archive: ${SCP_DEST}"
echo ""
cat "${EXP_FILE}"
echo ""

cd "${CLONE}/examples/hstu"
extra_args=()
if [[ "${DRY_RUN}" = "1" ]]; then
    extra_args+=(--dry-run)
fi

HSTU_BENCHMARK_BATCH_NAME="${BATCH_NAME}" \
    bash training/benchmark/scripts/submit_all_experiments_slurm.sh \
        --account="${ACCOUNT}" \
        --job-name="${JOBNAME_PREFIX}" \
        --container-image="${CONTAINER}" \
        --nodes="${NODES}" \
        --ranks-per-node="${RANKS_PER_NODE}" \
        --time="${TIME_LIMIT}" \
        --partition="${PARTITION}" \
        --nsys \
        --wait-and-analyze \
        --poll-interval="${POLL_INTERVAL}" \
        --scp-dest="${SCP_DEST}" \
        --results-dir="${RESULTS_BASE}" \
        --exp-file="${EXP_FILE}" \
        --hstu-root="${CLONE}/examples/hstu" \
        --branch="${BRANCH}" \
        "${extra_args[@]}" \
        -y

echo ""
echo "Submitted ${BATCH_NAME}."
echo "Monitor log:"
echo "  ${RESULTS_BASE}/${BATCH_NAME}/monitor.log"
