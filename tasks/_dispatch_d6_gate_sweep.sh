#!/bin/bash
# Submit a 2-node cw-dfw sweep for HSTU full_split d6 non-critical gates.
#
# Each row is one dashboard experiment. The third experiment-file column
# carries per-experiment env overrides; gate entries use "^" separators
# because Slurm --export uses commas.
set -euo pipefail

ACCOUNT="${ACCOUNT:-coreai_devtech_all}"
CONTAINER="${CONTAINER:-gitlab-master.nvidia.com/devtech-compute/distributed-recommender:devel_latest}"
NODES="${NODES:-2}"
RANKS_PER_NODE="${RANKS_PER_NODE:-8}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
PARTITION="${PARTITION:-batch_short}"
JOBNAME_PREFIX="${JOBNAME_PREFIX:-d6gate}"
CLONE="${CLONE:-/lustre/fsw/portfolios/coreai/users/junzhang/workspace/recsys-rework-mtms}"
RESULTS_BASE="${RESULTS_BASE:-/lustre/fsw/portfolios/coreai/users/junzhang/benchmark_runs}"
BATCH_NAME="${BATCH_NAME:-d6_gate_sweep_2n_cwdfw_$(date -u +%Y%m%d_%H%M%S)}"
SCP_DEST="${SCP_DEST:-vnc:/home/recsys-example-dashboard/benchmark/}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
DRY_RUN="${DRY_RUN:-0}"

COMMON_GIN="--balanced_shuffler --kernel_backend cutlass --recompute_layernorm --caching --evict lfu --ratio 0.1 --pipeline_type prefetch --value_dist zipf --value_dist_alpha 1.05"

EXP_FILE="/tmp/${BATCH_NAME}_experiments.txt"
cat > "${EXP_FILE}" <<EOF
# name,gin options,env overrides
all_gtok,${COMMON_GIN},HSTU_NONCRITICAL_GATE_DEFAULT=global_tokens_allreduce
all_outdist,${COMMON_GIN},HSTU_NONCRITICAL_GATE_DEFAULT=compute_output_dist
all_rankemb,${COMMON_GIN},HSTU_NONCRITICAL_GATE_DEFAULT=ranking_embedding_forward
all_forward,${COMMON_GIN},HSTU_NONCRITICAL_GATE_DEFAULT=forward
io_gtok,${COMMON_GIN},HSTU_NONCRITICAL_GATE_DEFAULT=forward;HSTU_NONCRITICAL_GATES=h2d=global_tokens_allreduce^start_shuffle=global_tokens_allreduce^finish_shuffle=global_tokens_allreduce
io_outdist,${COMMON_GIN},HSTU_NONCRITICAL_GATE_DEFAULT=forward;HSTU_NONCRITICAL_GATES=h2d=compute_output_dist^start_shuffle=compute_output_dist^finish_shuffle=compute_output_dist
io_rankemb,${COMMON_GIN},HSTU_NONCRITICAL_GATE_DEFAULT=forward;HSTU_NONCRITICAL_GATES=h2d=ranking_embedding_forward^start_shuffle=ranking_embedding_forward^finish_shuffle=ranking_embedding_forward
dist_outdist,${COMMON_GIN},HSTU_NONCRITICAL_GATE_DEFAULT=forward;HSTU_NONCRITICAL_GATES=start_input_dist=compute_output_dist^wait_input_dist=compute_output_dist
dist_rankemb,${COMMON_GIN},HSTU_NONCRITICAL_GATE_DEFAULT=forward;HSTU_NONCRITICAL_GATES=start_input_dist=ranking_embedding_forward^wait_input_dist=ranking_embedding_forward
pref_outdist,${COMMON_GIN},HSTU_NONCRITICAL_GATE_DEFAULT=forward;HSTU_NONCRITICAL_GATES=prefetch_embeddings=compute_output_dist
pref_rankemb,${COMMON_GIN},HSTU_NONCRITICAL_GATE_DEFAULT=forward;HSTU_NONCRITICAL_GATES=prefetch_embeddings=ranking_embedding_forward
stagger_early_late,${COMMON_GIN},HSTU_NONCRITICAL_GATE_DEFAULT=forward;HSTU_NONCRITICAL_GATES=h2d=global_tokens_allreduce^start_shuffle=global_tokens_allreduce^finish_shuffle=compute_output_dist^start_input_dist=compute_output_dist^wait_input_dist=ranking_embedding_forward
stagger_mid_late,${COMMON_GIN},HSTU_NONCRITICAL_GATE_DEFAULT=forward;HSTU_NONCRITICAL_GATES=h2d=compute_output_dist^start_shuffle=compute_output_dist^finish_shuffle=ranking_embedding_forward^start_input_dist=ranking_embedding_forward
EOF

echo "Batch:       ${BATCH_NAME}"
echo "Clone:       ${CLONE}"
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
RECSYS_PIPELINE_BACKEND=new \
HSTU_THREAD_MAP_VARIANT=io_data_dist_prefetch_compute \
HSTU_LA_DEPTH=6 \
HSTU_SPLIT_RANKING_FORWARD=1 \
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
        "${extra_args[@]}" \
        -y

echo ""
echo "Submitted ${BATCH_NAME}."
echo "Monitor log:"
echo "  ${RESULTS_BASE}/${BATCH_NAME}/monitor.log"
