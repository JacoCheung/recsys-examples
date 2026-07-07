#!/bin/bash

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_HSTU_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
TARGET_HSTU_DIR=/workspace/recsys-examples/examples/hstu
OUTPUT_DIR=${PROFILE_OUTPUT_DIR:?Set PROFILE_OUTPUT_DIR to a writable scratch path}
BENCH_ITERS=${BENCH_ITERS:-20}
PROFILER_START_ITER=${PROFILER_START_ITER:-5}
PROFILER_STOP_ITER=${PROFILER_STOP_ITER:-14}
PROJECT_PYTHONPATH=/workspace/recsys-examples/examples:/workspace/recsys-examples/examples/hstu
PROFILE_BACKENDS=${PROFILE_BACKENDS:-"cutlass triton"}
PROFILE_PHASES=${PROFILE_PHASES:-"fwd bwd"}
PROFILE_BATCH_SIZES=${PROFILE_BATCH_SIZES:-"16 256"}
PROFILE_SEQLENS=${PROFILE_SEQLENS:-"512 1024"}

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/benchmark_outputs" "${OUTPUT_DIR}/reports"
printf 'name\texit_code\treport\tlog\n' > "${OUTPUT_DIR}/manifest.tsv"

overall_status=0
cp "${SCRIPT_DIR}/hstu_attn_kernel_benchmark.py" \
    "${TARGET_HSTU_DIR}/training/benchmark/scripts/hstu_attn_kernel_benchmark.py"
cd "${TARGET_HSTU_DIR}"

for backend in ${PROFILE_BACKENDS}; do
    config="${SOURCE_HSTU_DIR}/training/configs/hstu_attn_profile_${backend}.gin"
    for phase in ${PROFILE_PHASES}; do
        for batch_size in ${PROFILE_BATCH_SIZES}; do
            for seqlen in ${PROFILE_SEQLENS}; do
                name="${backend}_${phase}_bs${batch_size}_sl${seqlen}_h4_d128"
                report="${OUTPUT_DIR}/reports/${name}"
                log="${OUTPUT_DIR}/logs/${name}.log"
                benchmark_output="${OUTPUT_DIR}/benchmark_outputs/${name}"

                echo "[$(date --iso-8601=seconds)] Starting ${name}" | tee "${log}"
                CUDA_VISIBLE_DEVICES=0 nsys profile \
                    --force-overwrite=true \
                    --sample=none \
                    --cpuctxsw=none \
                    --trace=cuda,nvtx \
                    --capture-range=cudaProfilerApi \
                    --capture-range-end=stop \
                    --cuda-graph-trace=node \
                    --env-var="PYTHONPATH=${PROJECT_PYTHONPATH}" \
                    --output="${report}" \
                    python -u training/benchmark/scripts/hstu_attn_kernel_benchmark.py \
                        --gin-config-file "${config}" \
                        --batch-sizes "${batch_size}" \
                        --seqlens "${seqlen}" \
                        --phase "${phase}" \
                        --warmup-iters 10 \
                        --bench-iters "${BENCH_ITERS}" \
                        --profiler-start-iter "${PROFILER_START_ITER}" \
                        --profiler-stop-iter "${PROFILER_STOP_ITER}" \
                        --cuda-graph \
                        --output-dir "${benchmark_output}" \
                    >> "${log}" 2>&1
                status=$?

                if [[ ${status} -ne 0 ]]; then
                    overall_status=1
                fi
                printf '%s\t%d\t%s.nsys-rep\t%s\n' \
                    "${name}" "${status}" "${report}" "${log}" \
                    >> "${OUTPUT_DIR}/manifest.tsv"
                echo "[$(date --iso-8601=seconds)] Finished ${name}: exit ${status}" \
                    | tee -a "${log}"
            done
        done
    done
done

exit "${overall_status}"
