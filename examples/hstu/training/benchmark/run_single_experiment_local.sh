#!/bin/bash
# ============================================================================
# Single Experiment Runner (Single Node)
# 
# Usage: ./training/benchmark/run_single_experiment_local.sh <exp_name> --config=<config_file> [options]
# 
# Environment Variables:
#   HSTU_ROOT         Path to examples/hstu directory (optional, defaults to pwd)
# 
# Options:
#   --config=PATH     Config file path (relative to examples/hstu or absolute)
#   --hstu-root=PATH  Specify examples/hstu directory path (overrides env var and pwd)
#   --nproc=N         Number of processes/GPUs (default: 8)
#   --nsys            Enable nsys profile sampling (traces all child processes/ranks)
#   --output-dir=PATH Output directory (default: results/{timestamp}/{exp_name}/)
#   --dry-run         Print commands only, do not execute
# 
# Output Directory Structure:
#   {output_dir}/
#   ├── {exp_name}_{timestamp}.log
#   └── {exp_name}_{timestamp}_{hostname}.nsys-rep  (if --nsys enabled)
# 
# Examples:
#   ./training/benchmark/run_single_experiment_local.sh exp0_baseline --config=training/benchmark/gin_configs/benchmark_exp0_baseline.gin
#   ./training/benchmark/run_single_experiment_local.sh exp0_baseline --config=training/benchmark/gin_configs/benchmark_exp0_baseline.gin --nproc=8
#   ./training/benchmark/run_single_experiment_local.sh exp0_baseline --config=training/benchmark/gin_configs/benchmark_exp0_baseline.gin --nsys
#   ./training/benchmark/run_single_experiment_local.sh --hstu-root=/path/to/examples/hstu exp0_baseline --config=training/benchmark/gin_configs/benchmark_exp0_baseline.gin
# ============================================================================

set -e

# Default values
NPROC=${NPROC:-8}
CONFIG_FILE=""
ENABLE_NSYS=0
CUSTOM_OUTPUT_DIR=""
DRY_RUN=0
CUSTOM_HSTU_ROOT=""

# Parse arguments
EXP_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config=*)
            CONFIG_FILE="${1#*=}"
            shift
            ;;
        --hstu-root=*)
            CUSTOM_HSTU_ROOT="${1#*=}"
            shift
            ;;
        --nproc=*)
            NPROC="${1#*=}"
            shift
            ;;
        --nsys)
            ENABLE_NSYS=1
            shift
            ;;
        --output-dir=*)
            CUSTOM_OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            head -29 "$0" | tail -26
            exit 0
            ;;
        -*)
            echo "❌ Error: Unknown option: $1"
            exit 1
            ;;
        *)
            if [ -z "$EXP_NAME" ]; then
                EXP_NAME="$1"
            fi
            shift
            ;;
    esac
done

# Argument validation
if [ -z "$EXP_NAME" ]; then
    echo "❌ Error: Missing experiment name"
    echo "Usage: $0 <exp_name> --config=<config_file> [options]"
    echo ""
    echo "Options:"
    echo "  --config=PATH     Config file path (required)"
    echo "  --hstu-root=PATH  Specify examples/hstu directory path"
    echo "  --nproc=N         Number of processes/GPUs (default: 8)"
    echo "  --nsys            Enable nsys profile sampling (traces all processes)"
    echo "  --output-dir=PATH Output directory (default: results/{timestamp}/{exp_name}/)"
    echo "  --dry-run         Print commands only, do not execute"
    exit 1
fi

if [ -z "$CONFIG_FILE" ]; then
    echo "❌ Error: Missing config file (--config=<path>)"
    exit 1
fi

# ============================================================================
# Set HSTU_ROOT (Priority: command line arg > env var > pwd)
# ============================================================================
if [ -n "$CUSTOM_HSTU_ROOT" ]; then
    HSTU_ROOT="$CUSTOM_HSTU_ROOT"
elif [ -z "$HSTU_ROOT" ]; then
    HSTU_ROOT=$(pwd)
fi

# Verify HSTU_ROOT directory exists
if [ ! -d "$HSTU_ROOT" ]; then
    echo "❌ Error: HSTU_ROOT directory does not exist: $HSTU_ROOT"
    exit 1
fi

# Verify directory structure
if [ ! -d "$HSTU_ROOT/training" ]; then
    echo "❌ Error: Invalid HSTU_ROOT - missing 'training' subdirectory"
    echo "  HSTU_ROOT: $HSTU_ROOT"
    echo ""
    echo "Please ensure HSTU_ROOT points to 'recsys-examples/examples/hstu'"
    exit 1
fi

# Path configuration
PROJECT_ROOT="${HSTU_ROOT}/../.."
SCRIPT_DIR="${HSTU_ROOT}/training/benchmark"
RESULTS_BASE="${SCRIPT_DIR}/results"

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Determine output directory
if [ -n "$CUSTOM_OUTPUT_DIR" ]; then
    # Use custom output directory
    if [[ ! "$CUSTOM_OUTPUT_DIR" = /* ]]; then
        # Relative path, relative to examples/hstu
        OUTPUT_DIR="${HSTU_ROOT}/${CUSTOM_OUTPUT_DIR}"
    else
        OUTPUT_DIR="${CUSTOM_OUTPUT_DIR}"
    fi
else
    # Default: results/{timestamp}/{exp_name}/
    OUTPUT_DIR="${RESULTS_BASE}/${TIMESTAMP}/${EXP_NAME}"
fi

# Only create directory in non-dry-run mode
if [ ${DRY_RUN} -eq 0 ]; then
    mkdir -p ${OUTPUT_DIR}
fi

# If config file is a relative path, make it relative to examples/hstu
if [[ ! "$CONFIG_FILE" = /* ]]; then
    CONFIG_FILE="${HSTU_ROOT}/${CONFIG_FILE}"
fi

# Check config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Color output
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "=========================================="
echo "🚀 Running Experiment: ${EXP_NAME}"
echo "=========================================="
echo "Config file: ${CONFIG_FILE}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "GPUs:        ${NPROC}"
echo ""
echo "NSYS Profiling: $([ ${ENABLE_NSYS} -eq 1 ] && echo 'ENABLED (all processes)' || echo 'DISABLED')"
if [ ${DRY_RUN} -eq 1 ]; then
    echo -e "${YELLOW}⚠️  DRY RUN MODE - Commands will be printed but not executed${NC}"
fi
echo "=========================================="
echo ""

# Environment variable setup
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Log file
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}_${TIMESTAMP}.log"

# ============================================================================
# DRY RUN Mode: Print commands only, do not execute
# ============================================================================
if [ ${DRY_RUN} -eq 1 ]; then
    HOSTNAME_SHORT=$(hostname -s)
    NSYS_OUTPUT="${OUTPUT_DIR}/${EXP_NAME}_${TIMESTAMP}_${HOSTNAME_SHORT}"
    
    echo -e "${CYAN}Would execute:${NC}"
    echo ""
    
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        echo "nsys profile \\"
        echo "    -o \"${NSYS_OUTPUT}\" \\"
        echo "    -f true \\"
        echo "    -s none \\"
        echo "    -t cuda,nvtx \\"
        echo "    -c cudaProfilerApi \\"
        echo "    --cpuctxsw none \\"
        echo "    --cuda-flush-interval 100 \\"
        echo "    --capture-range-end=stop \\"
        echo "    --cuda-graph-trace=node \\"
        echo "    torchrun \\"
        echo "        --standalone \\"
        echo "        --nproc_per_node=${NPROC} \\"
        echo "        training/pretrain_gr_ranking.py \\"
        echo "        --gin-config-file ${CONFIG_FILE} \\"
        echo "    2>&1 | tee ${LOG_FILE}"
    else
        echo "torchrun \\"
        echo "    --standalone \\"
        echo "    --nproc_per_node=${NPROC} \\"
        echo "    training/pretrain_gr_ranking.py \\"
        echo "    --gin-config-file ${CONFIG_FILE} \\"
        echo "    2>&1 | tee ${LOG_FILE}"
    fi
    
    echo ""
    echo -e "${YELLOW}DRY RUN completed. No commands were executed.${NC}"
    exit 0
fi

# ============================================================================
# Actual Execution Mode
# ============================================================================

# Start training
echo "📝 Logging to: ${LOG_FILE}"
echo "⏰ Started at: $(date)"
echo ""

# Already in examples/hstu directory, no need to cd

if [ ${ENABLE_NSYS} -eq 1 ]; then
    # ========================================================================
    # nsys profile mode enabled
    # Use nsys to wrap the entire torchrun command, nsys will automatically trace all child processes
    # Output file format: {exp_name}_{timestamp}_{hostname}
    # ========================================================================
    echo "🔬 Running with NVIDIA Nsight Systems profiling..."
    echo ""
    
    HOSTNAME_SHORT=$(hostname -s)
    NSYS_OUTPUT="${OUTPUT_DIR}/${EXP_NAME}_${TIMESTAMP}_${HOSTNAME_SHORT}"
    
    echo "📊 nsys output: ${NSYS_OUTPUT}.nsys-rep"
    echo ""
    
    # Use nsys to wrap torchrun, nsys will trace all child processes
    # Parameters consistent with slurm_job.sub
    nsys profile \
        -o "${NSYS_OUTPUT}" \
        -f true \
        -s none \
        -t cuda,nvtx \
        -c cudaProfilerApi \
        --cpuctxsw none \
        --cuda-flush-interval 100 \
        --capture-range-end=stop \
        --cuda-graph-trace=node \
        torchrun \
            --standalone \
            --nproc_per_node=${NPROC} \
            training/pretrain_gr_ranking.py \
            --gin-config-file ${CONFIG_FILE} \
        2>&1 | tee ${LOG_FILE}
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    echo ""
    echo "📊 nsys profile saved to: ${NSYS_OUTPUT}.nsys-rep"
    
else
    # ========================================================================
    # Standard training mode (no nsys profile)
    # ========================================================================
    torchrun \
        --standalone \
        --nproc_per_node=${NPROC} \
        training/pretrain_gr_ranking.py \
        --gin-config-file ${CONFIG_FILE} \
        2>&1 | tee ${LOG_FILE}
    
    EXIT_CODE=${PIPESTATUS[0]}
fi

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Experiment ${EXP_NAME} completed successfully!"
else
    echo "❌ Experiment ${EXP_NAME} failed with exit code: ${EXIT_CODE}"
fi
echo "⏰ Finished at: $(date)"
echo "📝 Log saved to: ${LOG_FILE}"
if [ ${ENABLE_NSYS} -eq 1 ]; then
    echo "📊 nsys profile: ${NSYS_OUTPUT}.nsys-rep"
fi
echo "=========================================="

exit $EXIT_CODE
