#!/bin/bash
# ============================================================================
# Batch Run All Experiments (Single Node)
# Usage: ./run_all_experiments.sh [options]
# 
# Options:
#   --nproc=N            Number of processes/GPUs (default: 8)
#   --experiments=LIST   Comma-separated list of experiments to run (default: all)
#   --help               Show help information
# 
# Notes:
#   - Runs exp0 to exp8 sequentially
#   - 10 seconds interval between each experiment
#   - All logs are saved to results/ directory
# 
# Examples:
#   ./run_all_experiments.sh                           # Run all experiments with default settings
#   ./run_all_experiments.sh --nproc=4
#   ./run_all_experiments.sh --experiments=exp0_baseline,exp8_full
# ============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR="${SCRIPT_DIR}/results"
mkdir -p ${OUTPUT_DIR}

# Default arguments
NPROC=8
EXPERIMENTS=""

# List of all experiments
ALL_EXPERIMENTS=(
    "exp0_baseline"
    "exp1_cutlass"
    "exp2_fusion"
    "exp3_recompute"
    "exp4_dynamicemb"
    "exp5_lfu"
    "exp6_pipeline"
    "exp7_tp"
    "exp8_full"
)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nproc=*)
            NPROC="${1#*=}"
            shift
            ;;
        --experiments=*)
            EXPERIMENTS="${1#*=}"
            shift
            ;;
        --help|-h)
            head -20 "$0" | tail -17
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Determine experiments to run
if [ -n "$EXPERIMENTS" ]; then
    IFS=',' read -ra experiments <<< "$EXPERIMENTS"
else
    experiments=("${ALL_EXPERIMENTS[@]}")
fi

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "🚀 HSTU Benchmark Suite (Single Node)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - GPUs/Processes:   ${NPROC}"
echo ""
echo "Total experiments: ${#experiments[@]}"
echo "Output directory:  ${OUTPUT_DIR}"
echo ""

# Record start time
START_TIME=$(date +%s)

# Run each experiment
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_EXPS=()

for i in "${!experiments[@]}"; do
    exp="${experiments[$i]}"
    exp_num=$((i + 1))
    
    echo ""
    echo "=========================================="
    echo -e "${YELLOW}[${exp_num}/${#experiments[@]}] Running ${exp}...${NC}"
    echo "=========================================="
    
    EXP_START=$(date +%s)
    
    # Run experiment
    if ${SCRIPT_DIR}/run_single_experiment.sh ${exp} --nproc=${NPROC}; then
        EXP_END=$(date +%s)
        EXP_DURATION=$((EXP_END - EXP_START))
        echo ""
        echo -e "${GREEN}✅ ${exp} completed successfully in ${EXP_DURATION}s${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        EXP_END=$(date +%s)
        EXP_DURATION=$((EXP_END - EXP_START))
        echo ""
        echo -e "${RED}❌ ${exp} failed after ${EXP_DURATION}s${NC}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_EXPS+=("${exp}")
        
        # Ask whether to continue
        echo ""
        echo "Do you want to continue with the next experiment? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Benchmark suite interrupted by user."
            break
        fi
    fi
    
    # Wait interval (except after last experiment)
    if [ $i -lt $((${#experiments[@]} - 1)) ]; then
        echo ""
        echo "⏱️  Waiting 10 seconds before next experiment..."
        sleep 10
    fi
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

# Print summary
echo ""
echo "=========================================="
echo "📊 Benchmark Suite Summary"
echo "=========================================="
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo -e "${GREEN}✅ Successful: ${SUCCESS_COUNT}${NC}"
echo -e "${RED}❌ Failed:     ${FAILED_COUNT}${NC}"

if [ ${FAILED_COUNT} -gt 0 ]; then
    echo ""
    echo "Failed experiments:"
    for exp in "${FAILED_EXPS[@]}"; do
        echo "  - ${exp}"
    done
fi

echo ""
echo "Results saved in: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Run visualize_results.py to generate plots"
echo "  2. Check individual logs in ${OUTPUT_DIR}"
echo "  3. Compare metrics across experiments"
echo ""

if [ ${FAILED_COUNT} -eq 0 ]; then
    echo -e "${GREEN}🎉 All experiments completed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some experiments failed. Please check the logs.${NC}"
    exit 1
fi
