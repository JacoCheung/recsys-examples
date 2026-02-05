#!/bin/bash
# ============================================================================
# Batch Submit All Experiments via SLURM sbatch
# 
# Usage: 
#   ./training/benchmark/submit_all_experiments_slurm.sh --exp-file=<file> [options]
# 
# Environment Variables:
#   HSTU_ROOT            Path to examples/hstu directory (optional, defaults to pwd)
# 
# Options:
#   --exp-file=FILE      Experiment list file (required, format: exp_name,gin_options)
#   --hstu-root=PATH     Specify examples/hstu directory path (overrides env var and pwd)
#   --results-dir=PATH   Output directory (default: training/benchmark/results)
#   --nsys               Enable nsys profile sampling
#   --sequential         Sequential execution (use dependencies, start next after previous completes)
#   --partition=NAME     SLURM partition name (default: batch)
#   --account=NAME       SLURM account name (optional, passed to sbatch -A)
#   --job-name=NAME      SLURM job name prefix (optional, passed to sbatch -J)
#   --container-image=IMAGE  Container image (default: gitlab-master.nvidia.com/devtech-compute/distributed-recommender:devel_latest)
#   --nodes=N            Number of nodes (default: 2)
#   --ranks-per-node=N   Number of ranks/processes per node (default: 8)
#   --time=HH:MM:SS      Job time limit (default: 04:00:00)
#   --dry-run            Print sbatch commands only, do not submit
#   --help               Show help information
# 
# Experiment List File Format:
#   # Comment lines start with #
#   exp_name,generate_gin_config_options
#   exp0_baseline,
#   exp1_cutlass,--kernel_backend cutlass
#   exp4_caching,--kernel_backend cutlass --recompute_layernorm --balanced_shuffler --caching
# 
# Output Directory Structure:
#   {results_dir}/
#   └── {batch_timestamp}/           # Timestamp of this batch submission
#       ├── exp0_baseline/           # First experiment
#       │   ├── exp0_baseline_*.log
#       │   ├── exp0_baseline_*.gin  # Generated config
#       │   ├── {job_name}_*.out     # SLURM stdout/stderr
#       │   └── exp0_baseline_*.nsys-rep  (if nsys enabled)
#       ├── exp1_cutlass/            # Second experiment
#       │   ├── ...
#       └── summary.txt              # Batch experiment summary
# 
# Examples:
#   # Run in examples/hstu directory
#   ./training/benchmark/submit_all_experiments_slurm.sh --exp-file=training/benchmark/experiments.txt
#   
#   # Use environment variable to specify HSTU_ROOT
#   export HSTU_ROOT=/path/to/recsys-examples/examples/hstu
#   ./submit_all_experiments_slurm.sh --exp-file=training/benchmark/experiments.txt
#   
#   # Use command line argument to specify HSTU_ROOT
#   ./submit_all_experiments_slurm.sh --hstu-root=/path/to/examples/hstu --exp-file=training/benchmark/experiments.txt
#   
#   # Other options
#   ./training/benchmark/submit_all_experiments_slurm.sh --exp-file=training/benchmark/experiments.txt --nsys
#   ./training/benchmark/submit_all_experiments_slurm.sh --exp-file=training/benchmark/experiments.txt --results-dir=/data/benchmark_results
# ============================================================================

set -e

# ============================================================================
# Default Parameters
# ============================================================================
ENABLE_NSYS=0
SEQUENTIAL=0
PARTITION="batch"
ACCOUNT=""
JOB_PREFIX=""
CONTAINER_IMAGE="gitlab-master.nvidia.com/devtech-compute/distributed-recommender:devel_latest"
NODES=2
RANKS_PER_NODE=8
TIME_LIMIT="04:00:00"
DRY_RUN=0
EXP_FILE=""
CUSTOM_RESULTS_DIR=""
CUSTOM_HSTU_ROOT=""

# ============================================================================
# Help Information
# ============================================================================
show_help() {
    head -58 "$0" | tail -57
    exit 0
}

# ============================================================================
# Parse Command Line Arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp-file=*)
            EXP_FILE="${1#*=}"
            shift
            ;;
        --hstu-root=*)
            CUSTOM_HSTU_ROOT="${1#*=}"
            shift
            ;;
        --results-dir=*)
            CUSTOM_RESULTS_DIR="${1#*=}"
            shift
            ;;
        --nsys)
            ENABLE_NSYS=1
            shift
            ;;
        --sequential)
            SEQUENTIAL=1
            shift
            ;;
        --partition=*)
            PARTITION="${1#*=}"
            shift
            ;;
        --account=*|-A=*)
            ACCOUNT="${1#*=}"
            shift
            ;;
        --job-name=*|-J=*)
            JOB_PREFIX="${1#*=}"
            shift
            ;;
        --container-image=*)
            CONTAINER_IMAGE="${1#*=}"
            shift
            ;;
        --nodes=*)
            NODES="${1#*=}"
            shift
            ;;
        --ranks-per-node=*)
            RANKS_PER_NODE="${1#*=}"
            shift
            ;;
        --time=*)
            TIME_LIMIT="${1#*=}"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Set HSTU_ROOT (Priority: command line arg > env var > pwd)
# ============================================================================
if [ -n "$CUSTOM_HSTU_ROOT" ]; then
    # Command line argument has highest priority
    HSTU_ROOT="$CUSTOM_HSTU_ROOT"
elif [ -z "$HSTU_ROOT" ]; then
    # If env var not set, use pwd
    HSTU_ROOT=$(pwd)
fi
# If env var is set, use it directly (no additional action needed)

# Verify HSTU_ROOT directory exists (skip in dry-run mode)
if [ ${DRY_RUN} -eq 0 ]; then
    if [ ! -d "$HSTU_ROOT" ]; then
        echo "❌ Error: HSTU_ROOT directory does not exist: $HSTU_ROOT"
        exit 1
    fi

    # Verify directory structure (check for training subdirectory)
    if [ ! -d "$HSTU_ROOT/training" ]; then
        echo "❌ Error: Invalid HSTU_ROOT - missing 'training' subdirectory"
        echo "  HSTU_ROOT: $HSTU_ROOT"
        echo ""
        echo "Please ensure HSTU_ROOT points to 'recsys-examples/examples/hstu'"
        exit 1
    fi
fi

# Path configuration
PROJECT_ROOT="${HSTU_ROOT}/../.."
SCRIPT_DIR="${HSTU_ROOT}/training/benchmark"

# ============================================================================
# Set Output Directory
# ============================================================================
if [ -n "$CUSTOM_RESULTS_DIR" ]; then
    # If relative path, make it relative to examples/hstu
    if [[ ! "$CUSTOM_RESULTS_DIR" = /* ]]; then
        RESULTS_BASE="${HSTU_ROOT}/${CUSTOM_RESULTS_DIR}"
    else
        RESULTS_BASE="${CUSTOM_RESULTS_DIR}"
    fi
else
    # Default directory
    RESULTS_BASE="${SCRIPT_DIR}/results"
fi

# Create timestamped batch experiment directory
BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_OUTPUT_DIR="${RESULTS_BASE}/${BATCH_TIMESTAMP}"

# ============================================================================
# Check Experiment List File
# ============================================================================
# If --exp-file is not provided, show help
if [ -z "$EXP_FILE" ]; then
    echo "⚠️  Missing experiment list file (--exp-file=<file>)"
    echo ""
    head -60 "$0" | tail -58
    exit 0
fi

# Read experiment list if provided
declare -a EXP_NAMES
declare -a GIN_OPTIONS

if [ -n "$EXP_FILE" ]; then
    # If relative path, make it relative to examples/hstu
    if [[ ! "$EXP_FILE" = /* ]]; then
        EXP_FILE="${HSTU_ROOT}/${EXP_FILE}"
    fi

    if [ ! -f "$EXP_FILE" ]; then
        if [ ${DRY_RUN} -eq 1 ]; then
            echo "⚠️  Experiment list file not found: $EXP_FILE"
            echo "   No experiments to run."
            exit 0
        else
            echo "❌ Error: Experiment list file not found: $EXP_FILE"
            exit 1
        fi
    fi

    # Read experiment list (skip comments and empty lines)
    while IFS=',' read -r exp_name gin_opts || [ -n "$exp_name" ]; do
        # Skip empty lines and comments
        [[ -z "$exp_name" || "$exp_name" =~ ^[[:space:]]*# ]] && continue
        # Trim leading/trailing whitespace
        exp_name=$(echo "$exp_name" | xargs)
        gin_opts=$(echo "$gin_opts" | xargs)
        EXP_NAMES+=("$exp_name")
        GIN_OPTIONS+=("$gin_opts")
    done < "$EXP_FILE"

    if [ ${#EXP_NAMES[@]} -eq 0 ]; then
        if [ ${DRY_RUN} -eq 1 ]; then
            echo "⚠️  No experiments found in $EXP_FILE"
            exit 0
        else
            echo "❌ Error: No experiments found in $EXP_FILE"
            exit 1
        fi
    fi
fi

# ============================================================================
# Color Output
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# Print Configuration Information
# ============================================================================
echo ""
echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}🚀 HSTU Benchmark - SLURM Submission${NC}"
echo -e "${CYAN}==========================================${NC}"
echo ""
echo -e "${BLUE}SLURM Configuration:${NC}"
echo "  Partition:        ${PARTITION}"
[ -n "$ACCOUNT" ] && echo "  Account:          ${ACCOUNT}"
[ -n "$JOB_PREFIX" ] && echo "  Job prefix:       ${JOB_PREFIX}"
echo "  Container:        ${CONTAINER_IMAGE}"
echo "  Nodes:            ${NODES}"
echo "  Ranks per node:   ${RANKS_PER_NODE}"
echo "  Total ranks:      $((NODES * RANKS_PER_NODE))"
echo "  Time limit:       ${TIME_LIMIT}"
echo "  Sequential mode:  $([ ${SEQUENTIAL} -eq 1 ] && echo 'YES' || echo 'NO')"
echo ""
echo -e "${BLUE}NSYS Profiling:${NC}"
echo "  Enabled:          $([ ${ENABLE_NSYS} -eq 1 ] && echo -e '${GREEN}YES${NC}' || echo 'NO')"
echo ""

if [ ${DRY_RUN} -eq 1 ]; then
    echo -e "${YELLOW}⚠️  DRY RUN MODE - Commands will be printed but not executed${NC}"
    echo ""
fi

echo -e "${BLUE}Batch timestamp:   ${BATCH_TIMESTAMP}${NC}"
echo -e "${BLUE}Output directory:  ${BATCH_OUTPUT_DIR}${NC}"
echo ""
echo -e "${BLUE}Experiment file: ${EXP_FILE}${NC}"
echo ""
echo -e "${BLUE}Experiments to run (${#EXP_NAMES[@]} total):${NC}"
for i in "${!EXP_NAMES[@]}"; do
    echo "  - ${EXP_NAMES[$i]}: ${GIN_OPTIONS[$i]:-'(defaults)'}"
done
echo ""

# ============================================================================
# Confirm Submission
# ============================================================================
if [ ${DRY_RUN} -eq 0 ]; then
    echo -e "${YELLOW}Do you want to submit these jobs? (y/n)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    echo ""
    
    # Create batch output directory
    mkdir -p ${BATCH_OUTPUT_DIR}
fi

# ============================================================================
# Submit Jobs
# ============================================================================
SUBMITTED_JOBS=()
PREV_JOB_ID=""

echo -e "${BLUE}Submitting jobs...${NC}"
echo ""

for i in "${!EXP_NAMES[@]}"; do
    exp="${EXP_NAMES[$i]}"
    gin_opts="${GIN_OPTIONS[$i]}"
    exp_num=$((i + 1))
    
    # Output directory for each experiment
    EXP_OUTPUT_DIR="${BATCH_OUTPUT_DIR}/${exp}"
    
    if [ ${DRY_RUN} -eq 0 ]; then
        mkdir -p ${EXP_OUTPUT_DIR}
    fi
    
    # Build sbatch command
    # Determine job name (with optional prefix)
    if [ -n "$JOB_PREFIX" ]; then
        FULL_JOB_NAME="${JOB_PREFIX}-hstu.${exp}"
    else
        FULL_JOB_NAME="hstu_${exp}"
    fi
    
    SBATCH_CMD="sbatch"
    SBATCH_CMD+=" --job-name=${FULL_JOB_NAME}"
    SBATCH_CMD+=" --output=${EXP_OUTPUT_DIR}/${FULL_JOB_NAME}_%j.out"
    SBATCH_CMD+=" --partition=${PARTITION}"
    
    # Add account if specified
    if [ -n "$ACCOUNT" ]; then
        SBATCH_CMD+=" --account=${ACCOUNT}"
    fi
    
    SBATCH_CMD+=" --nodes=${NODES}"
    SBATCH_CMD+=" --ntasks-per-node=${RANKS_PER_NODE}"
    SBATCH_CMD+=" --cpus-per-task=8"
    SBATCH_CMD+=" --mem=0"
    SBATCH_CMD+=" --time=${TIME_LIMIT}"
    SBATCH_CMD+=" --exclusive"
    SBATCH_CMD+=" --network=sharp"
    
    # Sequential execution mode: add dependency
    if [ ${SEQUENTIAL} -eq 1 ] && [ -n "$PREV_JOB_ID" ]; then
        SBATCH_CMD+=" --dependency=afterany:${PREV_JOB_ID}"
    fi
    
    # Export environment variables (including exp_name, gin_options, output_dir, HSTU_ROOT and CONTAINER_IMAGE)
    # Use single quotes around GIN_OPTIONS to preserve spaces
    SBATCH_CMD+=" --export=ALL,EXP_NAME=${exp},GIN_OPTIONS='${gin_opts}',EXP_OUTPUT_DIR=${EXP_OUTPUT_DIR},ENABLE_NSYS=${ENABLE_NSYS},HSTU_ROOT=${HSTU_ROOT},CONTAINER_IMAGE=${CONTAINER_IMAGE}"
    
    # Specify SLURM job script
    SBATCH_CMD+=" ${SCRIPT_DIR}/slurm_job.sub"
    
    echo -e "[${exp_num}/${#EXP_NAMES[@]}] ${YELLOW}${exp}${NC}"
    echo "  Options:    ${gin_opts:-'(defaults)'}"
    echo "  Output dir: ${EXP_OUTPUT_DIR}"
    
    if [ ${DRY_RUN} -eq 1 ]; then
        echo "  Command: ${SBATCH_CMD}"
        echo ""
    else
        # Submit job and get job ID
        JOB_OUTPUT=$(${SBATCH_CMD})
        JOB_ID=$(echo ${JOB_OUTPUT} | grep -oP '\d+$')
        
        if [ -n "$JOB_ID" ]; then
            echo -e "  ${GREEN}✅ Submitted: Job ID ${JOB_ID}${NC}"
            SUBMITTED_JOBS+=("${exp}:${JOB_ID}")
            PREV_JOB_ID=${JOB_ID}
        else
            echo -e "  ${RED}❌ Failed to submit${NC}"
            echo "  Output: ${JOB_OUTPUT}"
        fi
        echo ""
    fi
done

# ============================================================================
# Create Summary File
# ============================================================================
if [ ${DRY_RUN} -eq 0 ]; then
    SUMMARY_FILE="${BATCH_OUTPUT_DIR}/summary.txt"
    {
        echo "================================================================================"
        echo "HSTU Benchmark Suite - SLURM Submission Summary"
        echo "================================================================================"
        echo ""
        echo "Batch Timestamp: ${BATCH_TIMESTAMP}"
        echo "Submitted at:    $(date)"
        echo ""
        echo "SLURM Configuration:"
        echo "  Partition:        ${PARTITION}"
        [ -n "$ACCOUNT" ] && echo "  Account:          ${ACCOUNT}"
        [ -n "$JOB_PREFIX" ] && echo "  Job prefix:       ${JOB_PREFIX}"
        echo "  Container:        ${CONTAINER_IMAGE}"
        echo "  Nodes:            ${NODES}"
        echo "  Ranks per node:   ${RANKS_PER_NODE}"
        echo "  Time limit:       ${TIME_LIMIT}"
        echo "  Sequential:       $([ ${SEQUENTIAL} -eq 1 ] && echo 'YES' || echo 'NO')"
        echo "  NSYS Profiling:   $([ ${ENABLE_NSYS} -eq 1 ] && echo 'YES' || echo 'NO')"
        echo ""
        echo "Experiment File: ${EXP_FILE}"
        echo ""
        echo "Submitted Jobs (${#SUBMITTED_JOBS[@]} total):"
        echo "--------------------------------------------------------------------------------"
        for job_info in "${SUBMITTED_JOBS[@]}"; do
            exp_name=$(echo ${job_info} | cut -d: -f1)
            job_id=$(echo ${job_info} | cut -d: -f2)
            echo "  ${exp_name}: Job ID ${job_id}"
            echo "    Output: ${BATCH_OUTPUT_DIR}/${exp_name}/"
        done
        echo "================================================================================"
    } > ${SUMMARY_FILE}
fi

# ============================================================================
# Summary Information
# ============================================================================
echo ""
echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}📊 Submission Summary${NC}"
echo -e "${CYAN}==========================================${NC}"
echo ""

if [ ${DRY_RUN} -eq 1 ]; then
    echo -e "${YELLOW}DRY RUN completed. No jobs were submitted.${NC}"
else
    echo -e "${GREEN}Submitted ${#SUBMITTED_JOBS[@]} jobs:${NC}"
    echo ""
    for job_info in "${SUBMITTED_JOBS[@]}"; do
        exp_name=$(echo ${job_info} | cut -d: -f1)
        job_id=$(echo ${job_info} | cut -d: -f2)
        echo "  ${exp_name}: Job ID ${job_id}"
    done
    echo ""
    
    echo "📁 Output directory: ${BATCH_OUTPUT_DIR}"
    echo ""
    echo "Directory structure:"
    echo "  ${BATCH_OUTPUT_DIR}/"
    for exp in "${EXP_NAMES[@]}"; do
        # Determine job name pattern for display
        if [ -n "$JOB_PREFIX" ]; then
            JOB_NAME_PATTERN="${JOB_PREFIX}-${exp}"
        else
            JOB_NAME_PATTERN="hstu_${exp}"
        fi
        echo "  ├── ${exp}/"
        echo "  │   ├── ${exp}_*.log"
        echo "  │   ├── ${exp}_*.gin"
        if [ ${ENABLE_NSYS} -eq 1 ]; then
            echo "  │   ├── ${JOB_NAME_PATTERN}_*.out"
            echo "  │   └── ${exp}_*.nsys-rep"
        else
            echo "  │   └── ${JOB_NAME_PATTERN}_*.out"
        fi
    done
    echo "  └── summary.txt"
    echo ""
    
    echo "📝 Summary saved to: ${SUMMARY_FILE}"
    echo ""
    
    echo -e "${BLUE}Useful commands:${NC}"
    echo "  squeue -u \$USER              # View job queue"
    echo "  scancel <job_id>              # Cancel single job"
    echo "  scancel -u \$USER             # Cancel all jobs"
    echo "  scontrol show job <job_id>    # View job details"
    echo "  cat ${SUMMARY_FILE}"
    echo ""
    
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        echo -e "${BLUE}To analyze nsys profiles:${NC}"
        echo "  nsys stats ${BATCH_OUTPUT_DIR}/{exp_name}/*.nsys-rep"
        echo "  nsys-ui ${BATCH_OUTPUT_DIR}/{exp_name}/*.nsys-rep"
        echo ""
    fi
fi

echo -e "${GREEN}🎉 Done!${NC}"
echo ""
