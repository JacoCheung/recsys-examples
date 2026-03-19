#!/bin/bash
# ============================================================================
# Submit HSTU benchmark experiments on EOS cluster via SSH
#
# Usage:
#   ./training/benchmark/submit_remote.sh [options]
#
# This script SSHs into login-eos and runs submit_all_experiments_slurm.sh
# with pre-configured defaults. After submission, a background watcher polls
# the remote monitor log and pops up a desktop notification when all jobs
# finish. The script itself returns immediately so your terminal is not blocked.
#
# Run with --help / -h to see all available options and defaults.
# ============================================================================

set -e

# ============================================================================
# Defaults (edit these to match your environment)
# ============================================================================
LOGIN_HOST="login-eos"
REMOTE_HSTU_ROOT="/lustre/fsw/coreai_devtech_hugectr/junzhang/recsys-examples/examples/hstu"
SCP_DEST="vnc:/home/scratch.junzhang_sw/workspace/github/recsys-examples/examples/hstu/training/benchmark/results"
ACCOUNT="coreai_devtech_all"
JOB_NAME="coreai_devtech_all"
EXP_FILE="training/benchmark/experiments.txt"
NODES=2
CONTAINER_IMAGE="gitlab-master.nvidia.com/devtech-compute/distributed-recommender:devel_benchmark_e2e"
MONITOR_POLL_INTERVAL=60  # seconds between polling the remote monitor log
ENABLE_NSYS=true
GIT_BRANCH=""

usage() {
    cat <<'USAGE_EOF'
Usage: submit_remote.sh [options]

SSH into login-eos and run submit_all_experiments_slurm.sh with
pre-configured defaults. A background watcher polls the remote
monitor log and sends a desktop notification when all jobs finish.

Options:
  -h, --help                 Show this help message and exit
  --login-host=HOST          SSH login host           (default: login-eos)
  --remote-root=PATH         Remote HSTU root dir     (default: /lustre/fsw/.../examples/hstu)
  --exp-file=FILE            Experiment list file    (default: training/benchmark/experiments.txt)
  --nodes=N                  Number of nodes          (default: 2)
  --container-image=IMAGE    Container image          (default: see script)
  --account=NAME             SLURM account            (default: coreai_devtech_all)
  --job-name=NAME            SLURM job name prefix    (default: coreai_devtech_all)
  --scp-dest=DEST            SCP destination          (default: see script)
  --poll-interval=SEC        Watcher poll interval    (default: 60)
  --branch=BRANCH            Git branch to checkout (clones repo per job for isolation)
  --nsys / --no-nsys         Enable/disable nsys profiling (default: enabled)
  --dry-run                  Dry run mode (passed to remote script)

Any unrecognised options are forwarded to submit_all_experiments_slurm.sh.
-y and --wait-and-analyze are always enabled (SSH is non-interactive).

Examples:
  ./training/benchmark/submit_remote.sh
  ./training/benchmark/submit_remote.sh --nodes=4
  ./training/benchmark/submit_remote.sh --dry-run --no-nsys
USAGE_EOF
    exit 0
}

# ============================================================================
# Parse arguments
# ============================================================================
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)           usage ;;
        --login-host=*)      LOGIN_HOST="${1#*=}" ;;
        --remote-root=*)     REMOTE_HSTU_ROOT="${1#*=}" ;;
        --exp-file=*)        EXP_FILE="${1#*=}" ;;
        --nodes=*)           NODES="${1#*=}" ;;
        --container-image=*) CONTAINER_IMAGE="${1#*=}" ;;
        --account=*)         ACCOUNT="${1#*=}" ;;
        --job-name=*)        JOB_NAME="${1#*=}" ;;
        --scp-dest=*)        SCP_DEST="${1#*=}" ;;
        --poll-interval=*)   MONITOR_POLL_INTERVAL="${1#*=}" ;;
        --branch=*)          GIT_BRANCH="${1#*=}" ;;
        --nsys)              ENABLE_NSYS=true ;;
        --no-nsys)           ENABLE_NSYS=false ;;
        *)                   EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

# ============================================================================
# Build the remote command
# ============================================================================
if [ -n "${GIT_BRANCH}" ]; then
    # With --branch: clone repo to isolated directory, run submit from there.
    # This ensures experiments.txt and all scripts come from the target branch.
    REMOTE_REPO_ROOT="${REMOTE_HSTU_ROOT%/examples/hstu}"
    BRANCH_SAFE=$(echo "${GIT_BRANCH}" | tr '/' '-')
    CLONE_DIR="${REMOTE_REPO_ROOT}-runs/${BRANCH_SAFE}"

    REMOTE_CMD="set -e"
    REMOTE_CMD+=" && echo '📦 Cloning branch ${GIT_BRANCH} to ${CLONE_DIR}...'"
    REMOTE_CMD+=" && rm -rf ${CLONE_DIR}"
    REMOTE_CMD+=" && git clone -b ${GIT_BRANCH} --local ${REMOTE_REPO_ROOT} ${CLONE_DIR}"
    REMOTE_CMD+=" && { [ -d ${REMOTE_REPO_ROOT}/third_party/HierarchicalKV ] && cp -a ${REMOTE_REPO_ROOT}/third_party/HierarchicalKV ${CLONE_DIR}/third_party/ || true; }"
    REMOTE_CMD+=" && echo '✅ Clone ready: '\$(git -C ${CLONE_DIR} log --oneline -1)"
    REMOTE_CMD+=" && cd ${CLONE_DIR}/examples/hstu"
    REMOTE_CMD+=" && bash training/benchmark/submit_all_experiments_slurm.sh"
else
    # Without --branch: run directly from the remote working directory.
    REMOTE_CMD="cd ${REMOTE_HSTU_ROOT} && bash training/benchmark/submit_all_experiments_slurm.sh"
fi

# Common options
REMOTE_CMD+=" --scp-dest=${SCP_DEST}"
REMOTE_CMD+=" --account=${ACCOUNT}"
REMOTE_CMD+=" --job-name=${JOB_NAME}"
REMOTE_CMD+=" --exp-file=${EXP_FILE}"
REMOTE_CMD+=" --nodes=${NODES}"
REMOTE_CMD+=" --container-image ${CONTAINER_IMAGE}"
REMOTE_CMD+=" --wait-and-analyze"
REMOTE_CMD+=" -y"
if [ "${ENABLE_NSYS}" = true ]; then
    REMOTE_CMD+=" --nsys"
fi
if [ -n "${GIT_BRANCH}" ]; then
    REMOTE_CMD+=" --branch=${GIT_BRANCH}"
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    REMOTE_CMD+=" ${EXTRA_ARGS[*]}"
fi

echo "==========================================="
echo "🚀 Submitting HSTU benchmark on ${LOGIN_HOST}"
echo "==========================================="
echo ""
echo "Remote command:"
echo "  ${REMOTE_CMD}"
echo ""

# ============================================================================
# Submit via SSH, capture output to extract monitor log path
# ============================================================================
SSH_OUTPUT=$(ssh "${LOGIN_HOST}" "${REMOTE_CMD}" 2>&1) && SSH_EXIT=0 || SSH_EXIT=$?

# Print the SSH output (always, even on failure)
echo "${SSH_OUTPUT}"

if [ ${SSH_EXIT} -ne 0 ]; then
    echo ""
    echo "❌ SSH command failed with exit code ${SSH_EXIT}"
    exit ${SSH_EXIT}
fi

# ============================================================================
# Extract monitor log path from SSH output
# ============================================================================
# The submit script prints: "Monitor log: /path/to/monitor.log"
MONITOR_LOG=$(echo "${SSH_OUTPUT}" | grep -oP '(?<=Monitor log: )\S+' | head -1)

if [ -z "${MONITOR_LOG}" ]; then
    echo ""
    echo "⚠️  Could not detect monitor log path from output."
    echo "   Jobs may have been submitted without --wait-and-analyze, or submission failed."
    exit 0
fi

# ============================================================================
# Launch background watcher process
# ============================================================================
WATCHER_LOG="/tmp/hstu_benchmark_watcher_$$.log"

(
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watcher started (PID: $$)"
    echo "  Remote log: ${MONITOR_LOG}"
    echo "  Poll interval: ${MONITOR_POLL_INTERVAL}s"
    echo ""

    while true; do
        # Check if the monitor log contains the completion marker
        FINISHED=$(ssh "${LOGIN_HOST}" "grep -c 'Monitor script finished' '${MONITOR_LOG}' 2>/dev/null" 2>/dev/null || echo "0")

        if [ "${FINISHED}" -gt 0 ] 2>/dev/null; then
            echo ""
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Remote benchmark jobs completed!"
            echo ""

            # Capture last 20 lines of monitor log
            echo "--- Last 20 lines of monitor log ---"
            ssh "${LOGIN_HOST}" "tail -20 '${MONITOR_LOG}'" 2>/dev/null || true
            echo "--- End of monitor log ---"

            # ============================================================
            # Clean up remote clone directory (created by --branch)
            # ============================================================
            if [ -n "${CLONE_DIR}" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🧹 Cleaning up remote clone: ${CLONE_DIR}"
                ssh "${LOGIN_HOST}" "rm -rf '${CLONE_DIR}'" 2>/dev/null || true
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Clone directory cleaned up"
            fi

            # ============================================================
            # Desktop notification (try multiple methods)
            # ============================================================
            TITLE="🎉 HSTU Benchmark Completed"
            MSG="All benchmark jobs on ${LOGIN_HOST} have finished.\nMonitor log: ${MONITOR_LOG}"

            # Method 1: notify-send (Linux desktop with DBUS)
            if command -v notify-send &>/dev/null; then
                notify-send -u critical "${TITLE}" "${MSG}" 2>/dev/null || true
            fi

            # Method 2: zenity popup dialog (X11/Wayland)
            if command -v zenity &>/dev/null; then
                zenity --info --title="${TITLE}" --text="${MSG}" --width=400 2>/dev/null || true
            # Method 3: xmessage (basic X11)
            elif command -v xmessage &>/dev/null; then
                xmessage -center "${TITLE}: ${MSG}" 2>/dev/null || true
            fi

            # Method 4: terminal bell (write to original terminal)
            echo -e "\a" > /dev/tty 2>/dev/null || true

            # Method 5: wall message to all terminals
            echo "🎉 HSTU Benchmark on ${LOGIN_HOST} completed! Check: ${WATCHER_LOG}" | wall 2>/dev/null || true

            echo ""
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watcher finished."
            break
        fi

        # Log progress
        LATEST_STATUS=$(ssh "${LOGIN_HOST}" "grep 'Status:' '${MONITOR_LOG}' 2>/dev/null | tail -1" 2>/dev/null || echo "")
        if [ -n "${LATEST_STATUS}" ]; then
            echo "[$(date '+%H:%M:%S')] ${LATEST_STATUS}"
        else
            echo "[$(date '+%H:%M:%S')] Waiting..."
        fi

        sleep "${MONITOR_POLL_INTERVAL}"
    done
) >> "${WATCHER_LOG}" 2>&1 &

WATCHER_PID=$!
disown ${WATCHER_PID} 2>/dev/null || true

echo ""
echo "==========================================="
echo "👀 Background watcher started"
echo "==========================================="
echo "  Watcher PID:    ${WATCHER_PID}"
echo "  Watcher log:    ${WATCHER_LOG}"
echo "  Remote log:     ${MONITOR_LOG}"
echo "  Poll interval:  ${MONITOR_POLL_INTERVAL}s"
echo ""
echo "  You will get a desktop notification when all jobs complete."
echo ""
echo "  Useful commands:"
echo "    tail -f ${WATCHER_LOG}         # Follow watcher progress"
echo "    kill ${WATCHER_PID}                        # Stop watcher"
echo ""
echo "🎉 Terminal is free — you can continue working!"
echo ""
