#!/bin/bash
# Submit HSTU pipeline parity test on cw-dfw with the la-cascade code.
# Runs single test_hstu_pipeline_matches_none_pipeline (first param combo)
# to smoke-test the 6-la cascade for bit-exact correctness.
set -euo pipefail

CLONE="${CLONE:-/lustre/fsw/portfolios/coreai/users/junzhang/workspace/recsys-rework-mtms-la-cascade}"
ACCOUNT="${ACCOUNT:-coreai_devtech_all}"
CONTAINER="${CONTAINER:-gitlab-master.nvidia.com/devtech-compute/distributed-recommender:devel_latest}"
PARTITION="${PARTITION:-batch_short}"
RESULTS_DIR="${RESULTS_DIR:-/lustre/fsw/portfolios/coreai/users/junzhang/benchmark_runs/parity_la_cascade_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RESULTS_DIR}"

cat <<EOF > "${RESULTS_DIR}/run.sbatch"
#!/bin/bash
#SBATCH --account=${ACCOUNT}
#SBATCH --job-name=parity_la_cascade
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=${RESULTS_DIR}/parity_%j.out

srun --container-image=${CONTAINER} \\
     --container-mounts=${CLONE}:/workspace/clone,${RESULTS_DIR}:/workspace/results \\
     --no-container-mount-home \\
     bash -c "cd /workspace/clone/examples/hstu && \\
              export PYTHONPATH=/workspace/clone/examples:\${PYTHONPATH:-} && \\
              export PYTEST_FIRST_PARAM_ONLY=1 && \\
              echo '--- env ---' && \\
              python -c 'import torch; print(\"torch=\", torch.__version__); print(\"cuda=\", torch.cuda.is_available())' && \\
              echo '--- run pytest with torchrun ---' && \\
              torchrun --nproc_per_node=1 --master_port=12345 -m pytest -xvs test/test_hstu_pipeline_parity.py 2>&1 | tee /workspace/results/pytest.log; \\
              echo '--- exit=\$? ---'"
EOF

cd "${RESULTS_DIR}"
sbatch run.sbatch
echo "Results dir: ${RESULTS_DIR}"
