#!/bin/bash
set -e
export PYTHONPATH=/home/scratch.junzhang_sw/workspace/.local/lib/python3.12/site-packages:$PYTHONPATH
cd /home/scratch.junzhang_sw/workspace/github/recsys-hstu_cp/examples/hstu
echo "=== START $(date) ==="
echo "--- single-GPU ---"
python -m pytest test/cp/test_mask_func.py test/cp/test_jagged_dispatch.py test/cp/test_hstu_block_cp.py test/cp/test_cp_api_smoke.py --no-header 2>&1 | tail -5
echo "--- cp=2 het multi-GPU ---"
WORLD_SIZE=2 NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=2 -m pytest test/cp/test_cp_het_mask.py --no-header 2>&1 | tail -3
echo "--- cp=4 het multi-GPU ---"
WORLD_SIZE=4 NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 -m pytest test/cp/test_cp_het_mask.py --no-header 2>&1 | tail -3
echo "--- cp=4 fwd multi-GPU ---"
WORLD_SIZE=4 NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 -m pytest test/cp/test_cp_forward.py --no-header 2>&1 | tail -3
echo "--- cp=4 bwd multi-GPU ---"
WORLD_SIZE=4 NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 -m pytest test/cp/test_cp_backward.py --no-header 2>&1 | tail -3
echo "--- cp=2 E2E HSTUBlock ---"
WORLD_SIZE=2 NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=2 -m pytest test/cp/test_hstu_block_e2e.py --no-header 2>&1 | tail -3
echo "--- cp=4 E2E HSTUBlock ---"
WORLD_SIZE=4 NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 -m pytest test/cp/test_hstu_block_e2e.py --no-header 2>&1 | tail -3
echo "=== END $(date) ==="
