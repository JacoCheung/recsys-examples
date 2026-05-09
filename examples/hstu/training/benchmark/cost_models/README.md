# HSTU pipeline cost models (per-task GPU duration)

JSON files in this dir feed the fire-order auto-scheduler
(`commons.pipeline.engine.autosched.CostModel.from_json`). Each entry
maps `task_name` → `{cpu_us, gpu_us}`. `gpu_us` is what the scheduler
actually uses for critical-path / NCCL-chain math.

## File index

| file | scenario | source |
|---|---|---|
| `hstu_prefetch_caching_1node.json` | prefetch + DynamicEmb caching ratio=0.1 + balanced_shuffler + cutlass + recompute_layernorm + zipf 1.05, 1×8 H100 | nsys profile, rank 0, profile_step_start=150 to 200 (50 iters); commit `4a1568fb` |
| `hstu_prefetch_caching_2node.json` | same gin, 2×8 H100 + IB | TBD — populate after first 2-node nsys profile lands |

## How to (re)generate from a fresh nsys

1. Run benchmark with `NSYS=1 ALLOC=<jobid> bash mtms_runner.sh` (see
   `tasks/mtms_runner.sh`); produces 8 `.nsys-rep` files under
   `examples/hstu/training/benchmark/results/nsys/<variant>/`.
2. Convert + extract:
   ```
   srun --jobid=<alloc> --overlap --container-image=<image> bash -c \
     "python3 examples/hstu/training/benchmark/scripts/cost_model_from_nsys.py \
        path/to/rank0.nsys-rep -o cost_models/<scenario>.json"
   ```
   (script TBD — for now I generate by hand from `nsys_query.py`'s
   table.)
3. Commit the JSON; future auto-scheduler runs will pick it up.

## Sanity check

The default-stream chain GPU sum should approximately equal
`elapsed_time / 100` from the training log's profile region. Off-default
tasks should sum to "side-stream busy" reported by `nsys_query.py`.
