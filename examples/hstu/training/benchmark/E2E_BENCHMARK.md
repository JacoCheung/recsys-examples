# HSTU End-to-End Training Performance Benchmark

This document describes the HSTU end-to-end training benchmark: a set of **progressive experiments** that incrementally enable optimizations to quantify each one's contribution to training throughput (MFU).

## 1. Background

### Embedding in Large-Scale Recommendation

Production recommendation models use massive embedding tables (tens of millions to billions of rows). DynamicEmb stores these tables in host memory and serves lookups to GPU during training.

### Optimization Space

Each experiment below adds **one** optimization on top of the previous, so the speedup is cumulative:

| # | Optimization | What It Does |
|---|-------------|-------------|
| 0 | Baseline | Triton attention, DynamicEmb without caching, no shuffler, DP-only |
| 1 | **Workload-Balanced Shuffler** | Redistribute variable-length sequences across GPUs so that each GPU's total attention FLOPs are balanced. Eliminates GPU idle time caused by HSTU's O(n²) attention on skewed sequence lengths. |
| 2 | **CUTLASS Attention** | Replace Triton attention with a hand-tuned CUTLASS kernel optimized for HSTU's causal+context mask. Better register allocation and H100 utilization. |
| 3 | **DynamicEmb Caching** | Cache hot DynamicEmb rows in GPU HBM while keeping the full table in host memory. |
| 4 | **Hash-RoundRobin Sharding** | Use row-wise `hash_roundrobin` placement for DynamicEmb tables to distribute IDs more evenly across ranks. |
| 5 | **Prefetch Pipeline** | Enable the prefetch pipeline on top of caching and Hash-RoundRobin sharding. |

### Benchmark Configuration

**Hardware**: H100-SXM5-80GB (single-node 8 GPU or multi-node)

**Model hyperparameters** (fixed across all experiments):

| Parameter | Value |
|-----------|-------|
| Hidden size | 1024 |
| Num HSTU layers | 8 |
| Num attention heads | 4 |
| Head dimension (kv_channels / dim_per_head) | 256 |
| Item embedding dim | 128 |
| Contextual embedding dim | 128 |
| Prediction head | [512, 8] × 8 tasks |
| Optimizer | Adam (lr=1e-3) |

**Embedding tables**:

| Table | Rows | Dim | Type |
|-------|------|-----|------|
| item | 50M | 128 | DynamicEmb |
| action | 100 | 128 | Static (DP sharded) |
| user_id | 50M | 128 | DynamicEmb |
| user_age | 100 | 128 | Static (DP sharded) |
| item_category_l1 | 50 | 128 | Static (DP sharded) |

**Data distribution**:

| Parameter | Value |
|-----------|-------|
| Batch size per GPU | 32 |
| Max sequence length | 4096 |
| Sequence length distribution | Zipf (α=1.2), jagged |
| Key value distribution | Zipf (α=1.05) for `item` and `user_id` IDs |
| Training iterations | 1000 |
| Log interval | 100 |
| Eval interval | 1001 (disabled for 1000-iteration benchmark runs) |
| Profiling window | iterations 150-170 |

Synthetic data with Zipf-distributed sequence lengths simulates the heavy-tailed user-history patterns seen in production.

---

## 2. Results

**Hardware**: 2× H100-SXM5-80GB nodes (16 GPUs total).

**Run**: `cwdfw_benchmark_cont_full_nsys_20260528_202615`, commit `bc40d89c`. The table reports average TFLOPS/MFU from the logged intervals after the first warmup log, iter 199-999. Peak columns report the best logged interval. MFU uses 989 BF16 dense Tensor Core TFLOPS per H100 GPU, or 15.824 PFLOPS for the 16-GPU run.

| Exp | Name | Avg TFLOPS | Avg MFU (%) | Peak TFLOPS | Peak MFU (%) | Speedup vs Baseline | Notes |
|-----|------|-----------:|------------:|-------------:|-------------:|---------------------:|-------|
| 0 | Baseline | 1210 | 7.65 | 1240 | 7.84 | 1.00× | Triton attention, no shuffler/caching |
| 1 | +Shuffler | 1852 | 11.70 | 1933 | 12.22 | 1.53× | Balances Zipf-distributed sequence lengths |
| 2 | **+CUTLASS** | **4841** | **30.59** | **5270** | **33.30** | **4.00×** | Attention kernel swap, largest single-step gain |
| 3 | +DynamicEmb Caching | 4748 | 30.00 | 5156 | 32.58 | 3.93× | Enables HBM cache for DynamicEmb hot rows |
| 4 | +Hash-RoundRobin | 4938 | 31.21 | 5390 | 34.06 | 4.08× | Row-wise `hash_roundrobin` DynamicEmb placement |
| 5 | +Prefetch Pipeline | 4969 | 31.40 | 5410 | 34.19 | 4.11× | Prefetch pipeline on top of caching + Hash-RoundRobin |

### Key Takeaways

1. **CUTLASS attention is the main jump**: Replacing Triton attention with CUTLASS raises average throughput from 1852 TFLOPS to 4841 TFLOPS after the shuffler step, and reaches 5270 peak TFLOPS.

2. **Workload-balanced shuffler gives a clear baseline lift**: Zipf-distributed sequence lengths create load imbalance with O(n²) attention. The shuffler improves average throughput from 1210 TFLOPS to 1852 TFLOPS, a 1.53× speedup.

3. **DynamicEmb caching changes memory behavior more than raw throughput**: Caching slightly lowers average throughput compared with CUTLASS-only in this run, but it enables the intended host/HBM split for large DynamicEmb tables.

4. **Hash-RoundRobin recovers and improves throughput with caching enabled**: `hash_roundrobin` raises average throughput from 4748 TFLOPS to 4938 TFLOPS and reaches the best non-prefetch peak of 5390 TFLOPS.

5. **Prefetch is nearly flat in this run**: The prefetch pipeline adds a small average gain over Hash-RoundRobin, from 4938 TFLOPS to 4969 TFLOPS. Peak throughput reaches 5410 TFLOPS, or 34.19% MFU.

---

## 3. Reproducing the Benchmark

### Prerequisites

- Docker image built from `docker/Dockerfile`, or an equivalent environment with HSTU kernels and DynamicEmb compiled.
- All commands below assume **working directory** = `recsys-examples/examples/hstu`.

```bash
cd recsys-examples/examples/hstu
```

### Experiment definitions

Experiments are listed in `training/benchmark/experiments.txt`:

```
exp0_baseline,--value_dist zipf --value_dist_alpha 1.05
exp1_shuffler,--balanced_shuffler --value_dist zipf --value_dist_alpha 1.05
exp2_cutlass,--balanced_shuffler --kernel_backend cutlass --value_dist zipf --value_dist_alpha 1.05
exp3_caching,--balanced_shuffler --kernel_backend cutlass --caching --value_dist zipf --value_dist_alpha 1.05
exp4_caching_hr,--balanced_shuffler --kernel_backend cutlass --caching --dist_type hash_roundrobin --value_dist zipf --value_dist_alpha 1.05
exp5_prefetch,--balanced_shuffler --kernel_backend cutlass --caching --dist_type hash_roundrobin --pipeline_type prefetch --value_dist zipf --value_dist_alpha 1.05
```

Each line is `exp_name,options_for_generate_gin_config.py`. The script `generate_gin_config.py` produces a complete gin config file from these flags.

### Debug environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEM_DEBUG` | `0` | Log GPU physical memory (including NCCL buffers) after each optimizer step on all ranks |
| `CUDA_MEM_WATCHDOG` | `0` | Auto-call `torch.cuda.empty_cache()` when caching allocator fragmentation exceeds threshold |

Set before launching training, e.g. `export CUDA_MEM_WATCHDOG=1` in the SLURM job script or shell.

### Option A: Single experiment (local)

```bash
# Run one experiment on 8 GPUs
./training/benchmark/scripts/run_single_experiment_local.sh exp2_cutlass \
    --kernel_backend cutlass --nproc=8

# Dry-run (prints generated config, does not train)
./training/benchmark/scripts/run_single_experiment_local.sh exp2_cutlass \
    --kernel_backend cutlass --dry-run
```

### Option B: All experiments (local)

```bash
# Run every experiment in experiments.txt sequentially
./training/benchmark/scripts/run_all_experiments_local.sh \
    --exp-file=training/benchmark/experiments.txt \
    --nproc=8

# With nsys profiling
./training/benchmark/scripts/run_all_experiments_local.sh \
    --exp-file=training/benchmark/experiments.txt \
    --nproc=8 --nsys
```

### Option C: SLURM cluster

```bash
# Submit all experiments as SLURM jobs
./training/benchmark/scripts/submit_all_experiments_slurm.sh \
    --exp-file=training/benchmark/experiments.txt \
    --nodes=2 --ranks-per-node=8 --nsys

# Sequential execution (each job waits for the previous)
./training/benchmark/scripts/submit_all_experiments_slurm.sh \
    --exp-file=training/benchmark/experiments.txt \
    --nodes=2 --ranks-per-node=8 --nsys --sequential

# Dry-run
./training/benchmark/scripts/submit_all_experiments_slurm.sh \
    --exp-file=training/benchmark/experiments.txt --dry-run
```

Key `submit_all_experiments_slurm.sh` options:

| Flag | Default | Description |
|------|---------|-------------|
| `--exp-file=FILE` | *(required)* | Experiment list |
| `--nodes=N` | 2 | SLURM nodes |
| `--ranks-per-node=N` | 8 | GPUs per node |
| `--nsys` | off | Enable nsys profiling |
| `--sequential` | parallel | Chain jobs with dependencies |
| `--container-image=IMG` | *(see script)* | Override container image |
| `--partition=NAME` | batch | SLURM partition |
| `--time=HH:MM:SS` | 00:30:00 | Wall-time limit |
| `--wait-and-analyze` | off | Poll jobs and auto-run analysis |
| `--dry-run` | off | Print commands only |

### Running a subset

Create a custom experiment file:

```bash
cat > quick_test.txt << 'EOF'
exp0_baseline,--value_dist zipf --value_dist_alpha 1.05
exp2_cutlass,--balanced_shuffler --kernel_backend cutlass --value_dist zipf --value_dist_alpha 1.05
EOF

./training/benchmark/scripts/run_all_experiments_local.sh --exp-file=quick_test.txt --nproc=8
```

### Output directory structure

```
training/benchmark/results/
└── {batch_timestamp}/
    ├── exp0_baseline/
    │   ├── exp0_baseline_{timestamp}.gin     # generated config
    │   ├── exp0_baseline_{timestamp}.log     # training log
    │   └── exp0_baseline_*.nsys-rep          # nsys profiles (if --nsys)
    ├── exp1_shuffler/
    │   └── ...
    ├── exp2_cutlass/
    │   └── ...
    ├── exp3_caching/
    │   └── ...
    ├── exp4_caching_hr/
    │   └── ...
    ├── exp5_prefetch/
    │   └── ...
    ├── summary.txt                           # batch summary
    └── comparison.png                         # TFLOPS + MFU comparison chart
```

### Analyzing results

```bash
# Parse MFU from training logs
python training/benchmark/scripts/analyze_results.py \
    training/benchmark/results/{batch_timestamp}/

# Nsight Systems CLI stats
nsys stats training/benchmark/results/{batch_timestamp}/exp0_baseline/*.nsys-rep
```
