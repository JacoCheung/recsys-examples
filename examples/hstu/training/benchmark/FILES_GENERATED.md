# HSTU H100 16GPU Benchmark - Generated Files List

## 📋 Overview

For the H100 16 GPU benchmark, we have generated a complete set of configuration files, scripts, and documentation, including **9 progressive experiments** (with the new Workload Balancer optimization).

---

## 📁 Generated Files

### 1. Core Documentation

| File | Description |
|------|-------------|
| `H100_16GPU_BENCHMARK_PLAN.md` | **Main Document**: Complete benchmark plan including design philosophy, experiment configurations, execution plan, visualization approach, etc. |
| `FILES_GENERATED.md` | This file: Generated files list |

### 2. Experiment Configuration Files (10 files)

All configuration files are located at: `examples/hstu/training/configs/`

| Config File | Experiment | Description |
|------------|------------|-------------|
| `h100_16gpu_exp0_baseline.gin` | Exp 0 | Baseline (Triton attention, no optimization) |
| `h100_16gpu_exp1_cutlass.gin` | Exp 1 | +CUTLASS Attention |
| `h100_16gpu_exp2_fusion.gin` | Exp 2 | +Kernel Fusion |
| `h100_16gpu_exp3_recompute.gin` | Exp 3 | +Selective Recompute |
| `h100_16gpu_exp3.5_workload_balancer.gin` | **Exp 3.5** | **+Workload Balancer** 🔥 |
| `h100_16gpu_exp4_dynamicemb.gin` | Exp 4 | +DynamicEmb |
| `h100_16gpu_exp5_lfu.gin` | Exp 5 | +LFU Eviction |
| `h100_16gpu_exp6_pipeline.gin` | Exp 6 | +Pipeline Prefetch |
| `h100_16gpu_exp7_tp.gin` | Exp 7 | +Tensor Parallel (TP=2) |
| `h100_16gpu_exp8_full.gin` | Exp 8 | **Full Optimization** (all optimizations) |

### 3. Run Scripts (3 files)

All scripts are located at: `examples/hstu/training/benchmark/`

| Script | Description |
|--------|-------------|
| `run_single_experiment.sh` | Run a single experiment<br/>Usage: `./run_single_experiment.sh exp6_pipeline` |
| `run_all_experiments.sh` | **Batch run all 9 experiments**<br/>Auto sequential execution with progress display |
| `monitor_gpu.py` | GPU monitoring script<br/>Real-time recording of GPU utilization and memory to CSV |

---

## 🎯 Key Optimization Points

### New: Workload Balancer (Exp 3.5)

**Why is it important?**
- In **variable-length sequence** scenarios, computation workload varies greatly between samples (sequence length ranges from tens to thousands)
- HSTU attention complexity is O(n²), sequence length variance causes severe load imbalance
- Simple sample-count-based distribution makes some GPUs bottlenecks while others idle

**How does it work?**
- Calculates workload for each sample based on sequence length
- Dynamically redistributes samples across GPUs to balance total workload
- All GPUs complete computation simultaneously, reducing wait time

**Expected Results:**
- **Throughput improvement: 1.3-1.8x** (in variable-length sequence scenarios)
- GPU utilization: 60-70% → 85-95%
- **One of the most significant single optimizations**

**Configuration:**
```python
TrainerArgs.enable_balanced_shuffler = True
```

---

## 📊 Complete Experiment Sequence

```
Exp 0: Baseline (no optimization)
  ↓ +CUTLASS Attention
Exp 1: 1.3x speedup
  ↓ +Kernel Fusion
Exp 2: 1.56x cumulative speedup
  ↓ +Selective Recompute
Exp 3: 1.48x cumulative speedup (30% memory savings)
  ↓ +Workload Balancer 🔥
Exp 3.5: 2.22x cumulative speedup
  ↓ +DynamicEmb
Exp 4: 2.22x cumulative speedup (60% memory savings)
  ↓ +LFU Eviction
Exp 5: 2.44x cumulative speedup
  ↓ +Pipeline Prefetch
Exp 6: 3.17x cumulative speedup
  ↓ +Tensor Parallel
Exp 7: 3.17x cumulative speedup (75% memory savings)
  ↓ +Sequence Parallel
Exp 8: 3.01x final speedup (85% memory savings)
```

**Expected Overall Improvement:**
- 🚀 **3-3.5x end-to-end training speedup**
- 💾 **80-85% GPU memory savings**
- 📈 **Support 50M+ embedding table**
- 📏 **Support 4K+ sequence length**

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Enter benchmark directory
cd examples/hstu/training/benchmark

# Add execution permissions to scripts
chmod +x run_single_experiment.sh
chmod +x run_all_experiments.sh
chmod +x monitor_gpu.py
```

### 2. Run Single Experiment

```bash
# Set environment variables (modify according to actual situation)
export MASTER_ADDR="node0.cluster.local"
export NODE_RANK=0  # Use 0 for first node, 1 for second node

# Run experiment
./run_single_experiment.sh exp3.5_workload_balancer
```

### 3. Run All Experiments

```bash
# Start GPU monitoring in background (optional)
python monitor_gpu.py results/gpu_metrics.csv 5 &

# Run all experiments
./run_all_experiments.sh
```

### 4. Analyze Results

All logs are saved in the `results/` directory:
```
results/
├── exp0_baseline_node0.log
├── exp1_cutlass_node0.log
├── exp3.5_workload_balancer_node0.log
├── ...
└── gpu_metrics.csv
```

---

## 📈 Expected Benchmark Output

### Key Metrics

Each experiment outputs:

1. **Training throughput** (samples/sec)
2. **Iteration time** (ms/iter)
3. **GPU memory usage** (GB)
4. **Time breakdown**:
   - Embedding lookup time
   - Forward time
   - Backward time
   - Communication time

### Focus Comparisons

Key comparisons to focus on:

| Comparison | Description | Expected Improvement |
|------------|-------------|---------------------|
| Exp 0 vs Exp 1 | CUTLASS advantage | 1.3x |
| Exp 3 vs Exp 3.5 | **Workload Balancer effect** | **1.5x** 🔥 |
| Exp 5 vs Exp 6 | Pipeline effect | 1.3x |
| Exp 0 vs Exp 8 | Overall optimization effect | 3-3.5x |

---

## 💡 Usage Recommendations

### For Experiment Execution

1. **First run**: Recommend running a single experiment first to verify environment
   ```bash
   ./run_single_experiment.sh exp0_baseline
   ```

2. **Batch run**: If time is limited, run in batches:
   - First batch: Kernel optimization (exp0-exp3.5)
   - Second batch: Embedding + Pipeline (exp4-exp6)
   - Third batch: Parallelism (exp7-exp8)

3. **Monitoring**: Recommend running GPU monitoring simultaneously for later analysis

### For Demo Preparation

1. **Key highlights**:
   - Workload Balancer (exp 3.5): Key optimization for variable-length sequence scenarios
   - Pipeline (exp 6): Typical I/O hiding case
   - Final result (exp 8): 3x+ speedup + 85% memory savings

2. **Visualization**:
   - Throughput comparison bar chart (progressive)
   - Memory usage trend chart
   - Time breakdown stacked chart

3. **Customer concerns**:
   - Real scenario performance (variable-length sequences + large-scale embedding)
   - Ease of use (simple configuration, out-of-the-box)
   - ROI (improved hardware utilization, reduced training cost)

---

## ❓ FAQ

### Q: Why add Exp 3.5?

A: Workload Balancer is an important but often overlooked optimization:
- In real recommendation scenarios, user history lengths vary greatly (variable-length sequences)
- It can bring 1.3-1.8x significant improvement, worth showcasing separately
- Placed after exp 3 because it doesn't depend on DynamicEmb, can work with static embedding

### Q: What are the differences between all config files?

A: Each config file is **progressive**, changing only one optimization item at a time:
- exp0: Baseline (Triton attention)
- exp1: Switch to CUTLASS
- exp2: Add fusion
- exp3: Add recompute
- **exp3.5: Add workload balancer** 🆕
- exp4: Switch to DynamicEmb
- ...

This allows clearly seeing **each optimization's independent contribution**.

### Q: Why is Exp 8 called "full" instead of "sp"?

A: Exp 8 is **the culmination of all optimizations**, including:
- CUTLASS + Fusion + Recompute
- Workload Balancer
- DynamicEmb + LFU
- Pipeline Prefetch
- TP + SP

So "full" better represents its meaning.

### Q: Can some experiments be skipped?

A: Yes, but recommend running at least:
- Exp 0 (baseline)
- Exp 3.5 (workload balancer)
- Exp 6 (pipeline)
- Exp 8 (full optimization)

These 4 experiments can showcase the main optimization effects.

---

## 📞 Contact

If you have questions:
1. Check the detailed explanation in `H100_16GPU_BENCHMARK_PLAN.md`
2. Submit GitHub Issue: https://github.com/NVIDIA/recsys-examples/issues
3. Join community discussion: https://forums.developer.nvidia.com/

---

**Last Updated**: 2026-01-28  
**Version**: v1.1 (with Workload Balancer)
