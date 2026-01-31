# HSTU Training Optimization Comprehensive Benchmark Plan

## Executive Summary

This document provides a complete HSTU large-scale training benchmark plan for demonstrating all optimization achievements from project inception to present in an H100 dual-node 16 GPU environment. Through **progressive comparison experiments**, it clearly presents each optimization's contribution, providing strong support for customer demonstrations.

**Key Highlights**:
- ✅ Large-scale real scenario: H100 2 nodes × 8 GPUs = 16 GPUs
- ✅ Complete optimization stack: DynamicEmb + Pipeline + HSTU Kernel + Parallelism
- ✅ Systematic comparison: 8 progressive experiments showcasing end-to-end performance improvement
- ✅ Production-grade configuration: 50M embedding + 4K sequence length + 8 HSTU layers

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [Benchmark Design Philosophy](#2-benchmark-design-philosophy)
3. [Detailed Experiment Plan](#3-detailed-experiment-plan)
4. [Configuration Files and Scripts](#4-configuration-files-and-scripts)
5. [Execution Plan](#5-execution-plan)
6. [Result Visualization](#6-result-visualization)
7. [Demo Recommendations](#7-demo-recommendations)

---

## 1. Background and Motivation

### 1.1 Why Do We Need This Benchmark?

**Problem Background**:
- Since recsys-examples project creation, HSTU training has integrated multiple optimizations (DynamicEmb, Pipeline, Kernel Fusion, Parallelism, etc.)
- Lacking an **end-to-end large-scale benchmark** to systematically demonstrate all optimizations' combined effect
- Customers need to see performance in **real production scenarios**

**Goals**:
1. **Systematic Validation**: Validate all optimizations' effectiveness in real large-scale scenario (H100 16 GPUs)
2. **Quantify Benefits**: Clearly show performance improvements from each optimization (throughput, memory, convergence)
3. **Customer Confidence**: Generate strong customer interest through intuitive comparison experiments

### 1.2 Optimization Stack Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    HSTU Training Optimization Overview       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Embedding   │  │   Pipeline   │  │  Parallelism │      │
│  │  Optimization│  │  Optimization│  │  Optimization│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│        │                  │                  │               │
│        ▼                  ▼                  ▼               │
│  • DynamicEmb      • Prefetch       • Tensor Parallel       │
│  • LRU/LFU Evict   • Caching        • Sequence Parallel     │
│  • Admission       • Async          • Data Parallel         │
│                                      • Workload Balancer 🔥  │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │    HSTU Kernel Optimization (Kernel Fusion Default)   │   │
│  └──────────────────────────────────────────────────────┘   │
│        │                                                     │
│        ▼                                                     │
│  • CUTLASS Attention    • Kernel Fusion (Default)           │
│  • Selective Recompute  • Async Wgrad                       │
│  • FP8 Support          • Scaling SeqLen                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Expected Results**: Through systematic integration, achieve **3-5x end-to-end training speedup** compared to baseline.

---

## 2. Benchmark Design Philosophy

### 2.1 Design Principles

#### Principle 1: Progressive Comparison Experiments

**Why?**
- Simply showing "final optimized version" doesn't help customers understand each optimization's value
- Progressive comparison clearly shows optimization stack's **cumulative effect**

**How?**
```
Baseline → +Opt1 → +Opt2 → ... → Full Optimized
   ↓         ↓         ↓              ↓
 Perf A    Perf B    Perf C         Perf N
```

#### Principle 2: Production-Grade Real Scenarios

**Why?**
- Toy datasets cannot convince customers
- Real scenarios demonstrate technical value

**How?**
- Use power-law distributed synthetic data (simulating real recommendation scenarios)
- Large-scale embedding table (50M)
- Long sequence support (4K tokens)
- Multi-task learning (8 tasks)

#### Principle 3: Multi-Dimensional Metrics

**Why?**
- Single metric cannot fully evaluate system performance
- Customers care about throughput, latency, memory, and convergence

**How?**
- **Training throughput** (samples/sec): Core metric
- **Memory usage** (GPU Memory): Scalability metric
- **Convergence speed** (Loss curve): Model quality metric
- **Communication overhead** (Comm Time): Distributed efficiency metric

### 2.2 Experiment Flow Diagram

```mermaid
graph TD
    A[Start: Prepare Environment] --> B[Exp 0: Baseline<br/>DynamicEmb + Kernel Fusion]
    B --> C[Exp 1: +CUTLASS Attention]
    C --> D[Exp 2: CUTLASS Benefit Validation]
    D --> E[Exp 3: +Selective Recompute]
    E --> F[Exp 3.5: +Workload Balancer 🔥]
    F --> G[Exp 4: +DynamicEmb Caching 🔥]
    G --> H[Exp 5: +LFU Eviction]
    H --> I[Exp 6: +Pipeline Prefetch]
    I --> J[Exp 7: +Tensor Parallel]
    J --> K[Exp 8: +Sequence Parallel]
    K --> L[Result Summary and Visualization]
    
    style B fill:#ffcccc
    style F fill:#ffffcc
    style K fill:#ccffcc
```

### 2.3 Hardware and Software Environment

**Hardware Configuration**:
- GPU: H100-SXM5-80GB × 16 (2 nodes × 8 GPUs)
- CPU: Sufficient host memory (recommended 512GB+ per node)
- Network: NVLink + InfiniBand (for inter-node communication)

**Software Stack**:
- CUDA 12.1+
- PyTorch 2.1+
- TorchRec latest
- Megatron-Core 0.9.0
- DynamicEmb (from recsys-examples)
- HSTU kernels (CUTLASS-based)

---

## 3. Detailed Experiment Plan

### Experiment Configuration Baseline

**Fixed configuration** shared by all experiments:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model Parameters** | | |
| Hidden Size | 1024 | Hidden dimension |
| Num Layers | 8 | HSTU layer count |
| Num Heads | 4 | Attention head count |
| Head Dim | 256 | Dimension per head |
| **Data Configuration** | | |
| Sequence Length | 4096 | Maximum sequence length |
| Sequence Type | **Jagged** | **Variable-length sequences** |
| Item Vocab Size | 50M | Item feature table size |
| Action Vocab Size | 100 | Action feature table size |
| Batch Size | 32 | Batch size per GPU |
| **Training Configuration** | | |
| Total GPUs | 16 | H100 × 16 |
| Training Steps | 1000 | Training steps |
| Optimizer | Adam | Optimizer |
| Learning Rate | 1e-3 | Learning rate |
| Num Tasks | 8 | Multi-task count |

**Important Note**: Using **variable-length sequences (Jagged)** is typical of real recommendation scenarios where user history lengths vary greatly (from tens to thousands), making Workload Balancer optimization particularly important.

---

### Exp 0: Baseline

#### Why?
Establish performance baseline as comparison reference for all subsequent optimizations.

**Key Design**:
- ✅ Use **DynamicEmb** (production environment requirement)
- ❌ **No caching** (caching as independent optimization point)

#### How?
```yaml
Configuration:
- HSTU Attention: Triton (PyTorch fallback)
- Kernel Fusion: Enabled (fuse_norm_mul_dropout is Baseline standard)
- Recompute: Disabled
- Embedding: DynamicEmb WITHOUT caching  # 🔥 Key
- Pipeline: None
- Parallelism: Data Parallel only 
```

#### Key Configuration
```python
# Baseline configuration
NetworkArgs.kernel_backend = 'triton'  # or pytorch
# fuse_norm_mul_dropout enabled by default (Baseline standard)
NetworkArgs.recompute_input_layernorm = False
NetworkArgs.recompute_input_silu = False

# 🔥 Use DynamicEmb, but without caching
item_embedding/DynamicEmbeddingArgs.item_vocab_size_or_capacity = 50000000
item_embedding/DynamicEmbeddingArgs.item_vocab_gpu_capacity_ratio = 1.0  # Maximize GPU
item_embedding/DynamicEmbeddingArgs.caching = False  # No caching
item_embedding/DynamicEmbeddingArgs.evict_strategy = 'lru'

# No pipeline
TrainerArgs.pipeline_type = 'none'
TrainerArgs.enable_balanced_shuffler = False

# No TP/SP
TensorModelParallelArgs.tensor_model_parallel_size = 1
```

#### Expected Results
- Throughput: **Baseline performance** (denoted as 1.0x)
- Memory: May be high due to no recompute
- Serves as comparison baseline

---

### Exp 1: +CUTLASS Attention

#### Why?
CUTLASS implementation of HSTU attention has significant performance advantages over Triton:
- Better register allocation
- H100-specific optimizations
- Reduced shared memory contention

#### How?
```python
# Only modify this item
NetworkArgs.kernel_backend = 'cutlass'
```

#### Expected Results
- Throughput improvement: **1.2-1.4x** (vs Exp 0)
- Memory usage: Basically unchanged
- Attention computation time reduced 30-40%

---

### Exp 2: CUTLASS + Larger Batch (Validate CUTLASS Benefits)

#### Why?
Validate CUTLASS attention performance benefits by increasing batch size to leverage CUTLASS's efficient implementation.

**Note**: `fuse_norm_mul_dropout` is **baseline default enabled** feature, not a separate optimization point.

#### How?
```python
NetworkArgs.kernel_backend = 'cutlass'
# fuse_norm_mul_dropout enabled by default (Baseline standard)
```

#### Expected Results
- Throughput improvement: **1.15-1.2x** (vs Exp 1)
- Cumulative improvement: **1.38-1.68x** (vs Exp 0)
- Forward/backward computation time reduced 10-15% each

---

### Exp 3: +Selective Recompute

#### Why?
With long sequence lengths, activations consume significant memory:
- Trade compute time for memory space via recompute
- Allow larger batch size or longer sequences

#### How?
```python
NetworkArgs.kernel_backend = 'cutlass'
# fuse_norm_mul_dropout enabled by default
NetworkArgs.recompute_input_layernorm = True  # Selective recompute
NetworkArgs.recompute_input_silu = False      # Adjust as needed
```

#### Expected Results
- Memory usage: **Reduced 20-30%**
- Throughput: May slightly decrease 5-10% (recompute overhead)
- **Key benefit**: Support larger batch size → Overall throughput improvement

**Recommended Strategy**:
- If memory sufficient: Disable recompute
- If memory limited: Enable `recompute_input_layernorm`

---

### Exp 3.5: +Workload Balancer (🆕 Important Optimization)

#### Why?
In **variable-length sequence** scenarios, computation workload varies greatly between samples:

**Problem Scenario**:
```
GPU 0: [seq_len=4000, 3800, 3900, ...]  ← High workload, becomes bottleneck
GPU 1: [seq_len=500, 600, 400, ...]     ← Low workload, waits for GPU 0
...
```

Due to HSTU attention's O(n²) complexity, sequence length differences cause severe **load imbalance**:
- Simple sample-count distribution: Actual computation varies 10x+ between GPUs
- All-Reduce synchronization: All GPUs wait for the slowest one
- Resource waste: Fast GPUs idle waiting

**Workload Balancer Solution**:
- Calculate each sample's **workload** based on sequence length (considering attention complexity)
- Dynamically redistribute samples across GPUs to balance total workload per GPU
- Reduce inter-GPU wait time

#### How?
```python
# Enable workload balancer
TrainerArgs.enable_balanced_shuffler = True

# Internal implementation:
# - Calculate each sample's workload: O(seqlen²)
# - Use greedy algorithm to distribute samples to GPUs
# - Ensure total workload per GPU is as balanced as possible
```

#### Expected Results
- **Throughput improvement: 1.3-1.8x** (variable-length sequence scenarios)
  - Larger sequence length variance → Greater improvement
  - Minimal improvement for fixed-length sequences (but harmless)
- GPU utilization improvement: 60-70% → 85-95%
- Communication efficiency improvement: Reduced wait time

**Key Scenarios**:
- ✅ Real recommendation data (varying user history lengths)
- ✅ Multi-GPU training (DP ≥ 4)
- ✅ Variable-length sequences (`is_jagged=True`)
- ❌ Fixed-length sequences (limited improvement)

**Performance Comparison Example**:
```
Without Balancer (sequence lengths: [4000, 3800, ..., 500, 400]):
GPU 0: 100% | ███████████████████████████ | 250ms
GPU 1: 100% | ███████████████████████████ | 245ms
GPU 2:  45% | ████████████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ | 110ms (waiting)
GPU 3:  40% | ███████████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ | 95ms  (waiting)
→ Actual throughput limited by GPU 0, GPU 2/3 waste 50%+ compute

With Balancer (after redistribution):
GPU 0: 100% | ███████████████████████████ | 180ms
GPU 1: 100% | ███████████████████████████ | 175ms
GPU 2: 100% | ███████████████████████████ | 182ms
GPU 3: 100% | ███████████████████████████ | 178ms
→ All GPUs fully utilized, throughput improved ~1.4x
```

---

### Exp 4: +DynamicEmb Caching 🔥

#### Why?
Baseline already uses DynamicEmb, but **without caching**:
- Exp 0-3.5: DynamicEmb without caching (GPU-only or CPU-only)
- Problem: Either limited by GPU memory or slow CPU access

**DynamicEmb Caching Advantages**:
- **CPU Storage**: Massive embeddings (not limited by GPU memory)
- **GPU Cache**: Hot embeddings (low latency access)
- **Auto Eviction**: LRU/LFU strategy manages cache

This is **critical for production environments**!

#### How?
```python
# Enable DynamicEmb Caching (vs Baseline's caching=False)
item_embedding/DynamicEmbeddingArgs.item_vocab_size_or_capacity = 50000000
item_embedding/DynamicEmbeddingArgs.item_vocab_gpu_capacity_ratio = 0.1  # 10% in GPU cache
item_embedding/DynamicEmbeddingArgs.evict_strategy = 'lru'
item_embedding/DynamicEmbeddingArgs.caching = True  # 🔥 Enable caching!

# Keep other configs same as Exp 3.5
TrainerArgs.enable_balanced_shuffler = True  # Keep workload balancer
```

#### Expected Results
- GPU memory usage: **Reduced 70-80%** (vs Exp 0's GPU-only)
- Cache hit rate: **70-85%** (depends on data locality)
- Throughput: Slight decrease **5-10%** (CPU access on cache miss)
- **Key Benefits**:
  - Support **ultra-large-scale** embedding tables (billions of parameters)
  - Not limited by GPU memory
  - Maintain high performance (hot data in GPU cache)

---

### Exp 5: +LRU/LFU Eviction

#### Why?
Optimize DynamicEmb's eviction strategy:
- LRU: Suitable for scenarios with clear temporal locality
- LFU: Suitable for scenarios with frequency locality
- Improve GPU cache hit rate

#### How?
```python
# Test different eviction strategies
item_embedding/DynamicEmbeddingArgs.evict_strategy = 'lfu'  # Switch to LFU
item_embedding/DynamicEmbeddingArgs.score_strategy = 'frequency'
```

#### Expected Results
- GPU cache hit rate improvement: **10-20%**
- Lookup latency reduction: **5-10%**
- Throughput improvement: **1.05-1.1x** (vs Exp 4)

**Comparison Experiment**: Test LRU and LFU separately, choose better strategy.

---

### Exp 6: +Pipeline Prefetch

#### Why?
Training pipeline performance bottlenecks:
1. Embedding lookup (I/O bound)
2. Forward/Backward (Compute bound)

Overlap embedding lookup with compute via prefetch:
```
Traditional flow:
[Emb Lookup Batch N] → [Forward/Backward Batch N] → [Emb Lookup Batch N+1] → ...

Pipeline flow:
[Emb Lookup Batch N+1]  ← Execute ahead
        ↓
[Forward/Backward Batch N]
```

#### How?
```python
TrainerArgs.pipeline_type = 'prefetch'

# DynamicEmb needs caching enabled
item_embedding/DynamicEmbeddingArgs.caching = True
```

#### Expected Results
- Embedding lookup time **hidden** → Basically not on critical path
- Throughput improvement: **1.2-1.5x** (vs Exp 5)
- **Key Benefit**: Significantly improve end-to-end training efficiency

---

### Exp 7: +Tensor Parallel (TP)

#### Why?
When single GPU cannot fit large model or needs further acceleration:
- TP splits model parameters across multiple GPUs
- Suitable for accelerating dense computation (HSTU layers)

**TP Strategy**:
- Split HSTU's QKV linear layers and MLP
- Embedding layer still uses Data Parallel

#### How?
```python
# Enable TP=2 (2-way TP within each node)
TensorModelParallelArgs.tensor_model_parallel_size = 2

# Total parallelism: TP=2, DP=8
# 16 GPUs = 2 TP × 8 DP
```

#### Expected Results
- Per-GPU model memory: **Reduced ~50%**
- Throughput: May slightly decrease or stay flat due to communication overhead
- **Key Benefit**: Support larger models or batch sizes

---

### Exp 8: +Sequence Parallel (SP)

#### Why?
With long sequences, activation memory is huge:
- Sequence length = 4096, hidden = 1024 → 4KB × batch size per layer
- Sequence Parallel splits sequence dimension across GPUs

**SP Mechanism**:
```
Original: Each GPU processes full sequence
[GPU 0]: [Token 0-4095]
[GPU 1]: [Token 0-4095]

SP: Each GPU processes part of sequence
[GPU 0]: [Token 0-2047]
[GPU 1]: [Token 2048-4095]
```

#### How?
```python
# SP used with TP, configured via TensorModelParallelArgs
TensorModelParallelArgs.tensor_model_parallel_size = 2  # SP available with TP=2
# SP and TP share the same group
```

#### Expected Results
- Activation memory: **Reduced ~50%** (proportional to TP size)
- Throughput: Slight decrease (added all-gather/reduce-scatter communication)
- **Key Benefit**: Support ultra-long sequences (8K+)

---

### Experiment Summary Table

| Exp | Configuration | Relative Throughput | Cumulative Speedup | GPU Memory | Key Optimization |
|-----|--------------|--------------------|--------------------|-----------|------------------|
| 0 | Baseline (DynamicEmb no cache + Kernel Fusion) | 1.0x | 1.0x | 100% | Baseline (built-in Kernel Fusion) |
| 1 | +CUTLASS | 1.3x | 1.3x | 100% | Attention optimization |
| 2 | CUTLASS benefit validation | 1.2x | 1.56x | 100% | Confirm CUTLASS effect |
| 3 | +Recompute | 0.95x | 1.48x | 70% | Memory optimization |
| 3.5 | +Workload Balancer | 1.5x | **2.22x** | 70% | **Load balancing** 🔥 |
| 4 | +DynamicEmb **Caching** | 0.95x | **2.11x** | **30%** | **CPU+GPU hybrid** 🔥 |
| 5 | +LFU | 1.1x | 2.32x | 30% | Cache optimization |
| 6 | +Pipeline | 1.3x | 3.02x | 30% | I/O hiding |
| 7 | +TP | 1.0x | 3.02x | 20% | Model parallel |
| 8 | +SP | 0.95x | 2.87x | 15% | Sequence parallel |

**Expected Overall Improvement**: **2.9-3.2x end-to-end training speedup** + **85% memory savings**

**Key Notes**:
- **Baseline includes Kernel Fusion** (`fuse_norm_mul_dropout`), this is default feature
- **Workload Balancer** (Exp 3.5): **1.5x** improvement, most significant single optimization
- **DynamicEmb Caching** (Exp 4): 5% throughput decrease, but **40% memory savings**, supports ultra-large-scale embedding

---

## 4. Configuration Files and Scripts

### 4.1 Benchmark Configuration Files

8 gin configuration files will be created, each corresponding to one experiment.

#### File Structure
```
examples/hstu/training/configs/
├── h100_16gpu_exp0_baseline.gin
├── h100_16gpu_exp1_cutlass.gin
├── h100_16gpu_exp2_fusion.gin
├── h100_16gpu_exp3_recompute.gin
├── h100_16gpu_exp4_dynamicemb.gin
├── h100_16gpu_exp5_lfu.gin
├── h100_16gpu_exp6_pipeline.gin
├── h100_16gpu_exp7_tp.gin
└── h100_16gpu_exp8_sp.gin
```

#### Base Configuration Template

```python
# ========== h100_16gpu_base.gin ==========
# Common configuration for all experiments

# Trainer configuration
TrainerArgs.train_batch_size = 32
TrainerArgs.eval_batch_size = 32
TrainerArgs.log_interval = 10
TrainerArgs.eval_interval = 400
TrainerArgs.max_train_iters = 1000
TrainerArgs.max_eval_iters = 50
TrainerArgs.seed = 1234

# Enable profiling
TrainerArgs.profile = True
TrainerArgs.profile_step_start = 50
TrainerArgs.profile_step_end = 100

# Dataset configuration
item_and_action_feature/FeatureArgs.feature_names = ['item', 'action']
item_and_action_feature/FeatureArgs.max_sequence_length = 1024
item_and_action_feature/FeatureArgs.is_jagged = True  # Variable-length sequences

BenchmarkDatasetArgs.feature_args = [
    @item_and_action_feature/FeatureArgs(),
]
BenchmarkDatasetArgs.item_feature_name = 'item'
BenchmarkDatasetArgs.contextual_feature_names = []
BenchmarkDatasetArgs.action_feature_name = 'action'
BenchmarkDatasetArgs.max_num_candidates = 0

# Network configuration
NetworkArgs.item_embedding_dim = 128
NetworkArgs.contextual_embedding_dim = 256
NetworkArgs.num_layers = 8
NetworkArgs.num_attention_heads = 4
NetworkArgs.hidden_size = 1024
NetworkArgs.kv_channels = 256
# Note: fuse_norm_mul_dropout is baseline default feature, no explicit config needed

# Ranking Head
RankingArgs.prediction_head_arch = [512, 8]
RankingArgs.prediction_head_bias = True
RankingArgs.num_tasks = 8

# Optimizer
OptimizerArgs.optimizer_str = 'adam'
OptimizerArgs.learning_rate = 1e-3
# Note: weight_decay configured elsewhere, not OptimizerArgs parameter
```

### 4.2 Run Scripts

For detailed usage instructions, see [BENCHMARK_SCRIPTS_USAGE.md](./BENCHMARK_SCRIPTS_USAGE.md)

#### Script File Structure

```
examples/hstu/training/benchmark/
├── experiments.txt                    # Experiment list file
├── run_single_experiment_local.sh     # Single node single experiment run
├── run_all_experiments_local.sh       # Single node batch run
├── submit_all_experiments_slurm.sh    # SLURM batch submission
├── slurm_job.sub                      # SLURM job script
└── results/                           # Output directory
```

#### Experiment List File (experiments.txt)

```
# Format: exp_name,config_file_path
exp0_baseline,examples/hstu/training/configs/h100_16gpu_exp0_baseline.gin
exp1_cutlass,examples/hstu/training/configs/h100_16gpu_exp1_cutlass.gin
exp2_fusion,examples/hstu/training/configs/h100_16gpu_exp2_fusion.gin
exp3_recompute,examples/hstu/training/configs/h100_16gpu_exp3_recompute.gin
exp4_dynamicemb,examples/hstu/training/configs/h100_16gpu_exp4_dynamicemb.gin
exp5_lfu,examples/hstu/training/configs/h100_16gpu_exp5_lfu.gin
exp6_pipeline,examples/hstu/training/configs/h100_16gpu_exp6_pipeline.gin
exp7_tp,examples/hstu/training/configs/h100_16gpu_exp7_tp.gin
exp8_full,examples/hstu/training/configs/h100_16gpu_exp8_full.gin
```

#### Single Node Local Run

```bash
# Run single experiment
./run_single_experiment_local.sh exp0_baseline \
    --config=examples/hstu/training/configs/h100_16gpu_exp0_baseline.gin \
    --nproc=8

# Batch run all experiments
./run_all_experiments_local.sh --exp-file=experiments.txt --nproc=8
```

#### SLURM Cluster Submission

```bash
# Submit all experiments (dual node 16 GPU)
./submit_all_experiments_slurm.sh \
    --exp-file=experiments.txt \
    --nodes=2 \
    --ranks-per-node=8

# Enable nsys profile sampling
./submit_all_experiments_slurm.sh \
    --exp-file=experiments.txt \
    --nodes=2 \
    --ranks-per-node=8 \
    --nsys

# Sequential execution + nsys (run next after previous completes)
./submit_all_experiments_slurm.sh \
    --exp-file=experiments.txt \
    --nodes=2 \
    --ranks-per-node=8 \
    --nsys \
    --sequential

# Test mode (print commands only, don't submit)
./submit_all_experiments_slurm.sh \
    --exp-file=experiments.txt \
    --nsys \
    --dry-run
```

#### Run Only Some Experiments

```bash
# Create custom experiment list
cat > quick_test.txt << EOF
exp0_baseline,examples/hstu/training/configs/h100_16gpu_exp0_baseline.gin
exp8_full,examples/hstu/training/configs/h100_16gpu_exp8_full.gin
EOF

# Local run
./run_all_experiments_local.sh --exp-file=quick_test.txt --nproc=8

# Or submit to SLURM
./submit_all_experiments_slurm.sh --exp-file=quick_test.txt --nsys
```

#### Output Files

Logs and nsys profile files saved in `results/` directory, filenames include experiment name:

```
results/
├── {exp_name}_{timestamp}.log                    # Training log
├── {exp_name}_{jobid}.out                        # SLURM stdout
├── {exp_name}_{jobid}.err                        # SLURM stderr
└── nsys_profiles/
    └── {exp_name}_{timestamp}_job{id}_node{N}_rank{R}_{host}.nsys-rep
```

### 4.3 Performance Monitoring Script

```python
#!/usr/bin/env python3
# monitor_training.py
# Real-time monitoring of training performance metrics

import time
import subprocess
import json
from pathlib import Path

def get_gpu_metrics():
    """Get GPU utilization and memory"""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    lines = result.stdout.strip().split('\n')
    metrics = []
    for line in lines:
        util, mem_used, mem_total = line.split(',')
        metrics.append({
            'gpu_util': float(util),
            'mem_used_mb': float(mem_used),
            'mem_total_mb': float(mem_total),
            'mem_util': float(mem_used) / float(mem_total) * 100
        })
    return metrics

def monitor(output_file, interval=5):
    """Continuously monitor and record metrics"""
    with open(output_file, 'w') as f:
        f.write('timestamp,gpu_id,gpu_util,mem_used_mb,mem_total_mb,mem_util\n')
        
        while True:
            timestamp = time.time()
            metrics = get_gpu_metrics()
            
            for gpu_id, metric in enumerate(metrics):
                f.write(f"{timestamp},{gpu_id},"
                       f"{metric['gpu_util']},"
                       f"{metric['mem_used_mb']},"
                       f"{metric['mem_total_mb']},"
                       f"{metric['mem_util']}\n")
            f.flush()
            time.sleep(interval)

if __name__ == '__main__':
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else 'gpu_metrics.csv'
    monitor(output)
```

---

## 5. Execution Plan

### 5.1 Preparation Phase (1-2 days)

#### Step 1: Environment Setup
```bash
# 1. Clone code
git clone https://github.com/NVIDIA/recsys-examples.git
cd recsys-examples

# 2. Build Docker image
docker build -f docker/Dockerfile -t recsys-examples:h100 .

# 3. Start containers on both nodes
# Node 0:
docker run --gpus all --ipc=host --network=host \
    -v $PWD:/workspace \
    recsys-examples:h100 bash

# Node 1: Same as above

# 4. Verify GPUs
nvidia-smi
```

#### Step 2: Data Preparation
```bash
# Generate benchmark data (synthetic data, power-law distribution)
cd examples/hstu/training/benchmark
python generate_benchmark_data.py \
    --num_users 100000 \
    --num_items 50000000 \
    --avg_seq_length 4096 \
    --distribution power_law \
    --output_dir ./data/h100_benchmark
```

#### Step 3: Configuration Validation
```bash
# Single GPU run to verify configuration correctness
python examples/hstu/training/pretrain_gr_ranking.py \
    --gin-config-file examples/hstu/training/configs/h100_16gpu_exp0_baseline.gin
```

### 5.2 Experiment Execution (2-3 days)

#### Execution Time Estimate

| Phase | Time | Description |
|-------|------|-------------|
| Single experiment | 1-2 hours | 1000 iters + evaluation |
| 9 experiments | 9-18 hours | Sequential execution (including exp 3.5) |
| Result analysis | 4 hours | Data processing and visualization |
| Total | ~1.5 days | Excluding reruns |

#### Execution Strategies

**Strategy A: SLURM Batch Submission (Recommended)**
```bash
# Submit all experiments to SLURM cluster
./submit_all_experiments_slurm.sh \
    --exp-file=experiments.txt \
    --nodes=2 \
    --ranks-per-node=8 \
    --nsys \
    --sequential
```

**Strategy B: Single Node Local Run**
```bash
# Single node 8 GPU run all experiments
./run_all_experiments_local.sh --exp-file=experiments.txt --nproc=8
```

**Strategy C: Batch Execution**
```bash
# Create batch experiment lists
# First batch: Baseline + Kernel optimizations
cat > batch1.txt << EOF
exp0_baseline,examples/hstu/training/configs/h100_16gpu_exp0_baseline.gin
exp1_cutlass,examples/hstu/training/configs/h100_16gpu_exp1_cutlass.gin
exp2_fusion,examples/hstu/training/configs/h100_16gpu_exp2_fusion.gin
exp3_recompute,examples/hstu/training/configs/h100_16gpu_exp3_recompute.gin
EOF

# Second batch: Embedding + Pipeline optimizations
cat > batch2.txt << EOF
exp4_dynamicemb,examples/hstu/training/configs/h100_16gpu_exp4_dynamicemb.gin
exp5_lfu,examples/hstu/training/configs/h100_16gpu_exp5_lfu.gin
exp6_pipeline,examples/hstu/training/configs/h100_16gpu_exp6_pipeline.gin
EOF

# Third batch: Parallelism optimizations
cat > batch3.txt << EOF
exp7_tp,examples/hstu/training/configs/h100_16gpu_exp7_tp.gin
exp8_full,examples/hstu/training/configs/h100_16gpu_exp8_full.gin
EOF

# Submit batches
./submit_all_experiments_slurm.sh --exp-file=batch1.txt --nsys
./submit_all_experiments_slurm.sh --exp-file=batch2.txt --nsys
./submit_all_experiments_slurm.sh --exp-file=batch3.txt --nsys
```

#### Real-time Monitoring

During execution, start monitoring scripts:
```bash
# Terminal 1: View SLURM job status
watch -n 5 'squeue -u $USER'

# Terminal 2: Monitor GPUs
./monitor_training.py results/gpu_metrics.csv
```

### 5.3 Troubleshooting

**Common Issues**:

1. **OOM (Out of Memory)**
   - Check `train_batch_size`, reduce appropriately
   - Enable `recompute` or `gradient_checkpointing`
   
2. **NCCL Timeout**
   - Check network connection: `ping <other_node>`
   - Increase timeout: `export NCCL_TIMEOUT=1800`

3. **Abnormally Low Performance**
   - Check CPU frequency: `cpupower frequency-info`
   - Check GPU clock: `nvidia-smi -q -d CLOCK`

---

## 6. Result Visualization

### 6.1 Data Collection

After each experiment, collect the following data:

```python
# Extract key metrics from logs
{
    "exp_name": "exp6_pipeline",
    "config": {...},
    "metrics": {
        "throughput_samples_per_sec": 1234.5,
        "avg_iteration_time_ms": 256.3,
        "gpu_memory_used_gb": 45.2,
        "gpu_memory_peak_gb": 52.1,
        "loss": 0.234,
        "eval_auc": 0.756,
        "communication_time_ms": 12.5,
        "embedding_lookup_time_ms": 45.3,
        "forward_time_ms": 123.4,
        "backward_time_ms": 156.7
    }
}
```

### 6.2 Visualization Script

```python
#!/usr/bin/env python3
# visualize_results.py

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(results_dir):
    """Load all experiment results"""
    results = []
    for exp_file in sorted(Path(results_dir).glob('exp*_metrics.json')):
        with open(exp_file) as f:
            results.append(json.load(f))
    return results

def plot_throughput_comparison(results, output_path):
    """Plot throughput comparison chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    exp_names = [r['exp_name'] for r in results]
    throughputs = [r['metrics']['throughput_samples_per_sec'] for r in results]
    baseline = throughputs[0]
    speedups = [t / baseline for t in throughputs]
    
    # Bar chart
    x = np.arange(len(exp_names))
    bars = ax.bar(x, speedups, color=['red'] + ['steelblue'] * (len(exp_names)-1))
    bars[0].set_color('gray')  # Baseline in gray
    bars[-1].set_color('green')  # Final version in green
    
    # Add value labels
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Speedup (relative to baseline)', fontsize=12)
    ax.set_title('Training Throughput: Progressive Optimization', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")

def plot_memory_usage(results, output_path):
    """Plot memory usage comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    exp_names = [r['exp_name'] for r in results]
    memory_used = [r['metrics']['gpu_memory_used_gb'] for r in results]
    memory_peak = [r['metrics']['gpu_memory_peak_gb'] for r in results]
    
    x = np.arange(len(exp_names))
    width = 0.35
    
    ax.bar(x - width/2, memory_used, width, label='Average', color='steelblue')
    ax.bar(x + width/2, memory_peak, width, label='Peak', color='coral')
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('GPU Memory (GB)', fontsize=12)
    ax.set_title('GPU Memory Usage: Progressive Optimization', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")

def plot_breakdown(results, output_path):
    """Plot time breakdown comparison"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    exp_names = [r['exp_name'] for r in results]
    categories = ['embedding_lookup_time_ms', 'forward_time_ms', 
                  'backward_time_ms', 'communication_time_ms']
    labels = ['Embedding Lookup', 'Forward', 'Backward', 'Communication']
    
    data = []
    for cat in categories:
        data.append([r['metrics'].get(cat, 0) for r in results])
    
    data = np.array(data)
    
    # Stacked bar chart
    x = np.arange(len(exp_names))
    bottom = np.zeros(len(exp_names))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for i, (d, label, color) in enumerate(zip(data, labels, colors)):
        ax.bar(x, d, bottom=bottom, label=label, color=color)
        bottom += d
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Time (ms per iteration)', fontsize=12)
    ax.set_title('Training Time Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")

def generate_summary_table(results, output_path):
    """Generate summary table"""
    import pandas as pd
    
    data = []
    baseline_throughput = results[0]['metrics']['throughput_samples_per_sec']
    
    for r in results:
        m = r['metrics']
        data.append({
            'Experiment': r['exp_name'],
            'Throughput (samples/s)': f"{m['throughput_samples_per_sec']:.1f}",
            'Speedup': f"{m['throughput_samples_per_sec']/baseline_throughput:.2f}x",
            'Mem Used (GB)': f"{m['gpu_memory_used_gb']:.1f}",
            'Mem Peak (GB)': f"{m['gpu_memory_peak_gb']:.1f}",
            'Loss': f"{m.get('loss', 0):.4f}",
            'Eval AUC': f"{m.get('eval_auc', 0):.4f}"
        })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    df.to_csv(output_path.replace('.png', '.csv'), index=False)
    
    # Save as image
    fig, ax = plt.subplots(figsize=(14, len(results) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.15] * len(df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Header row style
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Baseline row highlight
    for i in range(len(df.columns)):
        table[(1, i)].set_facecolor('#FFE5E5')
    
    # Final version row highlight
    for i in range(len(df.columns)):
        table[(len(results), i)].set_facecolor('#E5FFE5')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

def main():
    results_dir = 'benchmark_results'
    output_dir = 'benchmark_results/figures'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = load_results(results_dir)
    
    print("Generating visualizations...")
    plot_throughput_comparison(results, f'{output_dir}/throughput_comparison.png')
    plot_memory_usage(results, f'{output_dir}/memory_usage.png')
    plot_breakdown(results, f'{output_dir}/time_breakdown.png')
    generate_summary_table(results, f'{output_dir}/summary_table.png')
    
    print("\nAll visualizations generated!")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    main()
```

### 6.3 Expected Visualization Results

#### Chart 1: Throughput Comparison (Core Chart)
```
┌─────────────────────────────────────────────┐
│  Training Throughput: Progressive Optimization  │
├─────────────────────────────────────────────┤
│                                              │
│  3.0x ┤                                  ▓▓▓│ 2.8x
│       │                              ▓▓▓    │
│  2.5x ┤                          ▓▓▓        │
│       │                      ▓▓▓            │
│  2.0x ┤                  ▓▓▓                │
│       │              ▓▓▓                    │
│  1.5x ┤          ▓▓▓                        │
│       │      ▓▓▓                            │
│  1.0x ┼──▓▓──────────────────────────────── │ Baseline
│       │  ▓▓                                 │
│       └──┬───┬───┬───┬───┬───┬───┬───┬────│
│         E0  E1  E2  E3  E4  E5  E6  E7  E8 │
└─────────────────────────────────────────────┘
```

#### Chart 2: Memory Usage Comparison
```
Shows Average and Peak GPU memory for each experiment
Highlights memory savings from DynamicEmb and Recompute
```

#### Chart 3: Time Breakdown Stacked Chart
```
Shows time breakdown for each experiment (Embedding, Forward, Backward, Comm)
Pipeline optimization significantly reduces Embedding time
```

---

## 7. Demo Recommendations

### 7.1 Demo Structure (Recommended 20-30 minutes)

#### Part 1: Background Introduction (5 minutes)
- **Problem Statement**: Challenges in large-scale recommendation system training
  - Massive embeddings (billion-scale)
  - Long sequence modeling (thousands of tokens)
  - Multi-task learning
- **NVIDIA Solution**: recsys-examples HSTU training stack

#### Part 2: Optimization Stack Overview (5 minutes)
- **Three-Layer Optimization Architecture** (show diagram)
  - Embedding layer: DynamicEmb
  - Kernel layer: CUTLASS + Fusion + Recompute
  - Parallelism layer: TP + SP + Pipeline
- **Key Innovations**
  - GPU-CPU hybrid embedding storage
  - Custom HSTU attention kernel
  - Pipeline compute and I/O overlap

#### Part 3: Benchmark Results Showcase (15 minutes)

**Presentation Order**:

1. **Overall Speedup** (Core selling point)
   ```
   Show chart: Throughput Comparison
   Focus: Baseline → Final 2-3x speedup
   ```
   - **Talking points**:
     > "Through systematic optimization, we achieved **2.8x end-to-end training speedup** on H100 16 GPU environment"

2. **Progressive Optimization Breakdown**
   ```
   Explain each optimization's contribution one by one
   ```
   - Exp 1-2: **Kernel Optimization** → 1.5x speedup
     > "Custom CUTLASS attention kernel and operator fusion provide 50% compute speedup"
   
   - Exp 3.5: **Workload Balancer** → 1.5x speedup 🔥
     > "In variable-length sequence scenarios, intelligent load balancing fully utilizes all GPUs, eliminating wait time"
   
   - Exp 4-5: **Embedding Optimization** → 80% memory savings
     > "DynamicEmb lets us support 50M embedding table with 20% memory"
   
   - Exp 6: **Pipeline Optimization** → 1.3x speedup
     > "Prefetch pipeline completely hides embedding lookup, no longer a bottleneck"
   
   - Exp 7-8: **Parallelism Optimization** → Support larger models
     > "TP and SP further reduce memory, supporting 8K+ sequence lengths"

3. **Time Breakdown Analysis**
   ```
   Show chart: Time Breakdown
   ```
   - Compare time breakdown between Baseline and Final
   - Highlight reduced Embedding time after Pipeline optimization

4. **Memory Optimization**
   ```
   Show chart: Memory Usage
   ```
   - Emphasize 85% memory savings
   - Explain how this enables larger scale training

#### Part 4: Customer Value (3 minutes)

**Key Messages**:
1. **Cost Reduction and Efficiency**
   - 2-3x training speedup on same hardware
   - Or achieve same performance with fewer GPUs

2. **Scalability**
   - Support billion-scale embedding tables
   - Support 8K+ sequence lengths

3. **Ease of Use**
   - Out-of-the-box configuration templates
   - Complete documentation and examples

#### Part 5: Q&A (5 minutes)

**Expected Questions**:
1. Q: "Are these optimizations universal? Can they transfer to other models?"
   - A: DynamicEmb and Pipeline are universal; HSTU kernel is specific but approach is reusable

2. Q: "What's the advantage over other frameworks (e.g., HugeCTR)?"
   - A: Better PyTorch ecosystem integration; supports ultra-long sequences

3. Q: "What are the production deployment requirements?"
   - A: CUDA 12.1+, H100/A100, see documentation for details

### 7.2 Demo Tips

#### Visual Design
- **Color Scheme**:
  - Baseline: Gray
  - Intermediate experiments: Blue gradient
  - Final: Green (success color)
  
- **Animation Effects**:
  - Bar chart appears one by one (showing progressive optimization)
  - Number jumping effect (attract attention)

#### Interactive Elements
- **Live Demo** (if conditions permit):
  - Run a mini benchmark (2-3 minutes)
  - Show real-time log output

- **Customer Engagement**:
  - "What scale are your recommendation systems?"
  - "How long does it take to train a model currently?"

### 7.3 Backup Slides

Prepare extra slides for deep-dive questions:

1. **Technical Depth**
   - DynamicEmb's HKV architecture
   - CUTLASS attention implementation details
   - Pipeline prefetch scheduling strategy

2. **Business Cases**
   - E-commerce customer deployment case
   - ROI calculation

3. **Roadmap**
   - Next steps (e.g., MOE support)
   - Community contribution opportunities

---

## 8. Checklist

### Pre-Execution Checklist

#### Environment Preparation
- [ ] H100 dual node 16 GPU available
- [ ] Docker image built
- [ ] Network configured correctly (InfiniBand)
- [ ] Sufficient storage space (> 500GB)

#### Code Preparation
- [ ] Latest code pulled
- [ ] 8 configuration files created
- [ ] Run scripts tested
- [ ] Monitoring scripts working

#### Data Preparation
- [ ] Benchmark data generated
- [ ] Data format validated
- [ ] Data access permissions correct

#### Demo Preparation
- [ ] Slides completed
- [ ] Visualization charts generated
- [ ] Demo environment tested
- [ ] Backup plan ready

### Post-Execution Checklist

#### Result Validation
- [ ] All 8 experiments completed successfully
- [ ] Log files complete
- [ ] Performance metrics reasonable
- [ ] No abnormal errors

#### Visualization
- [ ] All charts generated successfully
- [ ] Data accuracy verified
- [ ] Charts clear and visually appealing

#### Documentation
- [ ] Experiment report written
- [ ] Technical details documented
- [ ] Customer FAQ prepared

---

## 9. Appendix

### 9.1 References

- **Project Repository**: https://github.com/NVIDIA/recsys-examples
- **Issue #121**: https://github.com/NVIDIA/recsys-examples/issues/121
- **HSTU Paper**: Generative Recommenders (Meta)
- **DynamicEmb Documentation**: corelib/dynamicemb/README.md
- **Megatron-Core**: https://github.com/NVIDIA/Megatron-LM

### 9.2 Glossary

| Term | Full Name | Description |
|------|-----------|-------------|
| HSTU | Hierarchical Sequential Transduction Unit | Transformer variant for recommendation scenarios |
| DynamicEmb | Dynamic Embedding | GPU-CPU hybrid embedding storage |
| TP | Tensor Parallel | Tensor parallelism |
| SP | Sequence Parallel | Sequence parallelism |
| DP | Data Parallel | Data parallelism |
| HKV | Hierarchical Key-Value | Hierarchical key-value storage |
| LRU | Least Recently Used | Least recently used |
| LFU | Least Frequently Used | Least frequently used |

### 9.3 Contact

For questions:
- GitHub Issues: https://github.com/NVIDIA/recsys-examples/issues
- NVIDIA Developer Forums: https://forums.developer.nvidia.com/

---

## Summary

This benchmark plan systematically showcases the complete capabilities of the HSTU training optimization stack through **progressive comparison experiments**:

**Core Achievements**:
- ✅ **2-3x end-to-end training speedup**
- ✅ **80-85% GPU memory savings**
- ✅ **Support 50M+ embedding table**
- ✅ **Support 4K+ sequence length**

**Key Advantages**:
1. **Systematic**: Covers Embedding, Kernel, Parallelism full-stack optimization
2. **Realistic**: H100 16 GPU large-scale real scenario
3. **Intuitive**: Progressive comparison, each optimization clearly visible
4. **Reproducible**: Complete configs and scripts, customers can try immediately

With this benchmark, we believe we can fully demonstrate NVIDIA recsys-examples' technical capabilities at the conference, generating strong customer interest!

---

**Document Version**: v1.0  
**Last Updated**: 2026-01-28  
**Author**: NVIDIA RecSys Team
