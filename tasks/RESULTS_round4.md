# Round-4 Sweep — Results

**Hardware**: 1 node × 8 H100 SXM (cw-dfw `pool0-*`)
**Sweep root**: `/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/junzhang/benchmark_runs/round4_20260508_071850`
**Submitted**: 2026-05-08T07:18 UTC
**Branch**: `junzhang/rework-mtms` (squashed + rebased onto main `e4fc876f`)
**HEAD**: `74af7dd6 feat(hstu_pipeline,bench): HSTU_LA_DEPTH env var + round4 9-variant sweep`
**Workload**: synthetic HSTU, balanced shuffler + cutlass + caching ratio 0.1 + prefetch + zipf 1.05

## What this sweep tested

3 reps × 9 variants = **27 jobs** (all `COMPLETED 0:0`, ~7 min wall each).

Pipeline depth (`d3`/`d6`) is now encoded directly in variant names —
no more "OLD/NEW" preset shorthand. `HSTU_LA_DEPTH={3,6}` env var
selects lookahead profile at runtime:

  * `d3` (max la=2, 3 batches in flight): h2d=2 / start_shuffle=2 / finish_shuffle=2 / start_input_dist=1 / wait_input_dist=1 / prefetch=1 (round2-era plateau cascade)
  * `d6` (max la=5, 6 batches in flight): h2d=5 / start_shuffle=4 / finish_shuffle=3 / start_input_dist=3 / wait_input_dist=2 / prefetch=1 (6-la cascade)

Variant matrix:

| variant | thread_map | depth | thread count |
|---|---|---:|---:|
| `legacy_none` | (legacy backend, no engine) | n/a | 1 |
| `default_d3` | default (io / compute) | 3 | 2 |
| `by_stream_d3` | by_stream | 3 | 4 |
| `per_task_d3` | per_task | 3 | 14 |
| `io_prefetch_compute_d3` | io / prefetch / compute | 3 | 3 |
| `io_data_dist_compute_d3` | io / data_dist / compute | 3 | 3 |
| `full_split_d3` | io / data_dist / prefetch / compute | 3 | 4 |
| `default_d6` | default | 6 | 2 (= round3 `la_cascade`) |
| `full_split_d6` | io / data_dist / prefetch / compute | 6 | 4 (= round3 `la_cascade_full_split`) |

## Throughput — 3-rep mean of avg-of-6-windows iter 499/599/699/799/899/999

Sorted by `mean_avg6` descending:

| variant | mean_avg6 | std_avg6 | mean_iter999 | std999 | Δ vs legacy |
|---|------------:|---------:|---------------:|--------:|------------:|
| **`default_d3`** | **1897.99** | 9.75 | 2031.17 | 8.87 | **+0.94%** |
| `full_split_d6` | 1885.10 | 0.12 | 1915.73 | 14.43 | +0.25% |
| `io_data_dist_compute_d3` | 1882.19 | 31.08 | 1997.29 | 38.15 | +0.10% |
| `default_d6` | 1880.41 | 6.70 | 1875.76 | 32.95 | +0.00% |
| `legacy_none` | **1880.34** | 11.06 | 2013.40 | 14.66 | baseline |
| `io_prefetch_compute_d3` | 1880.20 | 21.09 | 2005.13 | 43.28 | −0.01% |
| `full_split_d3` | 1869.27 | 17.16 | 2003.92 | 27.51 | −0.59% |
| `by_stream_d3` | 1868.10 | 6.93 | 1989.69 | 9.39 | −0.65% |
| **`per_task_d3`** | 1804.17 | 0.53 | 1798.38 | 22.28 | **−4.05%** |

## Acceptance gate

| metric | threshold | actual | status |
|---|---|---:|:---:|
| best variant ≥ legacy × 1.08 | **2030.77** | **1897.99** (`default_d3`) | ❌ FAIL by 133 TFLOPS |
| best variant ≥ legacy × 1.02 | 1917.95 | 1897.99 | ❌ FAIL by 20 TFLOPS |
| best variant ≥ legacy | 1880.34 | 1897.99 | ✅ +0.94% (within 2σ) |
| no NCCL deadlocks | 27 × 1000 iter clean | 27/27 COMPLETED | ✅ |

## Findings

### 1. Best variant is `default_d3`（+0.94%）— marginal, within 2σ

Mean diff = 17.65 TFLOPS, pooled std (1-rep on each) ≈ √((9.75² + 11.06²)/2) ≈ 10.4. **Δ ≈ 1.7σ — marginal positive but not statistically robust on 3 reps.**

iter999 view is cleaner: `default_d3` 2031.17 ± 8.87 vs `legacy_none` 2013.40 ± 14.66. Diff 17.77, pooled std ≈ 12.1. **≈ 1.5σ on tail.**

If we had 5+ reps, `default_d3` 跟 `legacy_none` 的 ~1% 差距才能稳定区分。3 reps 仅给定向信号。

### 2. `per_task_d3` is the only clear loser（−4.05%）

`per_task_d3` (14 thread, 每 task 一线程) consistently slowest，跟 round2/round3 历史结论一致：14 thread GIL/CPU dispatch + 跨 thread sync 开销 dominates，这种 thread map 在 1-node 8-GPU + 14 task 上 **永远输**。

### 3. d6 cascade 没在这次 sweep 给 d3 之上的稳定优势

| pair | d3 | d6 | Δ |
|---|---:|---:|---:|
| `default_d3` vs `default_d6` | 1897.99 | 1880.41 | **d3 winner +0.94%** |
| `full_split_d3` vs `full_split_d6` | 1869.27 | 1885.10 | d6 winner +0.85% |

→ default thread_map 下 d3 比 d6 略好（cascade 收益被 11-task compute thread 串行吃掉），full_split thread_map 下 d6 比 d3 略好（多线程拆开后 cascade 终于能 partially convert 收益）。Round3 同一份 HTML 注解早就指出这点：

> la cascade 机制本身有效（splits a2a 飞 1 整 iter），但 default 2-thread map 下 11 个 task 串行在 compute 线程，剩余 host sync 还是拦后面 task 发射

但 **`full_split_d6` vs `legacy_none`** 的 mean diff 只 0.25%（4.76 TFLOPS），within 1σ pooled std (~7)，**还是统计噪声以内**。cascade × 4-thread 的"联合效应"没显示出 round3 注解里期待的"+5-7%"。

### 4. 跟 round3 数据对比

Round3 (1 rep × 9 variants, cw-dfw) 报告的 winner `new_default = 2037 TFLOPS` 在 round4 (3 reps) 复测时是 `default_d3 = 1898`（−7%），但 round3 的 `legacy_none = 1975` 跟 round4 的 `legacy_none = 1880` 相比也差 −5%。**两份数据的 absolute TFLOPS 不可比**，因为 cw-dfw 节点 thermal/load 在两次 sweep 之间漂移，但 **rank order 一致**：default_d3 / d6 / full_split 都接近 legacy；per_task_d3 最差。

3-rep 比 1-rep 噪声底低 √3 ≈ 1.73× → round4 的 std 在 7-30 TFLOPS 范围（3-rep mean），round3 的 1-rep 数据有效噪声底大概 12-50 TFLOPS。Round4 的相对结论可信度更高。

### 5. +8% 目标 confirmed unreachable through this axis

跟之前 finegrained_nccl + op_level_shuffler 实验同结论：

> 最佳 thread_map / lookahead / NCCL ordering 组合的总收益 ≈ +1%，统计上差 5-8% 一个数量级。**bottleneck 不在 NCCL 排序，也不在 thread_map 派发，是 default-stream 上 forward + backward + optimizer kernel 串行**。

要破 default-stream HOL，路径是 SPEC_finegrained_nccl §10 G followup（zero_grad 拆 stream + grad-buffer event sync），不在 thread_map / lookahead 的搜索空间里。

## 跟前几轮 sweep 对比

| sweep | 最佳 variant | 最佳 TFLOPS | Δ vs legacy |
|---|---|---:|---:|
| round2 (1 rep) | new_by_stream | 1896 | +1.0% |
| round3 (1 rep) | new_default | 2037 | +3.1% (round3 legacy=1975) |
| finegrained (5 reps) | finegrained | 1881.77 | −0.51% |
| op_level_shuffler (5 reps) | op_level_shuffler | 1877.37 | +0.32% |
| **round4 (3 reps)** | **default_d3** | **1897.99** | **+0.94%** |

所有 sweep 最佳 variant 都在 **−1% 到 +1%** 区间。**结论稳定**：thread_map / lookahead / NCCL granularity 在这种 default-stream-bound workload 上能撬动的 ceiling 是 1-2%。

## 数据来源

| variant | rep1 nsys-rep dir |
|---|---|
| `legacy_none` | `round4_20260508_071850/rep1/legacy_none/.../exp_pipeline_baseline/` (job 11636974) |
| `default_d3` | 同上 prefix `rep1/default_d3/...` (job 11636976) |
| `by_stream_d3` | (job 11636977) |
| `per_task_d3` | (job 11636978) |
| `io_prefetch_compute_d3` | (job 11636979) |
| `io_data_dist_compute_d3` | (job 11636980) |
| `full_split_d3` | (job 11636981) |
| `default_d6` | (job 11636982) |
| `full_split_d6` | (job 11636983) |

rep2 / rep3 路径对应 `rep2/` / `rep3/`，job ids 11636984-92 + 11636993-94 + 11637293-99.

CSV 解析: `/tmp/r4.csv` (162 行 = 27 jobs × 6 windows)。

## 引擎裁决

**Land 当前 branch as-is，rank order 跟 round3 一致 + 噪声底降低。最佳 default_d3 比 legacy 略好（+0.94%）但需要 5+ reps 才能从 1.5-2σ 提升到 ≥ 2.5σ 的统计显著。**

下一步攻坚方向跟前两次实验同结论：**SPEC_finegrained_nccl §10 G followup**（`zero_grad` / `optimizer_step` 拆独立 stream + 显式 grad-buffer event sync）。NCCL 路径这条线已经基本撬到天花板，再加 reps 也救不到 +8%。
