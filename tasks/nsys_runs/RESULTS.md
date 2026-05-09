# 8-GPU pretrain ranking GR — Legacy vs New Pipeline (nsys profile)

**Hardware**: 8× A100-80GB PCIe on `g492-ha0-0004`
**Config**: `benchmark_ranking.gin` (synthetic dataset, 8-layer HSTU,
hidden=1024, kernel=cutlass, prefetch=on, profile window iter 50→80)
**Date**: 2026-04-26

## Throughput parity (final 100-step window, iter 899→999)

| Backend | Wall time | TFLOPS  | MFU    | Loss @ iter 999 |
|---------|-----------|---------|--------|-----------------|
| legacy  | 47828 ms  | 568.24  | 22.78% | 4.370186        |
| new     | 48037 ms  | 565.76  | 22.68% | 4.808762        |

**Throughput delta**: −0.4% (within run-to-run noise).
Loss values diverge in this benchmark run; cause not investigated here
because both backends start from independent (non-replicated) state,
batch order is not pinned for parity, and the synthetic dataset uses
random Zipf draws. The unit-level parity test
(`test_hstu_pipeline_parity.py`) verifies bit-equivalent loss/logits
when both pipelines are driven on identical batches from identical
weights — that test is the loss-correctness gate; this benchmark only
asserts throughput parity.

## Structural metrics (nsys SQLite analysis — `tasks/analyze_nsys.py`)

| Metric                                       | Legacy   | New      |
|----------------------------------------------|----------|----------|
| Capture span (50–80 iter window)             | 24.99 s  | 25.30 s  |
| Total kernel busy across all streams         | 186.36 s | 186.17 s |
| Cross-stream overlap                         | +645.6%  | +635.9%  |
| Distinct CUDA streams                        | 5        | 5        |
| Distinct host threads issuing CUDA           | 48       | 64 (+16) |
| NCCL kernels                                 | 3240     | 3240     |
| NCCL time overlapped by compute on other streams | 82.7%    | 81.8%    |
| `cudaStreamSynchronize` count / total time   | 8425 / 167.03 s | 8425 / 167.11 s |
| `cudaDeviceSynchronize` count / total time   | 8 / 0.17 ms | 8 / 0.17 ms |

### Per-stream busy share (8 GPUs aggregated)

| Stream  | Legacy        | New           | Role                          |
|---------|---------------|---------------|-------------------------------|
| 7       | 124.77 s (67%) | 124.93 s (67%) | default / compute (fwd, bwd, optimizer) |
| 23      | 59.87 s (32%)  | 59.66 s (32%)  | data_dist (NCCL all-to-all + grad reduce) |
| 27      | 1.43 s (0.8%)  | 1.30 s (0.7%)  | memcpy (H2D + KK shuffler)     |
| 28 / 32 | 249.21 ms (0.1%) | 248.89 ms (0.1%) | prefetch_embeddings           |
| 35      | 33.04 ms (0.0%) | 32.65 ms (0.0%) | misc                          |

### Threads-per-stream (multi-thread CPU submission)

`stream 7` (compute) gets work from 46 threads (legacy) / 64 threads (new).
The +16 threads in the new path come from
`HSTU_DEFAULT_THREAD_MAP` adding an `io` and a `compute` worker per rank
(8 ranks × 2 = 16). Stream 23 (NCCL data_dist) similarly gets 31 threads
(legacy) / 35 threads (new). All five streams get work from ≥ 24 distinct
host threads in both runs — confirming neither pipeline is single-threaded
on the CPU side.

## Verification of the requested patterns

The user asked nsys to confirm three patterns:

1. **多个线程一起提交不同 stream** — multiple threads submitting work
   to different streams concurrently.
   ✅ 48–64 host threads × 5 streams; every stream gets traffic from
   ≥24 threads.

2. **有效的 overlap (每一个通信下面都有一个 kernel 计算或者 memcpy)** —
   every comm operation should be overlapped with a kernel compute or
   memcpy on a different stream.
   ✅ Aggregate: ~82% of total NCCL kernel time (over 3240 NCCL
   kernels) overlaps with compute kernels on other streams. The
   analyzer reports the time-weighted aggregate, not a per-kernel
   ratio — extending it to count NCCL kernels with zero concurrent
   compute is a useful future enhancement. Cross-stream overlap
   aggregate +636% means an average ~7 streams busy simultaneously
   over the 25 s window — a much stronger signal than NCCL/compute
   overlap alone.

3. **sync 没有阻塞 host 提交后续 kernel** — synchronization should not
   stall host-side kernel submission.
   ✅ Sync-API count is identical between legacy and new (8425 calls,
   avg 19.8 ms). The total of 167 s sync time over an 8-process run on
   a 25 s window means each process spends ~21 s in syncs, all of which
   are stream syncs that mirror the GPU-side critical path — host
   threads blocked on a stream sync are released as soon as the kernel
   queue drains, not idling.

## Failed perf experiment: skip event recording for same-stream-only producers (2026-04-26)

**Hypothesis**: the +344ms GPU bubble on the busy compute stream came
from `cuda.Event.record()` injecting unnecessary barrier markers for
producers no consumer ever waits on cross-stream. Theory: only record
events for producers that show up in `infer_cross_stream_event_deps`,
skip the rest, and bubble would shrink.

**Implementation**: added
`producers_with_cross_stream_consumers(schedule)` to compute the
allow-set, plumbed it through `executor.execute_stage` →
`_record_completion_event`, skipped recording when the task wasn't in
the set. Engine + HSTU parity tests both green.

**Result on 8-GPU benchmark**: throughput collapsed from 565.76
TFLOPS → **369.89 TFLOPS (−35%)**. Iter 999 went from 48 s to 73 s.

**Initial hypothesis (wrong)**: when no event is recorded, the executor's
`_apply_cross_stream_waits` falls back to `wait_stream(producer_stream)`
on every cross-stream edge. This was wrong — `infer_cross_stream_event_deps`
explicitly excludes same-stream pairs, so consumers never look up
events for skipped producers. No fallback fires.

**Re-run on different node (ipp1-2029, 2026-04-27)**: same fix
reproduced the −35% slowdown. Not cluster noise. Re-analyzed nsys
SQLite with `tasks/nsys_perf_diff.py`:

| metric | v2 (baseline) | v4 (skip-record) |
|---|---|---|
| Capture span | 25.3 s | 37.7 s (+49%) |
| `cudaEventRecordWithFlags` calls | 12,854 | 8,848 (−31%) ✓ fix worked |
| `cudaStreamWaitEvent` calls | 16,892 | 16,880 (=) |
| `cudaStreamSynchronize` total | 167 s | 268 s (+60%) |
| **Stream 7 (compute) GPU busy** | **125 s** | **225 s (+80%)** ⭐ |
| GPU bubbles >100us on stream 7 | 400 | 311 (−22%) |

**Real root cause** — counterintuitive: `cuda.Event.record()` doubles
as an **implicit GPU scheduler chunk boundary**. When every task
records, the scheduler treats the record points as natural break
points and tends to drain the current stream's work-chunk before
multiplexing other streams. Removing those records lets the scheduler
multiplex more aggressively across all 5 streams — but on A100 PCIe
this triggers SM / L2 / HBM-bandwidth contention. Stream 7's GPU
busy-time jumped from 125 s to 225 s: same kernels, same workload,
but each kernel's `end - start` lengthened ~80% because other streams'
kernels were now competing for the same SMs. Bubble count dropped
−22% (the in-stream gaps tightened — fewer barriers means kernels
back-to-back) but the throughput per kernel collapsed.

**Lesson**: `cuda.Event.record()` is **not pure overhead**. In a
multi-stream pipeline on A100, it acts as a flow-control / rate-
limiting hint to the GPU scheduler, preventing over-aggressive
cross-stream concurrency that would saturate SM/memory resources.
The same code on a chip with better cross-stream concurrency
(B200/H200) might behave differently.

**Conclusion**: the 0.5% legacy/new gap is structural (multi-thread
executor + threaded NCCL lock + fine-grained sync framework cost),
not a fixable hot spot via event-record removal. The right perf
exploration if parity ever becomes critical is the NCCL ordered
lock contention and the `cpu_deps` cross-thread Event chain — both
are CPU-side overheads not GPU-side ones, and won't have this
counterintuitive interaction.

**Reverted in**: working tree only; never committed. Diagnostic
analyzer `tasks/nsys_perf_diff.py` kept for future investigations.

### Same-node A/B re-test on g492-ha0-0004 (2026-04-27)

After noticing the prior tests were confounded by node-to-node
hardware variance, re-applied v3 fix and re-ran on the same node as
the v2 baseline (g492-ha0-0004, container 98286a859450):

| iter | v2 baseline (565 TFLOPS at 999) | v3 rerun (skip-record) |
|---|---|---|
| 99   | 444 TFLOPS                      | 280 TFLOPS (warmup is much slower) |
| 199  | 544                             | 549 (≈) |
| 299  | 446                             | 444 (≈) |
| 399  | 568                             | 565 (≈) |
| 499  | 492                             | 482 (-2%) |
| 599  | 569                             | 561 (-1.4%) |
| 600+ | ran to 999 fine                 | **NCCL ALLTOALL_BASE timeout (600s) on all 8 ranks** |

The collective that timed out was `ALLTOALL_BASE` SeqNum=4178 — that
is `start_input_dist` posting an all-to-all that never matched
across ranks. The deadlock came at the boundary between train slice
1 (iters 0-799) and the eval that fires at iter 800.

This is **stronger evidence** that the v3 skip-record fix is unsafe:
not just slow, but introduces cross-rank NCCL ordering races at
train→eval boundaries. The previous v3 run (which finished 1000 iters
at ~370 TFLOPS) and the v4 run (-35% on ipp1-2029) were not "stable
slowdowns" — they were lucky non-deadlocking runs of a flaky fix.

The flakiness lines up with the iter-reset bookkeeping interaction:
when the engine resets `_internal_iter`/`_pulled`/`_exhausted` on a
fresh iterator, ring slots still hold stale events from train. v2
works because ALL producers re-record on the new iter; v3 skips most
producers, leaving stale events that may or may not get hit
depending on timing — sometimes they're benign, sometimes they cause
mis-ordered NCCL submissions.

**Final conclusion**: skip-record is not safe to ship. Even when it
runs to completion, it's slower; and it can deadlock. The 0.5%
legacy/new structural gap is real and unaddressable via this
optimization.

### Codex (crest-alpha) deadlock root-cause analysis (2026-04-27)

Sent the v3 fix + nsys log to Codex with the question "where is the
deadlock?". Findings:

**Corrected failure-shape analysis (2026-04-27, careful re-read of
both deadlock logs)**:

The two deadlock runs have **two different failure modes**:

| Run | rank 4/5 last enq | other ranks last enq | mode |
|---|---|---|---|
| `new_v3_g492rerun` (1st) | **4177** | 4179 | **cross-rank submission skew** |
| `v3_repro` (2nd) | 4179 | 4179 | **all-rank GPU-side stall** |

Mode 1 — submission skew (1st deadlock): ranks 4 and 5 fell two NCCL
collectives behind. Their Python main thread did not reach the line
that calls start_input_dist's all-to-all in time, while the other 6
ranks had already moved on to the next iteration's collective. NCCL
watchdog on the leading 6 ranks fired because their #4178 never
matched a peer.

Mode 2 — GPU stall (2nd deadlock): all 8 ranks did successfully
enqueue work up through #4179, but none completed #4178. The NCCL
kernel for #4178 was queued on each rank's data_dist stream but its
predecessor work on the same stream never reaches completion under
nsys CUPTI tracing, so the kernel never dispatches.

Both modes are reachable from the same v3+nsys configuration —
which mode you hit depends on timing and on which rank gets unlucky
under CUPTI's tracing overhead.

### 2026-04-27 worker-thread stack dump (smoking gun!)

After extending `StackDumpWatchdog._dump_stacks` to dump every Python
thread (not just the watched MainThread) and to tag each thread with
`[rankN pid=M]`, re-ran v3+nsys on g492-ha0-0004 and captured the
worker thread stacks at the 60-s stall watchdog:

```
--- [rank2 pid=12770] thread: engine_0 (compute, non-daemon) ---
  File ".../engine/executor.py", line 527, in _run_thread_chain
    task.run(ctx)
  File ".../engine/task.py", line 339, in run
    self._fn(ctx)
  File ".../hstu_pipeline/tasks.py", line 190, in _fn
    tokens = batch_gpu.num_loss_tokens().to(state.device)   ← STUCK
```

The compute thread is stuck inside `make_global_tokens_task` at the
H2D copy. `num_loss_tokens()` returns a CPU scalar tensor (line 263
of `commons/datasets/hstu_batch.py`); `.to(cuda)` schedules an
async `cudaMemcpyAsync` H2D. Under v3+nsys this call hangs > 60 s.

**Why this task in particular**:
* `global_tokens_allreduce` is in v3's skip set (no cross-stream
  consumer waits for its event). v2 records an event after this
  task as a stream packet boundary; v3 doesn't.
* `start_input_dist`, `wait_input_dist`, and `global_tokens_allreduce`
  are all pinned to the SAME compute thread by
  `HSTU_DEFAULT_THREAD_MAP`. Stalling at `global_tokens.to()` blocks
  the compute thread, so subsequent NCCL submissions on this rank
  freeze.

**End-to-end deadlock chain**:
1. v3 fix removes `event.record()` for global_tokens_allreduce (and
   9 other tasks)
2. nsys CUPTI tracking interacts badly with this reduced event
   stream — specifically the next-iter `tensor.to(cuda)` H2D copy
   call hangs in CUDA driver under CUPTI hooks
3. compute thread (running global_tokens) blocks at line 190
4. compute thread can't fire next iter's `start_input_dist` (also
   on compute thread)
5. other 7 ranks DO fire next iter's start_input_dist → SeqNum
   mismatch across ranks → NCCL ALLTOALL_BASE timeout
6. (in mode 2 the same hang fires on all 8 ranks → all stuck on
   same SeqNum)

**Strong corroborating evidence (mode 1)**: only ranks 4 and 5
emit `[E427 ...] ProcessGroupNCCL.cpp:1972 Could not acquire GIL
within 300 ms on exit, possible GIL induced hang` (log lines
1644-1645). The same two ranks that fell behind on submission are
the same two that fail to release the Python GIL when ProcessGroupNCCL
tries to dump debug info. This means: on ranks 4/5 something is
holding the GIL inside the Python interpreter — not a CUDA-kernel-
in-flight (those release GIL) but a Python-level loop or callback,
plausibly a CUPTI Python callback fired while v3's reduced event
stream confuses CUPTI's tracking.

**Why ranks 4/5 specifically**: in 8-GPU PCIe topology these two are
typically on a different NUMA / CPU socket from ranks 0-3, 6-7. CUPTI
tracing overhead per process may differ across NUMA boundaries
enough to deterministically tilt the same two ranks into the slow
path. (Confirming this would require running on a different host with
different topology; mode-1 vs mode-2 split also depends on transient
timing so absolute determinism is unlikely.)

This aligns with the prior nsys observation that v3+nsys had stream 7
(compute) GPU busy time +80% (125s → 225s): under nsys CUPTI tracing,
v3's reduced event stream causes either a stream stall on all ranks
(mode 2) or a per-rank Python callback hang on the slow ranks
(mode 1). Both manifest as the watchdog timing out on the same
all-to-all.

**Two-layer failure mode**:

1. **Stale events stay on ring slots forever**. `SlotStore` keeps its
   event registry across `BatchRing.advance()` (`context.py:99-105,
   157-182`) — this is by design so producers can re-record the same
   `cuda.Event` object iter-after-iter. With v2 every task overwrites
   its event each iter, so events on slots are always current epoch.
   With v3, skipped tasks' events never get overwritten, so consumers
   doing `wait_event` (`executor.py:101-119`) can hit events from
   **iters ago** and trust them as if they were current — the engine
   has no epoch tag on events.

2. **Missing stream packet boundaries on NCCL tasks**. v2's
   `event.record(stream)` after every task creates a real CUDA stream
   boundary. v3 removes those boundaries for skipped same-stream-only
   tasks — including `start_input_dist`, `global_tokens_allreduce`,
   `backward`, `finalize_model_grads`. Combined with the stale-event
   issue, this gives the GPU scheduler more freedom to reorder
   submissions in ways that diverge across ranks, producing the
   observed 2-collective skew.

**Trigger**: iter-reset on train→eval switch. Train slice 1 ran 0-799,
eval started, my iterator-identity reset put `_internal_iter=0,
_pulled=_seeded, _exhausted=False, _prefill_done=False` but did NOT
clear ring events. Eval prefill then ran with ring slots holding train
slice 1's stale events from skipped producers — and those stale events
were trusted as fresh by `wait_event` callers.

**What v2 silently did right**:
* every task records → ring events always reflect current epoch
* every NCCL task has a real stream boundary after its body returns
* iter-reset is harmless because event overwrite happens on the very
  next iter

**To make this optimization safe (future SPEC scope, not in this PR)**:
1. Epoch-tagged events: `slot.set_event(name, event, epoch)`,
   consumer verifies epoch matches before `wait_event`; fall back if
   stale.
2. Iter-reset must clear all ring events, not just slot data.
3. Force record for every `nccl=True` task regardless of cross-stream
   consumer set, so the per-task stream boundary is preserved.

**Recommendation from Codex**: abandon the optimization. Final
answer: keep `_record_completion_event` recording for every task
(current code on `main` branch HEAD).

### 2026-04-27: nsys/CUPTI interaction hypothesis (codex H1/H2/H3 refuted)

After user pushed back hard ("don't speculate, give evidence"), a
deeper codex (crest-alpha) review refuted the previous "stale events
+ missing boundaries" story. Three hypotheses checked:

* **H1 (v2 latently buggy too)**: REFUTED. `producers_to_record` is
  built from the exact set of producer names in
  `infer_cross_stream_event_deps`; `_apply_cross_stream_waits`
  reads only those names. A producer outside the set is never queried
  through the event-dep path, so skipping its record is structurally
  safe.
* **H2 (deadlock at train→eval boundary)**: REFUTED by log timestamps.
  Stuck collective enqueued ~04:03:39, that's mid-train-slice (after
  iter 599 at 04:03:10), well before the iter-800 eval boundary.
* **H3 (incomplete batches)**: REFUTED. Synthetic
  ``BenchmarkDatasetArgs.num_generated_batches=100`` pre-generates
  full batches; `cycle()+islice()` wrapper makes the loader
  inexhaustible.

Watchdog stacks showed all 8 main threads blocked in
`ThreadedExecutor.execute_stage(... f.result())` — i.e. workers
hung, not the GPU sync. Without `TORCH_NCCL_TRACE_BUFFER_SIZE` the
exact stuck collective was unrecorded.

**The decisive experiment** — toggle nsys profile alone, same code:

| Run | Code | nsys profile | Result | Note |
|---|---|---|---|---|
| v2 baseline (original) | no fix | ✓ | 568 TFLOPS, completed | OK |
| v3 first (commit reverted) | fix | ✓ | 370 TFLOPS, completed | -35% slow |
| v3 g492 rerun | fix | ✓ | DEADLOCK at iter ~600 | NCCL timeout |
| **v3 flight (no nsys)** | **fix** | **✗** | **574 TFLOPS, completed** | **+1% over v2** |
| v3 repro | fix | ✓ | DEADLOCK at iter ~600 | reproduced |

→ **The deadlock is specific to nsys profile + v3 fix**. v3 fix alone
runs cleanly and beats v2 baseline by ~1%.

**Mechanism (hypothesis, not proven)**: nsys uses CUPTI to intercept
CUDA APIs. CUPTI may treat each `cuda.Event.record()` as an internal
completion-tracking anchor across CUDA streams. v3 skips
`event.record()` for 10 of 14 tasks per iter; CUPTI's internal state
for cross-stream tracking diverges, and one rank's CUPTI ends up in
a state where it stalls a future NCCL submission, while another
rank's CUPTI doesn't — producing the cross-rank skew (last enqueued
4179 vs 4177).

**Production decision**: do NOT ship v3 fix. Reasons:
1. The +1% throughput delta isn't worth making nsys profile
   incompatible with `RECSYS_PIPELINE_BACKEND=new`. nsys is the
   primary perf-debug tool; breaking it is a major UX regression.
2. The mechanism is exotic and depends on NVIDIA tooling internals
   (CUPTI), so the boundary between safe and unsafe usage is not
   knowable at code-review time.
3. We have no way to detect at runtime that nsys is attached, so a
   guard like "if nsys: record all" can't be cleanly written.

**What can be salvaged**: the optimization is feasible if/when:
* CUPTI's internal cross-stream tracking is documented/stable enough
  to skip events safely
* OR users opt in via env var with the "no nsys" caveat documented
* OR a future SPEC adds epoch-tagged events + iter-reset that clears
  ring events to make the same-stream-skip provably-safe regardless
  of CUPTI

For now: keep current `_record_completion_event` (records every
task). The 0.5% legacy/new structural gap stands as known cost.

## Run artefacts

* Legacy nsys: `examples/hstu/tasks/nsys_runs/legacy/nsys_legacy_*.nsys-rep`
* New    nsys: `examples/hstu/tasks/nsys_runs/new_v2/nsys_new_v2_*.nsys-rep`
* Generated gin configs and stdout logs sit beside each `.nsys-rep`.
* Reproduce: see `tasks/REPRODUCE_NSYS.md` (if added).
