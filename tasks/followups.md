# Deferred Follow-up Work

Items explicitly cut from a shipped problem's scope, tracked here so
they don't get lost. Each entry names the parent problem, why it was
deferred, and a concrete next-step trigger.

---

## From Problem #1 (schedulable pipeline engine)

### V7 — Explicit event escape hatch
- **Why deferred**: SPEC-level scope audit (2026-04-23). Not needed
  for any realistic T1/T2 loop in scope; `wait_stream` auto-inference
  covers the cases we care about.
- **Trigger to resume**: A concrete schedule where automatic cross-stream
  waits produce incorrect ordering and the user needs manual `Event`
  record/wait.

### V8 — BackwardHookTask
- **Why deferred**: same audit. Autograd is atomic in scope; no
  intra-backward decomposition needed for T1/T2.
- **Trigger to resume**: multi-task loss or per-layer pipeline
  parallelism that requires splitting backward into hooks.

### Problem #2 — FX forward decomposition beyond `ShardedModule`
- **Why deferred**: large surface area, needs its own spec. Scoped
  separately in SPEC.md §0.
- **Status**: Option A picked (HSTU-first adapter using torchrec's
  existing `_rewrite_model`). Full FX generalization is a later spec.

---

## From Problem #3 (multi-threaded executor)

### Real CUDA / NCCL ordering test
- **Why deferred**: codex-rescue round flagged that the NCCL ordering
  test (`test_threaded_executor_nccl_ordering`) only checks Python
  call order, not actual GPU collective launch order. A true test
  needs multi-process + real NCCL collectives.
- **Trigger to resume**: when HSTU pipeline (Problem #2) lands and we
  can hook into a multi-rank test fixture.

### NVTX integration
- **Why deferred**: `Task.nvtx_tag` field exists but the engine doesn't
  emit NVTX ranges. V10 audit left it as a followup.
- **Trigger to resume**: when profiling the real HSTU pipeline shows
  unlabeled CUDA activity we want to attribute.

---

## From Problem #2 (HSTU pipeline adapter — IN PROGRESS)

### Semantic B — autonomous / decoupled-rate data pipeline
- **Why deferred**: Problem #2 ships Semantic A (config-time
  `prefetch_depth`: deep batch_offset queue, still lockstep with
  `progress()`). Semantic B (input_dist runs at a rate independent
  of compute) needs engine-level API extension (`Task(mode="autonomous")`
  or separate `pipe.pump()` call).
- **Trigger to resume**: workload where input_dist latency has high
  variance and a bounded queue would smooth throughput. Design would
  be a followup spec on top of current engine.

### Non-torchrec FX generalization
- **Why deferred**: Option A uses torchrec's `_rewrite_model` directly.
  Generalizing to non-ShardedModule sharding (pure FSDP, custom
  sharders) is out of scope for this slice.
- **Trigger to resume**: when a non-torchrec user wants to adopt the
  engine and has a model whose forward graph can't be rewritten by
  torchrec's FX machinery.

### Non-Megatron DDP support
- **Why deferred**: the HSTU pipeline's `finalize_model_grads`,
  `zero_grad_buffer`, and loss-token normalization are Megatron-
  specific. Generic DDP users don't need these and current design
  hardcodes them in HSTU tasks.
- **Trigger to resume**: a non-Megatron user wants the same
  schedulable pipeline for vanilla PyTorch DDP.

### P2 bootstrap — 1-batch loss + abandoned input_dist awaitable
- **Status**: KNOWN LIMITATION in the v1 HSTU adapter. Codex flagged
  this as HIGH severity (A-HIGH-1 / A-HIGH-2 in the P2 review).
- **Why deferred**: fixing it cleanly requires either (a) pre-populating
  the engine's BatchRing slots + seeding `_pulled` counter (needs
  engine API extension) or (b) running one iteration synchronously
  outside the engine before delegating to engine.progress(). Both
  are doable but non-trivial.
- **Impact**: the new pipeline drops the first batch every training
  run. Not a numerical-correctness bug per-batch, but a dataset-
  coverage bug. The bootstrap's started input_dist collective is
  also abandoned — completes on the wire but its awaitable is
  dropped.
- **Trigger to resume**: before the new pipeline replaces legacy in
  HSTU training scripts.

### P2 prefetch variant — missing end-of-iter start_input_dist + pre-backward barrier
- **Status**: KNOWN LIMITATION. Codex B-HIGH-2 / B-HIGH-3.
- **Details**: legacy `JaggedMegatronPrefetchTrainPipelineSparseDist.
  progress()` (train_pipeline.py:993-999) does
  `current_stream().wait_stream(prefetch_stream)` before backward,
  and at end-of-iter fires `_start_sparse_data_dist(batch_ip2)` on the
  freshly shuffled batch. The task-graph port doesn't capture these.
- **Trigger to resume**: enabling the prefetch variant for HSTU
  training.

### P2 `attach()` / `detach()` lifecycle incomplete
- **Status**: KNOWN GAP. Codex B-MEDIUM-1.
- **Details**: legacy `attach()` re-pipelines the current batch's
  context or resets `_pipelined_modules` so the next `progress()`
  rewrites. The adapter only toggles `_model_attached`.
- **Trigger to resume**: when a user integrates the pipeline with a
  pause/resume flow that calls `detach()` then `attach()`.

### P2 full parity test
- **Status**: NOT YET WRITTEN. Codex F-MEDIUM-1.
- **Required setup**: multi-rank NCCL + torchrec sharded EBC +
  Megatron parallel state initialized. Current smoke tests only
  validate task metadata + schedule construction.
- **Trigger to resume**: when a 2-GPU test fixture is available in
  CI or on a dev machine; also needs the bootstrap-1-batch issue
  fixed first, otherwise parity is intrinsically impossible.

### P2 threaded executor for HSTU
- **Status**: Sequential executor is the current default for
  `HSTUPipeline`. Threaded is opt-in via `threaded=True`.
- **Why**: Codex D-CRITICAL-1 found a race — `start_input_dist`
  (data_dist thread) and `forward` (default thread) both call
  `module.forward.set_context(...)` and `postproc.set_context(...)`
  on shared state. Sequential avoids it.
- **Trigger to resume**: a perf profile showing the single-threaded
  CPU submission is the bottleneck. The fix would be pinning
  start_input_dist / wait_input_dist / forward to the same thread
  via `thread_map=...` or adding a lock around `set_module_context`.
