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

### P2 bootstrap — 1-batch loss + abandoned input_dist awaitable — RESOLVED
- **Status**: Fixed 2026-04-24 via engine `_seed_first_batch` API +
  idempotent HSTU tasks. The peek batch is now seeded into the
  engine ring as the first real batch and flows through forward
  normally. No dataset-coverage loss for non-caching cases.
- **Engine change**: `SchedulablePipeline._seed_first_batch(slot_contents)`
  pre-populates `ring.at(max_offset)` and skips the next
  auto-pull. Safe to call only before first `progress()`.
- **HSTU changes**: `h2d` / `start_shuffle` / `finish_shuffle` /
  `start_input_dist` tasks made idempotent — they skip if the
  slot they'd write is already populated, so the peek batch's
  pre-processed state (from `_rewrite_model` bootstrap) is
  preserved intact.
- **Residual**: none — the dynamicemb + prefetch combo (separate
  root cause) was subsequently resolved in d81593fc + ce5144d9
  (see entry below).

### dynamicemb + prefetch: pipeline-depth mismatch — RESOLVED
- **Status**: Fixed across d81593fc + ce5144d9 (2026-04-24..25).
- **Root cause** (re-verified): dynamicemb's outstanding-key
  counter is per-table (`BatchedDynamicEmbeddingTables.
  _prefetch_outstanding_keys`, `batched_dynamicemb_tables.py:358`),
  bumped at `prefetch()` and dropped at end of forward. Cache
  capacity is sized for **steady-state ~2 batches outstanding**,
  matching legacy's prefetch progress ordering (forward FIRST
  → prefetch SECOND). The original HSTU schedule declared
  `prefetch_embeddings@1` BEFORE `forward@0`, peaking at ~3
  batches outstanding and overflowing cache.
- **Fix 1** (d81593fc, schedule reorder): moved
  `prefetch_embeddings` AFTER `forward` in `_build_schedule`
  (pipeline.py:245-254). Steady-state outstanding drops from
  peak-3 to peak-2. Mirrors legacy
  `JaggedMegatronPrefetchTrainPipelineSparseDist.progress` at
  `train_pipeline.py:993-997`.
- **Fix 2** (ce5144d9, cache headroom + reset): bumped
  `global_hbm_for_values` 8 MiB → 32 MiB for the `item` dynamic
  table in `test_utils.py:587`, and added
  `reset_dynamicemb_cache_states()` helper (`test_utils.py:42`)
  called after ckpt save→load in the parity test so the loaded
  pipelined model starts with a clean cache state.
- **Verification**: parity test 24 passed / 8 xfailed / 0 failed
  on luna-prod-78-80gb 4×A100 (commit ce5144d9). 50 compare
  steps × 60 batches.

### P2 prefetch variant — missing end-of-iter start_input_dist + pre-backward barrier — RESOLVED
- **Status**: Fixed by d81593fc (2026-04-24). Codex B-HIGH-2 /
  B-HIGH-3.
- **Pre-backward barrier**: `make_backward_task` gained an
  optional `depends_on` kwarg; for the prefetch variant
  `_build_schedule` passes
  `depends_on=("prefetch_embeddings",)` (pipeline.py:259-262).
  Cross-stream depends_on triggers the engine's
  `_apply_cross_stream_waits` to emit
  `wait_stream(prefetch_stream)` on default stream before
  backward — mirrors legacy `train_pipeline.py:996-997`.
- **End-of-iter `start_input_dist(batch_ip2)`**: handled
  declaratively via the task DAG + ring rather than imperatively.
  `start_input_dist@1` runs once per iter on the slot at offset 1
  (= legacy's `_batch_ip2`, two iters ahead of `forward@0`); ring
  advance at end-of-iter shifts that slot down to offset 0 for the
  next iter's forward. Same data-flow as legacy
  `train_pipeline.py:1009-1013`, just expressed as topology
  instead of statement order.
- **Verification**: prefetch + dynamic_emb parity 4/4 PASS in
  ce5144d9 (was xfail before d81593fc).

### P2 `attach()` / `detach()` lifecycle incomplete
- **Status**: KNOWN GAP. Codex B-MEDIUM-1.
- **Details**: legacy `attach()` re-pipelines the current batch's
  context or resets `_pipelined_modules` so the next `progress()`
  rewrites. The adapter only toggles `_model_attached`.
- **Trigger to resume**: when a user integrates the pipeline with a
  pause/resume flow that calls `detach()` then `attach()`.

### P2 full parity test — RESOLVED
- **Status**: Shipped across 9a9a7f10 → ce5144d9 (2026-04-24..25).
  Codex F-MEDIUM-1.
- **Test file**: `examples/hstu/test/test_hstu_pipeline_parity.py`.
  Compares `HSTUPipeline` (Problem #2 adapter) against synchronous
  `JaggedMegatronTrainNonePipeline` baseline on bit-identical
  initial weights (ckpt save→load). Asserts
  `torch.allclose(reporting_loss, atol=1e-4)` and
  `torch.allclose(logits, atol=1e-4)` for 50 consecutive steps.
- **Coverage matrix**: 32 cases =
  `pipeline_type ∈ {prefetch, native}` ×
  `use_dynamic_emb ∈ {True, False}` ×
  `optimizer_type_str ∈ {sgd, adam}` ×
  `max_num_candidates ∈ {0, 10}` ×
  `contextual_feature_names ∈ {[user0,user1], []}`,
  dtype = bf16.
- **Result** (luna-prod-78-80gb 4×A100, 50 steps × 60 batches):
  24 passed / 8 xfailed / 0 failed in 152s. The 8 xfail are
  `use_dynamic_emb=False + adam`, blocked by torchrec FBGEMM
  TBE dropping Adam step state on ckpt save/load (upstream
  limitation, not HSTU — see notes inline at
  test_hstu_pipeline_parity.py:84-90).

### P2 threaded executor for HSTU — RESOLVED
- **Status**: Shipped 2026-04-24 (commit 23414493). Default is now
  `threaded=True` with `HSTU_DEFAULT_THREAD_MAP` pinning all
  set_context call-sites onto the "compute" thread. The original
  Codex D-CRITICAL-1 postproc race is resolved by construction.

### Engine: event-based cross-stream sync (replace wait_stream)
- **Status**: KNOWN PERFORMANCE LIMITATION. Codex review of commit
  23414493 (B-MEDIUM / C-HIGH).
- **What**: `engine/deps.py::infer_cross_stream_waits` and
  `engine/executor.py::_compute_cpu_deps` both match producer ↔
  consumer by **slot name** only (`writers_by_slot_name` at
  deps.py:55; `writers[slot.name] = task.name` at executor.py:115).
  They ignore `DataSlot.batch_offset`, and the resulting GPU sync
  uses **stream-granularity** `wait_stream(producer_stream)` —
  which blocks on ALL pending work on that stream, including the
  current iter's producer run for a FUTURE batch.
- **Concrete HSTU impact**: `forward@0 reads shuffled_batch@0`
  (data written by `finish_shuffle@2` in the PREVIOUS iter).
  Current analysis emits `wait_stream(memcpy)` on the default
  stream before forward. At the moment this wait fires, memcpy
  stream holds both:
    (a) prior iter's `finish_shuffle@2` → needed, correct
    (b) current iter's `finish_shuffle@2` for a NEW batch →
        unneeded but also waited on
  The io/compute overlap collapses because compute blocks on (b).
- **Correctness**: preserved. Ring advance guarantees the slot
  holds the right data; forward still reads correct values. Only
  wall-clock pipeline overlap is degraded.

- **Fix proposal (engine-level change)**:

  1. Each task records a post-execution CUDA event on its stream
     (`torch.cuda.Event` + `event.record(task_stream)`), stored
     in the slot store at the task's `batch_offset` under a
     reserved key like ``__done_event__{task_name}``.

  2. Events travel down the ring via `ring.advance()` alongside
     the data slots (so iter N+1's reader at offset K has access
     to the event recorded in iter N at offset K+1).

  3. `_apply_cross_stream_waits` changes from
     `wait_stream(producer_stream)` → `wait_event(prior_event)`,
     which waits on the SPECIFIC producer operation, not the
     whole stream. Under the hood this is `cudaStreamWaitEvent`
     — finer-grained than `cudaStreamSynchronize`/wait_stream.

  4. For CPU-side (`_compute_cpu_deps`), same idea: per-task
     completion events at the slot's ring-adjacent offset, not
     the current-iter task instance.

  5. DataSlot carries `batch_offset` already, so the producer
     lookup keys on `(slot.name, slot.batch_offset)`. A reader
     of `DataSlot(name, J)` pairs with the producer that wrote
     `DataSlot(name, K)` where K ≥ J; the event is the one
     recorded at offset K in iter (N-(K-J)) — ring advance has
     carried it down to offset J by iter N.

  6. Back-compat: if a task has no `.writes` declared but mutates
     in place (like our `start_input_dist` mutating torchrec_ctx),
     emit an event keyed by `(depends_on target name, offset)`
     instead of slot name. Users can then declare `depends_on`
     for non-slot-carried dependencies.

- **Scope**: engine-level; affects Problem #1 spec. Not HSTU-specific.
- **Trigger to resume**: either (a) perf profile shows significant
  io ↔ compute stream blocking under non-identity shuffler workloads,
  or (b) a new user needs tight fine-grained stream sync
  semantics. Current HSTU tests (identity shuffler) don't expose
  the perf loss.

### HSTU: torchrec_ctx mutation chain not modeled as DAG edges — PARTIALLY RESOLVED
- **Status**: Codex C-LOW. Runtime validator added 2026-04-24.
- **Resolution**: `HSTUPipeline._validate_set_context_colocation`
  refuses any `thread_map` that puts `start_input_dist` and
  `forward` on different worker threads. Fails at pipeline
  construction time (not mid-training). Added:
    - unit test: `test_set_context_colocation_custom_map_rejects_split`
    - integration test: `test_threaded_custom_bad_thread_map_rejected`
- **Residual**: the DAG itself still doesn't encode the
  torchrec_ctx mutation chain, so the validator is the seatbelt.
  Full DAG modeling is a larger refactor bundled with the
  event-based cross-stream sync followup above.

### HSTU threaded parity test: stress-mode missing — RESOLVED
- **Status**: Codex E-LOW. Fixed 2026-04-24.
- **Fix**: Added `test_threaded_stress_random_task_delays` which
  wraps every scheduled task's `.run` with a pre-call random
  0–3 ms delay (seeded per-rank for reproducibility). The 10-step
  parity check then runs under varied thread-scheduling
  interleavings, turning the suite from "flake detector" into a
  real race hunter.

<!-- stress-mode entry moved into the RESOLVED block above -->
