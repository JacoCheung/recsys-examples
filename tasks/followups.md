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
- **Status**: Fixed 2026-04-24 via engine `seed_first_batch` API +
  idempotent HSTU tasks. The peek batch is now seeded into the
  engine ring as the first real batch and flows through forward
  normally. No dataset-coverage loss for non-caching cases.
- **Engine change**: `SchedulablePipeline.seed_first_batch(slot_contents)`
  pre-populates `ring.at(max_offset)` and skips the next
  auto-pull. Safe to call only before first `progress()`.
- **HSTU changes**: `h2d` / `start_shuffle` / `finish_shuffle` /
  `start_input_dist` tasks made idempotent — they skip if the
  slot they'd write is already populated, so the peek batch's
  pre-processed state (from `_rewrite_model` bootstrap) is
  preserved intact.
- **Residual**: dynamicemb + prefetch combo still fails, but the
  root cause is now known to be different (see entry below).

### dynamicemb + prefetch: v1 per-batch context mismatch
- **Status**: KNOWN LIMITATION. NEW diagnosis (was previously
  conflated with the bootstrap issue, which is now fixed).
- **What**: Legacy `PrefetchTrainPipelineSparseDist` uses a SINGLE
  shared `self._context` (v0 legacy), so dynamicemb's cache
  accounting — which tracks outstanding prefetched keys
  per-context — sees one context recycled across all in-flight
  batches. Cache capacity is sized for that single-ctx model
  (cache_capacity ≈ 2 × per_batch_keys).
- **Our design**: engine's BatchRing gives each in-flight batch
  its own `torchrec_ctx` (v1 per-batch). With 3 batches in
  flight (prefetch depth=1), we hold 3 contexts → 3 × N_keys
  outstanding → overflow (12K > 7168).
- **Fix options**:
  1. Use a shared v0 context for all batches in HSTUPipeline's
     prefetch variant. Requires forking the context handling
     path so only prefetch variant does this — parallel to
     legacy's split between `TrainPipelineSparseDist` (v1) and
     `PrefetchTrainPipelineSparseDist` (v0).
  2. Patch dynamicemb upstream so outstanding is tracked per
     cache-table rather than per-context.
  3. Increase cache capacity in HSTU test config to 3+ batches
     worth — masks the limitation but doesn't solve it at scale.
- **Trigger to resume**: when production HSTU training enables
  `prefetch_type=prefetch` with `dynamic_emb` caching. The
  parity test marks this combo `xfail` for now.

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
