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

### NVTX integration — RESOLVED
- **Status**: Fixed in f77111e5 (2026-04-26).
- **Resolution**: ``engine/executor.py`` now wraps every
  ``task.run(ctx)`` call in ``nvtx.annotate(task.nvtx_tag or
  task.name)`` via the ``_nvtx_range`` context manager. Wrapping is
  applied at every callsite (SequentialExecutor, single-task fast
  path, all-same-thread fast path, per-thread chain with and without
  NCCL ticketing). ``nvtx`` is imported optionally so CPU-only test
  hosts and CUDA-less builds continue to work — the wrapper is a
  no-op when ``nvtx`` is unimportable or CUDA is unavailable.
- **Verification**: engine 183/2/0, HSTU parity 24/8/0, both
  unchanged from prior baselines — wrapping does not perturb
  behavior.

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

### P2 `attach()` / `detach()` lifecycle incomplete — RESOLVED
- **Status**: Fixed in 2026-04-26. Originally Codex B-MEDIUM-1.
- **Resolution**: ``HSTUPipeline.detach()`` now resets the engine
  bookkeeping after restoring the original sharded-module forwards:
  ``self._pipe`` is shut down and set to ``None``, and the
  ``pipelined_modules`` / ``pipelined_postprocs`` /
  ``original_forwards`` / ``original_kjt_dist_forwards`` lists are
  cleared. The next ``progress()`` after ``attach()`` will see
  ``self._pipe is None`` and run ``_ensure_pipe`` from scratch —
  ``_rewrite_model`` re-installs pipelined forwards on the
  (possibly re-attached) model and a fresh ``SchedulablePipeline``
  is built. ``attach()`` itself stays minimal: it only toggles
  ``_model_attached`` and accepts an optional new model reference.
- **Verification**: HSTU parity 24/8/0 unchanged (the parity test
  doesn't exercise detach/attach but verifies the happy path is
  unaffected by the bookkeeping reset).

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

### Engine: event-based cross-stream sync (replace wait_stream) — RESOLVED
- **Status**: Fixed across 25ca73fe → 6c25a1de → 05d08a45 → 43af9499
  (2026-04-26). Originally Codex review B-MEDIUM / C-HIGH on commit
  23414493.
- **Resolution**: SlotStore now carries a per-task event registry
  (``set_event``/``get_event``/``has_event``); BatchRing.advance
  rotates SlotStore objects in-place so the same
  ``torch.cuda.Event`` is reused across iterations. The executor
  records a completion event on each task's stream after
  ``task.run`` returns, and ``_apply_cross_stream_waits`` prefers
  ``wait_event(producer_event)`` from the ring slot over coarse
  ``wait_stream(producer_stream)``. Stream-level wait remains as
  a first-iter fallback when the slot has no event yet.
  ``deps.infer_cross_stream_event_deps`` returns
  ``(producer_task, producer_stream, slot_offset)`` triples
  including for cross-iter ``depends_on=("name", -N)`` edges, with
  redundancy checks against reads/writes data edges.
- **Verification**: engine 183/2/0; HSTU parity 24/8/0 unchanged.
  io/compute overlap improvement is workload-dependent — current
  HSTU tests use identity shuffler, so wall-clock parity does not
  exercise the perf delta. Re-profile under non-identity shuffler
  if quantification needed (`tasks/SPEC_p4_micro_repro.py` is a
  starter scaffold).

### HSTU: torchrec_ctx mutation chain not modeled as DAG edges — RESOLVED
- **Status**: Fixed in dc8e3ea4 (2026-04-26) on top of the
  earlier runtime-validator partial fix from 2026-04-24. Originally
  Codex C-LOW.
- **Resolution**: ``prefetch_embeddings`` now declares
  ``writes=("module_input_post_prefetch",
  "module_contexts_post_prefetch")`` as pseudo-slots — the task body
  still mutates ``torchrec_ctx`` in place, but the slot names give
  the engine a dependency identifier. ``forward`` (in the prefetch
  variant) declares the matching reads. ``deps.
  infer_cross_stream_event_deps`` now sees the
  ``prefetch_embeddings → forward`` cross-iter data edge and emits
  ``wait_event(prefetch_embeddings_event_at_offset_0)`` before
  forward, with stream-level fallback on iteration 1 when the ring
  slot has no event yet. SPEC_p4 v2 §8 documents the pseudo-slot
  mutation-chain syntax decision. The runtime
  ``_validate_set_context_colocation`` validator is preserved as
  belt-and-suspenders.
- **Verification**: HSTU parity 24/8/0; first-iter forward →
  prefetch race is now structurally covered by the explicit DAG
  edge instead of the cross-iter chain via
  ``backward.depends_on=prefetch_embeddings``.

### HSTU threaded parity test: stress-mode missing — RESOLVED
- **Status**: Codex E-LOW. Fixed 2026-04-24.
- **Fix**: Added `test_threaded_stress_random_task_delays` which
  wraps every scheduled task's `.run` with a pre-call random
  0–3 ms delay (seeded per-rank for reproducibility). The 10-step
  parity check then runs under varied thread-scheduling
  interleavings, turning the suite from "flake detector" into a
  real race hunter.

<!-- stress-mode entry moved into the RESOLVED block above -->
