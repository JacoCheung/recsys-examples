# TODO — Schedulable Train Pipeline (Problem #1)

Checkbox view of [plan.md](plan.md). Check off as you go.
Full acceptance criteria + verification live in plan.md — this file is
just for tracking.

---

## Phase 1 — Core API

- [x] **V0 — Scaffolding & invariants** (completed 2026-04-23)
  - [x] `engine/__init__.py` package marker
  - [x] Tests live at `examples/tests/commons/test_engine_*.py`
        (outside broken pipeline/ package tree)
  - [x] `test_engine_import_hygiene.py` rejects forbidden imports
        (verified via negative test: `import torchrec` → fail)
  - [x] `test_engine_legacy_untouched.py` asserts no diff on legacy
        pipeline files (verified via negative test)
  - [x] 3/3 tests pass in devel_latest container on ipp1-2029

- [x] **V1 — Minimal Task + Schedule + single-stream pipeline** (completed 2026-04-23)
  - [x] `engine/streams.py` — `StreamPool` (1 stream, `default` required)
  - [x] `engine/task.py` — `Task`, `DataSlot`, `Task.from_fn`
  - [x] `engine/context.py` — `TaskContext`, `BatchRing` (N=1)
  - [x] `engine/schedule.py` — `Schedule` (in_flight_batches derived),
        `Stage`
  - [x] `engine/pipeline.py` — `SchedulablePipeline` +
        `progress()` + RETURN_SLOT="step_result"
  - [x] `engine/__init__.py` exposes 7 symbols (Task, DataSlot,
        Schedule, Stage, StreamPool, SchedulablePipeline, TaskContext)
  - [x] 5/5 pipeline_smoke tests pass in container on ipp1-2029
        (single_stage_forward matches reference, in_flight_batches
        derived, StreamPool requires `default`, Task.from_fn binding,
        unknown-stream construction-time rejection)

- [x] **V2 — Backward + optimizer (full single-stream train loop)** (completed 2026-04-23)
  - [x] `engine/task.py` — `depends_on` already shipped in V1
  - [x] `engine/pipeline.py` — pre-V5 validator checks: unique names,
        stream slot existence, depends_on resolution (SPEC §4.2 rules 1, 3, 6)
  - [x] Backward is a plain `Task.from_fn` calling `.backward()` via
        preset's `_make_backward_task` — no dedicated class
  - [x] Optimizer uses `depends_on=("backward",)` — pure ordering edge
  - [x] `SchedulablePipeline.basic(...)` classmethod with 4 escape kwargs
        (forward_fn / loss_fn / backward_fn / optimizer_step_fn);
        prefetch / memcpy_stream raise `NotImplementedError`
  - [x] `engine/_presets.py` — 5 module-private makers + `_default_loss_extractor`
  - [x] `test_engine_autograd_stream_spike.py` — spike ran; plain-Linear
        ✅, multi-stream-forward ❌ (xfail-documented, out of #1 scope),
        DDP ⚠️ (runtime-conditional xfail — depends on DDP reducer config)
  - [x] `test_engine_full_train_loop.py::test_full_train_loop_matches_reference`
        — 20 steps, loss decreases, params match reference (atol=1e-5)
  - [x] 3 escape-kwarg wiring tests pass (forward_fn/backward_fn/optimizer_step_fn)
  - [x] 2 validator negative-case tests pass (duplicate-name, unresolved depends_on)
  - [x] **22 passed, 2 xfailed, 0 failed** on ipp1-2029 container

- [x] **CHECKPOINT A — resolved 2026-04-23** via V1 smoke + user verbal
  - [x] `Task.from_fn` ergonomics: accepted (V1 test uses it)
  - [x] `ctx.slots["x"]` / `.set("y", v)` read/write: accepted
  - [x] Three-layer dependency model: accepted (SPEC §4.2 unchanged)
  - [x] Preset form: `SchedulablePipeline.basic(model, optimizer)`
        classmethod chosen over `basic_train_step` module function

---

## Phase 2 — Parallelism

- [x] **V3 — Multi-stream + auto-inferred cross-stream waits** (completed 2026-04-23)
  - [x] `engine/streams.py` — N named streams supported (StreamPool
        already generalized in V1; V3 clarified None-handling semantics)
  - [x] `engine/deps.py` — `infer_cross_stream_waits(schedule)` returns
        `{consumer_name: (producer_stream_names, ...)}` from slot +
        depends_on edges; same-stream and unresolved edges skipped
  - [x] `engine/pipeline.py` — computes waits at construction; applies
        before each task via `current_stream().wait_stream(producer)`;
        resolves `None` to `default_stream()` only on CUDA
  - [x] 10/10 `test_engine_multi_stream.py` tests pass:
        6 analyzer unit tests (empty, same-stream, cross-stream slot,
        depends_on, multi-producer ordering, unresolved tolerance)
        + 4 runtime tests (race-prevention, independent-task
        concurrency, multi-stream train-loop parity, None-passthrough)
  - [x] Full V0-V3 suite green: **35 passed, 2 xfailed, 0 failed**

- [x] **V4 — Multi-batch in-flight + `batch_offset` + prefill/drain** (completed 2026-04-23)
  - [ ] `engine/context.py` — `BatchRing` to N batches + slot eviction
        on `advance()` (no cross-iter carry-over in v1)
  - [ ] `engine/task.py` — offset-aware slot lookups
  - [ ] `engine/pipeline.py` — `should_run(task, iter_count, pulled)`
        mask helper (SPEC §4.8), prefill absorption in first call,
        drain after iter-StopIteration, ring-empty termination
  - [ ] `SchedulablePipeline.basic(...)` — enable `prefetch=True` +
        `memcpy_stream=True` branches (T2 adoption)
  - [ ] `test_multi_batch.py::test_prefetch_correctness` — M in → M out
  - [ ] `test_multi_batch.py::test_prefill_mask` — §4.8 mask rule verified
  - [ ] `test_multi_batch.py::test_drain_mask` — §4.8 mask rule verified
  - [ ] `test_multi_batch.py::test_short_dataloader` — M < max_offset
  - [ ] `test_multi_batch.py::test_empty_dataloader` — M=0
  - [ ] `test_multi_batch.py::test_ring_eviction` — no CUDA leak
  - [ ] `test_multi_batch.py::test_preset_prefetch_parity` passes
  - [ ] `in_flight_batches` auto-derived from `max(batch_offset)+1`

- [ ] **CHECKPOINT B — review engine shape with user**
  - [ ] User reviews timing trace from V3
  - [ ] User reviews prefetch schedule from V4
  - [ ] Decide on `carries_over` default
  - [ ] Decide whether to proceed to V5–V8 or ship what's built

---

## Phase 3 — Hardening (V5 and V6 can run in parallel)

- [x] **V5 — Schedule validator + DAG correctness** (completed 2026-04-23)
  - [ ] `engine/autosched/validator.py` — 8 rules matching SPEC §4.2
        (unique names, non-negative offsets, stream existence, single
        writer per slot, reads resolve, depends_on resolves, acyclic,
        cross-stream wait insertion)
  - [ ] `SchedulablePipeline.__init__` calls validator
  - [ ] `test_schedule_validator.py` — one failing case per rule + smoke

- [x] **V6 — Determinism harness** (completed 2026-04-23)
  - [ ] `test_determinism.py::test_same_seed_same_loss` passes on V1–V4
  - [ ] `test_determinism.py::test_same_seed_same_grads` passes on V1–V4

> V7 (explicit event escape hatch) and V8 (BackwardHookTask) were
> **cut from scope** during the scope audit. Both tracked as follow-up
> specs. V9/V10 numbering preserved so the gap flags the cut.

- [ ] **CHECKPOINT C — review before auto-scheduler**
  - [ ] Full engine test suite green on 1-GPU runner
  - [ ] User reviews `engine/__init__.py` public surface
  - [ ] Decide whether to build V9 auto-scheduler or ship

---

## Phase 4 — Auto-scheduler + ship

- [x] **V9 — Auto-scheduler v1** (completed 2026-04-23)
  - [ ] `engine/autosched/cost_model.py` — `CostProfiler` + `CostModel`
  - [ ] `engine/autosched/list_scheduler.py` — critical-path scheduler
  - [ ] CLI: `python -m commons.pipeline.engine.autosched.profile`
  - [ ] CLI: `python -m commons.pipeline.engine.autosched`
  - [ ] `test_list_scheduler.py::test_synthetic_dag_optimal` passes
  - [ ] `test_list_scheduler.py::test_resource_feasibility` passes
  - [ ] `test_cost_model.py::test_profile_roundtrip` passes
  - [ ] Autosched output passes V5 validator

- [x] **V10 — Examples, preset polish, adoption docs** (completed 2026-04-23)
  - [ ] `engine/__init__.py` — clean public surface (≤ 10 symbols)
  - [ ] `SchedulablePipeline.basic(...)` + `_presets.py` final API polish + docstrings
  - [ ] `engine/examples/adopt_existing_loop.py` — T1 demo (≤ 8-line diff)
  - [ ] `engine/examples/minimal_mlp.py` — T2 demo (preset + prefetch)
  - [ ] `engine/README.md` — 1-page quickstart + 2-row adoption table
  - [ ] Integration test exercises both examples

---

## Final merge gate

- [ ] All of `engine/tests/` green in CI
- [ ] `test_import_hygiene.py` still enforced (engine stays framework-free)
- [ ] `test_legacy_untouched.py` still green (legacy byte-identical)
- [ ] `test_determinism.py` passes with every V3/V4 feature enabled
- [ ] Both examples run without error
- [ ] User sign-off on final `engine/__init__.py`
