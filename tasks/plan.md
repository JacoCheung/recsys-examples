# PLAN — Schedulable Train Pipeline (Problem #1)

Breakdown of [SPEC.md](../SPEC.md). Ten vertical slices, three checkpoints.

- Every slice ends with a runnable demo or a passing test file — not a
  horizontal layer of code. Stopping mid-slice still leaves a working
  engine (at an earlier capability level).
- Every slice enforces the SPEC's hard rule: **engine package imports
  only stdlib + `torch` (+ optional `nvtx`)**. A lint test in V0 checks
  this and runs on every subsequent slice.
- The `_rewrite_model` / TorchRec surface is **not touched** by any
  slice. Legacy `train_pipeline.py` remains byte-identical (verified by
  V0's git-diff check).

---

## 0. Dependency graph

```
          ┌────────────┐
          │ V0  Scaffold│  (dir, lint, byte-diff guard)
          └──────┬──────┘
                 │
          ┌──────▼──────┐
          │ V1  MinTask │  Task + Schedule + StreamPool (1 stream)
          │   pipeline  │  + smoke test: Linear forward
          └──────┬──────┘
                 │
          ┌──────▼──────┐
          │ V2  Backward │  plain Task calling .backward() + optimizer,
          │   + optim    │  full train loop, still single-stream
          └──────┬──────┘
                 │        ╭─── CHECKPOINT A — review API ergonomics ───╮
                 ▼
          ┌─────────────┐
          │ V3  Multi-  │  StreamPool.N streams, deps.py auto-insert
          │   stream    │  wait_stream on cross-stream slot edges
          └──────┬──────┘
                 │
          ┌──────▼──────┐
          │ V4  Multi-  │  batch_offset, BatchRing[In], prefill/drain,
          │   batch     │  prefetch demo
          └──────┬──────┘
                 │        ╭─── CHECKPOINT B — review engine shape ─────╮
                 ▼
          ┌─────────────┬─────────────┐
          │ V5  Validator│ V6 Determ.  │   (parallelizable)
          │  + DAG check │  harness    │
          └──────┬──────┴──────┬──────┘
                 │             │
                 │        ╭─── CHECKPOINT C — pre-autosched review ────╮
                 ▼
          ┌─────────────┐
          │ V9  Auto-   │  cost_model.py + list_scheduler.py
          │   scheduler │  + CLI tools (profile / autosched)
          └──────┬──────┘
                 │
          ┌──────▼──────┐
          │ V10 Examples│  adopt_existing_loop.py + minimal_mlp.py
          │   + API edge│  + engine/__init__.py public surface
          └─────────────┘
```

Legend: arrow = hard sequencing (downstream imports upstream).
V5 and V6 run on the engine produced by V4; V5/V6 can be done in
parallel by two agents if needed.

---

## 1. Phase map

| Phase | Slices | Deliverable | User gate |
|---|---|---|---|
| 1. Core API | V0–V2 | Single-stream training loop works end-to-end | Checkpoint A |
| 2. Parallelism | V3–V4 | Multi-stream + multi-batch overlap | Checkpoint B |
| 3. Hardening | V5, V6 | Validator, determinism | Checkpoint C |
| 4. Autosched + ship | V9, V10 | Auto-scheduler + runnable examples + public API | Merge readiness |

(V7 and V8 were cut from the plan during scope audit. V9/V10 numbering
preserved to keep git-blameable task history; the gap intentionally
signals the cut.)

---

## 2. Vertical slices

### V0 — Scaffolding & invariants

**Goal.** Land the directory skeleton and the non-negotiable guardrails
before any real code. Future slices piggy-back on these guards.

**Files created.**
- `examples/commons/pipeline/engine/__init__.py` (package marker +
  no-forbidden-imports docstring)
- `examples/tests/commons/test_engine_import_hygiene.py` (outside the
  engine package tree — see rationale below)
- `examples/tests/commons/test_engine_legacy_untouched.py`

**Why tests live at `examples/tests/commons/` and NOT inside
`engine/tests/`:** the parent package
`examples/commons/pipeline/__init__.py` eagerly does
`from . import train_pipeline`, which imports torch + nvtx +
commons.distributed. Any pytest collection rooted inside the engine
subtree walks up that package chain during test import, so even
pure-stdlib guard tests fail to collect without torch. Moving tests
to a sibling directory (`examples/tests/commons/`) bypasses the
parent-package walk entirely. No conftest in the engine subtree
either, for the same reason.

**Acceptance.**
1. `pytest examples/tests/commons/test_engine_*.py -x` runs clean.
2. `test_engine_import_hygiene.py` fails if any file under `engine/`
   imports `torchrec`, `megatron`, `fbgemm_gpu`, or
   `commons.distributed.*`.
3. `test_engine_legacy_untouched.py` asserts
   `git diff HEAD --
   examples/commons/pipeline/train_pipeline.py
   examples/commons/pipeline/train_pipeline_factory.py
   examples/commons/pipeline/utils.py`
   is empty.

**Verification.**
Run inside the `devtech-compute/distributed-recommender:devel_latest`
container on `ipp1-2029` (login-shell env lacks torch):
```bash
ssh ipp1-2029 'docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp \
    -e PYTHONPATH=/home/scratch.junzhang_sw/workspace/github/recsys-examples/examples \
    -v /home/scratch.junzhang_sw:/home/scratch.junzhang_sw \
    -w /home/scratch.junzhang_sw/workspace/github/recsys-examples \
    gitlab-master.nvidia.com:5005/devtech-compute/distributed-recommender:devel_latest \
    pytest examples/tests/commons/test_engine_import_hygiene.py \
           examples/tests/commons/test_engine_legacy_untouched.py -v'
```

Plus two negative-path sanity checks:
- Inject `import torchrec` into `engine/__init__.py`, rerun →
  `test_engine_import_hygiene` fails with the offending file + prefix
  named.
- `echo "# test" >> examples/commons/pipeline/utils.py`, rerun →
  `test_engine_legacy_untouched` fails with the git diff inline and
  instructions to revert.
Both were run and validated — V0 passes.

**Risks.** None — pure scaffold.

---

### V1 — Minimal Task + Schedule + single-stream pipeline

**Goal.** One `Task` wrapping a callable runs through
`SchedulablePipeline` on one stream, one stage. A `nn.Linear` forward
pass produces the expected output tensor.

**Files created / modified.**
- `engine/streams.py` — `StreamPool` (minimal: `{"default":
  torch.cuda.default_stream()}`).
- `engine/task.py` — `Task` dataclass + `DataSlot` + `Task.from_fn`.
- `engine/context.py` — `TaskContext`, `BatchRing[In]` (N=1 only).
- `engine/schedule.py` — `Schedule`, `Stage`, `ScheduledTask`.
- `engine/pipeline.py` — `SchedulablePipeline.__init__` +
  `progress()` (no cross-stream deps, no graph mode).
- `engine/tests/test_pipeline_smoke.py` — `test_single_stage_forward`.

**Acceptance.**
1. A user can write:
   ```python
   pool = StreamPool({"default": None})
   task = Task.from_fn(
       name="fwd", fn=lambda ctx: ctx.slots.set("y", model(ctx.slots["x"])),
       reads=(DataSlot("x"),), writes=(DataSlot("y"),))
   sched = Schedule(stages=(Stage(tasks=(task,)),),
                    stream_slots=("default",))
   # in_flight_batches is a derived @property — not authored
   pipe = SchedulablePipeline(sched, pool)
   y = pipe.progress(iter([x_batch])).slots["y"]
   ```
2. `y` matches `model(x_batch)` exactly (no autograd yet, just forward).

**Verification.**
```bash
pytest engine/tests/test_pipeline_smoke.py::test_single_stage_forward -x
```

**Risks.**
- Slot API shape — commit early, refactoring later hurts. Propose a
  `SlotStore` with `__getitem__` / `set()` and keep it small.
- `progress()` return type — per SPEC §4.4, returns whatever the last
  task stored in the designated "return" slot. v1: return the
  `TaskContext` directly; refine if awkward.

---

### V2 — Backward + optimizer step (full train loop, single stream)

**Goal.** H2D → forward → backward (plain Task) → optimizer. Twenty
steps of MLP training, loss decreases, final params match a
hand-written non-pipelined reference within `atol=1e-5`.

**Precondition — autograd-stream spike (ran 2026-04-23).** Spike
lives at
`examples/tests/commons/test_engine_autograd_stream_spike.py`.

Findings:

| Fixture | Result |
|---|---|
| (a) plain `nn.Linear` | ✅ pass — backward lands on declared stream |
| (b) multi-stream forward | ❌ fail — saved-tensor stream overrides declared backward stream |
| (c) DDP-wrapped single-rank | ⚠️ runtime-conditional xfail — test inspects grad stream at runtime and xfails only if DDP routed off the declared stream; behavior depends on DDP version / reducer config |

Per plan's earlier decision rule ("spike fails → narrow §4.7
claims"): SPEC §4.6 updated with an explicit scope table. V2/T1/T2
proceed since single-stream-forward case holds. Multi-stream forward
and DDP intra-backward overlap are out of Problem #1 scope (deferred
to Problem #2 / follow-up BackwardHookTask spec).

**Files created / modified.**
- `engine/tests/test_autograd_stream_spike.py` — spike fixtures.
- `engine/tests/test_pipeline_smoke.py` — add `test_full_train_loop`.

**No new engine code.** Backward is a plain `Task.from_fn` whose body
calls `ctx.slots["loss"].backward()`. The engine's existing stream-
context wrapping (from V1) makes autograd kernels land on the task's
declared stream (**subject to the spike verifying this assumption**).
No dedicated `BackwardTask` class.

**Acceptance.**
1. A plain Task with `fn=lambda ctx: ctx.slots["loss"].backward()`,
   `reads=(DataSlot("loss"),)`, `stream="default"` runs correctly
   (backward produces no slot — it's a pure ordering source).
2. Optimizer task with `depends_on=("backward",)` runs after backward
   — engine infers the ordering edge from `depends_on` per SPEC §4.2
   rule 6, not from a sentinel slot.
3. `test_full_train_loop` trains `nn.Linear(10→1)` on synthetic
   regression data for 20 steps, loss decreases overall (≥15/19
   adjacent pairs decreasing, allowing minibatch noise), final
   params within `atol=1e-5` of a non-pipelined reference loop with the
   same seed.
4. **Preset `SchedulablePipeline.basic(model, optimizer)`** ships in
   skeletal form (single-stream, no prefetch) and is what
   `test_full_train_loop` actually uses. Proves the T1 adoption path
   works end-to-end.

**Also lands.**
- `depends_on` field on Task (first slice to need pure-ordering edges).
- `SchedulablePipeline.RETURN_SLOT = "step_result"` + progress return
  pulls from this slot.
- `SchedulablePipeline.basic(model, optimizer, …)` classmethod ships
  in skeletal form — single-stream path only. `prefetch=True` and
  `memcpy_stream=True` branches raise `NotImplementedError` until V4.
  Component makers live in `engine/_presets.py` (module-private, not
  re-exported).
- Preset convention: `model(batch)` returns a scalar `loss` OR a
  tuple whose first element is the loss. The full return goes to
  `"step_result"` slot, loss goes to `"loss"` slot. For anything more
  complex, the user composes Tasks directly without the preset.

**Files modified.**
- `engine/task.py` — add `depends_on: tuple[str, ...] = ()`.
- `engine/pipeline.py` — DAG build uses slot edges + `depends_on`
  edges + same-stream declaration order. `RETURN_SLOT` plumbing.
- `engine/_presets.py` (new, 5 module-private makers + the classmethod body)

**Verification.**
```bash
pytest engine/tests/test_pipeline_smoke.py::test_full_train_loop -x
```

**Risks.**
- Sentinel-slot pattern (e.g. `grads_ready`) tempting as a shortcut —
  don't use it; use `depends_on`. Document in task module docstring.
- Preset API shape ossifies at V2 if we're not careful — Checkpoint A
  explicitly reviews it before V3.

---

### ╭── CHECKPOINT A ───────────────────────────────────────╮

**Gate.** User reviews the Task/Schedule + backward-as-plain-Task API
surface after V2. Any shape change lands here, *before* multi-stream
work compounds the cost.

**Artifacts to review.**
- `engine/__init__.py` (public exports so far).
- Tests `test_pipeline_smoke.py::test_single_stage_forward` and
  `::test_full_train_loop`.
- Code sample from V1 Acceptance #1.

**Explicit questions at this checkpoint.**
1. Is `Task.from_fn(name, fn, reads=..., writes=...)` ergonomic?
2. Is `TaskContext.slots["x"]` / `.set("y", ...)` the right data
   access pattern?
3. Three-layer dependency model OK — (a) same-stream declaration order
   implicit, (b) slot edges for real dataflow, (c) `depends_on=("name",)`
   for pure ordering? Any layer you want replaced or dropped?
4. *Resolved 2026-04-23: user chose `SchedulablePipeline.basic(model,
   optimizer)` classmethod form. Component makers stay
   module-private in `engine/_presets.py`.*

User answers before V3 starts.

╰──────────────────────────────────────────────────────────╯

---

### V3 — Multi-stream pipeline with auto-inferred cross-stream waits

**Goal.** Two streams (`default` + `memcpy`). An H2D task on `memcpy`
and a forward task on `default`. Engine auto-inserts
`default_stream.wait_stream(memcpy_stream)` before the forward reads
the H2D's output slot. Timing check confirms overlap.

**Files created / modified.**
- `engine/streams.py` — support N named streams with priorities.
- `engine/deps.py` — scan `Schedule` → for every slot edge where
  reader.stream ≠ writer.stream, emit a cross-stream dep record.
- `engine/pipeline.py` — apply dep records: before a task runs on
  stream S, `S.wait_stream(producer_stream)` for each dep.
- `engine/tests/test_multi_stream.py` — two tests:
  - `test_auto_wait_stream_inserted` (correctness)
  - `test_streams_overlap` (CUDA event timing: max(t_H2D, t_fwd) <
    t_H2D + t_fwd · 0.9).

**Acceptance.**
1. A two-stream schedule with an explicit `wait_stream` hand-inserted
   and the same schedule with it removed (but engine-inferred) produce
   identical per-step loss for 20 steps.
2. Timing test shows at least 10% overlap (CUDA events, deterministic
   sleep kernels).
3. Numerics remain correct (`atol=1e-5` vs single-stream reference).

**Verification.**
```bash
pytest engine/tests/test_multi_stream.py -x
```

**Risks.**
- `torch.cuda.current_stream()` vs explicit context — easy to get
  subtly wrong. Guard with a test that records the stream-id inside
  each task body and asserts it matches the schedule.
- `wait_stream` placement timing (before vs after `record_stream` on
  the producer tensor) — choose one convention, document it in
  `deps.py`.

---

### V4 — Multi-batch in-flight + `batch_offset` + prefill/drain

**Goal.** A prefetch-style schedule: task with `batch_offset=1` fetches
next batch's inputs while current batch's forward/backward runs.
`BatchRing[In]` holds N=2 in-flight batches. Prefill absorption in
first call + drain after iter-StopIteration — transparent to user.

**Files created / modified.**
- `engine/context.py` — `BatchRing` generalizes to N batches; slot
  eviction on `advance()` (no cross-iter carry-over in v1).
- `engine/task.py` — slot lookups honor `batch_offset`.
- `engine/pipeline.py`:
  - Internal iteration counter `_iter: int`
  - Task-mask helper: `should_run(task, iter_count, pulled) -> bool`
    implementing SPEC §4.8 formula. Single source of truth.
  - First-call prefill absorption (runs `max_offset` prefill iters +
    1 steady iter before returning first result)
  - `StopIteration`-on-empty-ring termination
- `engine/_presets.py` / `SchedulablePipeline.basic(...)` — enable
  `prefetch=True, memcpy_stream=True` code paths (T2 adoption tier).
  Stub branches from V2 filled in.
- `engine/tests/test_multi_batch.py`:
  - `test_prefetch_correctness` — M=20 dataloader, assert `progress`
    returns M results then StopIteration. Loss sequence matches
    single-batch reference (no prefetch) within `atol=1e-5`.
  - `test_prefill_mask` — instrument task bodies to record which
    iter they ran on; assert mask matches §4.8 rule on prefill phase.
  - `test_drain_mask` — same, on drain phase.
  - `test_short_dataloader` — M < max_offset case (e.g. N=3, M=1).
    Still produces exactly 1 result, then StopIteration.
  - `test_empty_dataloader` — M=0. First `progress` raises
    StopIteration. No crash.
  - `test_ring_eviction` — slot's large CUDA tensor is released on
    advance. No CUDA memory growth over 1000 iters.
  - `test_preset_prefetch_parity` — `SchedulablePipeline.basic(...,
    prefetch=True)` matches `SchedulablePipeline.basic(...)` in final
    params (`atol=1e-5`).

**Acceptance.**
1. A prefetch schedule with `batch_offset=0` fwd/bwd and
   `batch_offset=1` H2D trains the MLP for 20 steps and matches the
   single-stream single-batch reference.
2. `in_flight_batches` is auto-derived as `max(batch_offset)+1 = 2`
   from the schedule; the user does not pass it explicitly.
3. User-facing `progress()` contract matches SPEC §4.8: M batches in,
   M results out, (M+1)th call raises `StopIteration`. Validated in
   `test_prefetch_correctness`.
4. Prefill absorption: **first** `progress()` call internally runs
   `max_offset + 1` iterations before returning first result. No
   `None` leaks to the user. Validated in `test_prefill_mask`.
5. Drain: after `batch_iter` raises `StopIteration`, remaining
   `max_offset` `progress()` calls each return one buffered result.
   Validated in `test_drain_mask`.
6. Edge cases: short dataloader and empty dataloader both produce
   correct result counts and clean `StopIteration`.
7. Slot eviction: evicted ring slots are dropped on advance. No CUDA
   memory growth across 1000 iterations (monitored via
   `torch.cuda.memory_allocated`).

**Verification.**
```bash
pytest engine/tests/test_multi_batch.py -x
```

**Risks.**
- Ring-boundary off-by-one bugs are the classic pitfall. Invest in
  tests that log the (batch-offset, iteration) pair every task sees
  and diff against an expected sequence.
- Prefill/drain mask formula (SPEC §4.8): implement as a single
  helper `engine.pipeline._should_run(task, iter_count, pulled)` used
  by the driver. **Tests MUST NOT import this helper** — instead, the
  mask tests compute the expected (iter_count, task_k) execution set
  from SPEC §4.8's worked-example arithmetic directly, then assert the
  driver's observable behavior matches. Sharing the helper between
  driver and tests admits a tautological pass (a buggy formula passes
  its own test). This guard is explicit in `test_prefill_mask.py`
  docstring.
- CUDA memory leak if slot eviction isn't wired — `test_ring_eviction`
  catches this but write it early, not at V4's end.

---

### ╭── CHECKPOINT B ───────────────────────────────────────╮

**Gate.** User reviews engine after V4. By this point the engine has
everything needed to express HSTU's `prefetch_sparse_dist` pipeline
(minus validator/determinism hardening added in V5–V6). Stop here if
scope pressures ship earlier than full feature set.

**Artifacts to review.**
- Timing trace from V3's `test_streams_overlap`.
- Sample prefetch schedule from V4 test.
- `engine/__init__.py` public surface so far.

**Explicit questions.**
1. Does the multi-batch ring model fit how HSTU's prefetch really
   behaves, or does it need refinements (e.g. `batch_offset=2` for
   deeper prefetch)?
2. Proceed to V5–V6 + V9–V10, or stop and ship what's built now?

╰──────────────────────────────────────────────────────────╯

---

### V5 — Schedule validator + DAG correctness checks

**Goal.** Reject malformed schedules at construction time with clear
errors. No more silent wrong output.

**Files created / modified.**
- `engine/autosched/validator.py` — 8 checks matching SPEC §4.2:
  1. Unique task names across the whole schedule.
  2. `task.batch_offset >= 0`.
  3. `task.stream` ∈ `Schedule.stream_slots`.
  4. At most one task `writes` any given `DataSlot(name, batch_offset)`.
  5. Every `reads(slot)` resolves to a matching earlier `writes`.
  6. Every `depends_on=("name",)` resolves to a matching earlier task.
  7. Merged DAG (slot edges + `depends_on` + same-stream adjacency)
     is acyclic.
  8. Cross-stream `wait_stream` edges are emitted exactly for
     consumer→producer pairs bound to different streams.
- `engine/pipeline.py` — call validator in `__init__`.
- `engine/tests/test_schedule_validator.py` — one failing-case test
  per rule, plus a passing-case smoke.

**Acceptance.**
1. Each of the eight rules has a failing-case test that raises a
   `ScheduleValidationError` with a specific error message.
2. A legitimate V4 schedule passes the validator without touching its
   code.

**Verification.**
```bash
pytest engine/tests/test_schedule_validator.py -x
```

**Risks.**
- Error messages must be specific enough to be debuggable. Test them
  verbatim via `pytest.raises(match=...)`.

---

### V6 — Determinism harness

**Goal.** Prove that the engine is bit-identical across runs given
same seed + same schedule + same data.

**Files created / modified.**
- `engine/tests/test_determinism.py`:
  - `test_same_seed_same_loss` — two runs, 20 steps, compare per-step
    loss with `torch.equal`.
  - `test_same_seed_same_grads` — compare grad tensors byte-for-byte
    after one backward.

**Acceptance.**
1. Both tests pass on single-stream (V1/V2) and multi-stream (V3)
   schedules.
2. Both tests pass with multi-batch (V4).

**Verification.**
```bash
pytest engine/tests/test_determinism.py -x
```

**Risks.**
- Non-determinism from `torch.use_deterministic_algorithms(True)` not
  being set — ensure the test fixture sets it.
- Non-determinism from parallel CUDA kernels across streams — resolve
  by ensuring `wait_stream` barriers serialize consumers as expected.

---

### V7 and V8 — CUT from scope

V7 (explicit `wait_on` / `records` event escape hatch) and V8
(`BackwardHookTask`) were removed during scope audit. Both are
genuinely useful but were adding API surface beyond Problem #1's
original statement (decouple tasks from schedule + auto-scheduler).

Rationale for each cut:
- **V7 events**: the three-layer dependency model (same-stream
  declaration order / slot edges / `depends_on`) already covers every
  realistic dep pattern. A HugeCTR-style explicit event API is
  redundant in our auto-infer-by-default model. Re-open if a concrete
  use case forces it.
- **V8 BackwardHookTask**: DDP-bucket-allreduce-style overlap is its
  own self-contained feature. It requires threading model, slot-store
  thread safety, and a second Task-lifecycle (hook-time vs stage-time)
  — enough complexity to be a follow-up spec of its own.

Both are tracked as follow-up specs, not lost.

Numbering V9/V10 is preserved — the gap intentionally flags the cut.

---

### ╭── CHECKPOINT C ───────────────────────────────────────╮

**Gate.** User reviews before auto-scheduler work. Engine is
feature-complete for the handwritten path. Auto-scheduler is additive.

**Artifacts to review.**
- Full test suite pass on a single-GPU runner.
- `engine/__init__.py` surface.

**Explicit questions.**
1. Proceed to V9 (auto-scheduler), or ship engine as-is and revisit?
2. Auto-scheduler cost model: CUDA-event profile (v1 design), or
   something simpler like user-provided static hints?

╰──────────────────────────────────────────────────────────╯

---

### V9 — Auto-scheduler v1 (list scheduling + cost model)

**Goal.** Given a task set + a cost JSON + a stream inventory, emit a
valid `Schedule`. Cost comes from an offline warmup.

**Files created / modified.**
- `engine/autosched/cost_model.py`:
  - `CostProfiler` — instruments a user-supplied default schedule with
    CUDA events, runs N warmup iterations, dumps
    `{task_name: {cpu_us, gpu_us}}` JSON.
  - `CostModel` — load from JSON, provide per-task estimates.
- `engine/autosched/list_scheduler.py`:
  - Critical-path list scheduling. Priority = remaining CP length.
    Resource = (stream slot × batch-offset).
  - Post-pass: emit cross-stream `wait_stream` edges per SPEC §4.2
    rule 8 (reuse the V5 validator's emitter, not a re-implementation).
- `engine/autosched/__init__.py` — CLI entry points: `profile`,
  `autosched`.
- `engine/tests/test_list_scheduler.py`:
  - `test_synthetic_dag_optimal` — 8-task synthetic DAG with known
    optimum.
  - `test_resource_feasibility` — scheduler never double-assigns a
    stream at the same batch-offset in the same stage.
- `engine/tests/test_cost_model.py`:
  - `test_profile_roundtrip` — profile → JSON → reload → equal.
- CLI scripts referenced in SPEC §2:
  - `python -m commons.pipeline.engine.autosched.profile --tasks ...
    --schedule ... --steps 10 --out cost.json`
  - `python -m commons.pipeline.engine.autosched --tasks ... --cost
    cost.json --streams default,memcpy,comm --out schedule.json`

**Acceptance.**
1. Synthetic 8-task DAG produces known-optimal schedule (unit test).
2. Profile CLI produces a readable JSON on the V4 prefetch schedule.
3. Autosched CLI emits a `Schedule` that passes V5's validator.

**Verification.**
```bash
pytest engine/tests/test_list_scheduler.py engine/tests/test_cost_model.py -x
python -m commons.pipeline.engine.autosched.profile --help
python -m commons.pipeline.engine.autosched --help
```

**Risks.**
- Cost model accuracy — single warmup profile may not capture
  dataset-dependent cost variance. v1 accepts this; document as known
  limit.
- Scheduler complexity creep — resist the urge to add ILP, simulated
  annealing, etc. Ship list scheduler, measure, iterate.

---

### V10 — Examples, preset API polish, adoption docs

**Goal.** Runnable examples demonstrating the two v1 adoption tiers
(T1 drop-in + T2 prefetch), clean `engine/__init__.py` surface,
concrete migration guide. **T3 / T4 examples deferred** — composing
raw `Task`/`Schedule` is the public API and self-documents via
docstrings + unit tests.

**Files created / modified.**
- `engine/__init__.py` — public API: Task, Schedule, Stage,
  SchedulablePipeline, StreamPool, DataSlot, + `presets` submodule
  + autosched CLI entry points. ≤ 10 public symbols.
- `engine/pipeline.py` + `engine/_presets.py` — `.basic()`
  classmethod final polish, clear docstrings, deprecation path for
  future changes.
- `engine/examples/adopt_existing_loop.py` — **T1 demo.** Before /
  after diff: plain PyTorch loop → engine-hosted loop via
  `SchedulablePipeline.basic(...)` + `pipe.step(batch)`. ≤ 8-line
  `git diff` (6 deletions + 2 insertions).
- `engine/examples/minimal_mlp.py` — **T2 demo.** Same model +
  `prefetch=True, memcpy_stream=True`, shows overlap via timing.
- `engine/README.md` — 1-page quickstart + 2-row "which tier do I
  need?" table.

**Acceptance.**
1. Both examples run to completion on 1 GPU, print decreasing loss.
2. `adopt_existing_loop.py` proves the T1 story: `git diff` between
   "before" and "after" shows ≤ 8 lines changed (6 deletions + 2
   insertions), zero model code changes.
3. **Preset compatibility matrix** — `test_preset_matrix.py` exercises
   4 configurations, each migrating from a realistic non-engine loop:
   (a) **vanilla** — no escape kwargs (bar: ≤ 8-line diff)
   (b) **AMP + GradScaler** — uses `forward_fn` (wraps `model(b)` in
       `autocast`), `backward_fn=lambda l: scaler.scale(l).backward()`,
       `optimizer_step_fn=lambda: (scaler.step(optim), scaler.update())`
       (bar: ≤ 15-line diff). `scaler.update()` is mandatory and
       asserted in the test body.
   (c) **gradient clipping** — uses `optimizer_step_fn=lambda:
       (clip_grad_norm_(...), optim.step())` (bar: ≤ 15-line diff)
   (d) **LR scheduler** — uses `optimizer_step_fn=lambda:
       (optim.step(), scheduler.step())` (bar: ≤ 15-line diff)
   Each case asserts final params match the non-engine reference
   within `atol=1e-5`. If any case can't hit its bar, it's moved to
   T3/T4 publicly and the README updates accordingly.
4. `engine/__init__.py` exposes ≤ 10 public symbols — no internals.
5. README 2-row table: "no tuning" → T1, "want H2D overlap" → T2.
   Plus a "what about AMP/clipping/scheduler?" subsection with the
   escape-kwarg recipes.

**Verification.**
```bash
python examples/commons/pipeline/engine/examples/adopt_existing_loop.py
python examples/commons/pipeline/engine/examples/minimal_mlp.py
```

**Risks.**
- Examples often bitrot — tie each to an integration test that imports
  and runs a small version.
- Preset API shape cemented at V10 — any breaking change after this
  triggers a deprecation cycle, not a rewrite.

---

## 3. Risk register

| Risk | Mitigation | Owner |
|---|---|---|
| **Autograd does not honor caller's stream context** (SPEC §4.6 working assumption). Most likely blocker at V2. | Spike before V2 on 3 fixtures (nn.Linear / multi-stream forward / DDP). Failure → narrow §4.7 claims or pull hook-task back into scope. | V2 |
| **Preset escape-kwarg set is insufficient** to express a realistic loop. The 4 kwargs (`forward_fn`, `loss_fn`, `backward_fn`, `optimizer_step_fn`) are a *design claim* — until the V10 matrix passes, we don't know they actually cover AMP / GradScaler / grad-clip / LR scheduler loops at their stated line-count bars. | V10 `test_preset_matrix.py` is blocker-severity: each of 4 scenarios must pass its bar; any that fails is publicly retracted from the preset + documented in the README as T3/T4. | V10 |
| **§4.8 mask contract tautology risk.** If driver and tests share the `should_run` helper, a wrong formula passes its own tests. | V4 mask tests compute expected (iter_count, task_k) set directly from SPEC §4.8 worked example, not by calling the production helper. Enforced in test docstrings + V4 risks. | V4 |
| **Validator gaps poison auto-scheduler**: duplicate task names, duplicate slot writers, same-stage cross-stream ambiguity. | V5 validator ships all 8 rules (SPEC §4.2) — reviewed before V5 closes. | V5 |
| **Prefill/drain changes side-effect timing**: a user-side-effect task (logging, LR scheduler) fires at unexpected iterations because it lives at a particular `batch_offset`. | Document this clearly in §4.8; preset keeps zero_grad / optimizer at `batch_offset=0` so their CPU-observable order is identical to a vanilla loop. Any T3+ user writing side-effecting tasks owns this invariant. | V4, V10 |
| Slot API shape ossifies before we learn from real use | Checkpoint A forces early review after V2 | reviewer |
| Multi-batch ring off-by-one | Dedicated `test_ring_drain` + per-iteration (batch_offset, iter) logging | V4 |
| NCCL-order assumption: we claim single-threaded makes it safe; if HSTU tests surface an edge case, re-open | Problem #3 owns the fix; #1 ships with a doc note | #3 author |
| Auto-scheduler cost model varies across batches | Document as v1 limit, gate re-profile on iteration count | V9 |
| Engine gets silently coupled to TorchRec by a contributor's PR | `test_import_hygiene.py` from V0 runs on every PR | CI |

---

## 4. What is *not* in this plan

Directly lifted from SPEC §9 (non-goals) — pinned here so reviewers
don't ask:

- Pipeline parallelism (PP).
- Multi-threaded executor (Problem #3).
- `_rewrite_model` generalization (Problem #2).
- HSTU migration (separate future work item).
- Intra-backward decomposition (AOTAutograd / `torch.compile`).
- CUDA graph capture.
- YAML/DSL schedule authoring.
- Runtime-adaptive auto-scheduler.

---

## 5. Estimated sequencing (rough, not commitments)

| Slice | Rough effort | Can parallelize with |
|---|---|---|
| V0 | 0.5 d | — |
| V1 | 1–2 d | — |
| V2 | 1.5 d (incl. autograd spike) | — |
| V3 | 1–2 d | — |
| V4 | 1–2 d | — |
| V5 | 1 d | V6 |
| V6 | 0.5 d | V5 |
| V9 | 2–3 d | — |
| V10 | 2 d (incl. preset compat matrix) | — |

Total single-thread: ~10–14 engineering days. Parallelizing V5+V6
saves ~0.5 d. Prior 8–12 estimate was optimistic — it didn't account
for the §4.6 autograd spike or the V10 preset compatibility matrix
demanded by Finding 4.

---

## 6. Next action

Wait for Checkpoint A answers before starting V3. V0–V2 can proceed
immediately.

See [todo.md](todo.md) for the checkbox view.
