# SPEC — Schedulable Train Pipeline (Problem #1 of 3)

> **Scope note.** First of three sequential specs for reworking the training
> pipeline layer. Covers **Problem #1 only**: decouple what-to-do (tasks)
> from how-to-order-it (schedule), plus an auto-scheduler.
>
> - Problem #2 — generalize forward-decomposition (today: `_rewrite_model`
>   is tied to `ShardedModule`).
> - Problem #3 — multi-threaded NCCL-order-safe executor.
>
> These two get their own SPEC files. Problem #1 must leave clean seams
> for both, but ship and land without them.

> **Explicit design stance:** this engine is **greenfield and
> framework-agnostic**. It does not know about TorchRec, Megatron, DDP, or
> HSTU. It does not share code with, inherit from, or aim for parity with
> the existing `JaggedMegatronTrain*Pipeline` classes. Those stay untouched.
> Adapting HSTU to run on top of the new engine is a follow-on migration,
> not an acceptance criterion here.
>
> **Design reference — HugeCTR.** The engine is modeled closely on
> HugeCTR's `Pipeline` / `Scheduleable` abstractions
> (`HugeCTR/include/pipeline.hpp`,
> `HugeCTR/src/pybind/model_pipeline.cpp`). We take its core ideas:
> task = callable with a named stream, `init()`/`run()` split,
> event-based dependencies as an escape hatch. We add on top:
> (a) auto-inferred cross-stream deps from slot-level reader/writer edges
> (so users don't write `record_done` / `wait_event` for the common case),
> (b) an explicit `batch_offset` for multi-iteration in-flight pipelining,
> and (c) an auto-scheduler over a formal DAG.
>
> **Critical divergence — autograd.** HugeCTR has no autograd engine, so
> each layer's forward and backward are hand-written lambdas and *both*
> are schedulable. PyTorch's `loss.backward()` is a single atomic call
> into the autograd engine; we cannot reorder ops inside it. Our engine
> therefore treats **backward as one Task**, and relies on *around-
> backward* overlap (prefetch, H2D, optimizer-prep on other streams) —
> which is the same overlap HugeCTR actually exploits at the model-
> pipeline level. Intra-backward scheduling is a non-goal for this spec.
> §4.5 spells out the HugeCTR↔ours mapping; §4.6 details the autograd
> handling.
>
> **CUDA graph capture is out of scope for this spec** (deferred per user
> direction). The engine is eager-only in v1.

---

## 1. Objective

Provide a generic training-loop engine that consumes two user-supplied
inputs and nothing more:

1. **Task set** — a bag of pure Python callables, each declaring its input
   slots, output slots, preferred stream slot, and which in-flight batch it
   targets.
2. **Schedule** — a declarative plan saying *in what order* the tasks run
   and *on which CUDA stream*. The engine does no scheduling logic of its
   own; it executes the plan.

Also ship an **auto-scheduler**: given the task DAG, a cost model, and a
stream-inventory, emit a valid critical-path-minimizing `Schedule`.

The engine is designed to work with:
- arbitrary `torch.nn.Module` models (no `ShardedModule` assumption);
- arbitrary batch payload types (user defines `In`, `Out`);
- any or no distributed backend (single-GPU, DP, TP, FSDP — engine is
  oblivious);
- **no TorchRec, Megatron, or HSTU imports** in the engine module.

### Target users

1. A researcher with a custom PyTorch training step who wants async H2D +
   prefetch + compute overlap without writing a bespoke pipeline class.
2. A framework author who wants to declare "this work can overlap this
   collective" once and let the engine + scheduler figure it out.
3. (Later, out-of-scope migration) the recsys-examples HSTU trainer.

### Success looks like

- **Ease of use / minimal user-code intrusion is the primary bar** and
  applies to two bands. "Diff" = `git diff` insertions + deletions
  against the vanilla PyTorch loop:
  - **Vanilla** training loops (`model(batch) → loss → backward →
    step`): adopted with a **≤ 8-line diff** via
    `SchedulablePipeline.basic(model, optimizer)` constructed once +
    `pipe.step(batch)` inside the existing `for batch in loader:`
    loop (replacing the 6-line body).
  - **AMP / GradScaler / gradient clipping / LR scheduler** loops:
    adopted with a **≤ 15-line diff** via the same preset + its
    `forward_fn` / `loss_fn` / `backward_fn` / `optimizer_step_fn`
    escape kwargs.
  In both bands, the model class, optimizer, dataloader, loss
  function, and DDP/FSDP wrapping stay untouched. AMP `autocast`,
  `GradScaler`, clipping, and scheduler calls migrate into the
  preset's escape kwargs (one-liners each), not into the Task engine
  itself. See §4.7 for the contract and V10's preset compatibility
  matrix test for proof that all 4 band-crossing scenarios hit their
  line-count bars.
- A user can express a training step as ~5 `Task` objects + a ~20-line
  `Schedule`, get correct execution, and get parallelism the single-stream
  baseline doesn't.
- Adding a new variant (extra async step, different stream assignment) is
  a Schedule edit — no new pipeline subclass.
- The auto-scheduler, fed the same task DAG and a cost profile, produces
  a Schedule whose wall-clock step time is within a small margin of a
  hand-tuned one (target: 5%, [needs user confirmation]).

---

## 2. Commands

```bash
# Run the minimal example (single-GPU, no TorchRec, no Megatron)
python examples/commons/pipeline/engine/examples/minimal_mlp.py

# Profile a schedule's per-task cost and dump a cost.json
python -m commons.pipeline.engine.profile \
    --tasks path.to.my_tasks:TASKS \
    --schedule path.to.my_schedule:SCHEDULE \
    --steps 10 --out cost.json

# Auto-schedule given tasks + cost + stream inventory
python -m commons.pipeline.engine.autosched \
    --tasks path.to.my_tasks:TASKS \
    --cost cost.json \
    --streams default,memcpy,comm \
    --out schedule.json

# Engine unit + integration tests
pytest examples/tests/commons/test_engine_*.py -x
```

---

## 3. Project structure

All new code lives in a single, self-contained directory. No existing file
is modified.

```
examples/commons/pipeline/engine/        # NEW — entirety of Problem #1
├── __init__.py                          # public API surface
├── task.py                              # Task, DataSlot, StreamSlot, BatchOffset
├── schedule.py                          # Schedule, Stage, ScheduledTask
├── context.py                           # BatchRing[In] — N in-flight batches
├── streams.py                           # StreamPool, named stream slots + priorities
├── deps.py                              # cross-stream data-visibility inference
├── pipeline.py                          # SchedulablePipeline + SchedulablePipeline.basic(...) classmethod
├── _presets.py                          # module-private component makers for .basic()
│
├── autosched/
│   ├── __init__.py
│   ├── cost_model.py                    # CUDA-event profiling, JSON cache
│   ├── list_scheduler.py                # critical-path list scheduler
│   └── validator.py                     # DAG well-formedness + resource feasibility
│
└── examples/                            # minimal, framework-free examples
    ├── adopt_existing_loop.py           # T1: ≤8-line git-diff migration from plain PyTorch loop
    └── minimal_mlp.py                   # T2: single-GPU MLP with prefetch overlap
```

Tests live OUTSIDE the engine package tree at
`examples/tests/commons/test_engine_*.py` — NOT under
`examples/commons/pipeline/engine/tests/`. Rationale: the parent
package `examples/commons/pipeline/__init__.py` eagerly imports
`train_pipeline` (legacy), so pytest test-collection inside the
engine subtree would drag in torch / nvtx / commons.distributed even
for pure-stdlib tests. Placing tests at the sibling `examples/tests/`
hierarchy breaks that implicit package-tree walk.

```
examples/tests/commons/
├── test_engine_import_hygiene.py        # walks engine/ + rejects forbidden imports
├── test_engine_legacy_untouched.py      # asserts git diff HEAD on legacy files = ∅
├── test_engine_task_dag.py              # V1+: DAG construction, cycle detection
├── test_engine_schedule_validator.py    # V5: all 8 §4.2 rules
├── test_engine_list_scheduler.py        # V9: synthetic DAG optimality
├── test_engine_pipeline_smoke.py        # V1/V2: end-to-end on tiny nn.Module
├── test_engine_multi_stream.py          # V3: cross-stream overlap + timing
├── test_engine_multi_batch.py           # V4: prefill/drain + ring
└── test_engine_determinism.py           # same schedule + seed → bit-identical loss
```

> Existing `examples/commons/pipeline/train_pipeline.py`,
> `train_pipeline_factory.py`, `utils.py` are **not touched** by this
> spec. They continue to work as-is.

---

## 4. Core concepts

### 4.1 Task

Modeled on HugeCTR's `Scheduleable` + `StreamContextScheduleable`: a
callable workload with a named stream, optional event deps, and an
`init()`/`run()` split.

```python
class Task:
    name: str
    stream: str = "default"             # named stream slot; resolved by StreamPool
    batch_offset: int = 0               # 0 = current batch, 1 = next, …
    reads: tuple[DataSlot, ...] = ()
    writes: tuple[DataSlot, ...] = ()
    depends_on: tuple[str, ...] = ()    # other task names — pure ordering edge, no data
    nvtx_tag: str | None = None         # defaults to f"## {name} ##"

    def init(self, ctx: InitContext) -> None:
        """Called once when the pipeline is built — allocate buffers,
        register modules, cache references. Analogous to
        HugeCTR Scheduleable::init()."""

    def run(self, ctx: TaskContext) -> None:
        """Called every iteration — the actual workload. Analogous to
        HugeCTR Scheduleable::run(). Pure side effect on ctx."""
```

- Users may either subclass `Task` (for reusable tasks with state) or
  wrap a lambda via `Task.from_fn(name, fn, ...)` (matching HugeCTR's
  `StreamContextScheduleable(std::function<void()>)` ergonomics).
- `TaskContext` exposes the current `BatchRing`, the resolved
  `StreamPool`, and a typed slot store. It imports only `torch` — no
  framework symbols.
- `DataSlot = (name: str, batch_offset: int)`. Opaque, typed by
  convention at declaration. The engine matches reads to writes by slot
  identity.
- Stream names are user-defined strings (e.g. `"default"`, `"memcpy"`,
  `"comm"`), declared once by the `StreamPool`. Not a fixed enum.

### 4.2 Schedule

```python
@dataclass(frozen=True)
class Stage:
    tasks: tuple[ScheduledTask, ...]        # within-stage order = submission order

@dataclass(frozen=True)
class Schedule:
    stages: tuple[Stage, ...]
    stream_slots: tuple[str, ...]           # declared stream inventory

    @property
    def in_flight_batches(self) -> int:
        """Derived from tasks' batch_offsets — never authored by user."""
        return max((t.batch_offset for stage in self.stages
                    for t in stage.tasks), default=0) + 1
```

A `ScheduledTask` is a `Task` with a concrete stream binding. The v1
auto-scheduler preserves each task's declared stream; stream rebinding
would need a real resource model and is intentionally out of scope.

**Stages and cross-stream waits.** Stages are **organizational** — they
group tasks for visual clarity and for CPU-side submission order.
Cross-stream `wait_stream` insertion is driven by **consumer→producer
edges** (slot reads + `depends_on`), not by stage boundaries. A
consumer task with stream S reading a slot written by a producer task
with stream S' ≠ S gets `S.wait_stream(S')` inserted before its
submission, regardless of whether the producer is in the same stage
(earlier declaration order) or an earlier stage.

**Three layers of dependency** (from implicit to explicit):

| Layer | When | How expressed |
|---|---|---|
| 1. Same stream, same stage | Tasks share a CUDA stream and are adjacent in `Stage.tasks` | **Implicit** — declaration order = submission order. No syntax. Free. |
| 2. Real dataflow | Task B reads tensor written by Task A | **Slot** edges: `writes=(DataSlot("x"),)` / `reads=(DataSlot("x"),)`. Engine auto-inserts cross-stream `wait_stream` if needed. |
| 3. Pure ordering, no data | Task B must run after Task A but shares no tensor (e.g. `zero_grad` before forward, metric logging after loss, debug assertion before op) | **Explicit** `depends_on=("task_a_name",)`. Engine treats it identically to a slot edge for DAG / wait_stream purposes; no data is carried. |

**Validity rules:**
1. **Unique names.** Every task's `name` is unique across the whole
   schedule.
2. **Non-negative offsets.** `task.batch_offset >= 0` for every task.
3. **Stream existence.** Every `task.stream` appears in
   `Schedule.stream_slots`.
4. **Single writer per slot.** At most one task `writes` any given
   `DataSlot(name, batch_offset)`.
5. **Reads resolve.** Every `reads(slot)` resolves to a matching
   `writes(slot)` produced by a task whose combined position
   (stage index, within-stage position) is strictly earlier.
6. **`depends_on` resolves.** Every `depends_on=("name",)` names an
   existing task whose combined position is strictly earlier.
7. **Acyclic.** The inferred DAG (slot edges + `depends_on` edges +
   same-stream adjacent-in-stage ordering) is acyclic.
8. **Cross-stream wait insertion.** For every consumer→producer edge
   (from rules 5 or 6) where the two tasks bind to different streams,
   the engine auto-inserts `consumer_stream.wait_stream(producer_stream)`
   before the consumer submits. This is for **data visibility**, not
   NCCL ordering.

> **Explicitly not validated:** cross-rank NCCL submission ordering.
> Single-threaded execution makes the submission sequence identical on
> every rank by construction. The cross-rank NCCL-order hazard is a
> multi-threading problem and belongs to Problem #3.

### 4.3 Auto-scheduler (v1)

Inputs:
- Task DAG (edges = slot dependencies + user-declared ordering hints).
- `CostModel` — per-task CPU-time and GPU-time estimates.
- `ResourceInventory` — stream slot names + count, in-flight batches.

Algorithm (v1): list scheduling by priority = remaining critical-path
length, with resources = (stream slot × batch-offset). Post-pass:
auto-insert cross-stream dependencies per §4.2 rule 8.

Cost source (v1): offline warmup — run N iterations of a user-supplied
**default** schedule, record per-task CPU + GPU-event durations, cache
JSON. Users may hand-edit or ship static hints.

No runtime adaptivity in v1.

### 4.4 SchedulablePipeline

```python
class SchedulablePipeline(Generic[In, Out]):
    def __init__(
        self,
        schedule: Schedule,
        stream_pool: StreamPool,
        *,
        nvtx: bool = True,
    ) -> None:
        """Calls init() on every task and resolves named streams."""

    def progress(self, batch_iter: Iterator[In]) -> Out: ...
```

Each iteration, for each stage, each task's `run()` is called in
declaration order under its bound stream context. Cross-stream
`wait_stream` deps are inserted by the engine from slot edges.
Eager-only in v1; CUDA graph mode is deferred to a later spec.

The pipeline is **not** a subclass of the existing `TrainPipeline` ABC;
it's a new abstraction. If a user wants to drop it behind the legacy
`TrainPipelineFactory`, they write their own adapter — that's migration
work, not engine work.

`progress()` returns the value stored in a well-known slot named
`"step_result"` at end-of-iteration (current batch). Shape is
user-defined — whatever your forward Task writes there passes through
unchanged. No `(loss, tokens, output)` tuple baked in. If no task
writes `"step_result"`, `progress()` returns `None`. `SchedulablePipeline`
exposes `RETURN_SLOT: str = "step_result"` as a class attribute so T4
users can override.

### 4.5 Mapping to HugeCTR

For readers familiar with HugeCTR:

| HugeCTR | Our engine | Notes |
|---|---|---|
| `Scheduleable` | `Task` | We add `reads`/`writes`/`batch_offset`. Still has `init()` + `run()`. |
| `StreamContextScheduleable(lambda)` | `Task.from_fn(name, fn)` | Lambda convenience wrapper. |
| `set_stream("dp", priority)` | `StreamPool({"dp": torch.cuda.Stream(priority=...)})` + `Task(stream="dp")` | Stream priority belongs to the stream resource, not to each task. |
| `set_absolute_stream(name)` | `StreamPool({name: existing_stream})` + `Task(stream=name)` | Passing a concrete stream gives the same binding without a task-level flag. |
| `record_done()` / `wait_event(events)` | Auto-inferred from slot + `depends_on` edges; no explicit events API in v1. | User cost: one less escape hatch. Re-add if a use case forces it. |
| `GraphScheduleable(list)` | (deferred — no CUDA graph in v1) | Later spec. |
| `Pipeline::run()` | `SchedulablePipeline.progress(...)` | |
| `Pipeline::run_graph()` | (deferred) | |
| `omp parallel num_threads(local_gpu_count)` | (out of scope — engine is per-rank) | Multi-GPU dispatch is the trainer's job, not the engine's. |
| Inter-iter overlap via "prefetch" stream + manual cache swap | `batch_offset=1` tasks + `BatchRing[In]` | First-class. |
| No DAG, no scheduler | Formal DAG + `AutoScheduler` on top | Our additions. |
| Forward and backward each as separate `Scheduleable`s | Forward as user Task(s); **backward as a regular Task calling `.backward()`**. Intra-autograd hook integration deferred to a follow-up spec. | See §4.6 — forced by PyTorch autograd. |

### 4.6 PyTorch autograd handling

HugeCTR schedules per-layer `bprop` lambdas freely because it has no
autograd engine — each backward is hand-written. PyTorch's
`loss.backward()` is a single atomic call into the C++ autograd engine:
**we cannot reorder ops inside it, we cannot split it into a set of
sub-tasks, and we cannot insert cross-stream waits between its internal
ops.** This is a hard constraint, not a design choice.

Our answer — the one PyTorch-idiomatic approach:

**1. Backward is a regular Task. No dedicated class.**

The engine already wraps every `Task.run(ctx)` call in
`with stream_pool.use(self.stream):`. Empirically verified behavior
(V2 autograd-stream spike, 2026-04-23, `test_engine_autograd_stream_spike.py`):

| Fixture | Result | Implication |
|---|---|---|
| (a) Plain `nn.Linear` (single-stream forward) | ✅ Backward kernels land on declared stream | T1/T2 safe |
| (b) Multi-stream forward (layer on non-default) | ❌ Backward honors **saved-tensor forward stream**, not the declared backward-task stream | Multi-stream-forward models can't force all backward kernels onto one stream |
| (c) DDP-wrapped single-rank | ⚠️ **runtime-conditional xfail** — the test inspects the observed stream and only `pytest.xfail`s if DDP routed the grad kernel off the declared stream. Behavior depends on DDP version / reducer config | Flag for V10 compat matrix |

**What this means for #1 scope:**
- T1 (vanilla): `SchedulablePipeline.basic` default path uses
  single-stream forward → backward-on-declared-stream holds. ≤ 8-line
  adoption claim stands.
- T2 (prefetch + memcpy_stream): H2D is the only work on
  `memcpy_stream`; forward/backward both on `default`. Still
  single-stream forward. ≤ 15-line claim stands.
- **Multi-stream forward is not a #1 use case.** Users whose forward
  spans multiple streams (e.g., model parallelism, manual comm
  overlap during forward) cannot use the backward-Task-on-custom-stream
  pattern directly — their backward naturally runs on the
  forward-originating streams via saved-tensor semantics. This is a
  PyTorch invariant we cannot override. Such users drop to T3/T4 and
  accept whatever stream autograd chooses; or wait for the follow-up
  BackwardHookTask spec for finer control.
- DDP grad-bucket timing: spike shows grad kernels land on DDP's
  internal comm stream, not the user's declared backward stream.
  Verified not to break correctness — PyTorch's own stream
  coordination handles it. V10 preset compat matrix for DDP scenarios
  must assert correctness (numerical match), not stream placement.

So `loss.backward()` is a one-liner in a plain Task for the **common
case** (single-stream forward, no DDP intra-backward hooks):

```python
bwd = Task.from_fn(
    name="backward",
    fn=lambda ctx: ctx.slots["loss"].backward(),
    reads=(DataSlot("loss"),),
    stream="default",
)

opt = Task.from_fn(
    name="optimizer",
    fn=lambda ctx: optimizer.step(),
    depends_on=("backward",),       # pure ordering — no fake slot
    stream="default",
)
```

Backward carries no explicit `writes`; the optimizer declares
`depends_on=("backward",)` to encode the pure-ordering edge (see §4.2
rule 6 — `depends_on` resolution). Backward appears to the scheduler
as a single node in the DAG with a single cost estimate.

**2. Overlap happens *around* backward, not inside it.**

While backward runs on its stream, tasks on other streams (next batch's
H2D on `memcpy`, next batch's sparse-dist on `comm`, optimizer state
preallocation on `default`) run concurrently. This is exactly what
TorchRec's `PrefetchTrainPipelineSparseDist` and the current
`JaggedMegatron*` pipelines already do — the dominant win in recsys
training isn't intra-backward, it's cross-iteration and cross-task
overlap. HugeCTR's `model_pipeline.cpp` exploits the same kind of
overlap at the inter-stage level.

**3. What we give up vs HugeCTR.**

- Cannot reorder top-mlp `bprop` vs bottom-mlp `bprop`.
- Cannot schedule "grad-wrt-input" before "grad-wrt-weight".
- Cannot put `layer_N.bprop` on stream A and `layer_{N-1}.bprop` on
  stream B.
- Cannot fire per-bucket allreduce as grads materialize (the DDP
  overlap pattern) — this would require hooking into the autograd
  worker, which is a follow-up spec.

For 99% of recsys training, the around-backward overlap (point 2) is
where the actual performance win lives, so this is an acceptable v1
limit.

**4. Tests.**

- A regular Task calling `.backward()` puts grad kernels on its
  declared stream (verify via `torch.cuda.current_stream()` captured
  inside a `Tensor.register_hook` in the test).

**5. Out of scope for #1.**

- AOTAutograd / `torch.compile` integration.
- Per-layer backward task decomposition.
- Activation-checkpointing-aware scheduling.
- A dedicated `BackwardTask` class — plain Task suffices.
- Autograd-worker hook integration (`register_hook` /
  `register_multi_grad_hook` as first-class Task variety) — deferred
  to a follow-up spec once a concrete use case demands it.

### 4.7 Adoption path — how existing code opts in

Explicit non-goal: existing user code should require **zero
structural changes** to adopt the engine. Specifically, the following
must stay untouched:

- `class MyModel(nn.Module)` definitions
- optimizer construction
- `DataLoader` / custom iterators
- loss functions
- `model.train()` / `model.eval()` toggling
- DDP / FSDP wrapping
- `torch.amp.autocast` + `GradScaler`
- LR schedulers
- metrics / logging callbacks

**Scope of the ≤ 8-line bar.** The ≤ 8-line training-loop diff
applies to **vanilla** training loops that match this shape:

```python
for batch in dataloader:
    batch = batch.cuda()                      # -1
    optimizer.zero_grad()                      # -1
    out = model(batch)                         # -1
    loss = out if isinstance(out, torch.Tensor) else out[0]  # -1
    loss.backward()                            # -1
    optimizer.step()                           # -1
```

After engine adoption (T1):

```python
pipe = SchedulablePipeline.basic(model, optimizer)   # +1 (pre-loop)
for batch in dataloader:
    pipe.step(batch)                                  # +1 (replaces the 6 body lines)
```

That's **6 deletions + 2 insertions = 8-line `git diff`**. Model,
optimizer, dataloader, loss function, DDP/FSDP/AMP wrapping all stay
untouched. The `pipe.step(batch)` convenience wraps
`pipe.progress(iter([batch]))` for this single-batch-at-a-time pattern.

Loops that add AMP (`autocast` + `GradScaler`), gradient clipping, LR
scheduler `.step()`, gradient accumulation, or conditional optimizer
stepping **cannot hit 8 lines with only a model/optimizer input**;
they need to supply the mutated forward/backward/optimizer bodies via
the 4 escape kwargs. Typical AMP/scheduler loops land around ≤ 15
lines. Anything further (custom loss routing, multi-task losses,
postproc pipelines) drops the user into T3/T4 territory (compose raw
Tasks) — outside the preset's job.

Being explicit about this boundary is better than silently mis-selling
a ≤8-line migration to users whose loops don't fit the vanilla shape.

**Two adoption tiers for #1:**

| Tier | Effort | Tool | Covers |
|---|---|---|---|
| **T1 — vanilla preset** | ~8-line diff | `SchedulablePipeline.basic(model, optimizer)` + `pipe.step(batch)` | Plain `model(batch)→loss` or `(loss, …)`, vanilla `.backward()`, vanilla `.step()`. |
| **T2 — preset + knobs** | ~5–15-line diff | `SchedulablePipeline.basic(model, optimizer, prefetch=True, memcpy_stream=True, forward_fn=…, loss_fn=…, backward_fn=…, optimizer_step_fn=…)` | Adds H2D/prefetch overlap + escape hooks for AMP (`with autocast()` via `forward_fn`, `scaler.scale(loss).backward()` via `backward_fn`), grad clipping (`optimizer_step_fn`), LR scheduler (`optimizer_step_fn`), custom loss (`loss_fn`). |

Two more tiers are **deferred** to a follow-up spec (after #1 lands):

- T3 — custom Tasks with preset components
- T4 — full custom Schedule (HSTU-style heavy customization)

T3/T4 users in the interim can compose raw `Task` / `Schedule` /
`StreamPool` directly (all public). They just don't get curated
component helpers.

**Preset module contract** (`engine/presets/`):

The preset is a **classmethod on `SchedulablePipeline`** — single
public entry point. The 5 component makers (`_make_h2d_task`,
`_make_zero_grad_task`, `_make_forward_task`, `_make_backward_task`,
`_make_optimizer_task`) stay module-private — users don't need them
unless they drop to T3/T4, in which case they compose raw
`Task` / `Schedule` directly.

```python
class SchedulablePipeline(Generic[In, Out]):
    @classmethod
    def basic(
        cls,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        # overlap knobs
        prefetch: bool = False,
        memcpy_stream: bool = False,
        # escape hooks — each is a thin 1-line wrapper in the user's loop
        forward_fn: Callable[[torch.nn.Module, Any], Any] | None = None,
            # default: lambda m, b: m(b).  Override to wrap with
            # torch.amp.autocast(dtype=...) for AMP.
        loss_fn: Callable[[Any], torch.Tensor] | None = None,
            # model_output → scalar loss.  Default auto-extracts:
            # Tensor | tuple[0].  Override for custom routing.
        backward_fn: Callable[[torch.Tensor], None] | None = None,
            # default: lambda l: l.backward().  Override for
            # scaler.scale(loss).backward().
        optimizer_step_fn: Callable[[], None] | None = None,
            # default: optimizer.step.  Override for
            # clip_grad_norm_(params, max_norm) + scaler.step(optimizer)
            # + scaler.update() + scheduler.step().  (scaler.update() MUST
            # follow scaler.step() in every GradScaler loop — bake it in.)
    ) -> "SchedulablePipeline":
        """Assemble a canonical training-step pipeline.

        The four escape hooks cover the canonical pieces of a
        realistic PyTorch loop that the engine can't reach into:

          forward_fn       — forward-pass context wrapping (autocast)
          loss_fn          — loss extraction from model output
          backward_fn      — backward submission (GradScaler)
          optimizer_step_fn — post-backward state mutation
                              (clip / scaler.step / scheduler.step)

        The full model-return goes to slot "step_result" (passed
        through to pipe.progress()).
        """
```

Internal component helpers (not re-exported from `engine/__init__.py`):

```python
# engine/_presets.py (module-private)
def _make_h2d_task(device, *, stream="memcpy", batch_offset=0) -> Task: ...
def _make_zero_grad_task(optimizer, *, stream="default") -> Task: ...
def _make_forward_task(model, *, forward_fn=None, loss_fn=None, stream="default") -> Task: ...
def _make_backward_task(*, backward_fn=None, stream="default") -> Task: ...
def _make_optimizer_task(optimizer, *, optimizer_step_fn=None, stream="default") -> Task: ...
```

**Forward task's slot contract:** writes `"loss"` (scalar, for
backward to read) and `"step_result"` (whatever `model(batch)`
returned, for `pipe.progress()` to return to the caller).

**Protocol for `pipe.progress(batch_iter)`.** Matches the legacy
`TrainPipeline.progress` signature: accepts an iterator, consumes one
batch per call. The engine itself does the `next(batch_iter)` and
places the CPU batch into the ring at `DataSlot("batch_cpu",
batch_offset=max_offset)`. Tasks don't call `next()` themselves.

**HSTU / TorchRec adoption — explicitly out of scope.** The legacy
`JaggedMegatronTrain*Pipeline` classes stay byte-identical. Migration
is a separate PR, separately sized.

### 4.8 Prefill and drain

When `in_flight_batches = N > 1`, a single `progress()` call cannot run
every task — some tasks reference batches not yet loaded (prefill) or
already consumed (drain). The engine handles this transparently so
that **user code calling `progress()` in a loop matches the legacy
`TrainPipeline.progress()` contract one-for-one.**

**User-facing contract (matches legacy TorchRec):**

```python
it = iter(dataloader)          # M batches total
while True:
    try:
        result = pipe.progress(it)   # returns one result per call
    except StopIteration:
        break                        # raised after all M results delivered
```

M calls return M results. Call M+1 raises `StopIteration`. Drop-in
compatible with how users already drive
`JaggedMegatronTrainPipelineSparseDist`.

**Task execution mask rule — single source of truth.** For an
iteration with counter `iter_count` and a task with `batch_offset = k`:

```
run this task ⟺  (max_offset - k) ≤ iter_count < M_known + (max_offset - k)
```

where `max_offset = Schedule.in_flight_batches - 1` and `M_known` is
the total batches pulled so far (= `iter_count + 1` while pulling is
still going; = final `M` once the iterator raised `StopIteration`).

Every task follows this rule. No per-task escape hatch in v1. A
heartbeat / telemetry task attaches to some slot edge (e.g.
`reads=("loss",)`) and conditionally acts inside `fn` based on
`ctx.iter_count`.

**Phases** (observable side-effect of the mask — no separate code
paths):

- **prefill**: `iter_count < max_offset`. Some `k < max_offset -
  iter_count` tasks skip. First `progress()` call absorbs all prefill
  iters + 1 steady iter before returning, so users never see a `None`
  result.
- **steady**: `max_offset ≤ iter_count < M`. All tasks run.
- **drain**: `M ≤ iter_count < M + max_offset`. Tasks where the
  right-hand side of the mask fails (i.e., `iter_count ≥ M + max_offset
  - k`) skip.
- **end**: `iter_count ≥ M + max_offset`. No task satisfies the mask.
  `progress()` raises `StopIteration` to the caller.

**Worked example** — M=3 batches, `in_flight_batches=3` so
`max_offset=2`. Tasks: `h2d` (k=2), `compute` (k=1), `opt` (k=0).

Internal iteration counter vs user's `progress()` call count diverge
during prefill absorption. Tracking both:

| user call | internal iter_count | pulled | h2d (k=2) | compute (k=1) | opt (k=0) | phase | user sees |
|---|---|---|---|---|---|---|---|
| 1 | 0 → 1 → 2 (3 internal iters) | b0, b1, b2 | run, run, run | skip, run, run | skip, skip, run | prefill×2 + steady | **result₀** |
| 2 | 3 | — | skip | run | run | drain | **result₁** |
| 3 | 4 | — | skip | skip | run | drain | **result₂** |
| 4 | 5 | — | skip | skip | skip | end | **`StopIteration`** |

- M=3 batches pulled → 3 user-visible results → 4th (= M+1) user call
  raises `StopIteration`. Matches the user-facing contract above.
- Prefill absorbs `max_offset = 2` internal iterations into user call
  1. So user call 1 internally covers `iter_count = 0, 1, 2` (running
  tasks per their mask individually) before returning result₀ at
  end-of-iter-2.
- Each subsequent user call corresponds to exactly one internal
  iteration.
- **Mask formula applied independently per (internal iter_count,
  task k)** — the worked table above recomputes it row-by-row.

**Edge cases.**
- **Dataloader shorter than `max_offset`** (e.g. M=1, max_offset=2):
  fewer pulls, flip to drain earlier. Still produces exactly M
  results.
- **Empty dataloader** (M=0): first `progress()` raises
  `StopIteration`.
- **Infinite stream**: engine never enters drain; user controls loop
  exit externally.

**Slot lifecycle.** At end of each `progress()` call the `BatchRing`
advances one slot. The slot store of the evicted batch is dropped —
Python refcount releases any tensors it held. No cross-iteration slot
carry-over in v1.

---

## 5. Code style

- Apache-2.0 header on every new file.
- Type hints, `pyre-strict` compatible.
- Stdlib + `torch` only. **No imports** from `torchrec`, `megatron`,
  `fbgemm_gpu`, or repo-local `commons.distributed.*` inside the engine
  package. (Examples under `engine/examples/` may use `torch.distributed`,
  nothing else.)
- `logging.getLogger(__name__)`; no `print`.
- `nvtx.annotate` is opt-in — the engine depends on `nvtx` only if the
  constructor's `nvtx=True` flag is set (default True), and the import is
  lazy so environments without nvtx still work.
- Public API lives in `engine/__init__.py` re-exports; everything else
  is module-private.
- No new third-party runtime deps. Optional dev deps must degrade.

---

## 6. Testing strategy

### 6.1 Unit tests

- `test_task_dag.py` — DAG construction, cycle detection, unresolved
  slot detection.
- `test_schedule_validator.py` — all 8 validity rules of §4.2 each
  have a failing-case test; cross-stream `wait_stream` auto-insertion
  correctness.
- `test_list_scheduler.py` — synthetic DAGs (≤ 8 tasks, ≤ 3 stream
  slots) with known-optimal layouts; assert scheduler matches.

### 6.2 Functional tests

- `test_pipeline_smoke.py` — a tiny `nn.Linear` model trained for 20
  steps via `SchedulablePipeline`. No TorchRec, no Megatron, no NCCL.
  Asserts loss decreases and final params match a hand-written
  non-pipelined loop within `atol=1e-5`.
- `test_determinism.py` — same schedule + same seed on two runs →
  bit-identical loss every step.

### 6.3 Minimal multi-stream regression

- A 2-stream example (`default` + `memcpy`) with an H2D task on
  `memcpy` and a forward on `default`. Assert:
  - The engine inserts the `wait_stream` dep.
  - Nsight trace (captured on CI if available, otherwise synthetic
    CUDA-event timing) shows the two streams actually overlap.

### 6.3.1 Backward correctness (gates V2)

- **Autograd-stream spike** (ran 2026-04-23; see
  `test_engine_autograd_stream_spike.py`): three fixtures verifying
  the §4.6 assumption via `Tensor.register_hook` stream capture.
  - (a) Plain `nn.Linear`: **passes** — backward kernels submit on
    the backward Task's declared stream.
  - (b) Multi-stream forward: **xfail (strict)** — saved-tensor
    stream semantics override the declared backward stream. Out of
    #1 scope; §4.6 documents the limit.
  - (c) DDP-wrapped single-rank (world_size=1, one CUDA device):
    **runtime-conditional xfail** — the test body inspects the actual
    stream the hook fires on and calls `pytest.xfail` iff it differs
    from the declared stream. So behavior depends on DDP version /
    reducer config; we don't assert either outcome unconditionally.
    Flag for V10 compat matrix.
  Spike skip-guard: only when CUDA is absent.
- Numerical correctness: full MLP train loop with backward-on-non-
  default-stream matches a non-pipelined reference within `atol=1e-5`.

### 6.4 Performance acceptance

- On the minimal MLP example, `SchedulablePipeline` step time must be
  ≤ 1.05× a hand-coded multi-stream loop doing the same work. (This
  benchmarks engine overhead, not schedule quality.)
- Auto-scheduler vs. hand-written: within 5% on the same example.

> **No legacy-pipeline parity testing in this spec.** The engine is
> greenfield; it neither aims to match nor replace the `JaggedMegatron*`
> classes within the scope of Problem #1.

---

## 7. Boundaries

### Always do

- Keep the engine free of framework-specific imports — nothing from
  `torchrec`, `megatron`, `fbgemm_gpu`, or `commons.distributed.*`.
- Auto-infer cross-stream `wait_stream` dependencies from slot reader-
  writer stream pairs. Users should never have to call `wait_stream`
  themselves.
- Treat the in-flight batch count as a derived property of the schedule,
  never a pipeline constructor argument.
- Preserve determinism: same schedule + same seed + same data → bit-
  identical outputs across runs.

### Ask first

- Adding any dependency beyond stdlib + `torch` + `nvtx (optional)`.
- Introducing inheritance or a shared base class with the legacy
  `TrainPipeline` ABC — default is **no inheritance**; the new pipeline
  stands alone.
- Any API choice that would later force Problem #3's multi-threaded
  executor to break the Problem #1 API. (Flag the tension, propose
  options.)
- Exposing the engine under the existing `TrainPipelineFactory` string
  keys. Default is **no**; users opt in explicitly.

### Never do

- Touch existing `train_pipeline.py`, `train_pipeline_factory.py`, or
  `utils.py` as part of this spec.
- Import any TorchRec / Megatron / HSTU symbol from the engine package
  (even for type hints — use generics `In`, `Out`).
- Bake a specific batch type, loss shape, or optimizer contract into
  the engine.
- Bake a specific number of in-flight batches into the engine.
- Claim to solve cross-rank NCCL submission-order safety — that is
  Problem #3's job.
- Ship without the functional + determinism tests in §6.2 passing.

---

## 8. Assumptions (please correct)

| # | Assumption | Impact if wrong |
|---|---|---|
| A1 | A Task is a user-supplied Python callable declaring slot-level I/O + a preferred stream slot + a `batch_offset`. It is not bound to any specific module-method convention (no TorchRec `input_dist`/`output_dist`). | Task API shape changes entirely. |
| A2 | Schedule is authored as Python data (`Stage(...)` objects); JSON is only the serialized form for the auto-scheduler's output and cost cache. | Need a parser if YAML/DSL preferred. |
| A3 | Auto-scheduler is offline (one plan per job). Cost comes from an offline warmup profile. | Added complexity of runtime controller. |
| A4 | The engine is greenfield: no inheritance from `TrainPipeline`, no parity with `JaggedMegatron*Pipeline`, no changes to `TrainPipelineFactory`. Users opt in by constructing `SchedulablePipeline` directly. | If you want the new engine to also transparently replace legacy `pipeline_type="prefetch"`, that's a separate migration work item. |
| A5 | Cross-stream dependencies for data visibility are **auto-inferred** by the engine from slot reader/writer + `depends_on` edges. No explicit-event escape hatch in v1 (HugeCTR-style `wait_on` / `records` was cut from scope — re-open if a real use case forces it). | If auto-inference misses a case, we re-add `wait_on` / `records` as a v1.1 follow-up. |
| A6 | Cross-rank NCCL submission-order safety is **not** Problem #1's concern. Single-threaded execution guarantees it by construction. Problem #3 owns it. (Corrected per your note.) | — |
| A7 | In-flight batch count is **derived** from `max(batch_offset) + 1` across all tasks. | Minor — just API shape. |
| A8 | `_rewrite_model` / forward decomposition is entirely Problem #2's scope. The engine does not know how a user's forward was constructed; a user who wants their forward split into multiple tasks writes those tasks themselves. | If #1 must ship with helpers to auto-split a forward, scope expands. |
| A9 | SPEC.md in repo root is fine for now. Can move to `docs/design/` later. | Cosmetic. |
| A10 | CUDA graph capture is **out of scope** for this spec — eager only. (Per user direction.) | If graph capture is in scope, re-add §4 graph stage + `mode` argument. |
| A11 | Engine runs on **one rank at a time**. Multi-GPU dispatch (HugeCTR's `omp parallel num_threads(local_gpu_count)`) is out — each rank constructs its own `SchedulablePipeline`. | If you want the engine to dispatch across local GPUs internally, architecture changes. |
| A12 | `loss.backward()` is called from a **regular Task** (no dedicated `BackwardTask` class). The engine wraps every task's `run()` in its stream context, so autograd kernels naturally submit on the task's declared stream. No intra-backward decomposition, no autograd-worker hook integration — both deferred to follow-up specs. | If you need DDP-bucket-allreduce-style intra-backward overlap right now, that requires an autograd-hook Task and expands scope. |

---

## 9. Non-goals (explicitly out of scope)

- Pipeline parallelism (PP).
- Multi-threaded executor (Problem #3).
- Generalized forward decomposition (Problem #2).
- Dataloader, batch shuffler, mixed-precision, or DDP-internals changes.
- HSTU migration — existing HSTU training continues to use the legacy
  `JaggedMegatron*Pipeline` classes unchanged. Migrating HSTU to the new
  engine is a separate, later initiative.
- Inference pipeline — training only.

---

## 10. Acceptance criteria

Problem #1 lands when **all** of the following hold:

1. `SchedulablePipeline` trains a tiny `nn.Linear` model for 20 steps,
   loss decreases overall (≥15/19 adjacent pairs decreasing, allowing
   minibatch noise), final params match a hand-written
   non-pipelined loop within `atol=1e-5` (functional test).
2. Same pipeline run twice with identical seeds produces bit-identical
   per-step loss (determinism test).
3. A 2-stream example with user-declared `memcpy`-stream H2D overlapping
   `default`-stream compute shows overlap in a timing check.
4. Auto-scheduler on a ≤ 8-task synthetic DAG produces the known-optimal
   schedule (unit test).
5. Engine package imports cleanly with `torch` alone — no `torchrec`,
   `megatron`, or `fbgemm_gpu` import needed.
6. All tests `examples/tests/commons/test_engine_*.py` pass on CI.
7. `examples/commons/pipeline/train_pipeline.py` and its three legacy
   classes are byte-for-byte unchanged (confirmed by git diff).

---

## 11. Open questions for the user

1. **Task API form.** Ship both subclass (stateful, `init()`/`run()`)
   and `Task.from_fn(name, fn)` (lambda wrapper)?
   Default: both. Confirm.
2. **Schedule authoring.** Python-object schedule (A2) OK, or do you
   want a declarative YAML/JSON DSL for human-readable plans?
3. **Auto-scheduler scope.** Offline-only (A3) for v1?
4. **Spec sequencing.** Draft #2 and #3 specs now in parallel so the
   seams between them are visible, or land #1 first?
5. **Performance target.** Confirm 5% auto-scheduler-vs-hand-tuned gap
   and ≤ 1.05× engine-overhead vs hand-coded as provisional targets.
6. **Backend agnosticism.** Engine works for single-GPU, DDP, FSDP, TP
   — no distributed-backend-specific logic. Any hooks you want (e.g.,
   FSDP-aware stream hints)?

---

*Once you confirm/correct §8 and answer §11, I'll move to
`agent-skills:plan` to break this into verifiable tasks.*
