# Schedulable Pipeline Engine

A framework-agnostic PyTorch training-pipeline engine that decouples
**tasks** (what to do) from **schedules** (when to do it, on which
stream). HugeCTR-inspired; see [SPEC.md](../../../../../SPEC.md) for
the full design.

## TL;DR

```python
from commons.pipeline.engine import SchedulablePipeline

pipe = SchedulablePipeline.basic(model, optimizer, loss_fn=lambda out: out.sum())
for batch in dataloader:
    pipe.step(batch)
```

That's it for a vanilla loop. Model, optimizer, dataloader, and loss
convention stay untouched. The engine owns the training-step
orchestration.

## Which tier do I need?

| Your loop | Tier | Tool | Diff vs vanilla |
|---|---|---|---|
| `model(batch) → loss → backward → step` | **T1** | `SchedulablePipeline.basic(model, optimizer, loss_fn=…)` + `pipe.step(batch)` | ≤ 8 lines |
| + AMP / GradScaler | **T2** | Same, plus `forward_fn` (autocast wrapper) + `backward_fn` (scaler.scale) + `optimizer_step_fn` (scaler.step + scaler.update) | ≤ 15 lines |
| + gradient clipping | **T2** | Same, `optimizer_step_fn=lambda: (clip_grad_norm_(...), optimizer.step())` | ≤ 15 lines |
| + LR scheduler | **T2** | Same, `optimizer_step_fn=lambda: (optimizer.step(), scheduler.step())` | ≤ 15 lines |
| + H2D / compute overlap | **T2** | `prefetch=True, memcpy_stream=True` | ≤ 15 lines |
| + multi-threaded task submission | **T2** | `threaded=True` or `ThreadedExecutor(thread_map=...)` — threads and streams decoupled | + 1 line |
| Multi-task loss / custom routing / HSTU-style | **T3/T4** | Compose `Task` + `Schedule` + `StreamPool` directly | varies |

## Examples

- [`adopt_existing_loop.py`](examples/adopt_existing_loop.py) — **T1
  demo**: 8-line diff migration from a vanilla PyTorch loop.
- [`minimal_mlp.py`](examples/minimal_mlp.py) — **T2 demo**:
  prefetch + memcpy-stream overlap, with wall-clock comparison to
  T1.

Run either directly:

```bash
python examples/commons/pipeline/engine/examples/adopt_existing_loop.py
python examples/commons/pipeline/engine/examples/minimal_mlp.py
```

(Run inside the devel container on `ipp1-2029`; the engine needs
`torch` + `nvtx`.)

## Public API

Import surface (from `commons.pipeline.engine`):

| Symbol | Purpose |
|---|---|
| `SchedulablePipeline` | The driver. Use `.basic(...)` classmethod for T1/T2 or construct directly with `Schedule` + `StreamPool` for T3/T4. |
| `Task` | A schedulable unit of work. Subclass or `Task.from_fn(name, fn, …)`. |
| `DataSlot` | Opaque `(name, batch_offset)` handle for inter-task data flow. |
| `Schedule`, `Stage` | Declarative plan consumed by the pipeline. |
| `StreamPool` | Named stream registry. |
| `TaskContext` | Per-iteration handle passed to every `Task.run(ctx)`. Exposes `ctx.slots`, `ctx.stream_pool`, `ctx.iter_count`. |
| `ScheduleValidationError` | Raised when a `Schedule` violates [SPEC §4.2](../../../../../SPEC.md) rules. |
| `ThreadedExecutor` | Multi-threaded executor. Threads and CUDA streams are decoupled; `thread_map` controls task→thread assignment. NCCL ordering safety built in. |
| `SequentialExecutor` | Single-threaded executor (default). |

For the auto-scheduler and cost profiler:

```python
from commons.pipeline.engine.autosched import (
    CostProfiler,  # profile per-task cost via CUDA events
    CostModel,     # per-task cost store (JSON roundtrip)
    schedule_tasks,  # critical-path list scheduler
)
```

## Design constraints (never violated)

- **No framework-specific imports** inside the engine: zero
  `torchrec`, `megatron`, `fbgemm_gpu`, or `commons.distributed.*`.
  Enforced by
  [`test_engine_import_hygiene.py`](../../../../tests/commons/test_engine_import_hygiene.py).
- **Legacy pipeline untouched**:
  `examples/commons/pipeline/train_pipeline.py` +
  `train_pipeline_factory.py` + `utils.py` stay byte-identical on
  the engine branch. Enforced by
  [`test_engine_legacy_untouched.py`](../../../../tests/commons/test_engine_legacy_untouched.py).
- **Single-rank**: the engine is per-rank. Multi-GPU dispatch is
  the trainer's job, not the engine's.
- **Eager execution only**: no CUDA graph capture.
- **Backward is a plain Task**: PyTorch autograd is atomic; no
  intra-backward decomposition. See [SPEC §4.6](../../../../../SPEC.md).

## Scope boundaries

The engine decouples tasks from schedule, adds an auto-scheduler,
and ships a multi-threaded NCCL-safe executor. Generalizing
`_rewrite_model` / FX decomposition beyond `ShardedModule` is left
to per-framework adapters (e.g. `hstu_pipeline`).
