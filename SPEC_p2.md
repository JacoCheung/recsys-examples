# SPEC — Problem #2: HSTU Pipeline Adapter (Option A)

## 1. Objective

Port the HSTU-specific training pipeline (currently
`JaggedMegatronTrainPipelineSparseDist` + `JaggedMegatronPrefetchTrainPipelineSparseDist`
in `examples/commons/pipeline/train_pipeline.py`) onto the Problem #1
schedulable engine, **without** touching the legacy pipeline files.

The new pipeline reuses torchrec's `_rewrite_model`, `PipelinedForward`,
`KJTAllToAllForward`, and torchrec's `TrainPipelineContext` dataclass.
What changes is the **orchestration** — from hardcoded progress-loop
method sequencing to declarative `Schedule` + `Task` graph executed by
`SchedulablePipeline` + `ThreadedExecutor`.

## 2. Scope

**In scope:**
- Two pipeline variants: `hstu_sparse_dist` (non-prefetch) and
  `hstu_prefetch_sparse_dist`.
- HSTU-specific features preserved byte-for-byte in behavior:
  - 2-phase async Karmarkar–Karp shuffler
  - `num_loss_tokens` AllReduce + loss normalization
  - `finalize_model_grads` (Megatron TP)
  - `zero_grad_buffer` hook
  - Stream-discipline `wait_stream(memcpy)` before loss AllReduce
    (NCCL ordering safety)
- Factory registration (new, separate from legacy factory).
- `prefetch_depth` kwarg → Semantic A deep in-flight queue.
- Parity test vs legacy (final params match `atol=1e-5`).

**Out of scope (tracked in `tasks/followups.md`):**
- Semantic B (autonomous / rate-decoupled data pipeline).
- Non-torchrec FX generalization.
- Non-Megatron DDP support.
- Migrating `pretrain_gr_retrieval.py` to default to the new pipeline
  (new pipeline is opt-in via env var or CLI flag).

## 3. File Layout

```
examples/commons/pipeline/hstu_pipeline/         # NEW package
    __init__.py                 # public exports: HSTUPipeline, factory
    pipeline.py                 # HSTUPipeline class (wraps SchedulablePipeline)
    tasks.py                    # HSTU-specific Task factories
    shuffle_tasks.py            # KK shuffle task factories
    factory.py                  # Independent factory with name → class map
    context_mgr.py              # TrainPipelineContext lifecycle within ring slot

examples/tests/commons/
    test_hstu_pipeline_parity.py    # legacy vs new parity (cuda-only)
    test_hstu_pipeline_tasks.py     # unit tests for each task factory
```

**Hard invariants:**
- No edits to `train_pipeline.py`, `train_pipeline_factory.py`,
  `utils.py` — enforced by the existing
  `test_engine_legacy_untouched.py` (extended to cover these files).
- No edits to `engine/` — Problem #1/#3 engine stays framework-free.
- No torchrec / megatron imports from `engine/`; those live in
  `hstu_pipeline/` only.

## 4. Public API

### 4.1 `HSTUPipeline`

```python
class HSTUPipeline:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        # Variant selection
        prefetch: bool = False,           # whether to use prefetch task
        prefetch_depth: int = 1,          # Semantic A: in-flight batches
                                          # beyond the legacy depth of 2/3
        # HSTU features
        batch_shuffler: BaseTaskBalancedBatchShuffler = IdentityBalancedBatchShuffler(),
        assert_nan_loss: bool = False,
        # torchrec _rewrite_model knobs
        apply_jit: bool = False,
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[Callable[[In], Tuple[Tensor, Out]]] = None,
        # Executor knobs
        threaded: bool = True,
        thread_map: Optional[ThreadMap] = None,  # default "by_stream"
    ) -> None: ...

    def progress(self, dataloader_iter: Iterator[In]) -> Tuple[Tensor, Tensor, Out]:
        """Returns (local_loss_sum.detach(), global_tokens, model_output).
        Matches legacy return signature."""

    def attach(self, model) -> None: ...
    def detach(self) -> torch.nn.Module: ...
    def shutdown(self) -> None: ...
    def __enter__(self): ...
    def __exit__(self, *exc): self.shutdown()
```

### 4.2 Factory

```python
# hstu_pipeline/factory.py

class HSTUPipelineFactory:
    _registry: Dict[str, Type[HSTUPipeline]] = {}

    @classmethod
    def register(cls, name: str, klass: Type) -> None: ...

    @classmethod
    def create(cls, name: str, **kwargs) -> HSTUPipeline: ...

# Pre-registered at module import:
# "hstu_sparse_dist"          → HSTUPipeline(prefetch=False)
# "hstu_prefetch_sparse_dist" → HSTUPipeline(prefetch=True)
# "hstu_none"                 → HSTUNonePipeline (synchronous, no engine)
```

HSTU training scripts opt in via env var
`HSTU_USE_SCHEDULABLE_PIPELINE=1` or CLI flag `--pipeline-engine schedulable`.
Default path remains the legacy factory — zero risk to existing users.

## 5. Task Graph

### 5.1 Stream layout

| Stream | Purpose | Priority |
|---|---|---|
| `default` | compute, backward, optimizer | 0 |
| `memcpy` | H2D, shuffle AllGather + index_select | -1 |
| `data_dist` | input_dist splits + tensors all-to-all | -1 |
| `prefetch` (prefetch variant only) | `sharded_module.prefetch()` | -1 |

### 5.2 Slot layout

Stored in `ctx.slots` at each batch's `batch_offset`:

| Slot | Holds | Written by |
|---|---|---|
| `raw_batch_cpu` | CPU batch pulled from loader | engine-populated (`batch_cpu`) |
| `raw_batch_gpu` | H2D'd CPU→GPU batch, pre-shuffle | h2d task |
| `shuffled_batch` | post-shuffle batch ready for input_dist | finish_shuffle task |
| `torchrec_ctx` | `TrainPipelineContext` per batch | h2d task (fresh ctx) |
| `global_tokens` | scalar AllReduce result | global_tokens task |
| `shuffle_handle` | `ShuffleHandle` from start_shuffle_async | start_shuffle task |
| `losses` | forward output tensor | forward task |
| `output` | forward secondary output | forward task |
| `step_result` | `(local_loss.detach(), global_tokens, output)` | optimizer task |

### 5.3 Task list (non-prefetch variant, `prefetch_depth=1`)

`max_offset = 2`, so ring has 3 slots. Legacy-equivalent depth.

| # | Task name | batch_offset | stream | nccl | depends_on / reads | writes |
|---|---|---|---|---|---|---|
| 1 | `h2d` | 2 | memcpy | - | reads `raw_batch_cpu` | `raw_batch_gpu`, `torchrec_ctx` |
| 2 | `start_shuffle` | 2 | memcpy | - | reads `raw_batch_gpu` | `shuffle_handle` |
| 3 | `finish_shuffle` | 1 | memcpy | ✓ (AllGather) | reads `shuffle_handle` | `shuffled_batch` |
| 4 | `start_input_dist` | 1 | data_dist | ✓ (all_to_all) | reads `shuffled_batch`, `torchrec_ctx` | (mutates torchrec_ctx) |
| 5 | `wait_input_dist` | 1 | data_dist | - | depends_on=`start_input_dist` | (mutates torchrec_ctx) |
| 6 | `zero_grad_buffer` | 0 | default | - | - | - |
| 7 | `zero_grad` | 0 | default | - | depends_on=`zero_grad_buffer` | - |
| 8 | `global_tokens_allreduce` | 0 | default | ✓ | reads `raw_batch_cpu@0` (for num_loss_tokens) | `global_tokens` |
| 9 | `wait_for_batch` | 0 | default | - | depends_on=`wait_input_dist` (cross-offset via carry) | - |
| 10 | `wait_memcpy_for_nccl_safety` | 0 | default | - | depends_on=`finish_shuffle` (cross-offset) | - |
| 11 | `forward` | 0 | default | - | reads `torchrec_ctx@0`, `raw_batch_cpu@0` | `losses`, `output` |
| 12 | `backward` | 0 | default | ✓ (DDP grad AllReduce) | reads `losses`, `global_tokens` | (grads) |
| 13 | `finalize_model_grads` | 0 | default | ✓ (TP AllReduce) | depends_on=`backward` | - |
| 14 | `optimizer_step` | 0 | default | - | depends_on=`finalize_model_grads` | `step_result` |
| 15 | `watchdog_step` | 0 | default | - | depends_on=`optimizer_step` | - |

**NCCL ordering**: tasks 3, 4, 8, 12, 13 are `nccl=True`. Declaration
order inside the schedule equals legacy call order → cross-rank
submission sequence preserved.

### 5.4 Task list (prefetch variant)

Adds one task, uses `max_offset = 3`:

| # | Task name | batch_offset | stream | nccl | Notes |
|---|---|---|---|---|---|
| 3.5 | `prefetch_embeddings` | 1 | prefetch | - | After `wait_input_dist`, before `forward` |

All other tasks shift `batch_offset` up by 1 for h2d/shuffle/input_dist
rows so that compute still lives at `batch_offset=0`.

### 5.5 Deep queue (`prefetch_depth > 1`)

Increases `max_offset` by `prefetch_depth - 1`. The extra depth sits
between `input_dist` and `compute`, meaning `wait_input_dist` results
buffer up in the ring before compute consumes them. Tasks from table
5.3 with offset > 0 all shift up; compute-side offset stays 0.

## 6. Context Lifecycle

`TrainPipelineContext` is torchrec's per-batch state (splits
awaitables, module contexts, postproc results). It travels with the
batch through the ring:

1. `h2d` task creates a fresh `TrainPipelineContext` and stores it in
   slot `torchrec_ctx` at the batch's top offset.
2. `start_input_dist` mutates it (fills `input_dist_splits_requests`).
3. `wait_input_dist` mutates it (fills `input_dist_tensors_requests`
   → awaited results cached back into the context).
4. `forward` task, before invoking `model(batch)`, calls
   `_set_module_context(ctx.slots["torchrec_ctx"])` — this walks all
   `PipelinedForward` instances installed by `_rewrite_model` and
   gives them the right context for popping input_dist results.
5. `ring.advance()` at iteration boundary evicts the slot containing
   the just-consumed context.

This is the "dual context" design — our engine's `SlotStore` holds
torchrec's `TrainPipelineContext` as one of the slot values. There's
no separate `deque` in `HSTUPipeline`.

## 7. `_rewrite_model` Integration

Called exactly once at `HSTUPipeline.__init__`:

```python
from commons.pipeline.utils import _rewrite_model, _override_input_dist_forwards

pipelined_modules, model, original_forwards, postproc_modules, pipelined_module_names = \
    _rewrite_model(
        model=model,
        context=TrainPipelineContext(),  # stub; real ones created per batch
        dist_stream=stream_pool.get("data_dist"),
        batch=peek_batch,
        apply_jit=apply_jit,
        pipelined_forward=PipelinedForward if not prefetch else PrefetchPipelinedForward,
        pipeline_postproc=pipeline_postproc,
        default_stream=stream_pool.get("default"),
    )

# Monkeypatch KJTAllToAllForward on the rewritten modules:
original_kjt_dist_forwards = _override_input_dist_forwards(pipelined_modules)

# Save for detach():
self._pipelined_modules = pipelined_modules
self._original_forwards = original_forwards
self._original_kjt_dist_forwards = original_kjt_dist_forwards
```

The peek_batch is obtained from a user-provided optional `peek_batch`
kwarg (matching the legacy pattern of running one iteration through
FX to identify pipelineable ops).

## 8. Validation

### 8.1 Parity test

`test_hstu_pipeline_parity.py::test_new_matches_legacy`:
- Construct identical model, optimizer, same torch seed.
- Legacy: `JaggedMegatronTrainPipelineSparseDist(model_a, opt_a, ...)`.
- New:    `HSTUPipeline(model_b, opt_b, ...)`.
- Drive both for 20 batches from the same deterministic dataloader.
- Assert `torch.allclose(p_a, p_b, atol=1e-5, rtol=0)` for every
  parameter.
- Assert returned `(loss, global_tokens)` match across all 20 steps.

Marked `@pytest.mark.skipif(not torch.cuda.is_available())` and
requires multi-process NCCL fixture (at least 2 ranks). Uses a tiny
sharded embedding model (non-HSTU, synthetic) so it runs on any 2-GPU
box without real HSTU data.

### 8.2 Unit tests

`test_hstu_pipeline_tasks.py`:
- each task factory (h2d, start_input_dist, wait_input_dist, prefetch,
  start_shuffle, finish_shuffle, forward, backward, optimizer_step)
  exercised in isolation with mocked dependencies.
- validator rejects invalid schedules (e.g. compute at offset > 0).

### 8.3 Factory test

- `HSTUPipelineFactory.create("hstu_sparse_dist", ...)` returns the
  right class.
- Unknown name → `KeyError`.

## 9. Non-Goals

- Performance beats legacy — parity is the bar. Any speedup from
  multi-threaded submission is bonus, not required.
- Removing legacy variants — `jagged_sparse_dist` etc. stay as-is.
- Adding new features (e.g. gradient accumulation, pipeline parallelism).

## 10. Assumptions

1. torchrec's `_rewrite_model`, `PipelinedForward`,
   `KJTAllToAllForward`, `TrainPipelineContext` are import-stable
   within the pinned torchrec version.
2. Schedule construction is cheap enough to do at `HSTUPipeline.__init__`
   time (no dynamic stage reshuffling per-batch).
3. All NCCL tasks' relative declaration order is the ONLY ordering
   constraint across ranks. Within a rank, NCCL ticket lock enforces it.
4. `ThreadedExecutor(thread_map="by_stream")` is the default; users can
   override to pin NCCL-heavy tasks to a single thread for extra safety.

## 11. Acceptance Checklist

- [ ] `hstu_pipeline/` package created with all 6 files.
- [ ] `HSTUPipeline` init succeeds on a 2-GPU sharded embedding model.
- [ ] `progress()` returns legacy-matching `(loss, tokens, output)`.
- [ ] Parity test green (20 steps, atol=1e-5).
- [ ] `_override_input_dist_forwards` correctly restored on `detach()`.
- [ ] `HSTUPipelineFactory` pre-registers both variants.
- [ ] Existing `test_engine_legacy_untouched.py` extended to also guard
      `train_pipeline.py`, `train_pipeline_factory.py`, `utils.py` from
      edits (no-op change since they're already guarded — verify).
- [ ] No new imports of torchrec/megatron from `engine/`.
- [ ] Codex review passes after full implementation.
