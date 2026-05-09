# SPEC — Problem #3 slice 2: Event-based cross-stream synchronization

Status: **draft**. Promoted from `tasks/followups.md` (Codex B-MEDIUM /
C-HIGH on commit 23414493). Engine-level refactor; affects all
pipeline users.

## 1. Objective

Replace the current **stream-granularity** `wait_stream(producer_stream)`
cross-stream sync with **task-granularity** `wait_event(producer_event)`.
This gives the engine the ability to express "consumer waits for this
specific producer operation" rather than "consumer waits for everything
queued on the producer stream up to this submission point".

Same change applies to the CPU-side ordering via
`threading.Event` in `ThreadedExecutor._compute_cpu_deps`.

## 2. Motivation

See `tasks/followups.md` entry "Engine: event-based cross-stream sync".
Short version: HSTU's steady-state pipeline has

    iter N: [finish_shuffle@2 (on memcpy)]  ← current iter, for batch N+2
            [forward@0         (on default)] ← reads shuffled_batch@0 from iter N-1 producer

`forward` calls `wait_stream(memcpy)` which waits for ALL pending
memcpy work, including (a) the prior iter's shuffle for forward's
batch (needed) and (b) the CURRENT iter's shuffle for a FUTURE
batch (not needed). The io/compute overlap collapses.

With event-based sync:
  - Each task records an event on its stream at end of `task.run`
  - Events travel down the ring via `ring.advance()` alongside slots
  - Consumer waits only on the SPECIFIC producer event that wrote
    the data it reads — ignoring concurrent producer activity on
    the same stream

## 3. Scope

**In scope:**
- Replace `wait_stream` with `wait_event` in `_apply_cross_stream_waits`.
- Per-task post-event recording in `SchedulablePipeline._run_one_internal_iter`.
- Event storage in slot store (alongside data slots); travels with
  `ring.advance()`.
- `infer_cross_stream_waits` / `_compute_cpu_deps` re-key on
  `(slot.name, batch_offset)` instead of `slot.name`.
- Producer resolution uses offset arithmetic: reader@J pairs with
  writer@K (K ≥ J); the event recorded at offset K in iter
  (N-(K-J)) has been carried down to offset J by iter N.

**Out of scope (future slices):**
- Engine ring pre-population API (that's a separate followup for
  the HSTU bootstrap fix — interaction surface is small).
- Re-pointing existing Problem #1 smoke/unit tests if any rely on
  stream-granularity sync behavior (unlikely — they're coarse).
- Performance micro-benchmarks (nsys trace collection) — orthogonal.

## 4. Design

### 4.1 Event storage

Each slot store gains a parallel "event store" keyed by task name:

```python
class SlotStore:
    _data: Dict[str, Any]        # existing
    _events: Dict[str, torch.cuda.Event]   # NEW — per producer task
```

When a task with `writes=(DataSlot("X", K),)` finishes, the engine
records an event on the task's stream and stores it in
`ring.at(K).events[task.name]`. `ring.advance()` already shifts slot
stores by one position; the `_events` dict travels with them.

### 4.2 Wait-event inference

`engine/deps.py::infer_cross_stream_waits` becomes
`infer_cross_stream_event_waits` returning:

```python
Dict[str, Tuple[Tuple[str, int], ...]]   # consumer_task_name -> [(producer_task_name, producer_offset), ...]
```

Consumer at `reads=(DataSlot("shuffled_batch", 0),)` with producer
`finish_shuffle` writing `DataSlot("shuffled_batch", 2)` yields a
wait entry `(finish_shuffle, 2)` — the consumer's wait target is
"whatever event was recorded at offset 2 by finish_shuffle, now
available at offset 0 after ring advance".

### 4.3 Wait site

In `_run_one_internal_iter`, before `task.run(ctx)`:

```python
if waits := cross_stream_event_waits.get(task.name, ()):
    consumer_stream = torch.cuda.current_stream()
    for (producer_name, producer_offset_at_write_time) in waits:
        # After ring advance, the event has moved from the producer
        # offset to the slot at ``task.batch_offset``. Look it up
        # in the current slot store's event dict.
        event = ctx.slots._events.get(producer_name)
        if event is not None:
            consumer_stream.wait_event(event)   # wait_event, NOT wait_stream
```

### 4.4 Post-task event recording

After `task.run(ctx)` returns (CUDA kernels enqueued on its stream):

```python
if device_type == "cuda":
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream())
    ring.at(task.batch_offset)._events[task.name] = event
```

### 4.5 CPU-side ordering (`ThreadedExecutor._compute_cpu_deps`)

Same (name, offset) keying. `threading.Event` lookup uses
`(producer_task_name, slot_offset_after_advance)` instead of
current-iter producer task instance. Producer events from prior
iters have ALREADY been set (they came from the prior iter's
`task.run` completion), so the `Event.wait()` returns immediately
except in the genuine edge case where a slow producer hasn't yet
completed on wall-clock.

## 5. API changes

**Non-breaking:**
- `SlotStore.events` property (new, additive).
- `Task.writes` semantics unchanged — still declares slot writes.
- `DataSlot(name, batch_offset)` unchanged.

**Internal:**
- `infer_cross_stream_waits` → `infer_cross_stream_event_waits`.
  Old function removed. Only internal callers affected.
- Executor's `_apply_cross_stream_waits` renamed to
  `_apply_cross_stream_event_waits`.

**Config switch (transition period):**
- `SchedulablePipeline(..., cross_stream_sync="event")` default.
- `cross_stream_sync="stream"` available as fallback for
  ~one release to validate.

## 6. Test plan

### 6.1 Unit

- `test_engine_event_sync.py` — 5-10 tests for the new helpers
  (event recording, offset-keyed producer lookup, ring advance of
  events, wait_event emission in threaded mode).

### 6.2 Integration

- Existing `test_hstu_pipeline_parity.py` + `test_hstu_pipeline_threaded.py`
  must pass unchanged under the new sync. Numerical parity preserved.

### 6.3 Perf

- NSYS trace comparison: HSTU native variant with non-identity
  shuffler, depth=1, 100 iters. Metric: idle time on default
  stream between memcpy-stream collectives. Expected improvement
  depends on shuffler cost; even 10-20% overlap improvement is
  meaningful.

### 6.4 Regression

- `test_engine_determinism.py` — bit-exact determinism must
  survive event-based sync (events carry natural ordering, so
  this should be a no-op assertion).

## 7. Acceptance criteria

- [ ] All existing engine/HSTU tests pass under the new sync
      (smoke + multi-stream + multi-batch + threaded + parity).
- [ ] New unit tests for event mechanism (§6.1).
- [ ] No regression on determinism tests (§6.4).
- [ ] HSTU parity suite still 21 pass / 4 xfail baseline.
- [ ] NSYS trace shows narrower cross-stream dependency edges
      (§6.3).
- [ ] Codex review pass.

## 8. Non-goals

- Rewrite `wait_stream` callers outside the engine's
  `_apply_cross_stream_*` helper (e.g., user code that directly
  calls `wait_stream`). Those are user responsibility.
- Provide a C++ / CUDA-graph-compatible variant. Event recording
  in eager mode only.

## 9. Open questions

1. **Event pool reuse**: creating `torch.cuda.Event()` per task per
   iteration is ~cheap but not free. Consider a pool-backed
   allocator in `streams.py` if profiling shows allocation
   overhead. Default: create fresh; pool is a micro-opt.

2. **Cross-iter events on CPU**: `threading.Event` has no
   "record event on stream" analog; it's strictly one-shot
   set/wait. For cross-iter CPU deps, need to track events by
   (producer_name, iter_count). Easier to keep CPU deps
   slot-name-based for now since the race was on GPU
   granularity.

3. **Error paths**: what if an upstream task raises before
   recording its event? The consumer's wait_event on the
   unrecorded event blocks forever. Need a fallback (timeout
   or sentinel event that signals failure). `ThreadedExecutor`'s
   cancellation flag can be consulted before wait_event.

4. **Graph-capture compatibility (deferred)**: torch CUDA graphs
   don't currently capture `cudaEventRecord` reliably. If a
   future slice wants CUDA-graph support, revisit.

## 10. Rollout

1. Land engine changes under `cross_stream_sync="event"` default
   with `"stream"` as opt-out.
2. Run full regression — engine + HSTU + existing training jobs.
3. 1 week soak.
4. Remove `cross_stream_sync` kwarg (events-only).
