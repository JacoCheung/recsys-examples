# SPEC: Auto-convert `cross_iter_depends_on` Δ=0 to same-progress sync

**Status**: **Implemented** 2026-04-28 (UTC+8). 233 engine tests pass; GPU end-to-end correctness test parametrized over both forms (`same_progress_sync` explicit vs `cross_iter` Δ=0 auto-promoted) confirmed equivalent logits and final weights to a manual la=1-pipelined SGD baseline.
**Author**: junzhang
**Date**: 2026-04-28 (UTC+8)

## Background

Today the engine rejects `cross_iter_depends_on=((X, -N),)` whenever
the closed-form Δ = `X.la + N − consumer.la` evaluates to 0. The error
message tells the user to rewrite as `same_progress_sync=(X,)`.

**Two independent reviews** (gpt-5.4 yolo, gpt-5.5 yolo + stage-descending
reference doc) verified that the *mechanical contract* needed to make
Δ=0 `cross_iter_depends_on` work is **identical** to what
`same_progress_sync` already provides:

1. Add a topological-sort edge X → self (so X fires before self in
   the same progress).
2. For threaded execution, add a CPU-side `threading.Event` wait
   (cross-thread case).
3. For cross-stream, emit `wait_event` at `slot[producer.la]`;
   same-stream relies on CUDA stream FIFO.

The remaining difference is **only API labeling**: the user might
naturally write `cross_iter_depends_on` ("I want X from N batches
ago") when expressing batch-flow logic, while `same_progress_sync`
reads as "wait for X in this progress." Both should compile to the
same engine actions when Δ=0.

## Goal

Eliminate the Δ=0 rejection. The engine internally maps Δ=0
`cross_iter_depends_on` to the same enforcement contract as
`same_progress_sync`, so the user keeps a single consistent mental
model of `cross_iter_depends_on=((X, -N),)` regardless of whether
N happens to make Δ=0 or Δ≥1.

`same_progress_sync` remains a public field for the original use cases
where the user does NOT want to express batch logic (out-of-slot
state coherency, aux logging).

## Non-goals

- Removing `same_progress_sync` as a public field. Use cases (a) and
  (c) in §3.7 stay valid and natural.
- Rewriting the topological-sort core algorithm.
- Stage/period scheduler from `sw_pipeline_api.html` (separate effort
  if ever needed).

## Current behavior (snapshot)

| Δ | Cross-stream behavior | Same-stream behavior |
|---|---|---|
| Δ < 0 | `raise ValueError("future-read")` at construction | same |
| **Δ = 0** | **`raise ValueError(... use same_progress_sync ...)`** | **same** |
| Δ ≥ 1, slot_offset ≥ 0 | emit `(producer, producer.stream, consumer.la − N)` triple | short-circuit `continue` (FIFO) |
| Δ ≥ 1, slot_offset < 0 | `raise ValueError("rotated out of ring")` | short-circuit `continue` (FIFO) |

`_build_same_progress_dag_edges` excludes `cross_iter_depends_on`
entirely; only `reads/writes` exact slot, matching-la `depends_on`,
and `same_progress_sync` contribute topo edges.

## Proposed behavior

### Validator (in `infer_cross_stream_event_deps`, deps.py around L437)

Replace the Δ=0 rejection branch with an **auto-conversion** path:

```
N = -neg_offset
delta = producer.la + N - consumer.la

if delta < 0: raise (future-read, unchanged)

if delta == 0:
    # Auto-convert: same semantics as same_progress_sync.
    # Skip the ring-rotation slot_offset emit; instead emit at
    # producer.la (the same slot where producer recorded *this*
    # progress).
    if producer.stream != consumer.stream:
        triples.append((producer.name, producer.stream, producer.batch_offset))
    # same-stream: no triple needed; FIFO + topo edge handles it.
    continue   # done with this dep_name

# delta >= 1: existing path unchanged
slot_offset = consumer.la - N
if producer.stream == consumer.stream:
    continue   # FIFO short-circuit
if slot_offset < 0:
    raise (rotated out, unchanged)
triples.append((producer.name, producer.stream, slot_offset))
```

### Topo-DAG edge inclusion (in `_build_same_progress_dag_edges`)

When iterating cross_iter_depends_on entries, ADD a topo edge
producer → consumer **iff Δ=0**:

```
for task in tasks:
    for dep_name, neg_offset in task.cross_iter_depends_on:
        producer = name_to_task.get(dep_name)
        if producer is None or producer.name == task.name:
            continue
        N = -neg_offset
        delta = producer.batch_offset + N - task.batch_offset
        if delta == 0:
            incoming[task.name].add(producer.name)
        # delta != 0: no topo edge (existing behavior; ring handles)
```

This is the mechanical equivalent of how `same_progress_sync` is
handled in the same function.

### CPU-side dep edge (in executor `_compute_cpu_deps`)

`_compute_cpu_deps` should also recognize Δ=0 `cross_iter_depends_on`
and add a CPU `threading.Event` cross-thread edge — same pattern as
`same_progress_sync`. Since the topo edge in `_build_same_progress_dag_edges`
already establishes producer-before-consumer, the CPU edge is the
threading-side completion-event wait.

(The existing `_compute_cpu_deps` already iterates `same_progress_sync`;
extend the same loop to also pick up Δ=0 cross_iter entries, or
factor a helper.)

### Task.__init__ field-overlap check (task.py)

No change. The existing rule "a producer name cannot appear in two of
{depends_on, cross_iter_depends_on, same_progress_sync}" still holds.
Authors who write `cross_iter_depends_on=((X,-N),)` with Δ=0 will
have it auto-promoted; if they ALSO try `same_progress_sync=(X,)`
that's a collision and raises (correctly — they're saying the same
thing twice).

## Test plan

1. **Existing rejection tests for Δ=0 must be rewritten** to assert
   *acceptance* + correct topo order (update before fwd) +
   correct event_deps emission.
   - `test_engine_cross_iter_depends.test_cross_iter_dep_delta_zero_rejected_directs_to_same_progress_sync`
     → flip to `test_cross_iter_dep_delta_zero_accepted_as_same_progress`.
   - `test_engine_cross_iter_depends.test_cross_iter_dep_delta_zero_same_stream_still_rejected`
     → flip to `test_cross_iter_dep_delta_zero_same_stream_accepted_via_topo_edge`.

2. **Static-analysis test** confirming Δ=0 cross_iter NOW contributes
   a topo edge. Update
   `test_cross_iter_does_not_contribute_topo_or_cpu_edge` to be
   parametrized: cross_iter Δ≥1 contributes no topo edge (existing
   behavior); cross_iter Δ=0 DOES contribute one (new behavior).

3. **End-to-end correctness on GPU**: parametrize the new
   `test_engine_same_progress_sync_correctness.py` to also run with
   `fwd.cross_iter_depends_on=(("update", -1),)` instead of
   `fwd.same_progress_sync=("update",)`. Both must produce
   identical logits / weights as the manual baseline.

4. **GPU repro script** `tasks/cross_iter_delta0_repro.py` (the
   monkey-patched bypass demonstrator) becomes obsolete; replace
   with `tasks/cross_iter_delta0_correctness.py` showing both
   field forms produce equal outputs.

5. **Future-read and rotated-out tests** unchanged — those rejection
   paths stay.

## Doc updates needed

1. **§3.6 changelog "Cross-iter strict-cross-progress + future-read
   validators"** — drop the Δ=0 strict-rejection language. Replace
   with "Δ=0 cross_iter is auto-promoted to same_progress_sync
   semantics (same topo edge + slot[producer.la] event lookup)."

2. **§3.7 "same_progress_sync — when to use it"** — add a 4th use
   case row labeled "(d) implicit, via cross_iter Δ=0 auto-promotion"
   pointing back to `cross_iter_depends_on` syntax.

3. **Appendix A.3 constraint table** — remove the `Δ=0 → REJECTED`
   row, replace with `Δ=0 → auto-promoted (no ring rotation; topo
   edge + slot[producer.la])`.

4. **Appendix A.4 plain-language summary** — drop the "Δ=0: not
   really cross-iter, use same_progress_sync" bullet; replace with
   "Δ=0: same-progress equivalent — engine auto-handles it the same
   way same_progress_sync would."

5. **Appendix A.5 examples table** — flip the Δ=0 row from
   `REJECTED` to `OK (auto-promoted)`.

6. **task.py docstring** for `cross_iter_depends_on` — note that
   Δ=0 is a valid implicit form of same-progress sync.

## Backward compatibility

- **Looser, not stricter.** Configurations that previously raised
  now succeed. Existing valid configurations are unchanged.
- No public API change. Field signatures unchanged.
- Field-overlap rule still prevents duplicate declarations.

## Risks

1. **Validator simplicity loss.** The cross_iter handler grows a
   new auto-promote branch. Mitigated by extracting a small helper
   that mirrors the same_progress_sync emit code.

2. **User confusion: "is my dep cross-progress or same-progress?"**
   Users won't directly see the difference at runtime; both produce
   correct ordering. Documentation must explicitly call out the
   auto-promotion so authors aren't surprised when Δ=0 doesn't error.

3. **Test churn.** Several rejection tests need polarity flipped.
   Acceptable cost for cleaner UX.

4. **Field-overlap edge.** A user might write
   `fwd.cross_iter_depends_on=(("update",-1),)` AND
   `fwd.same_progress_sync=("update",)` — the existing overlap
   guard still raises. (Already handled.)

## Open questions for review

- Q1: Should the engine emit a `logging.debug` line on auto-promotion
  so power-users can see the conversion happened? Probably not —
  noisy and unnecessary.
- Q2: Is "Δ=0 auto-promote" a permanent contract or a transitional
  ergonomic? Probably permanent — there's no value in ever rejecting it.
- Q3: Should the doc still recommend `same_progress_sync` for use
  cases (a) and (c)? Yes — for those cases, batch-flow language is
  awkward. Auto-promotion is a convenience for authors whose mental
  model is batch-centric.

## Implementation order (when approved)

1. Update `_build_same_progress_dag_edges` to add Δ=0 cross_iter
   topo edges.
2. Update `_compute_cpu_deps` to add Δ=0 cross_iter CPU edges.
3. Update `infer_cross_stream_event_deps` cross_iter handler:
   replace Δ=0 raise with auto-promote emit.
4. Flip / extend the affected unit tests.
5. Update doc sections per "Doc updates needed".
6. Re-run full engine test suite + GPU correctness test.
7. Sync doc to viz repo and push Pages.

---

## Ready-first considered and rejected

**Recommendation:** do not adopt ready-first scheduling as part of this Δ=0 `cross_iter_depends_on` auto-convert spec. The Δ=0 fix only needs the same mechanical contract already used by `same_progress_sync`: add a same-progress topo edge, add the cross-thread CPU event edge, and emit cross-stream event lookup at `slot[producer.la]`.

Source reviewed: [SWPipeline API, Ready-First Scheduling Algorithm](https://jacocheung.github.io/visualization/pipeline/sw_pipeline_api.html#ready-first), re-fetched on 2026-04-28 (UTC+8). Review by gpt-5.5 + xhigh effort with curl access.

### Extracted ready-first algorithm

The SWPipeline doc describes ready-first as an `_build_enqueue_order` strategy that "builds a period-local dependency graph" and sorts ready nodes by "`(stall_cost, name)` priority".

Inputs / state:

- `self._defs`: task-name to `PipelineTask`.
- `self._stage_map`: task-name to SWPipeline `stage`.
- `self._stream_map`: task-name to CUDA stream.
- `self._intra_iter_deps`: same-iteration logical deps.
- `self._inter_iter_deps`: previous-iteration deps.

Output:

- `List[PipelineTask]`: one construction-time enqueue order for each SWPipeline period.

Algorithm, paraphrased from the doc:

```python
adj = {task_name: []}
in_degree = {task_name: 0}

# Period-local graph.
for task, deps in intra_iter_deps:
    for dep in deps:
        if stage[dep] == stage[task]:
            add edge dep -> task

for task, deps in inter_iter_deps:
    for dep in deps:
        if stage[dep] == stage[task] + 1:
            add edge dep -> task

# Priority.
stall_cost[task] = count of period-local incoming edges dep -> task
                   where stream[dep] != stream[task]

# Kahn topo sort.
ready = zero_indegree_tasks sorted by (stall_cost[task], task_name)
while ready:
    pop first ready task
    append to order
    release successors
    re-sort ready by (stall_cost[task], task_name)

return tasks in order
```

Edges excluded by the SWPipeline algorithm are stage-separated dependencies whose producer ran in an earlier period; those are assumed to have CPU and CUDA events already recorded by the time the consumer period is enqueued.

### Comparison to this engine

This engine does not have SWPipeline's `period / iter / stage` scheduler. Construction currently reorders single-stage schedules with `topological_sort(schedule)` and leaves declaration order only as the tie-breaker for DAG-independent tasks (`pipeline.py:84-101`, `deps.py:126-183`). The DAG is built from exact-slot `reads/writes`, matching-lookahead `depends_on`, and `same_progress_sync` (`deps.py:52-123`). `cross_iter_depends_on` is currently excluded from that DAG (`deps.py:70-80`), which is exactly the Δ=0 bug this spec fixes.

At runtime, each internal progress computes the §4.8 active-task mask (`pipeline.py:225-237`) and passes the filtered active list to the executor (`pipeline.py:275-287`). `ThreadedExecutor` filters active tasks (`executor.py:432`), partitions them by `thread_map` while preserving the already-sorted task order (`executor.py:454-463`), runs the all-same-thread path serially in that order (`executor.py:464-480`), or runs per-thread FIFO chains with CPU dependency waits (`executor.py:489-568`). CPU-side deps already include exact-slot reads, `depends_on`, `same_progress_sync`, and same-stream predecessor ordering (`executor.py:189-270`). GPU event waits are applied immediately before each task (`executor.py:73-129`), and `same_progress_sync` already emits event lookup at `producer.batch_offset` (`deps.py:534-556`).

### Adoption value

Ready-first can solve a real but separate performance problem: if a task with a cross-stream same-progress wait is ordered before independent work on the same CUDA stream, that independent work can sit behind a `wait_event`.

Concrete example in this engine:

```text
Q: stream Y, no deps
A: stream X, same_progress_sync=("Q",)   # cross-stream waiter
P: stream X, no deps
```

If declaration order and current topo tie-break produce `Q, A, P`, stream X receives `A.wait_event(Q)` before `P`, so `P` idles behind A's wait. A ready-first-like tie-break would prefer zero-stall `P` before cross-stream-waiting `A`, producing `Q, P, A`.

Users would see changed fire order for DAG-independent tasks. That can improve latency for independent same-stream work and may reduce GPU stream idle time, but it can also change log order, NVTX order, side-effect timing, and tests that intentionally rely on declaration-order tie-breaks. Steady-state throughput only improves when this ordering-induced wait is on the critical path; if the main compute stream is already saturated, ready-first is mostly cosmetic.

### Compatibility

Full SWPipeline ready-first is not directly compatible with the current `lookahead` / `BatchRing` model. Its edge inclusion rules depend on explicit SWPipeline `stage` and `period` identities. Mapping `lookahead` to SWPipeline `stage` would be wrong: this engine's same-progress relations are defined by active tasks in a `progress()` call plus ring offsets, and `same_progress_sync` can intentionally connect tasks with different lookaheads.

A limited subset would be compatible: keep this engine's existing same-progress DAG and change Kahn's ready tie-break from declaration order to `(stall_cost, declaration_order)` or `(stall_cost, task_name)`, where `stall_cost` counts cross-stream incoming edges in `_build_same_progress_dag_edges`. That subset is a scheduler heuristic, not required for Δ=0 correctness.

### Cost / benefit

Do not include ready-first in this spec. It would broaden a narrow correctness/UX change into a user-visible scheduling policy change. Implementation would touch `deps.py:126-178`, likely `_compute_cpu_deps` assumptions through the changed active order (`executor.py:189-270`), and tests that assert declaration-order tie-break behavior. Runtime overhead would be small if computed at construction time, but the behavioral test surface is nontrivial.

For this spec, keep the implementation scoped to Δ=0 auto-conversion:

- add Δ=0 `cross_iter_depends_on` edges to `_build_same_progress_dag_edges`;
- mirror `same_progress_sync` in `_compute_cpu_deps`;
- replace the current Δ=0 rejection in `infer_cross_stream_event_deps` (`deps.py:472-489`) with same-progress event lookup at `producer.batch_offset`, matching the existing `same_progress_sync` path (`deps.py:541-553`).

Ready-first can be reconsidered later as a separate profiler-driven performance spec.
