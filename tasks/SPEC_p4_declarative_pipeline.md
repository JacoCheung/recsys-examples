# SPEC_p4: Declarative Pipeline — User-Facing DAG with Engine-Inferred Cross-Iter Semantics

> Status: **DRAFT v2**
>
> Relates:
> [SPEC_p1 / schedulable pipeline engine](../SPEC.md),
> [SPEC_p2 / HSTU adapter](../SPEC_p2.md),
> [SPEC_p3 / event-based cross-stream synchronization](../SPEC_p3.md).
>
> Note: no `tasks/SPEC_p1*.md`, `tasks/SPEC_p2*.md`, or
> `tasks/SPEC_p3*.md` files currently exist. Links point to the root
> specs that define the same problem sequence.
>
> Supersedes in part:
> [tasks/followups.md](followups.md) followup #1
> ("Engine: event-based cross-stream sync") and followup #11
> ("HSTU: torchrec_ctx mutation chain not modeled as DAG edges").

## 1. Title + Frontmatter

This document proposes a fourth pipeline design slice:
**Declarative Pipeline — User-Facing DAG with Engine-Inferred
Cross-Iter Semantics**.

The current engine separates tasks from execution, but adapter authors
still encode cross-iteration meaning directly in schedules. They choose
offsets, depend on declaration order, and add same-iteration dependencies
that stand in for prior-iteration semantics.

SPEC_p4 changes the user contract:

- users declare a single-batch DAG;
- users still choose streams;
- users declare reads, writes, dependencies, and per-task lookahead;
- the engine derives ring depth, offsets, cross-iteration waits, and
  first-iteration prefill.

The intended result is not a new executor. The intended result is a
cleaner contract: adapters describe dataflow, and the engine owns
cross-iteration scheduling mechanics.

## 2. Motivation

The immediate trigger is the first-iteration prefetch race in the HSTU
prefetch pipeline. `forward` consumes data that prior
`prefetch_embeddings` work should have produced, but iteration 1 has no
prior `backward` to bootstrap ordering. Correctness can become
timing-luck-only for the first consume.

That race is a symptom. The root issue is a leaky abstraction: the
current API requires the HSTU adapter to encode engine-level semantics.

### Leak #1: Manual Offsets and Declaration Order

`examples/commons/pipeline/hstu_pipeline/pipeline.py` lines 225-254
hand-author cross-iteration layout:

- line 225 reads `self._prefetch_depth`;
- lines 226-228 compute `h2d_offset=depth+1`,
  `input_dist_offset=depth`, and `prefetch_offset=1`;
- lines 230-254 build the task list using those offsets;
- lines 237-244 state that `prefetch_embeddings` must be declared
  **after** `forward` in the prefetch variant.

The adapter author must know that `prefetch_embeddings@1` prepares a
future batch while `forward@0` consumes the current batch. They must also
know declaration order affects dynamic embedding cache pressure. That is
steady-state ring protocol leaking into adapter code.

### Leak #2: Backward Dependency as Sync Workaround

Commit `d81593fc` added
`backward.depends_on=("prefetch_embeddings",)` for the prefetch variant.
The current code in
`examples/commons/pipeline/hstu_pipeline/pipeline.py` shows this at
lines 255-262:

- lines 255-258 explain the prefetch-only dependency;
- line 259 computes `backward_deps`;
- line 262 passes the dependency to `make_backward_task`.

This is not fundamentally a backward data dependency. It works around a
missing `forward -> prefetch_embeddings` cross-stream semantic edge. In
steady state, surrounding stream order may hide the issue. On iteration
1, there is no previous `backward`, so the intended ordering is not
structural.

### Leak #3: In-Place Mutation Chain Hidden from the DAG

`examples/commons/pipeline/hstu_pipeline/tasks.py` lines 415-423 mutate
TorchRec context state:

- `module_input_post_prefetch[fwd._name]`;
- `module_contexts_post_prefetch[fwd._name]`.

The task declaration at lines 425-431 names the task, stream, offset,
and `depends_on=("wait_input_dist",)`, but declares no `writes=` for
those mutated context fields. The engine is blind to the logical
`prefetch_embeddings -> forward` data edge.

`tasks/followups.md` lines 247-258 confirms the residual issue: the DAG
does not encode the TorchRec context mutation chain, and a runtime
colocation validator only partially compensates. SPEC_p4 treats that as
a design bug. In-place mutation chains must become explicit DAG edges.

### Leak #4: Prefetch Stream Naming Debt

The requested path `examples/commons/pipeline/hstu_pipeline/utils.py`
does not exist in this tree. The relevant implementation is
`examples/commons/pipeline/utils.py`; line ranges below are exact for
that file and approximate relative to the requested path.

In `examples/commons/pipeline/utils.py`:

- lines 648-655 define `PrefetchPipelinedForward.__init__`; the fifth
  positional argument is named `prefetch_stream`;
- lines 656-662 store it as `stream=prefetch_stream`;
- lines 674-677 wait on `self._stream` inside `__call__`;
- lines 1519-1525 pass `dist_stream` positionally from `_rewrite_model`.

So a parameter named `prefetch_stream` receives the data distribution
stream in the HSTU rewrite path. The wait is therefore on data
distribution, not necessarily on a stream conceptually named prefetch.
That may be behaviorally intentional, but the naming makes the
dependency model harder to audit.

### Structural Diagnosis

The first-iteration race should not be fixed by adding one more edge to
today's imperative schedule. The engine should understand prepare tasks,
consume tasks, prepare-chain distance, cross-stream producer events,
first-iteration prefill, and context mutations as dataflow.

Once those concepts are in the engine contract, the HSTU adapter no
longer needs hand-written offset arithmetic or declaration-order
workarounds.

## 3. Design Goals

- User declares a per-batch task DAG only: `reads`, `writes`, `stream`,
  `depends_on`, and `lookahead`.
- Engine derives ring depth, per-task `batch_offset`, cross-iteration
  wait edges, and first-iteration prefill.
- Mutation chains through context objects are modeled as explicit DAG
  edges, so no implicit runtime validator is needed for correctness.
- Stream sync is event-granular, not stream-granular, converging with
  SPEC_p3 and `tasks/followups.md` lines 182-245.
- Migration is backward compatible: existing HSTU scheduling keeps
  working until the adapter is ported.

## 4. Non-Goals

- **Auto stream affinity**: users still pick streams. The engine validates
  and synchronizes stream edges; it does not place tasks automatically.
- **Multi-process / distributed concerns**: DDP, Megatron TP, rank-level
  ordering, checkpoint broadcast, and process lifecycle stay in the
  distributed framework layer.
- **Replacing TorchRec `_rewrite_model`**: the HSTU adapter may continue
  to consume TorchRec sharded modules, `PipelinedForward`,
  `PrefetchPipelinedForward`, and TorchRec contexts. SPEC_p4 is about
  the engine layer above them.

## 5. User-Facing API Sketch

The user-facing descriptor is lookahead-based:

```python
Task(
    name: str,
    stream: str,
    reads: tuple[str, ...] = (),
    writes: tuple[str, ...] = (),
    depends_on: tuple[str | tuple[str, int], ...] = (),
    lookahead: int = 0,
)

Pipeline(tasks=[...])
```

`lookahead = N` means the batch this task processes is `N` iterations
earlier than the batch the current forward processes in the same
`progress()` call. This is an absolute distance to forward, not an
increment between adjacent tasks.

- `lookahead = 0` is current-iteration work: forward, backward,
  finalization, optimizer step, watchdogs, and other post-forward work.
- `lookahead > 0` is early preparation: H2D copy, shuffle, input
  distribution, embedding prefetch, and context mutation that prepares a
  later forward.

**Cross-iter sync is mostly automatic.** The user describes the per-batch
DAG once, in single-batch terms — `reads`, `writes`, and `depends_on`
all express what holds for one batch flowing through the pipeline. The
engine combines that single-batch DAG with per-task `lookahead` to
infer cross-iter sync edges, ring layout, and ring advance behavior on
its own. Concretely, when a writer task with `lookahead = K_w` writes a
slot read by a task with `lookahead = K_r < K_w`, the engine knows the
data the reader sees today was produced `K_w − K_r` iterations earlier
and emits the corresponding cross-iter `wait_event` automatically. The
user does not annotate that.

`depends_on` accepts two forms; the user almost always wants the
first:

- a bare task name `"X"` is the **default** form — declares an
  ordering edge from `X` to this task. Whether the edge is within-iter
  or cross-iter is **derived from the lookahead diff**: if
  `X.lookahead == self.lookahead`, the engine treats it as within-iter
  (X completes before self in the same iteration); if
  `X.lookahead > self.lookahead`, the engine treats it as cross-iter
  with N = `X.lookahead - self.lookahead` (X's completion event from
  N iterations earlier, recovered through the ring slot that has
  rotated down to self.lookahead). If `X.lookahead < self.lookahead`,
  construction fails: the producer has not yet run by the consumer's
  iteration.

  The user **does not write the `-N`** number. The lookahead diff
  carries the cross-iter semantics implicitly; cross-iter is just
  "ordering edge between two tasks with different lookahead."

- a tuple `("X", -N)` is an **escape hatch** for the rare case where
  the user wants to override the auto-derived offset (e.g. wait for
  X's output from a deeper iteration than the lookahead diff would
  imply). Negative offsets only; positive rejected at construction
  time.

In particular, `forward.depends_on=("prefetch_embeddings",)` works on
a schedule where `forward.lookahead=0` and
`prefetch_embeddings.lookahead=1`: the engine derives N=1 and emits
`forward.wait_event(prefetch_embeddings_event_at_offset_0)` — fully
equivalent to writing the explicit `("prefetch_embeddings", -1)`,
without burdening the user with the iteration arithmetic.

If a data-flow edge already exists via `reads`/`writes`, the user
does **not** restate it as a cross-iter `("X", -N)` escape-hatch
form — that would be redundant and the engine rejects it.

The user does not declare `batch_offset`, ring depth, ring advance,
first-iteration prefill, cross-iteration wait edges, roles, stages, or
prepare chains. `batch_offset` remains an internal engine concept.

The mental model is three-dimensional:

- the per-batch DAG: `reads`, `writes`, and `depends_on`;
- per-task stream affinity: `stream`;
- per-task distance from forward: `lookahead`.

### HSTU Prefetch Translation

The HSTU prefetch variant can be described without user-authored
offsets:

```python
h2d = Task(
    "h2d", "memcpy",
    reads=("batch_cpu",), writes=("batch_gpu",), lookahead=2,
)
start_shuffle = Task(
    "start_shuffle", "memcpy",
    reads=("batch_gpu",), writes=("shuffle_handle",), lookahead=2,
)
finish_shuffle = Task(
    "finish_shuffle", "memcpy",
    reads=("shuffle_handle",), writes=("shuffled_batch",), lookahead=2,
)
start_input_dist = Task(
    "start_input_dist", "data_dist",
    reads=("shuffled_batch",), writes=("torchrec_ctx",), lookahead=1,
)
wait_input_dist = Task(
    "wait_input_dist", "data_dist",
    reads=("torchrec_ctx",), writes=("torchrec_ctx",), lookahead=1,
)
prefetch_embeddings = Task(
    "prefetch_embeddings", "prefetch",
    reads=("shuffled_batch", "torchrec_ctx"),
    writes=("module_input_post_prefetch",
            "module_contexts_post_prefetch"),
    lookahead=1,
)
forward = Task(
    "forward", "default",
    reads=("batch_gpu", "torchrec_ctx", "shuffled_batch",
           "module_input_post_prefetch",
           "module_contexts_post_prefetch"),
    writes=("losses", "output"),
    lookahead=0,
)
backward = Task(
    "backward", "default", depends_on=("forward",), lookahead=0,
)
finalize_grads = Task(
    "finalize_grads", "default",
    depends_on=("backward",), lookahead=0,
)
optimizer_step = Task(
    "optimizer_step", "default",
    depends_on=("finalize_grads",), lookahead=0,
)
watchdog = Task(
    "watchdog", "default",
    depends_on=("optimizer_step",), lookahead=0,
)

pipeline = Pipeline(tasks=[
    h2d,
    start_shuffle,
    finish_shuffle,
    start_input_dist,
    wait_input_dist,
    prefetch_embeddings,
    forward,
    backward,
    finalize_grads,
    optimizer_step,
    watchdog,
])
```

### Reads, Writes, and Mutation Edges

The DAG uses declared reads, writes, and `depends_on`.
`depends_on` remains available for ordering not naturally modeled as slot
dataflow, but context mutation chains should not rely only on
`depends_on`.

If a task mutates a context object consumed by another task, the mutation
must be represented as a logical write/read pair. Section 8 lists open
syntax choices.

## 6. Inference Algorithm

This is a semantic skeleton, not an implementation prescription.

1. Build the single-batch DAG from the user task list.

   Edges fall into three classes:

   - **Within-iter edges** come from `depends_on` entries that are bare
     task names or `(name, 0)` tuples, plus read-after-write,
     write-after-read where mutation ordering matters, and
     write-after-write on the same logical slot — all involving tasks
     that share the same `lookahead`.
   - **Cross-iter data edges (auto-inferred)** come from `reads`/
     `writes` where the producer and consumer have **different
     lookahead**. If `writer.lookahead = K_w > K_r = reader.lookahead`,
     the engine knows the consumer's read of that slot today was
     produced `K_w − K_r` iterations earlier (the writer wrote it at
     ring offset `K_w` and ring advance carried the slot down to offset
     `K_r` over `K_w − K_r` iterations). The user **does not** annotate
     this as cross-iter — it is implied by the lookahead difference.
   - **Cross-iter pure-control edges** come from `depends_on` entries
     with negative iteration offset `("name", -N)`. Used only when
     ordering is required between iterations but no slot expresses the
     relationship through `reads`/`writes`. Restating an already-
     inferred data edge here is rejected as redundant.

   The input graph contains no user-authored `batch_offset`.

2. Topologically sort with stream-affinity grouping.

   The engine sorts the DAG, then groups adjacent tasks by stream when it
   can do so without violating dependencies. Grouping is an execution
   planning detail and must not change the logical graph.

3. Identify consume sinks.

   Tasks with `lookahead = 0` are current-iteration sinks. For HSTU,
   `forward` and `backward` are natural sinks, and `forward` is the
   primary consumer for prepared input and context state. `backward` must
   not be used as a proxy for missing forward/prefetch synchronization.

4. Identify prepare chains feeding each sink.

   For each task with `lookahead > 0`, the engine walks forward through
   read/write edges and explicit dependencies to find the
   `lookahead = 0` consumers reached by that prepared data or context
   state.

5. Compute longest prepare-chain length `L`.

   `L = max(t.lookahead for t in tasks)`. The declared lookahead already
   carries the absolute distance to forward, so no chain-distance
   inference is required.

6. Set ring depth.

   Ring depth is `L + 1`. The engine may still expose resource-specific
   validation, but the user-facing API does not carry `pipeline_depth`.

7. Assign `batch_offset`.

   Internally, `t.batch_offset = t.lookahead`. Validate that every task
   with `lookahead > 0` has a path through reads/writes or explicit
   dependencies to at least one task with `lookahead = 0`; otherwise it is
   orphan preparation and should fail construction.

8. Emit cross-stream wait edges.

   Emit a `wait_event` wherever a dependency crosses streams. The lookup
   key for the producer event depends on the edge class established in
   step 1:

   - **Within-iter (same lookahead)**: the producer recorded its event
     on the slot at offset `lookahead`; the consumer reads from the
     same slot in the same iteration, so the engine fetches the event
     at ring offset `consumer.lookahead`.
   - **Cross-iter data (auto from reads/writes)**: the consumer's
     read at ring offset `K_r` corresponds to the writer's earlier
     record. After ring advance, the producer's event has migrated
     down to offset `K_r`, so the engine fetches the event there too —
     the consumer's `wait_event` lookup is uniformly keyed at
     `consumer.lookahead`, regardless of which iteration recorded the
     event.
   - **Cross-iter pure control (`depends_on=(name, -N)`)**: the
     producer's event was recorded at offset `producer.lookahead`
     iteration `N` ago, so after ring advance it sits at
     `producer.lookahead − N`. The engine fetches it from that offset.

   In all three cases the wait targets a specific producer event
   carried by the ring slot, not the entire producer stream. This is
   where SPEC_p4 subsumes SPEC_p3/followup #1.

9. Synthesize first-iteration prefill.

   Before any `lookahead = 0` task fires, run `L` thin prefill
   iterations. Prefill iteration `i` dispatches only tasks with
   `lookahead >= i`. Iteration 1 must see the same invariants as steady
   state: the forward input exists, post-prefetch context mutation has
   occurred, required producer events exist, and no current-iteration task
   relies on prior backward to create ordering.

## 7. Migration Path

### Phase A: Add Declarative API Beside Imperative API

Add the declarative API without removing existing
`Schedule`/`Task-with-batch_offset` construction. Both paths work.

Initially, the engine can translate declarative pipelines into the
existing schedule representation before execution. This phase should
also expose an inspection report showing inferred offsets, ring depth,
prefill tasks, and wait edges.

### Phase B: Port HSTU Adapter

Rewrite the HSTU adapter to use the declarative API. The adapter should
no longer compute `h2d_offset`, `input_dist_offset`, or
`prefetch_offset`, and should no longer depend on declaration order for
`forward` versus `prefetch_embeddings`.

Internally, the engine may still translate to the old `Schedule` for the
executor.

### Phase C: Deprecate Imperative API

Once all in-tree adapters use the declarative API, deprecate direct
imperative schedule construction for normal users. The old API may remain
temporarily as an expert escape hatch, but it should not be the
recommended integration surface.

## 8. Open Questions

- Lookahead validity rules: should the engine reject
  `reader.lookahead > writer.lookahead` as a future-read error at
  construction time, mirroring existing `engine/autosched/validator.py`
  constraints?
- Consistency between `reads`/`writes` and `lookahead`: if
  `forward` (`lookahead = 0`) reads a slot only written by a task with
  `lookahead = 5`, is the user committing to a 6-batch ring? Should the
  engine warn, fail, or auto-correct?
- Mutation-chain syntax — **decided: bare-name `depends_on` between
  the producing and consuming tasks**, with cross-iter offset
  auto-derived from lookahead diff (§5).

  The HSTU prefetch chain is the canonical case: `prefetch_embeddings`
  mutates `torchrec_ctx.module_input_post_prefetch[fwd._name]` in
  place; `forward` reads it. Both tasks have lookahead values, and
  the consumer simply lists the producer in `depends_on`:

  ```python
  prefetch_embeddings = Task(name="prefetch_embeddings", lookahead=1, ...)
  forward = Task(name="forward", lookahead=0,
                 depends_on=("prefetch_embeddings",), ...)
  ```

  The engine sees `producer.lookahead - consumer.lookahead = 1 > 0` and
  emits the cross-iter `wait_event` exactly as if `forward.reads` had
  declared a pseudo-slot written by `prefetch_embeddings`. Adapter
  schedule code stays in single-batch terms and never has to spell out
  internal attribute names like `module_input_post_prefetch`.

  Alternatives considered and rejected:

  - **Pseudo-slot per mutated attribute** (an earlier interim choice,
    landed in dc8e3ea4): worked, but leaked torchrec-internal field
    names into HSTU adapter schedule definitions. Reverted in
    9a2c7a12 in favor of bare-name `depends_on` auto-inference.
  - **Re-listing `writes=("torchrec_ctx",)`**: conflated three
    distinct mutators on one slot, broke single-writer-per-slot.
  - **Dedicated `mutates=...` field**: pure redundancy with
    reads/writes, more API surface for no expressive gain.

  Open implementation detail: when several tasks chain in-place
  mutation (e.g. start_input_dist → wait_input_dist →
  prefetch_embeddings → forward) the user lists each link via
  `depends_on=("predecessor",)` independently. The engine handles
  same-stream chains via stream FIFO and cross-stream chains via the
  ring's event carrier; users do not see this distinction.
- Multi-stream stage with same `lookahead`: when several tasks share
  `lookahead = N` but run on different streams, the engine still must
  emit cross-stream waits among them. Verify Section 6 step 8 covers this
  case in implementation.
- Interaction with the multi-thread executor thread map tracked under
  Problem #3 in `tasks/followups.md`: does the declarative API need
  thread affinity, or can executor policy derive it from stream affinity?
- How does first-iteration prefill interact with optimizer state,
  checkpoint load, and dynamic embedding cache state, similar to the
  `ce5144d9` `reset_dynamicemb_cache_states` issue described in
  `tasks/followups.md` lines 113-118?
- Backwards compatibility: if a user gives the engine a `Schedule` with
  explicit `batch_offset`, do we support it forever, sunset it after
  migration, or keep it for tests/internal adapters only?

## 9. Acceptance Criteria

This draft can move to STABLE when:

- HSTU adapter is rewritten using the declarative API;
- HSTU parity remains at least `24 passed / 8 xfailed`;
- the first-iteration `forward -> prefetch_embeddings` race is
  structurally impossible, not only covered by a test;
- no call site in `examples/commons/pipeline/hstu_pipeline/` references
  `batch_offset` directly;
- the TorchRec context mutation chain is captured as a first-class DAG
  edge type;
- mutation-chain correctness no longer depends on a runtime colocation
  validator;
- inferred cross-iteration stream waits use event granularity;
- the engine can explain inferred offsets, ring depth, prefill tasks, and
  cross-iteration waits in a construction-time report.

## 10. Cross-References

### Leak #1 and Leak #2

`examples/commons/pipeline/hstu_pipeline/pipeline.py`

- lines 225-228: manual offset computation;
- lines 230-254: ordered task construction;
- lines 237-244: prefetch must be declared after forward;
- lines 255-262: `backward` depends on `prefetch_embeddings` in the
  prefetch variant.

### Leak #3

`examples/commons/pipeline/hstu_pipeline/tasks.py`

- lines 415-423: `prefetch_embeddings` mutates TorchRec context fields;
- lines 425-431: task declaration has no `writes=` entry for those
  mutated fields.

### Leak #4

Requested path:

- `examples/commons/pipeline/hstu_pipeline/utils.py`

Current tree:

- that file does not exist;
- actual file is `examples/commons/pipeline/utils.py`.

Actual citations:

- lines 648-655: `PrefetchPipelinedForward.__init__` names the fifth
  positional argument `prefetch_stream`;
- lines 656-662: the argument is stored as the base forward stream;
- lines 674-677: `__call__` waits on `self._stream`;
- lines 1519-1525: `_rewrite_model` passes `dist_stream` positionally
  into `pipelined_forward`.

Line range approximate relative to the requested path; exact relative to
`examples/commons/pipeline/utils.py`.

### Followup #1: Event-Based Cross-Stream Sync

`tasks/followups.md` lines 182-245 describes replacing stream-granularity
`wait_stream` with event-granularity waits.

SPEC_p4 subsumes that followup by making event-based waits the required
implementation for inferred cross-iteration wait edges. A consumer waits
on the producer event for the data it reads, not on unrelated work queued
on the same stream.

### Followup #11: Mutation Chain Not Modeled as DAG Edges

`tasks/followups.md` lines 247-258 describes the residual problem: the
TorchRec context mutation chain is not encoded in the DAG, and a runtime
validator only partially compensates.

SPEC_p4 subsumes that followup by requiring mutation chains to become
first-class DAG edges. The HSTU adapter should declare which task mutates
post-prefetch context, which task consumes it, which stream/event edge
protects it, and which ring slot carries the mutated state.

The concrete mutation-chain syntax remains an open question in Section 8.
