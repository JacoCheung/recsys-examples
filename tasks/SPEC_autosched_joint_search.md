# SPEC: HSTU Auto-Scheduler v2 Joint Search

> Status: **DRAFT**
>
> Scope: heuristic design only. This document contains pseudocode and
> scheduling math, not a Python implementation.
>
> Inputs studied:
>
> - `examples/commons/pipeline/engine/autosched/fire_order.py`
> - `examples/commons/pipeline/hstu_pipeline/pipeline.py`
> - `examples/hstu/training/benchmark/cost_models/hstu_prefetch_caching_1node.json`

## Algorithm Choice: Beam Search

Pick **beam search** for the v2 joint scheduler.

Beam search is a good fit because the HSTU space is small but coupled:
thread placement, stream placement, and fire-order tie breaks interact
through the same DAG, overlap matrix, and NCCL communicator constraints.
It lets the scheduler keep several plausible schedules alive without
requiring an exact solver or a benchmark loop, and it composes cleanly
with the existing `auto_assign_lookaheads` function as a per-candidate
lookahead oracle.

Tradeoffs for alternatives:

- **Local search**: cheaper per step, but too sensitive to the initial
  hand-authored schedule. A single stream or thread move can look bad
  until its companion fire-order move is also applied, so local search
  can stop early in a coupled space.
- **ILP**: attractive for exactness, but awkward here because the
  objective includes resource-constrained list scheduling, ring
  lookahead validity, NCCL ordered-lock serialization, and topological
  tie-break effects. Linearizing those would make the model more complex
  than the current scheduler warrants.
- **Hybrid**: useful later, for example beam search followed by a local
  polish pass. For a first v2 design it adds knobs and debug surface
  without changing the key safety requirements.

## Search Space

The joint search state is:

```text
Candidate =
  thread_map     # CPU worker assignment
  stream_map     # CUDA stream assignment
  fire_ordering  # declaration / topological tie-break priority
  lookahead_map  # derived by auto_assign_lookaheads, not searched directly
```

The search is over `thread_map x stream_map x fire_ordering`.
`lookahead_map` is deliberately not a fourth free axis. It is derived by
the existing `auto_assign_lookaheads(schedule, cost_model,
max_in_flight=...)` implementation after each candidate's thread and
stream choices are materialized into a provisional schedule.

Thread-map candidates come from `HSTU_THREAD_MAP_PRESETS`:

- `default`
- `by_stream`
- `io_prefetch_compute`
- `io_data_dist_compute`
- `io_data_dist_prefetch_compute`

`per_task` is rejected by the v2 hard constraint because the HSTU task
set has 14 tasks and `max_threads <= 4`.

The stream-map axis starts from the current HSTU stream layout:

| Task group | Current stream | Search treatment |
| --- | --- | --- |
| `h2d`, `start_shuffle`, `finish_shuffle` | `memcpy` | movable prep group |
| `start_input_dist`, `wait_input_dist` | `data_dist` | movable comm-prep group |
| `prefetch_embeddings` | `prefetch` | movable prefetch group, prefetch variant only |
| `forward`, `backward`, `finalize_model_grads`, `optimizer_step` | `default` | hard frozen for bit-exactness |
| `zero_grad`, `global_tokens_allreduce`, `nccl_safety_barrier`, `watchdog_step` | `default` | kept on default by HSTU policy unless a future validator broadens support |

Fire ordering is the priority used for incomparable tasks before the
engine's topological sort. It never removes DAG edges. The v2 scheduler
generates a small set of resource-aware tie-break policies, such as:

- critical-default-chain first;
- longest off-default task first;
- NCCL queue early, preserving the existing declaration/ticket order;
- PCIe prep early;
- slack first, where slack is the estimated time before the default
  stream consumer needs the producer's result.

## Objective Function Definition

The objective is the estimated steady-state step latency from the cost
model seeds plus small scheduler overhead terms:

```text
minimize score(candidate)

score =
  steady_step_us(candidate)
  + thread_overhead_us(candidate)
  + cross_thread_wait_overhead_us(candidate)
  + invalidity_penalty(candidate)
```

`invalidity_penalty` is infinite for any hard-constraint violation.
Otherwise it is zero.

The latency model uses the JSON cost model:

```text
gpu_us(t) = cost_model[t].gpu_us
cpu_us(t) = cost_model[t].cpu_us
```

For the current HSTU seed file every `cpu_us` entry is `0.0`, so GPU
resource occupancy dominates. The model still carries the CPU term so a
future cost file can distinguish thread-map choices directly.

For each task instance `(progress_index, task)` that passes the engine
mask formula, the evaluator performs resource-constrained list
scheduling:

```text
host_start =
  max(all dependency completion times visible to this task,
      cpu_thread_free_time[thread_map[task]])

host_done =
  host_start + cpu_us(task) + submit_overhead_us

gpu_start =
  max(host_done,
      cuda_stream_free_time[stream_map[task]],
      nccl_comm_free_time[nccl_comm(task)] if task is NCCL,
      pcie_free_time if task is PCIe-bound)

gpu_done =
  gpu_start + gpu_us(task)
```

The overlap matrix from `fire_order.py` supplies the resource-conflict
labels:

- `stream`: same CUDA stream FIFO serialization;
- `nccl`: same NCCL communicator serialization, even across different
  CUDA streams;
- `pcie`: host/device transfer bandwidth contention;
- `ok`: no modeled resource conflict.

`steady_step_us(candidate)` is the mean or median interval between
successive `optimizer_step` completions after warmup and before drain:

```text
steady_step_us =
  robust_average(
    finish_time[optimizer_step, batch_i]
    - finish_time[optimizer_step, batch_(i-1)]
    for middle steady-state batches
  )
```

The evaluator should simulate at least `2 * max_in_flight + 4`
progress calls so warmup, steady state, and drain are all represented.
For HSTU v2 the hard cap is `max_in_flight <= 5`.

## Constraint Encoding

Hard constraints:

1. **Bit-exact default-stream anchors**

   These tasks must remain on `stream="default"` and `lookahead=0`:

   ```text
   forward
   backward
   finalize_model_grads
   optimizer_step
   ```

   This is the same safety contract enforced by
   `DEFAULT_BIT_EXACT_TASKS` in `fire_order.py`. Moving these tasks or
   giving them stale lookahead would change the model-update semantics.

2. **In-flight cap**

   ```text
   max_in_flight <= 5
   lookahead(task) in [0, max_in_flight - 1]
   ```

   The joint search passes this cap into `auto_assign_lookaheads`. A
   candidate is invalid if the existing authored lookahead or the
   lookahead oracle requires a value outside the cap.

3. **Thread cap**

   ```text
   number_of_distinct_threads(thread_map) <= 4
   ```

   `per_task` is filtered out for the 14-task HSTU schedule. `by_stream`
   is counted after resolving the candidate's stream map, because stream
   merging can change the number of worker groups.

4. **NCCL communicator serialization**

   If two tasks share the same `nccl_comm`, they serialize even when
   assigned to different CUDA streams. The overlap matrix labels this
   conflict as `N` / `nccl`. The evaluator represents each NCCL
   communicator as an exclusive resource:

   ```text
   gpu_start(task) >= nccl_comm_free_time[nccl_comm(task)]
   nccl_comm_free_time[nccl_comm(task)] = gpu_done(task)
   ```

   The fire-order generator also preserves declaration/ticket order for
   tasks on the same communicator so it remains compatible with
   `_NcclOrderedLock`.

5. **DAG and ring-slot legality**

   The scheduler must not remove or invert edges from `reads`, `writes`,
   `depends_on`, `same_progress_sync`, or derived cross-iteration
   dependencies. Existing cross-iteration caps are honored by
   `auto_assign_lookaheads`; a candidate that causes that function to
   reject is pruned.

6. **Compose with lookahead-only search**

   The existing lookahead logic is authoritative for lookahead
   assignment. V2 never reimplements the overlap matrix or the
   `DEFAULT_BIT_EXACT_TASKS` checks.

## Pseudocode

The pseudocode below is intentionally Python-like but not runnable
Python. Helper names describe scheduler steps, not real APIs.

```python
procedure JOINT_HSTU_SEARCH(
    base_schedule,
    cost_model,
    hstu_thread_presets,
    beam_width,
    max_threads = 4,
    max_in_flight = 5,
):
    require max_threads <= 4
    require max_in_flight <= 5

    frozen = {
        "forward",
        "backward",
        "finalize_model_grads",
        "optimizer_step",
    }

    legal_thread_presets = []
    for preset_name, preset in hstu_thread_presets:
        if preset_name == "per_task":
            continue
        legal_thread_presets.append((preset_name, preset))

    canonical = Candidate(
        thread_map = HSTU_DEFAULT_THREAD_MAP,
        stream_map = CURRENT_HSTU_STREAMS(base_schedule),
        fire_ordering = DECLARATION_ORDER(base_schedule),
        lookahead_map = CURRENT_LOOKAHEADS(base_schedule),
    )

    canonical = NORMALIZE_AND_SCORE(
        canonical,
        base_schedule,
        cost_model,
        frozen,
        max_threads,
        max_in_flight,
    )

    beam = TOP_BY_SCORE([canonical], beam_width)
    best = canonical

    for round in 1..SEARCH_ROUNDS:
        expanded = []

        for candidate in beam:
            for next_candidate in MUTATE_THREAD_MAP(candidate, legal_thread_presets):
                expanded.append(next_candidate)

            for next_candidate in MUTATE_STREAM_MAP(candidate):
                expanded.append(next_candidate)

            for next_candidate in MUTATE_FIRE_ORDERING(candidate):
                expanded.append(next_candidate)

        scored = []
        for candidate in UNIQUE(expanded):
            normalized = NORMALIZE_AND_SCORE(
                candidate,
                base_schedule,
                cost_model,
                frozen,
                max_threads,
                max_in_flight,
            )
            if normalized is valid:
                scored.append(normalized)

        beam = TOP_BY_SCORE(DOMINANCE_PRUNE(scored), beam_width)

        if beam is empty:
            break

        if SCORE(beam[0]) < SCORE(best):
            best = beam[0]

        if NO_SCORE_IMPROVEMENT_FOR_PATIENCE_ROUNDS():
            break

    return best
```

Candidate normalization is where v2 composes with the existing
lookahead-only scheduler:

```python
procedure NORMALIZE_AND_SCORE(
    candidate,
    base_schedule,
    cost_model,
    frozen,
    max_threads,
    max_in_flight,
):
    if VIOLATES_FROZEN_STREAM_OR_LOOKAHEAD(candidate, frozen):
        return invalid

    resolved_thread_map = RESOLVE_THREAD_MAP(candidate.thread_map,
                                            candidate.stream_map)
    if COUNT_THREADS(resolved_thread_map) > max_threads:
        return invalid

    provisional_schedule = REBUILD_SCHEDULE(
        base_schedule,
        stream_map = candidate.stream_map,
        fire_ordering = candidate.fire_ordering,
        lookahead_map = CURRENT_LOOKAHEADS(base_schedule),
    )

    try:
        derived_lookahead = auto_assign_lookaheads(
            provisional_schedule,
            cost_model,
            max_in_flight = max_in_flight,
        )
    except SchedulerRejectsCandidate:
        return invalid

    if ANY(derived_lookahead[t] != 0 for t in frozen):
        return invalid

    scored_schedule = REBUILD_SCHEDULE(
        base_schedule,
        stream_map = candidate.stream_map,
        fire_ordering = candidate.fire_ordering,
        lookahead_map = derived_lookahead,
    )

    overlap_matrix = compute_overlap_matrix(scored_schedule.tasks)

    latency = RESOURCE_LIST_SCHEDULE_LATENCY(
        scored_schedule,
        cost_model,
        resolved_thread_map,
        overlap_matrix,
        max_in_flight,
    )

    candidate.lookahead_map = derived_lookahead
    candidate.score = latency
    return candidate
```

Thread-map mutations:

```python
procedure MUTATE_THREAD_MAP(candidate, legal_thread_presets):
    for preset_name, preset in legal_thread_presets:
        yield candidate with thread_map = preset
```

Stream-map mutations are bounded, group-oriented moves rather than a
full Cartesian product:

```python
procedure MUTATE_STREAM_MAP(candidate):
    movable_groups = [
        {"h2d", "start_shuffle", "finish_shuffle"},
        {"start_input_dist", "wait_input_dist"},
        {"prefetch_embeddings"},
    ]

    for group in movable_groups:
        if group is absent from schedule:
            continue

        for stream_choice in LEGAL_STREAM_CHOICES(group):
            next_map = COPY(candidate.stream_map)

            for task in group:
                if task in BIT_EXACT_FROZEN_TASKS:
                    continue
                next_map[task] = stream_choice

            yield candidate with stream_map = next_map
```

Fire-order mutations generate topological tie-break policies; the
engine still owns final topological sorting:

```python
procedure MUTATE_FIRE_ORDERING(candidate):
    policies = [
        CRITICAL_DEFAULT_CHAIN_FIRST,
        LONGEST_OFF_DEFAULT_FIRST,
        NCCL_QUEUE_EARLY_WITH_TICKET_ORDER,
        PCIE_PREP_EARLY,
        MIN_SLACK_FIRST,
    ]

    for policy in policies:
        priority = BUILD_PRIORITY_VECTOR(candidate, policy)
        yield candidate with fire_ordering = priority

    for adjacent_pair in INCOMPARABLE_ADJACENT_PAIRS(candidate.fire_ordering):
        yield candidate with that pair swapped
```

Dominance pruning is conservative:

```python
procedure DOMINANCE_PRUNE(candidates):
    grouped = GROUP_BY(
        candidates,
        keys = (thread_map, stream_map, lookahead_map),
    )

    for group in grouped:
        keep the lowest-score fire_ordering
        discard candidates with same or worse score and no lower thread count
```

## Complexity Bound Analysis

Let:

- `n` be the number of tasks. For the HSTU prefetch schedule, `n = 14`.
- `p` be legal thread-map presets. Here `p <= 5` after filtering
  `per_task`.
- `m` be movable stream groups. Here `m <= 3`.
- `s` be legal stream choices per group. Here `s <= 4`.
- `k` be generated fire-order policies plus local swaps per candidate.
- `B` be beam width.
- `R` be the number of beam rounds.
- `H` be the number of simulated progress calls.
- `E` be the number of DAG edges.

The exhaustive upper bound over stream maps alone is `O(p * s^m)`, and
full fire-order enumeration can be exponential in the number of
incomparable tasks. Beam search avoids enumerating all topological
orders.

Per candidate:

- `auto_assign_lookaheads` builds task resources and the overlap matrix
  in `O(n^2)`, then propagates dependency constraints over the task DAG.
  A conservative bound is `O(n^2 + n * E)`.
- The resource-list latency model simulates `H` progress windows with
  `n` tasks each. With small fixed resource sets, the bound is
  `O(H * n log n + H * E)` if ready tasks are kept in a priority queue.

Beam-search cost:

```text
O(
  R * B * (p + m * s + k)
  * (n^2 + n * E + H * n log n + H * E)
)
```

For the HSTU constants (`n = 14`, `p <= 5`, `m <= 3`, `s <= 4`,
`max_in_flight <= 5`), this is small enough for an offline planner or a
startup-time heuristic pass. The practical cost is dominated by the
latency evaluator, not by the overlap matrix.

Space usage is:

```text
O(B * n + H * n + n^2)
```

for the beam candidates, simulated task instances, and overlap matrix.

## Worked Example: 14 HSTU Cost-Model Tasks

The seed file
`examples/hstu/training/benchmark/cost_models/hstu_prefetch_caching_1node.json`
contains these task durations:

| # | Task | cpu_us | gpu_us | Current lookahead | Current stream |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `h2d` | 0.0 | 300.0 | 2 | `memcpy` |
| 2 | `start_shuffle` | 0.0 | 4600.0 | 2 | `memcpy` |
| 3 | `finish_shuffle` | 0.0 | 2300.0 | 2 | `memcpy` |
| 4 | `start_input_dist` | 0.0 | 8600.0 | 1 | `data_dist` |
| 5 | `wait_input_dist` | 0.0 | 400.0 | 1 | `data_dist` |
| 6 | `prefetch_embeddings` | 0.0 | 1100.0 | 1 | `prefetch` |
| 7 | `zero_grad` | 0.0 | 90.0 | 0 | `default` |
| 8 | `global_tokens_allreduce` | 0.0 | 120.0 | 0 | `default` |
| 9 | `nccl_safety_barrier` | 0.0 | 20.0 | 0 | `default` |
| 10 | `forward` | 0.0 | 28850.0 | 0 | `default` |
| 11 | `backward` | 0.0 | 12320.0 | 0 | `default` |
| 12 | `finalize_model_grads` | 0.0 | 290.0 | 0 | `default` |
| 13 | `optimizer_step` | 0.0 | 66610.0 | 0 | `default` |
| 14 | `watchdog_step` | 0.0 | 80.0 | 0 | `default` |

Step 1: classify tasks.

The bit-exact frozen set is:

```text
forward, backward, finalize_model_grads, optimizer_step
```

The HSTU prep groups are:

```text
memcpy group:    h2d, start_shuffle, finish_shuffle
data_dist group: start_input_dist, wait_input_dist
prefetch group:  prefetch_embeddings
```

Step 2: filter thread maps.

`HSTU_THREAD_MAP_PRESETS` exposes six named choices. Under
`max_threads <= 4`, the search keeps `default`, `by_stream`,
`io_prefetch_compute`, `io_data_dist_compute`, and
`io_data_dist_prefetch_compute`. It rejects `per_task` because it would
create one worker per task, i.e. 14 workers for this schedule.

Step 3: compute the default-stream critical path used by the existing
lookahead oracle.

`default_stream_critical_path_us` sums every default-stream task at the
minimum default lookahead, which is `0`:

```text
zero_grad                  90
global_tokens_allreduce   120
nccl_safety_barrier        20
forward                 28850
backward                12320
finalize_model_grads      290
optimizer_step          66610
watchdog_step              80
--------------------------------
default-stream total   108380 us
```

The hard bit-exact four-task core is:

```text
forward + backward + finalize_model_grads + optimizer_step
= 108070 us
```

Step 4: account for NCCL serialization.

For a non-identity shuffler, the modeled DP NCCL queue is:

```text
start_shuffle             4600
finish_shuffle            2300
start_input_dist          8600
wait_input_dist            400
global_tokens_allreduce    120
backward                 12320
finalize_model_grads       290
--------------------------------
same-comm NCCL queue     28630 us
```

For an identity shuffler, `start_shuffle` and `finish_shuffle` are not
real collectives, so the queue drops to `21730 us`. In either case the
overlap matrix marks same-communicator pairs as `N`, so they serialize
even if the stream map puts them on different CUDA streams.

Step 5: evaluate a representative candidate.

Consider:

```text
thread_map    = io_data_dist_prefetch_compute
stream_map    = current HSTU streams
fire_ordering = NCCL_QUEUE_EARLY_WITH_TICKET_ORDER
```

The candidate is materialized into a provisional schedule and passed to
`auto_assign_lookaheads(..., max_in_flight=5)`.

The existing lookahead oracle sees:

- off-default task durations are much smaller than the `108380 us`
  default-stream window;
- same-comm NCCL queue, `28630 us` in the non-identity case, is also
  below one default-stream window;
- the authored floors are `2` for the memcpy group and `1` for
  data-dist/prefetch;
- default-stream bit-exact tasks are already at lookahead `0`.

So the derived lookahead map remains:

```text
h2d                       2
start_shuffle             2
finish_shuffle            2
start_input_dist          1
wait_input_dist           1
prefetch_embeddings       1
zero_grad                 0
global_tokens_allreduce   0
nccl_safety_barrier       0
forward                   0
backward                  0
finalize_model_grads      0
optimizer_step            0
watchdog_step             0
```

Step 6: score the candidate.

The latency evaluator simulates several progress calls. In steady
state, work from three batch positions is present at once:

```text
lookahead 2: h2d, start_shuffle, finish_shuffle
lookahead 1: start_input_dist, wait_input_dist, prefetch_embeddings
lookahead 0: zero_grad, global_tokens_allreduce, barrier, forward,
             backward, finalize_model_grads, optimizer_step, watchdog
```

The overlap matrix enforces:

- `S` for same-stream FIFO conflicts, for example the memcpy prep group;
- `N` for same-communicator NCCL conflicts, for example
  `start_input_dist` versus `backward`;
- `P` for PCIe-style contention such as `h2d` and
  `prefetch_embeddings`;
- `.` / `ok` where cross-batch overlap is allowed.

Because the optimizer-heavy default chain is about `108 ms`, the prep
work can usually be hidden if the stream and thread map expose enough
parallelism and the NCCL queue is not placed behind avoidable tie-break
delays. The beam keeps this candidate if its modeled
`optimizer_step`-to-`optimizer_step` interval beats the alternatives
after thread and cross-thread-wait overheads.

Step 7: produce a plan, not code.

The v2 scheduler output is a schedule descriptor:

```text
best thread_map
best stream_map
best fire_ordering priority
derived lookahead_map from auto_assign_lookaheads
modeled steady_step_us
```

This spec intentionally does not implement that descriptor in Python.
It defines how an implementation should search and how it must remain
compatible with the existing lookahead-only scheduler.
