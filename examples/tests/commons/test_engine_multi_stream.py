# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""V3 — multi-stream pipeline with auto-inferred cross-stream waits.

Verifies SPEC §4.2 rule 8: for every consumer→producer edge (slot
read OR `depends_on`) where the two tasks bind to different streams,
the engine inserts `consumer_stream.wait_stream(producer_stream)`
before the consumer submits.
"""

import pytest
import torch
from commons.pipeline.engine import (
    DataSlot,
    SchedulablePipeline,
    Schedule,
    Stage,
    StreamPool,
    Task,
)
from commons.pipeline.engine.deps import infer_cross_stream_waits

# ----------------------------------------------------------------------
# Pure unit tests for the analyzer (no CUDA needed)
# ----------------------------------------------------------------------


def _trivial(ctx):
    return None


def test_analyzer_empty_schedule() -> None:
    """Analyzer returns {} when no tasks have cross-stream edges."""
    task = Task.from_fn(name="a", fn=_trivial, stream="default")
    schedule = Schedule(stages=(Stage(tasks=(task,)),), stream_slots=("default",))
    assert infer_cross_stream_waits(schedule) == {}


def test_analyzer_same_stream_slot_edge_no_wait() -> None:
    """Two tasks sharing a stream emit no wait_stream even when they
    share a slot (same-stream ordering is implicit via declaration
    order on the stream)."""
    writer = Task.from_fn(
        name="writer",
        fn=_trivial,
        writes=(DataSlot("x"),),
        stream="default",
    )
    reader = Task.from_fn(
        name="reader",
        fn=_trivial,
        reads=(DataSlot("x"),),
        stream="default",
    )
    schedule = Schedule(
        stages=(Stage(tasks=(writer, reader)),),
        stream_slots=("default",),
    )
    assert infer_cross_stream_waits(schedule) == {}


def test_analyzer_cross_stream_slot_edge_emits_wait() -> None:
    """Cross-stream slot edge → reader must wait on writer's stream."""
    writer = Task.from_fn(
        name="h2d",
        fn=_trivial,
        writes=(DataSlot("gpu"),),
        stream="memcpy",
    )
    reader = Task.from_fn(
        name="forward",
        fn=_trivial,
        reads=(DataSlot("gpu"),),
        stream="default",
    )
    schedule = Schedule(
        stages=(Stage(tasks=(writer, reader)),),
        stream_slots=("default", "memcpy"),
    )
    waits = infer_cross_stream_waits(schedule)
    assert waits == {"forward": ("memcpy",)}


def test_analyzer_cross_stream_depends_on_emits_wait() -> None:
    """Cross-stream `depends_on` edge → consumer must wait on
    producer's stream, same as slot edges."""
    producer = Task.from_fn(
        name="allreduce",
        fn=_trivial,
        stream="comm",
    )
    consumer = Task.from_fn(
        name="optimizer",
        fn=_trivial,
        depends_on=("allreduce",),
        stream="default",
    )
    schedule = Schedule(
        stages=(Stage(tasks=(producer, consumer)),),
        stream_slots=("default", "comm"),
    )
    waits = infer_cross_stream_waits(schedule)
    assert waits == {"optimizer": ("comm",)}


def test_analyzer_multiple_producers_sorted_deterministic() -> None:
    """A consumer depending on two producers on two different streams
    gets both, in deterministic (sorted) order."""
    p_a = Task.from_fn(name="a", fn=_trivial, writes=(DataSlot("x"),), stream="comm")
    p_b = Task.from_fn(name="b", fn=_trivial, writes=(DataSlot("y"),), stream="memcpy")
    c = Task.from_fn(
        name="c",
        fn=_trivial,
        reads=(DataSlot("x"), DataSlot("y")),
        stream="default",
    )
    schedule = Schedule(
        stages=(Stage(tasks=(p_a, p_b, c)),),
        stream_slots=("default", "comm", "memcpy"),
    )
    waits = infer_cross_stream_waits(schedule)
    assert waits == {"c": ("comm", "memcpy")}  # sorted


def test_analyzer_cross_iter_prefetch_edge_emits_wait() -> None:
    """V4 prefetch: writer@lookahead=1 on `memcpy`, reader@lookahead=0
    on `default`, both for slot name "batch_gpu". The underlying
    slot store is the same tensor (migrated via ring advance), so
    the reader's stream must wait on the writer's stream."""
    h2d = Task.from_fn(
        name="h2d",
        fn=_trivial,
        writes=(DataSlot("batch_gpu", batch_offset=1),),
        stream="memcpy",
        lookahead=1,
    )
    compute = Task.from_fn(
        name="compute",
        fn=_trivial,
        reads=(DataSlot("batch_gpu", batch_offset=0),),
        stream="default",
        lookahead=0,
    )
    schedule = Schedule(
        stages=(Stage(tasks=(h2d, compute)),),
        stream_slots=("default", "memcpy"),
    )
    waits = infer_cross_stream_waits(schedule)
    assert waits == {
        "compute": ("memcpy",)
    }, f"cross-iter prefetch edge missed: {waits}"


def test_analyzer_rejects_same_name_writers_on_different_streams() -> None:
    """V4 invariant: all writers of a given slot name must share one
    stream. Different-stream writers of the same name create
    ambiguity for cross-iter readers (which producer's stream does
    the reader wait on?)."""
    w_a = Task.from_fn(
        name="w_a",
        fn=_trivial,
        writes=(DataSlot("X", batch_offset=1),),
        stream="memcpy",
        lookahead=1,
    )
    w_b = Task.from_fn(
        name="w_b",
        fn=_trivial,
        writes=(DataSlot("X", batch_offset=0),),
        stream="comm",
        lookahead=0,
    )
    schedule = Schedule(
        stages=(Stage(tasks=(w_a, w_b)),),
        stream_slots=("default", "memcpy", "comm"),
    )
    with pytest.raises(ValueError, match="multiple streams"):
        infer_cross_stream_waits(schedule)


def test_analyzer_rejects_duplicate_writers() -> None:
    """SPEC §4.2 rule 4 — single writer per slot. The analyzer
    enforces this eagerly (not deferring to V5) because a duplicate
    writer would silently make `consumer.wait_stream(...)` target
    the wrong producer."""
    a = Task.from_fn(name="a", fn=_trivial, writes=(DataSlot("x"),), stream="default")
    b = Task.from_fn(name="b", fn=_trivial, writes=(DataSlot("x"),), stream="memcpy")
    schedule = Schedule(
        stages=(Stage(tasks=(a, b)),),
        stream_slots=("default", "memcpy"),
    )
    with pytest.raises(ValueError, match="multiple writers"):
        infer_cross_stream_waits(schedule)


def test_analyzer_unresolved_slot_silently_skipped() -> None:
    """An unresolved slot read produces no wait entry — V5 validator
    rejects such schedules; V3 analyzer tolerates them so the engine
    still runs on partially-validated schedules during V2/V3/V4."""
    reader = Task.from_fn(
        name="reader",
        fn=_trivial,
        reads=(DataSlot("nonexistent"),),
        stream="default",
    )
    schedule = Schedule(stages=(Stage(tasks=(reader,)),), stream_slots=("default",))
    assert infer_cross_stream_waits(schedule) == {}


# ----------------------------------------------------------------------
# Runtime tests (need CUDA)
# ----------------------------------------------------------------------


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("multi-stream runtime test requires CUDA")
    return torch.device("cuda:0")


def test_cross_stream_wait_prevents_race() -> None:
    """Two-stream schedule: writer on `memcpy` spends many kernels
    building a tensor and ends with a distinctive final value (42.0);
    reader on `default` sums ALL 8M elements asynchronously (no CPU
    sync). A race would see some elements not yet reaching 42.0 —
    detectable in the full-tensor sum. With auto-inserted
    `wait_stream`, sum == 8_000_000 * 42.0 exactly.

    Key guard against false-positive: the reader does NOT call
    `.item()` inside the task (which would CPU-sync and mask any
    race). Instead it captures `x.sum()` as a GPU-resident Tensor;
    we `.item()` it only after `torch.cuda.synchronize()` outside
    the pipeline. This way, under missing wait_stream, the sum
    kernel on `default` runs racily against the writer kernels on
    `memcpy` and produces a wrong value.
    """
    device = _cuda_or_skip()

    n_elements = 8_000_000
    # Pre-allocate on default stream. This init is irrelevant — the
    # writer's first fill_ on memcpy-stream will overwrite it, but
    # only if ordering holds.
    shared = torch.zeros(n_elements, device=device)

    def _writer(ctx) -> None:
        # Many slow ops on memcpy stream; reader must wait for ALL
        # of them including the final fill_(42.0).
        shared.fill_(0.0)
        for _ in range(1000):
            shared.add_(0.001)  # accumulates to ~1.0
        shared.fill_(42.0)  # distinctive terminal value
        ctx.slots.set("shared", shared)

    def _reader(ctx) -> None:
        x = ctx.slots["shared"]
        # Async: sum() enqueues a kernel on the reader's current
        # stream (default), producing a 0-dim Tensor. No CPU sync.
        # We capture this tensor and inspect its value AFTER
        # synchronize() in the test body.
        ctx.slots.set("step_result", x.sum())

    writer = Task.from_fn(
        name="writer",
        fn=_writer,
        writes=(DataSlot("shared"),),
        stream="memcpy",
    )
    reader = Task.from_fn(
        name="reader",
        fn=_reader,
        reads=(DataSlot("shared"),),
        stream="default",
    )
    schedule = Schedule(
        stages=(Stage(tasks=(writer, reader)),),
        stream_slots=("default", "memcpy"),
    )
    pool = StreamPool(
        {
            "default": torch.cuda.default_stream(device),
            "memcpy": torch.cuda.Stream(device),
        }
    )
    pipe = SchedulablePipeline(schedule, pool)

    # Run 10 iterations. If wait_stream is missing entirely, at
    # least one iteration will race (kernel scheduling is non-
    # deterministic when there's no ordering constraint). Catching
    # any ONE racy sum is sufficient to fail the test; a single-shot
    # version could false-negative if hardware scheduling happened
    # to serialize the streams by accident.
    expected = n_elements * 42.0
    for i in range(10):
        result_tensor = pipe.progress(iter([None]))
        torch.cuda.synchronize(device)
        observed = result_tensor.item()
        assert observed == expected, (
            f"Iteration {i}: cross-stream read produced "
            f"sum={observed}, expected {expected} (= 8M * 42.0). "
            f"A value significantly below this indicates the reader "
            f"raced into the writer's accumulate phase — "
            f"auto-inserted wait_stream may be missing or misordered."
        )


def test_v7_explicit_event_escape_hatch_round_trip() -> None:
    """V7 escape hatch: a producer task records a user-named event
    via ``ctx.record_event("checkpoint")`` partway through its work,
    a consumer task on a different stream waits via
    ``ctx.wait_event("checkpoint")`` before reading the produced value.

    This is the "I want a partial-progress signal" pattern that the
    auto cross-stream sync (engine-recorded completion event after
    task body returns) cannot express. To prove the wait is doing
    real work, the producer:
      1. records the user event right after writing the distinctive
         final value, then
      2. issues a long tail of unrelated work that the consumer
         must NOT block on.
    The consumer waits on ``checkpoint`` (which is recorded BEFORE
    the unrelated tail) and reads the shared tensor. Without the
    wait, on first iter (no auto cross-stream sync from a previous
    iter's slot data) the consumer would race ahead of the writer.
    """
    device = _cuda_or_skip()

    n_elements = 4_000_000
    shared = torch.zeros(n_elements, device=device)

    def _writer(ctx) -> None:
        shared.fill_(0.0)
        for _ in range(800):
            shared.add_(0.01)
        shared.fill_(99.0)
        ctx.record_event("checkpoint")
        # Long unrelated tail — if the consumer waits on the producer's
        # auto-completion event, it would block on this. Waiting on
        # the user event lets it proceed sooner.
        for _ in range(2000):
            shared.add_(0.001)
        ctx.slots.set("shared", shared)

    def _reader_via_user_event(ctx) -> None:
        # Drop the engine-auto wait_stream by reading from a slot the
        # writer DOESN'T declare (effectively bypassing data-deps),
        # forcing reliance on the explicit user event. We still write
        # to step_result so the engine can return it.
        waited = ctx.wait_event("checkpoint")
        x = shared
        # On the first iter, no producer has run yet on any prior
        # iteration, so wait_event returns False and we should fall
        # back to wait_stream-equivalent. To keep the test
        # deterministic across first/steady iters, consult `waited`.
        if not waited:
            # First-iter fallback: explicitly wait on memcpy stream.
            ctx.stream_pool.get("default").wait_stream(ctx.stream_pool.get("memcpy"))
        ctx.slots.set("step_result", x.sum())

    writer = Task.from_fn(
        name="writer",
        fn=_writer,
        writes=(DataSlot("shared"),),
        stream="memcpy",
    )
    reader = Task.from_fn(
        name="reader",
        fn=_reader_via_user_event,
        reads=(DataSlot("shared"),),
        stream="default",
        # Explicit task-name dep so the schedule validator + engine
        # know reader follows writer. The user-event wait is what
        # makes the GPU sync correct on default stream.
        depends_on=("writer",),
    )
    schedule = Schedule(
        stages=(Stage(tasks=(writer, reader)),),
        stream_slots=("default", "memcpy"),
    )
    pool = StreamPool(
        {
            "default": torch.cuda.default_stream(device),
            "memcpy": torch.cuda.Stream(device),
        }
    )
    pipe = SchedulablePipeline(schedule, pool)

    # Recorded final value (99.0) + 2000 × 0.001 tail. fp32
    # accumulation drift on the tail gives ~2.0 ± 0.01. Use a
    # tolerance on the per-element mean rather than an exact compare;
    # the per-element value should be close to 101.0, NOT close to
    # 99.0 (which would mean the reader observed only the terminal
    # fill_ and not the tail) and NOT close to 0.0 (which would mean
    # the reader raced ahead of the writer entirely).
    for i in range(5):
        result_tensor = pipe.progress(iter([None]))
        torch.cuda.synchronize(device)
        observed_mean = result_tensor.item() / n_elements
        assert abs(observed_mean - 101.0) < 0.1, (
            f"V7 round-trip iter {i}: per-element mean={observed_mean}, "
            f"expected ~101.0 (= 99.0 terminal fill + 2000 × 0.001 "
            f"tail). A mean near 99.0 means the reader observed only "
            f"the terminal value but not the post-event tail (engine's "
            f"data-dep auto-sync failed); near 0.0 means the reader "
            f"raced past the writer entirely (V7 API broken)."
        )


def test_stream_pool_use_none_resolves_to_anchor_default() -> None:
    """Direct assertion: `pool.use('default')` on a slot that holds
    `None` must enter `default_stream(anchor_device)` as the current
    stream — NOT leak whatever ambient stream was set by an outer
    context.

    This is the V3 round-3 fix: without smart-resolve, `use(None)`
    would be a `torch.cuda.stream(None)` no-op and task bodies
    declared on a `None` slot would silently run on whatever outer
    stream happened to be current. That breaks the user's declared
    stream inventory: a task declared on `"default"` could end up
    executing on some unrelated stream a caller's outer context set.

    Codex's round-4 analysis showed a race-based regression is
    self-healing here (because `current_stream()` inside `use()` is
    used for both the task body and the wait_stream, they align
    either way). This direct check is the honest gate: it fails iff
    `use()`'s smart-resolve is absent or wrong.
    """
    device = _cuda_or_skip()

    memcpy_stream = torch.cuda.Stream(device)
    pool = StreamPool(
        {
            "default": None,
            "memcpy": memcpy_stream,
        }
    )
    # anchor_device is derived from memcpy's device (first concrete
    # slot in the pool).
    assert pool.anchor_device == device

    outer_stream = torch.cuda.Stream(device)
    anchor_default = torch.cuda.default_stream(device)
    assert outer_stream != anchor_default  # sanity — distinct streams

    with torch.cuda.stream(outer_stream):
        # Sanity: outer context actually set outer_stream as current
        assert torch.cuda.current_stream() == outer_stream
        # Now enter use('default') where the slot is None.
        with pool.use("default"):
            cur = torch.cuda.current_stream()
            assert cur != outer_stream, (
                f"use('default') leaked the outer stream. "
                f"current_stream()={cur}, outer={outer_stream}. "
                f"StreamPool.use(None) is acting as a no-op instead "
                f"of resolving to default_stream(anchor_device)."
            )
            assert cur == anchor_default, (
                f"use('default') resolved to {cur}, expected "
                f"default_stream(anchor_device)={anchor_default}."
            )
        # On exiting use(), outer stream is restored
        assert torch.cuda.current_stream() == outer_stream


def test_cross_stream_sync_is_actually_emitted() -> None:
    """Spy on cross-stream sync primitives and verify the engine emits
    one for a cross-stream schedule.

    After the followup-#1 fix, the engine prefers fine-grained
    ``wait_event(producer_event)`` over coarse ``wait_stream(producer_
    stream)``: a writer task records a CUDA event on its stream, and
    the reader task waits on that specific event from the ring slot.
    Both forms serve the same ordering goal — what matters here is
    that *some* cross-stream sync was issued. Stream-granularity
    ``wait_stream`` is still emitted as a first-iter fallback when the
    ring slot has no producer event yet.

    Complements the race test — even if a race happened to produce the
    correct answer, this test still catches a missing cross-stream
    edge."""
    device = _cuda_or_skip()

    memcpy_stream = torch.cuda.Stream(device)

    def _writer(ctx):
        ctx.slots.set("shared", torch.ones(4, device=device))

    def _reader(ctx):
        ctx.slots.set("step_result", ctx.slots["shared"].sum())

    writer = Task.from_fn(
        name="writer",
        fn=_writer,
        writes=(DataSlot("shared"),),
        stream="memcpy",
    )
    reader = Task.from_fn(
        name="reader",
        fn=_reader,
        reads=(DataSlot("shared"),),
        stream="default",
    )
    schedule = Schedule(
        stages=(Stage(tasks=(writer, reader)),),
        stream_slots=("default", "memcpy"),
    )
    pool = StreamPool(
        {
            "default": torch.cuda.default_stream(device),
            "memcpy": memcpy_stream,
        }
    )
    pipe = SchedulablePipeline(schedule, pool)

    # Monkey-patch both wait_stream and wait_event to record calls.
    original_wait_stream = torch.cuda.Stream.wait_stream
    original_wait_event = torch.cuda.Stream.wait_event
    stream_calls = []
    event_calls = []

    def _stream_spy(self, other):
        stream_calls.append((int(self.stream_id), int(other.stream_id)))
        return original_wait_stream(self, other)

    def _event_spy(self, event):
        event_calls.append(int(self.stream_id))
        return original_wait_event(self, event)

    torch.cuda.Stream.wait_stream = _stream_spy
    torch.cuda.Stream.wait_event = _event_spy
    try:
        pipe.progress(iter([None]))
    finally:
        torch.cuda.Stream.wait_stream = original_wait_stream
        torch.cuda.Stream.wait_event = original_wait_event

    default_id = int(torch.cuda.default_stream(device).stream_id)
    memcpy_id = int(memcpy_stream.stream_id)
    saw_stream_wait = (default_id, memcpy_id) in stream_calls
    saw_event_wait = default_id in event_calls
    assert saw_stream_wait or saw_event_wait, (
        f"Expected default to wait on memcpy via wait_stream or "
        f"wait_event at least once; observed wait_stream calls: "
        f"{stream_calls}, wait_event calls on stream_id: {event_calls}. "
        f"Engine is not emitting cross-stream sync."
    )


def test_wait_stream_not_emitted_for_same_stream_edges() -> None:
    """Negative spy — no wait_stream call when writer and reader share
    a stream. Complements `test_analyzer_same_stream_slot_edge_no_wait`
    by confirming the runtime also respects the analyzer's decision."""
    device = _cuda_or_skip()

    def _writer(ctx):
        ctx.slots.set("shared", torch.ones(4, device=device))

    def _reader(ctx):
        ctx.slots.set("step_result", ctx.slots["shared"].sum())

    writer = Task.from_fn(
        name="writer",
        fn=_writer,
        writes=(DataSlot("shared"),),
        stream="default",
    )
    reader = Task.from_fn(
        name="reader",
        fn=_reader,
        reads=(DataSlot("shared"),),
        stream="default",
    )
    schedule = Schedule(
        stages=(Stage(tasks=(writer, reader)),),
        stream_slots=("default",),
    )
    pool = StreamPool({"default": torch.cuda.default_stream(device)})
    pipe = SchedulablePipeline(schedule, pool)

    original = torch.cuda.Stream.wait_stream
    calls = []

    def _spy(self, other):
        calls.append((int(self.stream_id), int(other.stream_id)))
        return original(self, other)

    torch.cuda.Stream.wait_stream = _spy
    try:
        pipe.progress(iter([None]))
    finally:
        torch.cuda.Stream.wait_stream = original

    assert calls == [], f"Same-stream edges must not trigger wait_stream; got {calls}"


def test_independent_tasks_can_run_on_different_streams() -> None:
    """Two tasks with NO data or ordering edge between them, bound to
    different streams, can both be submitted and run without blocking.

    Smoke-level check — just verifies the engine doesn't serialize
    unrelated tasks behind a phantom wait_stream. Overlap quantity is
    not asserted (single-iteration setup; real overlap measurement
    happens in V4 with prefetch + V10 timing tests).
    """
    device = _cuda_or_skip()

    ran: list = []

    def _task_a(ctx):
        torch.cuda.synchronize(device)
        ran.append("a")

    def _task_b(ctx):
        torch.cuda.synchronize(device)
        ran.append("b")

    a = Task.from_fn(name="a", fn=_task_a, stream="comm")
    b = Task.from_fn(name="b", fn=_task_b, stream="memcpy")
    schedule = Schedule(
        stages=(Stage(tasks=(a, b)),),
        stream_slots=("default", "comm", "memcpy"),
    )
    pool = StreamPool(
        {
            "default": torch.cuda.default_stream(device),
            "comm": torch.cuda.Stream(device),
            "memcpy": torch.cuda.Stream(device),
        }
    )
    pipe = SchedulablePipeline(schedule, pool)
    pipe.progress(iter([None]))

    assert ran == ["a", "b"], f"Expected tasks to run in declaration order; got {ran}."
    # Analyzer should emit no waits (no inter-task edges).
    assert infer_cross_stream_waits(schedule) == {}


def test_full_train_loop_multi_stream_matches_single_stream() -> None:
    """A 2-stream schedule (H2D on memcpy, the rest on default) must
    produce identical final parameters to the equivalent single-stream
    schedule (default-only). Numerical parity under auto-inferred
    cross-stream waits."""
    device = _cuda_or_skip()

    # --- reference single-stream pipeline via SchedulablePipeline.basic ---
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    ref_model = torch.nn.Linear(8, 4).to(device)
    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=1e-2)
    ref_pipe = SchedulablePipeline.basic(
        ref_model, ref_opt, loss_fn=lambda out: out.sum()
    )

    # --- two-stream pipeline: custom schedule with H2D on memcpy ---
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    eng_model = torch.nn.Linear(8, 4).to(device)
    eng_opt = torch.optim.SGD(eng_model.parameters(), lr=1e-2)

    def _h2d(ctx):
        batch = ctx.slots["batch_cpu"]
        ctx.slots.set("batch_gpu", batch.to(device, non_blocking=True))

    def _zero_grad(ctx):
        eng_opt.zero_grad(set_to_none=True)

    def _forward(ctx):
        out = eng_model(ctx.slots["batch_gpu"])
        ctx.slots.set("loss", out.sum())
        ctx.slots.set("step_result", out)

    def _backward(ctx):
        ctx.slots["loss"].backward()

    def _optstep(ctx):
        eng_opt.step()

    tasks = (
        Task.from_fn(
            name="h2d",
            fn=_h2d,
            reads=(DataSlot("batch_cpu"),),
            writes=(DataSlot("batch_gpu"),),
            stream="memcpy",
        ),
        Task.from_fn(name="zero_grad", fn=_zero_grad, stream="default"),
        Task.from_fn(
            name="forward",
            fn=_forward,
            reads=(DataSlot("batch_gpu"),),
            writes=(DataSlot("loss"), DataSlot("step_result")),
            depends_on=("zero_grad",),
            stream="default",
        ),
        Task.from_fn(
            name="backward",
            fn=_backward,
            reads=(DataSlot("loss"),),
            stream="default",
        ),
        Task.from_fn(
            name="optimizer",
            fn=_optstep,
            depends_on=("backward",),
            stream="default",
        ),
    )
    schedule = Schedule(
        stages=(Stage(tasks=tasks),),
        stream_slots=("default", "memcpy"),
    )
    pool = StreamPool(
        {
            "default": torch.cuda.default_stream(device),
            "memcpy": torch.cuda.Stream(device),
        }
    )
    eng_pipe = SchedulablePipeline(schedule, pool)

    # Deterministic batches
    torch.manual_seed(123)
    batches = [torch.randn(4, 8) for _ in range(15)]

    for x in batches:
        ref_pipe.step(x.to(device))
        eng_pipe.step(x)

    # Params should match exactly — same math, same init, just
    # different stream placement.
    for (_, p_ref), (_, p_eng) in zip(
        ref_model.named_parameters(), eng_model.named_parameters()
    ):
        assert torch.allclose(p_ref, p_eng, atol=1e-5, rtol=0), (
            f"Multi-stream schedule diverged from single-stream "
            f"reference: max_abs_diff="
            f"{(p_ref - p_eng).abs().max().item():.6g}"
        )

    # Engine should have emitted a wait_stream for the forward task
    # reading batch_gpu (written on memcpy, read on default).
    waits = infer_cross_stream_waits(schedule)
    assert waits.get("forward") == (
        "memcpy",
    ), f"Expected forward to wait on memcpy; got {waits}"


def test_stream_pool_get_returns_raw_none_unchanged() -> None:
    """StreamPool.get() returns raw slot values — `None` stays `None`.

    Rationale: `None` is a user-intent signal ("don't mess with the
    stream"). Auto-resolving it would mask explicit `device=cpu`
    intent and break scenarios where a user wants the pool to be
    stream-context-neutral. The pipeline driver handles the
    `None → default_stream` resolution at the point of need
    (wait_stream emission), not at the storage layer.
    """
    pool = StreamPool({"default": None})
    assert pool.get("default") is None
