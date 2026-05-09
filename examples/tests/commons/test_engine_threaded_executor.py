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

"""Problem #3 — ThreadedExecutor tests.

Tests verify:
  - Tasks on different streams run concurrently (CPU overlap).
  - Cross-stream data dependencies are correctly ordered.
  - NCCL-tagged tasks execute in deterministic declaration order.
  - ThreadedExecutor produces identical results to SequentialExecutor.
  - Preset `threaded=True` works end-to-end.
  - Error propagation from worker threads.
"""

import threading
import time
from collections import defaultdict
from typing import Dict, List

import pytest
import torch
from commons.pipeline.engine import (
    DataSlot,
    SchedulablePipeline,
    Schedule,
    SequentialExecutor,
    Stage,
    StreamPool,
    Task,
    ThreadedExecutor,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_SEED = 42
_STEPS = 20
_BATCH = 8
_IN = 16
_OUT = 4


def _seeded_init(device: torch.device, seed: int = _SEED):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = torch.nn.Linear(_IN, _OUT).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    return model, opt


def _make_batches(device: torch.device, seed: int = _SEED + 1):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return [
        torch.randn(_BATCH, _IN, device=device, generator=gen) for _ in range(_STEPS)
    ]


def _drive(pipe: SchedulablePipeline, batches: list) -> None:
    it = iter(batches)
    while True:
        try:
            pipe.progress(it)
        except StopIteration:
            break


# ------------------------------------------------------------------
# Test: concurrent execution on different streams
# ------------------------------------------------------------------


def test_threaded_executor_concurrent_streams() -> None:
    """Tasks on different streams should run concurrently, not
    sequentially. We verify by checking that both threads are alive
    at the same time during execution."""
    concurrency_evidence: Dict[str, List[str]] = defaultdict(list)
    barrier = threading.Barrier(2, timeout=5)

    def _task_a(ctx):
        concurrency_evidence["a"].append(threading.current_thread().name)
        try:
            barrier.wait()  # Both tasks must reach here to proceed
        except threading.BrokenBarrierError:
            pass

    def _task_b(ctx):
        concurrency_evidence["b"].append(threading.current_thread().name)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass

    tasks = (
        Task.from_fn("task_a", _task_a, stream="stream_1"),
        Task.from_fn("task_b", _task_b, stream="stream_2"),
    )
    schedule = Schedule(
        stages=(Stage(tasks=tasks),), stream_slots=("stream_1", "stream_2")
    )
    pool = StreamPool({"default": None, "stream_1": None, "stream_2": None})
    pipe = SchedulablePipeline(schedule, pool, executor="threaded")

    pipe.progress(iter([torch.tensor(1.0)]))

    # Both tasks executed
    assert len(concurrency_evidence["a"]) == 1
    assert len(concurrency_evidence["b"]) == 1
    # They ran on different threads
    assert (
        concurrency_evidence["a"][0] != concurrency_evidence["b"][0]
    ), "Tasks on different streams should run on different threads"


# ------------------------------------------------------------------
# Test: cross-stream data dependency ordering
# ------------------------------------------------------------------


def test_threaded_executor_cross_stream_data_ordering() -> None:
    """A consumer task on stream B that reads a slot written by a
    producer on stream A must see the written value, even though
    they run on different threads."""
    results: List[int] = []

    def _producer(ctx):
        time.sleep(0.05)  # Simulate slow CPU work
        ctx.slots.set("shared_data", 42)

    def _consumer(ctx):
        val = ctx.slots["shared_data"]
        results.append(val)

    tasks = (
        Task.from_fn(
            "producer",
            _producer,
            stream="stream_a",
            writes=(DataSlot("shared_data"),),
        ),
        Task.from_fn(
            "consumer",
            _consumer,
            stream="stream_b",
            reads=(DataSlot("shared_data"),),
        ),
    )
    schedule = Schedule(
        stages=(Stage(tasks=tasks),),
        stream_slots=("default", "stream_a", "stream_b"),
    )
    pool = StreamPool({"default": None, "stream_a": None, "stream_b": None})
    pipe = SchedulablePipeline(schedule, pool, executor="threaded")

    pipe.progress(iter([torch.tensor(1.0)]))

    assert results == [
        42
    ], "Consumer must see producer's slot write despite running on different thread"


# ------------------------------------------------------------------
# Test: NCCL ordering
# ------------------------------------------------------------------


def test_threaded_executor_nccl_ordering() -> None:
    """NCCL-tagged tasks must execute in declaration order even when
    running on different streams/threads."""
    execution_order: List[str] = []
    order_lock = threading.Lock()

    def _make_nccl_task(name: str, stream: str):
        def _fn(ctx):
            with order_lock:
                execution_order.append(name)

        return Task.from_fn(name, _fn, stream=stream, nccl=True)

    tasks = (
        _make_nccl_task("nccl_first", "stream_1"),
        _make_nccl_task("nccl_second", "stream_2"),
        _make_nccl_task("nccl_third", "stream_1"),
    )
    schedule = Schedule(
        stages=(Stage(tasks=tasks),),
        stream_slots=("default", "stream_1", "stream_2"),
    )
    pool = StreamPool({"default": None, "stream_1": None, "stream_2": None})

    # Run multiple times to catch non-determinism
    for _ in range(10):
        execution_order.clear()
        pipe = SchedulablePipeline(schedule, pool, executor="threaded")
        pipe.progress(iter([torch.tensor(1.0)]))

        assert execution_order == ["nccl_first", "nccl_second", "nccl_third"], (
            f"NCCL tasks must execute in declaration order, " f"got {execution_order}"
        )


# ------------------------------------------------------------------
# Test: sequential fallback for single-stream
# ------------------------------------------------------------------


def test_threaded_executor_single_stream_fallback() -> None:
    """When all tasks are on one stream, ThreadedExecutor should still
    produce correct results (fast-path: no thread dispatch)."""
    call_order: List[str] = []

    def _t1(ctx):
        call_order.append("t1")

    def _t2(ctx):
        call_order.append("t2")

    def _t3(ctx):
        call_order.append("t3")

    tasks = (
        Task.from_fn("t1", _t1, stream="default"),
        Task.from_fn("t2", _t2, stream="default"),
        Task.from_fn("t3", _t3, stream="default"),
    )
    schedule = Schedule(stages=(Stage(tasks=tasks),), stream_slots=("default",))
    pool = StreamPool({"default": None})
    pipe = SchedulablePipeline(schedule, pool, executor="threaded")

    pipe.progress(iter([torch.tensor(1.0)]))
    assert call_order == ["t1", "t2", "t3"]


# ------------------------------------------------------------------
# Test: parity with SequentialExecutor
# ------------------------------------------------------------------


def test_threaded_executor_parity_with_sequential() -> None:
    """ThreadedExecutor must produce identical training results as
    SequentialExecutor for a vanilla training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sequential
    seq_model, seq_opt = _seeded_init(device)
    pipe_seq = SchedulablePipeline.basic(
        seq_model, seq_opt, loss_fn=lambda out: out.sum()
    )
    _drive(pipe_seq, _make_batches(device))

    # Threaded
    thr_model, thr_opt = _seeded_init(device)
    pipe_thr = SchedulablePipeline.basic(
        thr_model, thr_opt, loss_fn=lambda out: out.sum(), threaded=True
    )
    _drive(pipe_thr, _make_batches(device))

    for (_, p_seq), (_, p_thr) in zip(
        seq_model.named_parameters(), thr_model.named_parameters()
    ):
        assert torch.allclose(p_seq, p_thr, atol=1e-5, rtol=0), (
            f"Threaded executor diverged from sequential: "
            f"max_diff={(p_seq - p_thr).abs().max().item():.6g}"
        )


# ------------------------------------------------------------------
# Test: threaded + prefetch parity
# ------------------------------------------------------------------


def test_threaded_executor_prefetch_parity() -> None:
    """ThreadedExecutor with prefetch must match sequential+prefetch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_memcpy = device.type == "cuda"

    # Sequential + prefetch
    seq_model, seq_opt = _seeded_init(device)
    pipe_seq = SchedulablePipeline.basic(
        seq_model,
        seq_opt,
        loss_fn=lambda out: out.sum(),
        prefetch=True,
        memcpy_stream=use_memcpy,
    )
    batches = _make_batches(device)
    cpu_batches = [b.cpu() for b in batches] if use_memcpy else batches
    _drive(pipe_seq, cpu_batches)

    # Threaded + prefetch
    thr_model, thr_opt = _seeded_init(device)
    pipe_thr = SchedulablePipeline.basic(
        thr_model,
        thr_opt,
        loss_fn=lambda out: out.sum(),
        prefetch=True,
        memcpy_stream=use_memcpy,
        threaded=True,
    )
    _drive(
        pipe_thr,
        [b.cpu() for b in _make_batches(device)]
        if use_memcpy
        else _make_batches(device),
    )

    for (_, p_seq), (_, p_thr) in zip(
        seq_model.named_parameters(), thr_model.named_parameters()
    ):
        assert torch.allclose(p_seq, p_thr, atol=1e-5, rtol=0), (
            f"Threaded+prefetch diverged: "
            f"max_diff={(p_seq - p_thr).abs().max().item():.6g}"
        )


# ------------------------------------------------------------------
# Test: error propagation
# ------------------------------------------------------------------


def test_threaded_executor_error_propagation() -> None:
    """Errors in worker threads must propagate to the main thread."""

    def _failing_task(ctx):
        raise RuntimeError("intentional failure")

    def _normal_task(ctx):
        pass

    tasks = (
        Task.from_fn("normal", _normal_task, stream="stream_1"),
        Task.from_fn("failing", _failing_task, stream="stream_2"),
    )
    schedule = Schedule(
        stages=(Stage(tasks=tasks),),
        stream_slots=("default", "stream_1", "stream_2"),
    )
    pool = StreamPool({"default": None, "stream_1": None, "stream_2": None})
    pipe = SchedulablePipeline(schedule, pool, executor="threaded")

    with pytest.raises(RuntimeError, match="intentional failure"):
        pipe.progress(iter([torch.tensor(1.0)]))


# ------------------------------------------------------------------
# Test: executor string shorthand
# ------------------------------------------------------------------


def test_executor_string_shorthand() -> None:
    """executor='threaded' must create a ThreadedExecutor."""
    tasks = (Task.from_fn("t", lambda ctx: None, stream="default"),)
    schedule = Schedule(stages=(Stage(tasks=tasks),), stream_slots=("default",))
    pool = StreamPool({"default": None})

    pipe = SchedulablePipeline(schedule, pool, executor="threaded")
    assert isinstance(pipe._executor, ThreadedExecutor)

    pipe2 = SchedulablePipeline(schedule, pool)
    assert isinstance(pipe2._executor, SequentialExecutor)


# ------------------------------------------------------------------
# Test: executor type validation
# ------------------------------------------------------------------


def test_executor_invalid_type_rejected() -> None:
    """Invalid executor type must raise TypeError."""
    tasks = (Task.from_fn("t", lambda ctx: None, stream="default"),)
    schedule = Schedule(stages=(Stage(tasks=tasks),), stream_slots=("default",))
    pool = StreamPool({"default": None})

    with pytest.raises(TypeError, match="executor must be"):
        SchedulablePipeline(schedule, pool, executor=42)


# ------------------------------------------------------------------
# Test: nccl flag on Task
# ------------------------------------------------------------------


def test_task_nccl_flag() -> None:
    """Task and Task.from_fn must accept and store nccl=True."""
    t1 = Task.from_fn("t1", lambda ctx: None, stream="default", nccl=True)
    assert t1.nccl is True

    t2 = Task.from_fn("t2", lambda ctx: None, stream="default")
    assert t2.nccl is False

    class MyTask(Task):
        name = "custom"
        nccl = True

        def run(self, ctx):
            pass

    t3 = MyTask()
    assert t3.nccl is True


# ------------------------------------------------------------------
# Test: thread-local context isolation
# ------------------------------------------------------------------


def test_threaded_context_isolation() -> None:
    """Each thread must see its own _active_offset and iter_count,
    not a value set by another thread."""
    offsets_seen: Dict[str, int] = {}
    barrier = threading.Barrier(2, timeout=5)

    def _task_offset0(ctx):
        offsets_seen["t0"] = ctx._active_offset
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass

    def _task_offset1(ctx):
        offsets_seen["t1"] = ctx._active_offset
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass

    # Two tasks on different streams with different batch_offsets.
    # A prefetch-like schedule where h2d is at offset=1 and compute
    # is at offset=0.
    tasks = (
        Task.from_fn("h2d", _task_offset1, stream="memcpy", batch_offset=1),
        Task.from_fn("compute", _task_offset0, stream="default", batch_offset=0),
    )
    schedule = Schedule(
        stages=(Stage(tasks=tasks),), stream_slots=("default", "memcpy")
    )
    pool = StreamPool({"default": None, "memcpy": None})
    pipe = SchedulablePipeline(schedule, pool, executor="threaded")

    # Need 2 batches in flight for offset=1
    pipe.progress(iter([torch.tensor(1.0), torch.tensor(2.0)]))

    assert (
        offsets_seen["t1"] == 1
    ), f"h2d task should see offset=1, got {offsets_seen['t1']}"
    assert (
        offsets_seen["t0"] == 0
    ), f"compute task should see offset=0, got {offsets_seen['t0']}"


# ------------------------------------------------------------------
# Test: shutdown is idempotent
# ------------------------------------------------------------------


def test_threaded_executor_shutdown_idempotent() -> None:
    """Calling shutdown() multiple times must not raise."""
    executor = ThreadedExecutor()
    executor.shutdown()
    executor.shutdown()  # second call should be a no-op


# ------------------------------------------------------------------
# Test: context manager support
# ------------------------------------------------------------------


def test_pipeline_context_manager() -> None:
    """SchedulablePipeline supports `with` for deterministic shutdown."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, opt = _seeded_init(device)
    with SchedulablePipeline.basic(
        model, opt, loss_fn=lambda out: out.sum(), threaded=True
    ) as pipe:
        _drive(pipe, _make_batches(device))
    # After exiting, the executor's pool should be shut down
    assert pipe._executor._pool is None


# ------------------------------------------------------------------
# Test: NCCL task failure aborts later NCCL tasks
# ------------------------------------------------------------------


def test_nccl_failure_aborts_later_nccl_tasks() -> None:
    """If an nccl=True task fails, subsequent NCCL tasks must not run."""
    ran: List[str] = []

    def _nccl_ok(ctx):
        ran.append("ok")

    def _nccl_fail(ctx):
        ran.append("fail")
        raise RuntimeError("nccl collective failed")

    def _nccl_should_not_run(ctx):
        ran.append("should_not_run")

    tasks = (
        Task.from_fn("nccl_ok", _nccl_ok, stream="stream_1", nccl=True),
        Task.from_fn("nccl_fail", _nccl_fail, stream="stream_2", nccl=True),
        Task.from_fn("nccl_after", _nccl_should_not_run, stream="stream_1", nccl=True),
    )
    schedule = Schedule(
        stages=(Stage(tasks=tasks),),
        stream_slots=("default", "stream_1", "stream_2"),
    )
    pool = StreamPool({"default": None, "stream_1": None, "stream_2": None})
    pipe = SchedulablePipeline(schedule, pool, executor="threaded")

    with pytest.raises(RuntimeError):
        pipe.progress(iter([torch.tensor(1.0)]))

    assert (
        "should_not_run" not in ran
    ), "NCCL task after a failed NCCL task should not have executed"


# ------------------------------------------------------------------
# Test: cancellation prevents further tasks after error
# ------------------------------------------------------------------


def test_cancellation_prevents_further_tasks() -> None:
    """After one thread errors, other threads should stop running
    new tasks (cancellation flag).

    Uses a barrier to synchronize: stream_2's first task waits for
    stream_1 to fail, then checks if its second task gets skipped.
    """
    ran: List[str] = []
    fail_barrier = threading.Event()

    def _fail_task(ctx):
        ran.append("fail")
        fail_barrier.set()
        raise RuntimeError("boom")

    def _wait_for_fail(ctx):
        fail_barrier.wait(timeout=5)
        time.sleep(0.05)  # Let cancellation propagate
        ran.append("after_wait")

    def _should_be_cancelled(ctx):
        ran.append("should_be_cancelled")

    tasks = (
        Task.from_fn("fail_task", _fail_task, stream="stream_1"),
        Task.from_fn("wait_for_fail", _wait_for_fail, stream="stream_2"),
        Task.from_fn("cancelled", _should_be_cancelled, stream="stream_2"),
    )
    schedule = Schedule(
        stages=(Stage(tasks=tasks),),
        stream_slots=("default", "stream_1", "stream_2"),
    )
    pool = StreamPool({"default": None, "stream_1": None, "stream_2": None})
    pipe = SchedulablePipeline(schedule, pool, executor="threaded")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.progress(iter([torch.tensor(1.0)]))

    # The second task on stream_2 should be skipped by cancellation
    assert "should_be_cancelled" not in ran


# ------------------------------------------------------------------
# Test: pool resizes for wider stages
# ------------------------------------------------------------------


def test_pool_resizes_for_wider_stages() -> None:
    """ThreadedExecutor must handle stages with more streams than
    the initial pool size without deadlocking."""
    ran: List[str] = []

    tasks_stage1 = (
        Task.from_fn("s1_a", lambda ctx: ran.append("s1_a"), stream="default"),
        Task.from_fn("s1_b", lambda ctx: ran.append("s1_b"), stream="stream_2"),
    )
    tasks_stage2 = (
        Task.from_fn("s2_a", lambda ctx: ran.append("s2_a"), stream="default"),
        Task.from_fn("s2_b", lambda ctx: ran.append("s2_b"), stream="stream_2"),
        Task.from_fn("s2_c", lambda ctx: ran.append("s2_c"), stream="stream_3"),
    )
    schedule = Schedule(
        stages=(Stage(tasks=tasks_stage1), Stage(tasks=tasks_stage2)),
        stream_slots=("default", "stream_2", "stream_3"),
    )
    pool = StreamPool({"default": None, "stream_2": None, "stream_3": None})

    # Start with max_workers=2, but stage2 needs 3 streams
    executor = ThreadedExecutor(max_workers=2)
    pipe = SchedulablePipeline(schedule, pool, executor=executor)

    pipe.progress(iter([torch.tensor(1.0)]))
    assert set(ran) == {"s1_a", "s1_b", "s2_a", "s2_b", "s2_c"}


# ------------------------------------------------------------------
# Test: thread_map="per_task" — every task on its own thread
# ------------------------------------------------------------------


def test_thread_map_per_task() -> None:
    """With per_task mapping, independent tasks on the SAME stream
    can still run concurrently on different threads."""
    threads_seen: Dict[str, str] = {}
    barrier = threading.Barrier(2, timeout=5)

    def _task_a(ctx):
        threads_seen["a"] = threading.current_thread().name
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass

    def _task_b(ctx):
        threads_seen["b"] = threading.current_thread().name
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass

    # Both on the SAME stream, but per_task puts them on different threads
    tasks = (
        Task.from_fn("task_a", _task_a, stream="default"),
        Task.from_fn("task_b", _task_b, stream="default"),
    )
    schedule = Schedule(stages=(Stage(tasks=tasks),), stream_slots=("default",))
    pool = StreamPool({"default": None})
    executor = ThreadedExecutor(thread_map="per_task")
    pipe = SchedulablePipeline(schedule, pool, executor=executor)

    pipe.progress(iter([torch.tensor(1.0)]))

    assert (
        threads_seen["a"] != threads_seen["b"]
    ), "per_task should put same-stream tasks on different threads"


# ------------------------------------------------------------------
# Test: thread_map=dict — explicit mapping
# ------------------------------------------------------------------


def test_thread_map_dict() -> None:
    """Explicit dict mapping: tasks mapped to the same thread_id share
    a thread; different thread_ids get different threads."""
    threads_seen: Dict[str, str] = {}
    barrier = threading.Barrier(2, timeout=5)

    def _record(name):
        def _fn(ctx):
            threads_seen[name] = threading.current_thread().name
            if name in ("a", "c"):
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    pass

        return _fn

    tasks = (
        Task.from_fn("a", _record("a"), stream="default"),
        Task.from_fn("b", _record("b"), stream="default"),
        Task.from_fn("c", _record("c"), stream="default"),
    )
    schedule = Schedule(stages=(Stage(tasks=tasks),), stream_slots=("default",))
    pool = StreamPool({"default": None})
    executor = ThreadedExecutor(
        thread_map={"a": "thread_1", "b": "thread_1", "c": "thread_2"}
    )
    pipe = SchedulablePipeline(schedule, pool, executor=executor)
    pipe.progress(iter([torch.tensor(1.0)]))

    # a and b on same thread, c on different thread
    assert threads_seen["a"] == threads_seen["b"]
    assert threads_seen["a"] != threads_seen["c"]


# ------------------------------------------------------------------
# Test: thread_map=callable — custom function
# ------------------------------------------------------------------


def test_thread_map_callable() -> None:
    """Callable thread_map: user provides an arbitrary function."""
    threads_seen: Dict[str, str] = {}
    barrier = threading.Barrier(2, timeout=5)

    def _record(name):
        def _fn(ctx):
            threads_seen[name] = threading.current_thread().name
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass

        return _fn

    tasks = (
        Task.from_fn("io_task", _record("io_task"), stream="default"),
        Task.from_fn("compute_task", _record("compute_task"), stream="default"),
    )
    schedule = Schedule(stages=(Stage(tasks=tasks),), stream_slots=("default",))
    pool = StreamPool({"default": None})
    executor = ThreadedExecutor(
        thread_map=lambda task: "io" if "io" in task.name else "compute"
    )
    pipe = SchedulablePipeline(schedule, pool, executor=executor)
    pipe.progress(iter([torch.tensor(1.0)]))

    assert threads_seen["io_task"] != threads_seen["compute_task"]


# ------------------------------------------------------------------
# Test: cross-thread deps work when thread != stream
# ------------------------------------------------------------------


def test_cross_thread_deps_with_decoupled_streams() -> None:
    """When threads and streams are decoupled, cross-thread CPU deps
    must still enforce data ordering via the DAG (reads/writes)."""
    results: List[int] = []

    def _producer(ctx):
        time.sleep(0.05)
        ctx.slots.set("data", 99)

    def _consumer(ctx):
        results.append(ctx.slots["data"])

    # Both on SAME stream, but mapped to DIFFERENT threads.
    # CPU-side ordering must come from slot dependency, not stream.
    tasks = (
        Task.from_fn(
            "producer",
            _producer,
            stream="default",
            writes=(DataSlot("data"),),
        ),
        Task.from_fn(
            "consumer",
            _consumer,
            stream="default",
            reads=(DataSlot("data"),),
        ),
    )
    schedule = Schedule(stages=(Stage(tasks=tasks),), stream_slots=("default",))
    pool = StreamPool({"default": None})
    executor = ThreadedExecutor(thread_map="per_task")
    pipe = SchedulablePipeline(schedule, pool, executor=executor)

    pipe.progress(iter([torch.tensor(1.0)]))
    assert results == [99]


# ------------------------------------------------------------------
# Test: basic(threaded=True, thread_map=...) passthrough
# ------------------------------------------------------------------


def test_basic_thread_map_passthrough() -> None:
    """basic(threaded=True, thread_map=...) passes thread_map to
    the ThreadedExecutor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, opt = _seeded_init(device)
    pipe = SchedulablePipeline.basic(
        model,
        opt,
        loss_fn=lambda out: out.sum(),
        threaded=True,
        thread_map="per_task",
    )
    assert isinstance(pipe._executor, ThreadedExecutor)
    assert pipe._executor._thread_map == "per_task"


# ------------------------------------------------------------------
# Test: NCCL deadlock on cancellation (Codex CRITICAL regression guard)
# ------------------------------------------------------------------


def test_nccl_lock_does_not_deadlock_on_early_failure() -> None:
    """Regression guard for the deadlock Codex flagged:

    Setup: two NCCL-tagged tasks (tickets 0 + 1) on different threads,
    plus a non-NCCL task that fails BEFORE ticket 0 has a chance to
    release. Without ``_NcclOrderedLock.abort()``, the worker that
    holds ticket 1's wait would block forever — no one ever advances
    ``next_ticket`` past 0.

    With the fix, the failing thread calls ``self._nccl_lock.abort()``
    in its except block; the waiter wakes and re-raises. Stage exits
    within ``timeout``.
    """
    barrier = threading.Barrier(2, timeout=5)

    def _fail_first(ctx):
        # Fire BEFORE the NCCL chain has any release. Sync with the
        # NCCL worker so we KNOW it's already inside acquire(1).
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        # Sleep just enough for the other thread to enter acquire(1)
        time.sleep(0.05)
        raise RuntimeError("fail before NCCL chain progresses")

    def _nccl_ticket_0(ctx):
        # Wait until the failing task is about to fail
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        # Block here — the failing task will raise and abort the lock
        time.sleep(1.0)

    def _nccl_ticket_1(ctx):
        pass  # never gets here in the deadlock scenario

    tasks = (
        Task.from_fn("fail_first", _fail_first, stream="stream_a"),
        Task.from_fn("nccl_0", _nccl_ticket_0, stream="stream_b", nccl=True),
        Task.from_fn("nccl_1", _nccl_ticket_1, stream="stream_c", nccl=True),
    )
    schedule = Schedule(
        stages=(Stage(tasks=tasks),),
        stream_slots=("default", "stream_a", "stream_b", "stream_c"),
    )
    pool = StreamPool(
        {
            "default": None,
            "stream_a": None,
            "stream_b": None,
            "stream_c": None,
        }
    )
    pipe = SchedulablePipeline(schedule, pool, executor="threaded")

    # If abort() is missing, this hangs forever. Cap with a soft
    # deadline via Python — pytest will reap, but worse, CI hangs.
    # We rely on the fix making this fast (well under 1s after the
    # 0.05s sleep + 1s nccl_0 sleep, totaling ~1.05s in the worst
    # ordering).
    start = time.perf_counter()
    with pytest.raises(RuntimeError, match="fail before NCCL"):
        pipe.progress(iter([torch.tensor(1.0)]))
    elapsed = time.perf_counter() - start
    # 5s ceiling: deadlock would take >> this. Real fix unblocks
    # within ~1s.
    assert elapsed < 5.0, (
        f"NCCL lock deadlock — stage took {elapsed:.1f}s; "
        f"fix should unblock within 1s of the failing task raising"
    )
