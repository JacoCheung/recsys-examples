# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for ``ThreadedExecutor``."""

import threading
import time

import pytest
from commons.pipeline.engine import (
    DataSlot,
    SameProgressSyncSide,
    SchedulablePipeline,
    Schedule,
    SequentialExecutor,
    Stage,
    StreamPool,
    Task,
    ThreadedExecutor,
)
from commons.pipeline.engine.executor import _compute_cpu_deps


def _pipe(tasks, *, executor="threaded", stream_slots=None) -> SchedulablePipeline:
    if stream_slots is None:
        stream_slots = tuple(sorted({"default"} | {t.stream for t in tasks}))
    schedule = Schedule(stages=(Stage(tasks=tuple(tasks)),), stream_slots=stream_slots)
    pool = StreamPool({name: None for name in stream_slots})
    return SchedulablePipeline(schedule, pool, executor=executor)


def test_threaded_executor_runs_independent_streams_on_different_threads() -> None:
    barrier = threading.Barrier(2, timeout=5)
    threads = {}

    def _record(name):
        def _fn(ctx):
            threads[name] = threading.current_thread().name
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass

        return _fn

    pipe = _pipe(
        (
            Task.from_fn("a", _record("a"), stream="stream_a"),
            Task.from_fn("b", _record("b"), stream="stream_b"),
        )
    )
    pipe.progress(iter([None]))

    assert threads["a"] != threads["b"]


def test_cross_thread_slot_dependency_orders_consumer() -> None:
    observed = []

    def _producer(ctx):
        time.sleep(0.05)
        ctx.slots.set("x", 42)

    def _consumer(ctx):
        observed.append(ctx.slots["x"])

    pipe = _pipe(
        (
            Task.from_fn("producer", _producer, stream="a", writes=("x",)),
            Task.from_fn("consumer", _consumer, stream="b", reads=("x",)),
        )
    )
    pipe.progress(iter([None]))

    assert observed == [42]


def test_nccl_tasks_run_in_declaration_order() -> None:
    order = []
    lock = threading.Lock()

    def _nccl(name):
        def _fn(ctx):
            with lock:
                order.append(name)

        return Task.from_fn(name, _fn, stream=name, nccl=True)

    pipe = _pipe((_nccl("a"), _nccl("b"), _nccl("c")))
    pipe.progress(iter([None]))

    assert order == ["a", "b", "c"]


def test_worker_error_propagates_and_skips_later_chain_tasks() -> None:
    ran = []
    release = threading.Event()

    def _fail(ctx):
        ran.append("fail")
        release.set()
        raise RuntimeError("boom")

    def _wait(ctx):
        release.wait(timeout=5)
        ran.append("wait")

    def _skip(ctx):
        ran.append("skip")

    pipe = _pipe(
        (
            Task.from_fn("fail", _fail, stream="a"),
            Task.from_fn("wait", _wait, stream="b"),
            Task.from_fn("skip", _skip, stream="b"),
        )
    )

    with pytest.raises(RuntimeError, match="boom"):
        pipe.progress(iter([None]))
    assert "skip" not in ran


def test_nccl_failure_aborts_later_nccl_tasks() -> None:
    ran = []

    def _ok(ctx):
        ran.append("ok")

    def _fail(ctx):
        ran.append("fail")
        raise RuntimeError("collective failed")

    def _after(ctx):
        ran.append("after")

    pipe = _pipe(
        (
            Task.from_fn("ok", _ok, stream="a", nccl=True),
            Task.from_fn("fail", _fail, stream="b", nccl=True),
            Task.from_fn("after", _after, stream="c", nccl=True),
        )
    )

    with pytest.raises(RuntimeError, match="collective failed"):
        pipe.progress(iter([None]))
    assert "after" not in ran


def test_executor_shorthand_and_type_validation() -> None:
    task = Task.from_fn("t", lambda ctx: None)
    threaded = _pipe((task,), executor="threaded", stream_slots=("default",))
    sequential = _pipe((task,), executor=None, stream_slots=("default",))

    assert isinstance(threaded._executor, ThreadedExecutor)
    assert isinstance(sequential._executor, SequentialExecutor)

    with pytest.raises(TypeError, match="executor must be"):
        _pipe((task,), executor=object(), stream_slots=("default",))


def test_thread_map_dict_groups_tasks_by_thread_id() -> None:
    barrier = threading.Barrier(2, timeout=5)
    threads = {}

    def _record(name):
        def _fn(ctx):
            threads[name] = threading.current_thread().name
            if name in {"a", "c"}:
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    pass

        return _fn

    executor = ThreadedExecutor(thread_map={"a": "io", "b": "io", "c": "compute"})
    pipe = _pipe(
        (
            Task.from_fn("a", _record("a"), stream="stream_a"),
            Task.from_fn("b", _record("b"), stream="stream_b"),
            Task.from_fn("c", _record("c"), stream="stream_c"),
        ),
        executor=executor,
        stream_slots=("default", "stream_a", "stream_b", "stream_c"),
    )
    pipe.progress(iter([None]))

    assert threads["a"] == threads["b"]
    assert threads["a"] != threads["c"]


def test_cross_thread_deps_work_when_thread_map_decouples_from_stream() -> None:
    observed = []

    def _producer(ctx):
        time.sleep(0.05)
        ctx.slots.set("x", 99)

    def _consumer(ctx):
        observed.append(ctx.slots["x"])

    executor = ThreadedExecutor(thread_map="per_task")
    pipe = _pipe(
        (
            Task.from_fn("producer", _producer, writes=(DataSlot("x"),)),
            Task.from_fn("consumer", _consumer, reads=(DataSlot("x"),)),
        ),
        executor=executor,
        stream_slots=("default",),
    )
    pipe.progress(iter([None]))

    assert observed == [99]


def test_cross_lookahead_depends_on_does_not_create_cpu_cycle() -> None:
    t1 = Task.from_fn(
        "T1",
        fn=lambda ctx: None,
        lookahead=0,
        stream="stream_a",
        depends_on=("T2",),
    )
    t2 = Task.from_fn(
        "T2",
        fn=lambda ctx: None,
        lookahead=2,
        stream="stream_b",
        same_progress_sync=("T1",),
    )
    completion = {"T1": threading.Event(), "T2": threading.Event()}
    cpu_deps = _compute_cpu_deps(
        [t1, t2],
        {"T1": "stream_a", "T2": "stream_b"},
        completion,
    )

    assert completion["T1"] in cpu_deps.get("T2", [])
    assert completion["T2"] not in cpu_deps.get("T1", [])


def test_gpu_only_same_progress_sync_does_not_create_cpu_dep() -> None:
    producer = Task.from_fn("producer", fn=lambda ctx: None, stream="stream_a")
    consumer = Task.from_fn(
        "consumer",
        fn=lambda ctx: None,
        stream="stream_b",
        same_progress_sync=(("producer", SameProgressSyncSide.GPU),),
    )
    completion = {"producer": threading.Event(), "consumer": threading.Event()}

    cpu_deps = _compute_cpu_deps(
        [producer, consumer],
        {"producer": "stream_a", "consumer": "stream_b"},
        completion,
    )

    assert completion["producer"] not in cpu_deps.get("consumer", [])
