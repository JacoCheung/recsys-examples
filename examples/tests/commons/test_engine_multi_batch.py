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

"""V4 — multi-batch in-flight + prefill/drain (SPEC §4.8).

Tests:
  - `test_prefetch_correctness`: M batches in → M results out
  - `test_prefill_mask`: §4.8 mask formula during prefill
  - `test_drain_mask`: §4.8 mask formula during drain
  - `test_short_dataloader`: M < max_offset
  - `test_empty_dataloader`: M = 0
  - `test_ring_eviction`: no CUDA memory growth over many iters
  - `test_preset_prefetch_parity`: basic(prefetch=True) matches basic()

The mask tests intentionally do NOT import the production helper
(`pipeline._should_run`) — they compute expected (iter_count, k)
execution sets independently from §4.8's formula. This prevents a
tautological pass if the formula were wrong in both places.
"""

import gc
from typing import Iterator

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

# ----------------------------------------------------------------------
# Mask-formula tests — computed independently from SPEC §4.8
# ----------------------------------------------------------------------


def _expected_runs(max_offset: int, k: int, M: int):
    """Independent computation of which internal `iter_count` values
    satisfy `(max_offset - k) ≤ iter < M + (max_offset - k)`.

    Does NOT call `pipeline._should_run` — keeps the oracle
    independent of the code under test.
    """
    lo = max_offset - k
    hi = M + (max_offset - k)
    return list(range(lo, hi))


def _make_instrumented_schedule(max_offset: int, ks: list, record: dict):
    """Build a schedule where every task records which iter_count it
    ran on. Tasks' batch_offset = k, each appending to
    `record[k]`."""
    tasks = []
    for k in ks:

        def _make_fn(k=k):
            def _fn(ctx):
                record.setdefault(k, []).append(ctx.iter_count)

            return _fn

        tasks.append(
            Task.from_fn(
                name=f"task_k{k}",
                fn=_make_fn(),
                stream="default",
                batch_offset=k,
            )
        )
    return Schedule(
        stages=(Stage(tasks=tuple(tasks)),),
        stream_slots=("default",),
    )


def test_prefill_mask() -> None:
    """For `M=3, max_offset=2` (k∈{0,1,2}):
    Task k=2 runs at iter 0,1,2 (prefill + steady, no drain)
    Task k=1 runs at iter 1,2,3 (one prefill + steady + one drain)
    Task k=0 runs at iter 2,3,4 (steady + drain; no prefill)
    """
    M, max_offset = 3, 2
    record: dict = {}
    schedule = _make_instrumented_schedule(max_offset, [0, 1, 2], record)
    pool = StreamPool({"default": None})
    pipe = SchedulablePipeline(schedule, pool)

    # Drive to exhaustion.
    batches = iter(list(range(M)))
    results = []
    while True:
        try:
            r = pipe.progress(batches)
            results.append(r)
        except StopIteration:
            break

    # User sees M results (one per batch).
    assert len(results) == M, f"expected {M} results, got {len(results)}"
    # Mask rule for each k:
    for k in [0, 1, 2]:
        assert record.get(k, []) == _expected_runs(max_offset, k, M), (
            f"k={k}: ran on {record.get(k, [])}, "
            f"expected {_expected_runs(max_offset, k, M)}"
        )


def test_drain_mask() -> None:
    """Same as prefill mask but with a larger M (5) to give a more
    distinct steady-state + drain pattern."""
    M, max_offset = 5, 3
    record: dict = {}
    schedule = _make_instrumented_schedule(max_offset, [0, 1, 2, 3], record)
    pool = StreamPool({"default": None})
    pipe = SchedulablePipeline(schedule, pool)

    batches = iter(list(range(M)))
    results = []
    while True:
        try:
            results.append(pipe.progress(batches))
        except StopIteration:
            break

    assert len(results) == M
    for k in [0, 1, 2, 3]:
        expected = _expected_runs(max_offset, k, M)
        assert (
            record.get(k, []) == expected
        ), f"k={k}: got {record.get(k, [])}, expected {expected}"


def test_prefetch_correctness_m_in_m_out() -> None:
    """Basic driver contract: M calls → M results → (M+1)th raises
    StopIteration. Uses a schedule with batch_offset=1 prefetch."""
    record: dict = {}

    def _h2d(ctx):
        record.setdefault("h2d_iters", []).append(ctx.iter_count)
        batch = ctx.slots["batch_cpu"]
        ctx.slots.set("batch_gpu", batch)  # cpu passthrough

    def _compute(ctx):
        record.setdefault("compute_iters", []).append(ctx.iter_count)
        ctx.slots.set("step_result", ctx.slots["batch_gpu"])

    h2d = Task.from_fn(
        name="h2d",
        fn=_h2d,
        reads=(DataSlot("batch_cpu", batch_offset=1),),
        writes=(DataSlot("batch_gpu", batch_offset=1),),
        stream="default",
        batch_offset=1,
    )
    compute = Task.from_fn(
        name="compute",
        fn=_compute,
        reads=(DataSlot("batch_gpu"),),
        writes=(DataSlot("step_result"),),
        stream="default",
        batch_offset=0,
    )
    schedule = Schedule(
        stages=(Stage(tasks=(h2d, compute)),), stream_slots=("default",)
    )
    pool = StreamPool({"default": None})
    pipe = SchedulablePipeline(schedule, pool)

    M = 5
    batches = iter(range(M))
    results = []
    while True:
        try:
            results.append(pipe.progress(batches))
        except StopIteration:
            break

    # M results, (M+1)th call raised
    assert results == [0, 1, 2, 3, 4]
    # h2d ran on iters 0..4 (prefill + steady, not drain)
    assert record["h2d_iters"] == [0, 1, 2, 3, 4]
    # compute ran on iters 1..5 (steady + drain, not prefill)
    assert record["compute_iters"] == [1, 2, 3, 4, 5]


def test_short_dataloader_m_less_than_max_offset() -> None:
    """Dataloader shorter than max_offset (M=1, max_offset=2). Engine
    must still produce exactly M=1 result."""
    record: dict = {}
    schedule = _make_instrumented_schedule(2, [0, 1, 2], record)
    pool = StreamPool({"default": None})
    pipe = SchedulablePipeline(schedule, pool)

    results = []
    batches = iter([0])  # M=1
    while True:
        try:
            results.append(pipe.progress(batches))
        except StopIteration:
            break

    assert len(results) == 1, f"M=1 should yield 1 result, got {len(results)}"


def test_empty_dataloader_first_call_raises() -> None:
    """M=0: first `progress` call raises StopIteration without running
    any task with batch_offset=0."""
    record: dict = {}
    schedule = _make_instrumented_schedule(1, [0, 1], record)
    pool = StreamPool({"default": None})
    pipe = SchedulablePipeline(schedule, pool)

    batches: Iterator[torch.Tensor] = iter([])  # M=0
    with pytest.raises(StopIteration):
        pipe.progress(batches)
    # No k=0 task should ever have run (no current batch ever arrives)
    assert record.get(0, []) == []


def test_ring_eviction_no_memory_growth() -> None:
    """Run 200 iterations of a schedule that writes a large CUDA
    tensor to a slot each step. Ring advance must drop the old slot
    store so CUDA memory doesn't grow unboundedly."""
    if not torch.cuda.is_available():
        pytest.skip("memory test needs CUDA")
    device = torch.device("cuda:0")

    # Tensor sizing: 1_000_000 × float32 = 4,000,000 bytes = ~4 MB
    # per iteration. Chosen so retaining even one extra slot store
    # pushes memory growth past the 10 MB bound below.
    size = 1_000_000

    def _write_big(ctx):
        # dtype=torch.float32 explicit — growth arithmetic below
        # relies on this.
        t = torch.ones(size, dtype=torch.float32, device=device)
        ctx.slots.set("step_result", t)

    task = Task.from_fn(
        name="big_write",
        fn=_write_big,
        writes=(DataSlot("step_result"),),
        stream="default",
    )
    schedule = Schedule(stages=(Stage(tasks=(task,)),), stream_slots=("default",))
    pool = StreamPool({"default": torch.cuda.default_stream(device)})
    pipe = SchedulablePipeline(schedule, pool)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)

    # Drive 200 iterations
    batches = iter(range(200))
    while True:
        try:
            pipe.progress(batches)
        except StopIteration:
            break

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    final = torch.cuda.memory_allocated(device)

    # Per-iter allocation: 1M × float32 = 4.0 MB exactly.
    # Expected retained after advance: 0 (the offset=0 slot is
    # dropped on advance — its tensor refcount drops to zero).
    # Allocator overhead + any single transient block: <6 MB
    # observed; 10 MB threshold leaves no room for a second
    # retained 4 MB slot (which would push over the bound).
    growth = final - baseline
    assert growth < 10 * 1024 * 1024, (
        f"CUDA memory grew by {growth / 1024 / 1024:.1f} MB over 200 "
        f"iterations. Each iter allocates 1M×float32 = 4.0 MB; ring "
        f"should evict on advance so growth reflects only allocator "
        f"overhead. A >10 MB growth indicates the ring is holding "
        f"onto a second (or more) slot store worth of tensors."
    )


def test_preset_prefetch_parity() -> None:
    """`SchedulablePipeline.basic(..., prefetch=True, memcpy_stream=True)`
    must give the same final params as `basic(...)` (no prefetch) on
    the same model/seed/data. Only the scheduling differs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    model_a = torch.nn.Linear(8, 4).to(device)
    opt_a = torch.optim.SGD(model_a.parameters(), lr=1e-2)
    pipe_a = SchedulablePipeline.basic(model_a, opt_a, loss_fn=lambda o: o.sum())

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    model_b = torch.nn.Linear(8, 4).to(device)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=1e-2)
    pipe_b = SchedulablePipeline.basic(
        model_b,
        opt_b,
        loss_fn=lambda o: o.sum(),
        prefetch=True,
        memcpy_stream=(device.type == "cuda"),
    )

    torch.manual_seed(123)
    batches = [torch.randn(4, 8) for _ in range(15)]

    # Drive each pipe to exhaustion on the same batches.
    def _run(pipe, data):
        it = iter(data)
        results = []
        while True:
            try:
                results.append(pipe.progress(it))
            except StopIteration:
                break
        return results

    res_a = _run(pipe_a, [b.to(device) for b in batches])
    res_b = _run(
        pipe_b, [b.to(device) if device.type == "cuda" else b for b in batches]
    )

    # Both paths see M=15 results
    assert len(res_a) == 15
    assert len(res_b) == 15

    # Final parameters match (prefetch is schedule-equivalent, not
    # computation-different).
    for (_, p_a), (_, p_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert torch.allclose(p_a, p_b, atol=1e-5, rtol=0), (
            f"prefetch diverged from single-stream: max_abs_diff="
            f"{(p_a - p_b).abs().max().item():.6g}"
        )


def test_in_flight_batches_derived_from_max_offset() -> None:
    """`Schedule.in_flight_batches` is a `@property` — never authored.
    Confirms §4.2 rule 4 derivation survives V4 multi-batch."""
    t0 = Task.from_fn(name="t0", fn=lambda ctx: None, stream="default", batch_offset=0)
    t2 = Task.from_fn(name="t2", fn=lambda ctx: None, stream="default", batch_offset=2)
    schedule = Schedule(stages=(Stage(tasks=(t0, t2)),), stream_slots=("default",))
    assert schedule.in_flight_batches == 3  # max(0, 2) + 1
