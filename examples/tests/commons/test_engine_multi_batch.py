# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for multi-batch prefill/drain behavior."""

from typing import Dict, Iterable, List

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


def _drain(pipe: SchedulablePipeline, batches: Iterable):
    out = []
    it = iter(batches)
    while True:
        try:
            out.append(pipe.progress(it))
        except StopIteration:
            return out


def _instrumented_schedule(offsets, record):
    tasks = []
    for offset in offsets:

        def _make_fn(k=offset):
            def _fn(ctx):
                record.setdefault(k, []).append(ctx.iter_count)

            return _fn

        tasks.append(
            Task.from_fn(
                name=f"task_{offset}",
                fn=_make_fn(),
                lookahead=offset,
            )
        )
    return Schedule(stages=(Stage(tasks=tuple(tasks)),), stream_slots=("default",))


def _expected_iters(max_offset: int, offset: int, n_batches: int):
    lo = max_offset - offset
    return list(range(lo, lo + n_batches))


def test_prefill_and_drain_mask_for_multiple_offsets() -> None:
    n_batches = 4
    max_offset = 2
    record: Dict[int, List[int]] = {}
    pipe = SchedulablePipeline(
        _instrumented_schedule([0, 1, 2], record),
        StreamPool({"default": None}),
    )

    results = _drain(pipe, range(n_batches))

    assert len(results) == n_batches
    for offset in (0, 1, 2):
        assert record[offset] == _expected_iters(max_offset, offset, n_batches)


def test_prefetch_schedule_returns_one_result_per_input_batch() -> None:
    seen: Dict[str, List[int]] = {}

    def _prefetch(ctx):
        seen.setdefault("prefetch", []).append(ctx.iter_count)
        ctx.slots.set("batch_gpu", ctx.slots["batch_cpu"])

    def _compute(ctx):
        seen.setdefault("compute", []).append(ctx.iter_count)
        ctx.slots.set("step_result", ctx.slots["batch_gpu"])

    schedule = Schedule(
        stages=(
            Stage(
                tasks=(
                    Task.from_fn(
                        "prefetch",
                        _prefetch,
                        lookahead=1,
                        reads=(DataSlot("batch_cpu", 1),),
                        writes=(DataSlot("batch_gpu", 1),),
                    ),
                    Task.from_fn(
                        "compute",
                        _compute,
                        reads=("batch_gpu",),
                        writes=("step_result",),
                    ),
                )
            ),
        ),
        stream_slots=("default",),
    )
    pipe = SchedulablePipeline(schedule, StreamPool({"default": None}))

    assert _drain(pipe, range(5)) == [0, 1, 2, 3, 4]
    assert seen["prefetch"] == [0, 1, 2, 3, 4]
    assert seen["compute"] == [1, 2, 3, 4, 5]


def test_empty_and_short_dataloaders_have_clear_stop_behavior() -> None:
    record: Dict[int, List[int]] = {}
    pipe = SchedulablePipeline(
        _instrumented_schedule([0, 1, 2], record),
        StreamPool({"default": None}),
    )

    assert len(_drain(pipe, [0])) == 1

    pipe = SchedulablePipeline(
        _instrumented_schedule([0, 1], {}),
        StreamPool({"default": None}),
    )
    with pytest.raises(StopIteration):
        pipe.progress(iter([]))


def test_progress_resets_state_on_new_iterator_after_drain() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    pipe = SchedulablePipeline.basic(model, opt, loss_fn=lambda out: out.sum())

    first = [torch.randn(2, 4, device=device) for _ in range(3)]
    second = [torch.randn(2, 4, device=device) for _ in range(3)]

    assert len(_drain(pipe, first)) == len(first)
    assert len(_drain(pipe, second)) == len(second)


def test_progress_rejects_iterator_change_while_batches_are_in_flight() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Linear(4, 2).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    pipe = SchedulablePipeline.basic(
        model,
        opt,
        loss_fn=lambda out: out.sum(),
        prefetch=True,
        memcpy_stream=(device.type == "cuda"),
    )

    first = iter([torch.randn(2, 4, device=device) for _ in range(4)])
    second = iter([torch.randn(2, 4, device=device) for _ in range(4)])
    pipe.progress(first)

    with pytest.raises(RuntimeError, match="before the previous one drained"):
        pipe.progress(second)
