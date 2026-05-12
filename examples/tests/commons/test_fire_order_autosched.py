# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for fire-order lookahead assignment."""

from __future__ import annotations

import pytest
from commons.pipeline.engine import Schedule, Stage
from commons.pipeline.engine.autosched import (
    DEFAULT_BIT_EXACT_TASKS,
    CostModel,
    TaskCost,
    TaskResource,
    auto_assign_lookaheads,
    compute_overlap_matrix,
    default_stream_critical_path_us,
    task_resources,
)
from commons.pipeline.engine.task import Task


def _t(name: str, *, stream: str = "default", la: int = 0, **kwargs) -> Task:
    return Task.from_fn(
        name=name,
        fn=lambda ctx: None,
        stream=stream,
        lookahead=la,
        **kwargs,
    )


def _schedule(tasks, stream_slots=("default", "memcpy", "data_dist")) -> Schedule:
    return Schedule(stages=(Stage(tasks=tuple(tasks)),), stream_slots=stream_slots)


def _cost_model(tasks, default_gpu_us: float = 1000.0, **overrides) -> CostModel:
    return CostModel(
        {t.name: TaskCost(0.0, overrides.get(t.name, default_gpu_us)) for t in tasks}
    )


def test_task_resources_and_overlap_matrix_cover_resource_labels() -> None:
    tasks = [
        _t("h2d", stream="memcpy", la=2),
        _t("prefetch_embeddings", stream="prefetch", la=1),
        _t("start_input_dist", stream="data_dist", la=1, nccl=True),
        _t("backward", stream="default", nccl=True),
        _t("forward", stream="default"),
    ]
    resources = task_resources(tasks)

    assert resources["h2d"].pcie is True
    assert resources["prefetch_embeddings"].pcie is True
    assert resources["start_input_dist"].nccl_comm == "dp"
    assert resources["forward"].nccl_comm is None

    matrix = compute_overlap_matrix(tasks, resources)
    assert matrix[("forward", "forward")] == "self"
    assert matrix[("forward", "backward")] == "stream"
    assert matrix[("start_input_dist", "backward")] == "nccl"
    assert matrix[("h2d", "prefetch_embeddings")] == "pcie"
    assert matrix[("h2d", "forward")] == "ok"
    assert (
        matrix[("backward", "start_input_dist")]
        == matrix[("start_input_dist", "backward")]
    )


def test_taskresource_conflict_priority_is_stream_then_nccl_then_pcie() -> None:
    same_stream_a = TaskResource(stream="default", nccl_comm="dp", pcie=True)
    same_stream_b = TaskResource(stream="default", nccl_comm="ep", pcie=True)
    assert same_stream_a.conflicts_with(same_stream_b) == "stream"

    same_comm_a = TaskResource(stream="memcpy", nccl_comm="dp", pcie=True)
    same_comm_b = TaskResource(stream="data_dist", nccl_comm="dp", pcie=True)
    assert same_comm_a.conflicts_with(same_comm_b) == "nccl"

    pcie_a = TaskResource(stream="memcpy", nccl_comm=None, pcie=True)
    pcie_b = TaskResource(stream="prefetch", nccl_comm=None, pcie=True)
    assert pcie_a.conflicts_with(pcie_b) == "pcie"

    independent = TaskResource(stream="prefetch", nccl_comm=None, pcie=False)
    assert pcie_a.conflicts_with(independent) is None


def test_default_stream_critical_path_counts_only_current_default_chain() -> None:
    tasks = [
        _t("zero_grad"),
        _t("forward"),
        _t("backward"),
        _t("default_la1", la=1),
        _t("h2d", stream="memcpy", la=2),
    ]
    costs = _cost_model(
        tasks,
        zero_grad=10,
        forward=100,
        backward=200,
        default_la1=9999,
        h2d=9999,
    )

    assert default_stream_critical_path_us(tasks, costs) == 310


def test_auto_assign_keeps_default_chain_and_bumps_offdefault_floor() -> None:
    tasks = [
        _t("h2d", stream="memcpy", la=2, writes=("batch_gpu",)),
        _t(
            "start_input_dist",
            stream="data_dist",
            la=1,
            reads=("batch_gpu",),
            nccl=True,
        ),
        _t("forward", reads=("batch_gpu",), writes=("losses",)),
        _t("backward", reads=("losses",), depends_on=("forward",)),
        _t("finalize_model_grads", depends_on=("backward",)),
        _t("optimizer_step", depends_on=("finalize_model_grads")),
    ]
    out = auto_assign_lookaheads(
        _schedule(tasks),
        _cost_model(tasks, start_input_dist=50_000, forward=1000),
        max_in_flight=8,
    )

    assert out["forward"] == 0
    assert out["backward"] == 0
    assert out["finalize_model_grads"] == 0
    assert out["optimizer_step"] == 0
    assert out["start_input_dist"] >= 1
    assert out["h2d"] >= out["start_input_dist"]


def test_auto_assign_rejects_authored_floor_above_budget() -> None:
    tasks = [
        _t("h2d", stream="memcpy", la=4, writes=("batch_gpu",)),
        _t("forward", reads=("batch_gpu",)),
    ]
    with pytest.raises(ValueError, match="exceeds max_in_flight"):
        auto_assign_lookaheads(_schedule(tasks), _cost_model(tasks), max_in_flight=3)


def test_auto_assign_rejects_async_optimizer_shapes() -> None:
    tasks = [
        _t("forward", writes=("losses",)),
        _t("backward", reads=("losses",), depends_on=("forward",)),
        _t("finalize_model_grads", depends_on=("backward")),
        _t("optimizer_step", stream="opt_side", depends_on=("finalize_model_grads")),
    ]
    with pytest.raises(ValueError, match="optimizer_step.*default"):
        auto_assign_lookaheads(
            _schedule(tasks, stream_slots=("default", "opt_side")),
            _cost_model(tasks),
        )

    tasks[-1] = _t("optimizer_step", la=1, depends_on=("finalize_model_grads"))
    with pytest.raises(ValueError, match="optimizer_step.*lookahead=0"):
        auto_assign_lookaheads(_schedule(tasks), _cost_model(tasks))


def test_auto_assign_cross_iter_consumer_is_not_bumped() -> None:
    tasks = [
        _t("producer", stream="memcpy", la=2),
        _t(
            "consumer",
            stream="data_dist",
            la=1,
            nccl=True,
            cross_iter_depends_on=(("producer", -1),),
        ),
        _t("forward"),
    ]
    out = auto_assign_lookaheads(
        _schedule(tasks),
        _cost_model(tasks, consumer=50_000, forward=100),
        max_in_flight=5,
    )

    assert out["consumer"] == 1


def test_auto_assign_cross_iter_cap_conflict_raises() -> None:
    tasks = [
        _t("seed", stream="memcpy", la=2),
        _t(
            "producer",
            stream="memcpy",
            la=1,
            writes=("p_out",),
            cross_iter_depends_on=(("seed", -1),),
        ),
        _t("consumer", stream="data_dist", la=2, reads=("p_out",), nccl=True),
        _t("forward"),
    ]
    with pytest.raises(ValueError, match="cross-iter cap"):
        auto_assign_lookaheads(
            _schedule(tasks),
            _cost_model(tasks, consumer=50_000, forward=100),
            max_in_flight=5,
        )


def test_default_bit_exact_tasks_exported() -> None:
    assert {"forward", "backward", "finalize_model_grads", "optimizer_step"} <= set(
        DEFAULT_BIT_EXACT_TASKS
    )
