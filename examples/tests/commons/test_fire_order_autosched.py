# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real unit tests for the fire-order auto-scheduler.

Builds real :class:`Schedule`/:class:`Task` objects (no mocks) so the
tests exercise the same path the engine uses. Each test enumerates the
expected outcome by name (no lumped parametrize) per CLAUDE.md.
"""
from __future__ import annotations

from typing import List

import pytest
from commons.pipeline.engine import Schedule, Stage
from commons.pipeline.engine.autosched import (
    CostModel,
    TaskCost,
    TaskResource,
    auto_assign_lookaheads,
    compute_overlap_matrix,
    default_stream_critical_path_us,
    describe_overlap_matrix,
    task_resources,
)
from commons.pipeline.engine.task import Task

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------


def _T(name: str, *, stream: str, la: int = 0, **kw) -> Task:
    return Task.from_fn(name, fn=lambda ctx: None, stream=stream, lookahead=la, **kw)


def _hstu_like_schedule(*, prefetch: bool = True) -> Schedule:
    """Build a tiny HSTU-shaped schedule; the values mirror the
    factory in ``hstu_pipeline/pipeline.py`` but trimmed to the bits
    the auto-scheduler exercises (stream / lookahead / nccl flag /
    reads / writes / depends_on).
    """
    h2d_la = 2
    id_la = 1
    pf_la: int = 1
    tasks: List[Task] = [
        _T(
            "h2d",
            stream="memcpy",
            la=h2d_la,
            reads=("batch_cpu",),
            writes=("batch_gpu", "torchrec_ctx"),
        ),
        _T(
            "start_shuffle",
            stream="memcpy",
            la=h2d_la,
            reads=("batch_gpu",),
            writes=("shuffle_handle",),
            nccl=True,
        ),
        _T(
            "finish_shuffle",
            stream="memcpy",
            la=h2d_la,
            reads=("batch_gpu", "shuffle_handle"),
            writes=("shuffled_batch",),
            nccl=True,
        ),
        _T(
            "start_input_dist",
            stream="data_dist",
            la=id_la,
            reads=("shuffled_batch", "torchrec_ctx"),
            nccl=True,
        ),
        _T(
            "wait_input_dist",
            stream="data_dist",
            la=id_la,
            depends_on=("start_input_dist",),
            nccl=True,
        ),
        _T("zero_grad", stream="default", la=0),
        _T(
            "global_tokens_allreduce",
            stream="default",
            la=0,
            reads=("batch_gpu",),
            writes=("global_tokens",),
            nccl=True,
        ),
        _T(
            "nccl_safety_barrier",
            stream="default",
            la=0,
            depends_on=("finish_shuffle",),
        ),
        _T(
            "forward",
            stream="default",
            la=0,
            reads=("batch_gpu", "torchrec_ctx", "shuffled_batch"),
            writes=("losses", "output"),
            depends_on=(
                ("prefetch_embeddings", "nccl_safety_barrier")
                if prefetch
                else ("wait_input_dist", "nccl_safety_barrier")
            ),
        ),
    ]
    if prefetch:
        tasks.append(
            _T(
                "prefetch_embeddings",
                stream="prefetch",
                la=pf_la,
                depends_on=("wait_input_dist",),
            )
        )
    bw_sps = ("prefetch_embeddings",) if prefetch else ()
    tasks += [
        _T(
            "backward",
            stream="default",
            la=0,
            reads=("losses", "global_tokens"),
            writes=("local_loss_sum",),
            depends_on=("zero_grad",),
            same_progress_sync=bw_sps,
            nccl=True,
        ),
        _T(
            "finalize_model_grads",
            stream="default",
            la=0,
            depends_on=("backward",),
            nccl=True,
        ),
        _T(
            "optimizer_step",
            stream="default",
            la=0,
            depends_on=("finalize_model_grads",),
            writes=("step_result",),
        ),
        _T("watchdog_step", stream="default", la=0, depends_on=("optimizer_step",)),
    ]
    streams = ("default", "memcpy", "data_dist") + (("prefetch",) if prefetch else ())
    return Schedule(stages=(Stage(tasks=tuple(tasks)),), stream_slots=streams)


def _uniform_cost_model(schedule: Schedule, *, ms_per_task: float = 1.0) -> CostModel:
    return CostModel(
        {
            t.name: TaskCost(cpu_us=0.0, gpu_us=ms_per_task * 1000.0)
            for t in schedule.all_tasks()
        }
    )


# --------------------------------------------------------------------
# task_resources
# --------------------------------------------------------------------


def test_task_resources_default_assignment_marks_nccl_comm_for_nccl_tasks() -> None:
    sched = _hstu_like_schedule()
    res = task_resources(list(sched.all_tasks()))
    assert res["start_shuffle"].nccl_comm == "dp"
    assert res["finish_shuffle"].nccl_comm == "dp"
    assert res["start_input_dist"].nccl_comm == "dp"
    assert res["global_tokens_allreduce"].nccl_comm == "dp"
    assert res["finalize_model_grads"].nccl_comm == "dp"
    # Non-NCCL stays None.
    assert res["h2d"].nccl_comm is None
    assert res["forward"].nccl_comm is None
    assert res["optimizer_step"].nccl_comm is None


def test_task_resources_pcie_default_set_covers_h2d_and_prefetch_only() -> None:
    sched = _hstu_like_schedule(prefetch=True)
    res = task_resources(list(sched.all_tasks()))
    assert res["h2d"].pcie is True
    assert res["prefetch_embeddings"].pcie is True
    # Everything else must be False — explicit list to avoid silent drift
    for name in [
        "start_shuffle",
        "finish_shuffle",
        "start_input_dist",
        "wait_input_dist",
        "zero_grad",
        "global_tokens_allreduce",
        "nccl_safety_barrier",
        "forward",
        "backward",
        "finalize_model_grads",
        "optimizer_step",
        "watchdog_step",
    ]:
        assert res[name].pcie is False, name


def test_task_resources_override_nccl_comm_per_task() -> None:
    sched = _hstu_like_schedule()
    custom = {
        "start_input_dist": "input_dist_comm",
        "finalize_model_grads": "grad_comm",
    }
    res = task_resources(list(sched.all_tasks()), nccl_comm_of=custom)
    assert res["start_input_dist"].nccl_comm == "input_dist_comm"
    assert res["finalize_model_grads"].nccl_comm == "grad_comm"
    # An overridden task that is NOT in the override map and NOT
    # NCCL-flagged stays None.
    assert res["forward"].nccl_comm is None


def test_task_resources_override_extending_pcie_set() -> None:
    sched = _hstu_like_schedule()
    res = task_resources(
        list(sched.all_tasks()), pcie_tasks=frozenset({"h2d", "finalize_model_grads"})
    )
    assert res["finalize_model_grads"].pcie is True
    # Default prefetch mark dropped because we replaced the set.
    assert res["prefetch_embeddings"].pcie is False


# --------------------------------------------------------------------
# TaskResource.conflicts_with
# --------------------------------------------------------------------


def test_taskresource_same_stream_returns_stream() -> None:
    a = TaskResource(stream="default", nccl_comm=None, pcie=False)
    b = TaskResource(stream="default", nccl_comm=None, pcie=False)
    assert a.conflicts_with(b) == "stream"


def test_taskresource_same_nccl_comm_different_streams_returns_nccl() -> None:
    a = TaskResource(stream="memcpy", nccl_comm="dp", pcie=False)
    b = TaskResource(stream="data_dist", nccl_comm="dp", pcie=False)
    assert a.conflicts_with(b) == "nccl"


def test_taskresource_pcie_pair_returns_pcie() -> None:
    a = TaskResource(stream="memcpy", nccl_comm=None, pcie=True)
    b = TaskResource(stream="prefetch", nccl_comm=None, pcie=True)
    assert a.conflicts_with(b) == "pcie"


def test_taskresource_independent_returns_none() -> None:
    a = TaskResource(stream="memcpy", nccl_comm=None, pcie=False)
    b = TaskResource(stream="default", nccl_comm=None, pcie=False)
    assert a.conflicts_with(b) is None


def test_taskresource_priority_stream_then_nccl_then_pcie() -> None:
    # Same stream + different NCCL comms still returns "stream".
    a = TaskResource(stream="default", nccl_comm="dp", pcie=False)
    b = TaskResource(stream="default", nccl_comm="ep", pcie=False)
    assert a.conflicts_with(b) == "stream"
    # Different stream + same NCCL + both PCIe → "nccl" wins over pcie.
    c = TaskResource(stream="memcpy", nccl_comm="dp", pcie=True)
    d = TaskResource(stream="data_dist", nccl_comm="dp", pcie=True)
    assert c.conflicts_with(d) == "nccl"


# --------------------------------------------------------------------
# compute_overlap_matrix
# --------------------------------------------------------------------


def test_overlap_matrix_self_diagonal() -> None:
    sched = _hstu_like_schedule()
    matrix = compute_overlap_matrix(list(sched.all_tasks()))
    for t in sched.all_tasks():
        assert matrix[(t.name, t.name)] == "self"


def test_overlap_matrix_symmetric() -> None:
    sched = _hstu_like_schedule()
    tasks = list(sched.all_tasks())
    matrix = compute_overlap_matrix(tasks)
    for a in tasks:
        for b in tasks:
            assert matrix[(a.name, b.name)] == matrix[(b.name, a.name)], (a, b)


def test_overlap_matrix_default_stream_pairs_serialize() -> None:
    sched = _hstu_like_schedule()
    matrix = compute_overlap_matrix(list(sched.all_tasks()))
    default_pairs = [
        ("forward", "backward"),
        ("backward", "optimizer_step"),
        ("zero_grad", "optimizer_step"),
        ("global_tokens_allreduce", "forward"),
        ("nccl_safety_barrier", "watchdog_step"),
    ]
    for a, b in default_pairs:
        assert matrix[(a, b)] == "stream", (a, b)


def test_overlap_matrix_dp_nccl_cross_stream_pairs_serialize() -> None:
    """Same DP comm tasks on different streams still cannot overlap."""
    sched = _hstu_like_schedule()
    matrix = compute_overlap_matrix(list(sched.all_tasks()))
    nccl_cross_stream_pairs = [
        ("start_shuffle", "start_input_dist"),
        ("finish_shuffle", "global_tokens_allreduce"),
        ("start_input_dist", "finalize_model_grads"),
        ("backward", "start_input_dist"),
    ]
    for a, b in nccl_cross_stream_pairs:
        assert matrix[(a, b)] == "nccl", (a, b)


def test_overlap_matrix_independent_pair_marked_ok() -> None:
    """forward (default, compute) and prefetch_embeddings (prefetch,
    no NCCL) should be marked overlap-feasible."""
    sched = _hstu_like_schedule(prefetch=True)
    matrix = compute_overlap_matrix(list(sched.all_tasks()))
    assert matrix[("forward", "prefetch_embeddings")] == "ok"
    assert matrix[("optimizer_step", "h2d")] == "ok"
    assert matrix[("forward", "h2d")] == "ok"


def test_overlap_matrix_pcie_pair_marked_pcie() -> None:
    sched = _hstu_like_schedule(prefetch=True)
    matrix = compute_overlap_matrix(list(sched.all_tasks()))
    assert matrix[("h2d", "prefetch_embeddings")] == "pcie"


def test_overlap_matrix_describe_renders_legend() -> None:
    sched = _hstu_like_schedule()
    rendered = describe_overlap_matrix(list(sched.all_tasks()))
    assert "legend" in rendered
    assert "stream" in rendered
    # First column has task indices and names; sanity-check a known row.
    assert "forward" in rendered
    assert "optimizer_step" in rendered


# --------------------------------------------------------------------
# default_stream_critical_path_us
# --------------------------------------------------------------------


def test_critical_path_sums_only_lookahead_zero_default_stream_tasks() -> None:
    sched = _hstu_like_schedule(prefetch=True)
    cm = CostModel(
        {
            "h2d": TaskCost(0, 300),
            "start_shuffle": TaskCost(0, 4600),
            "finish_shuffle": TaskCost(0, 2300),
            "start_input_dist": TaskCost(0, 8600),
            "wait_input_dist": TaskCost(0, 400),
            "prefetch_embeddings": TaskCost(0, 1100),
            "zero_grad": TaskCost(0, 90),
            "global_tokens_allreduce": TaskCost(0, 120),
            "nccl_safety_barrier": TaskCost(0, 20),
            "forward": TaskCost(0, 28850),
            "backward": TaskCost(0, 12320),
            "finalize_model_grads": TaskCost(0, 290),
            "optimizer_step": TaskCost(0, 66610),
            "watchdog_step": TaskCost(0, 80),
        }
    )
    cp = default_stream_critical_path_us(list(sched.all_tasks()), cm)
    expected = 90 + 120 + 20 + 28850 + 12320 + 290 + 66610 + 80
    assert cp == pytest.approx(expected, rel=1e-9)


def test_critical_path_excludes_higher_lookahead_default_tasks() -> None:
    """If somehow a default-stream task is at lookahead>0, the CP only
    counts the chain at the smallest la (the active iter)."""
    tasks: List[Task] = [
        _T("zero_grad", stream="default", la=0),
        _T("forward", stream="default", la=0, depends_on=("zero_grad",)),
        _T("aux_la1", stream="default", la=1),  # not part of the la=0 chain
    ]
    sched = Schedule(stages=(Stage(tasks=tuple(tasks)),), stream_slots=("default",))
    cm = CostModel(
        {
            "zero_grad": TaskCost(0, 100),
            "forward": TaskCost(0, 200),
            "aux_la1": TaskCost(0, 9999),
        }
    )
    assert default_stream_critical_path_us(list(sched.all_tasks()), cm) == 300


def test_critical_path_zero_when_no_default_stream_tasks() -> None:
    tasks = [
        _T(
            "only_memcpy",
            stream="memcpy",
            la=2,
            reads=("batch_cpu",),
            writes=("batch_gpu",),
        )
    ]
    sched = Schedule(stages=(Stage(tasks=tuple(tasks)),), stream_slots=("memcpy",))
    cm = CostModel({"only_memcpy": TaskCost(0, 1000)})
    assert default_stream_critical_path_us(list(sched.all_tasks()), cm) == 0.0


# --------------------------------------------------------------------
# auto_assign_lookaheads
# --------------------------------------------------------------------


def test_auto_la_default_stream_tasks_keep_existing_la() -> None:
    sched = _hstu_like_schedule(prefetch=True)
    cm = _uniform_cost_model(sched, ms_per_task=1.0)
    out = auto_assign_lookaheads(sched, cm, max_in_flight=5)
    for t in sched.all_tasks():
        if (t.stream or "default") == "default":
            assert out[t.name] == t.batch_offset, t.name


def test_auto_la_offdefault_at_least_one_when_compute_dominates() -> None:
    """When the compute chain dwarfs every off-default task, every
    off-default task should get la >= 1 (so its data is ready an iter
    early) but never above its existing la when the existing la is
    already correct."""
    sched = _hstu_like_schedule(prefetch=True)
    # Compute on default stream is huge (108ms), every other task tiny.
    cm = CostModel(
        {
            "h2d": TaskCost(0, 300),
            "start_shuffle": TaskCost(0, 200),
            "finish_shuffle": TaskCost(0, 200),
            "start_input_dist": TaskCost(0, 200),
            "wait_input_dist": TaskCost(0, 100),
            "prefetch_embeddings": TaskCost(0, 200),
            "zero_grad": TaskCost(0, 100),
            "global_tokens_allreduce": TaskCost(0, 100),
            "nccl_safety_barrier": TaskCost(0, 100),
            "forward": TaskCost(0, 30000),
            "backward": TaskCost(0, 12000),
            "finalize_model_grads": TaskCost(0, 200),
            "optimizer_step": TaskCost(0, 60000),
            "watchdog_step": TaskCost(0, 50),
        }
    )
    out = auto_assign_lookaheads(sched, cm, max_in_flight=5)
    # h2d existing la=2 — recommended is at least 2 (we do not shrink).
    assert out["h2d"] >= 2
    assert out["start_shuffle"] >= 2
    # off-default with existing la=1 stays >= 1.
    assert out["start_input_dist"] >= 1
    assert out["prefetch_embeddings"] >= 1


def test_auto_la_pushes_la_when_nccl_chain_exceeds_critical_path() -> None:
    """When DP-comm NCCL chain exceeds the default critical path, NCCL
    tasks must be pushed deeper to hide the chain."""
    sched = _hstu_like_schedule(prefetch=True)
    cm = CostModel(
        {
            "h2d": TaskCost(0, 300),
            # NCCL chain dominates: shuffle 30ms + input_dist 30ms + bwd 30ms NCCL
            # + finalize 30ms + gtok 1ms = ~121ms; default chain is 50ms.
            "start_shuffle": TaskCost(0, 30000),
            "finish_shuffle": TaskCost(0, 30000),
            "start_input_dist": TaskCost(0, 30000),
            "wait_input_dist": TaskCost(0, 100),
            "prefetch_embeddings": TaskCost(0, 1000),
            "zero_grad": TaskCost(0, 100),
            "global_tokens_allreduce": TaskCost(0, 1000),
            "nccl_safety_barrier": TaskCost(0, 100),
            "forward": TaskCost(0, 20000),  # 20ms
            "backward": TaskCost(
                0, 10000
            ),  # 10ms (incl. DDP NCCL at high la → contributes to chain)
            "finalize_model_grads": TaskCost(0, 30000),
            "optimizer_step": TaskCost(0, 19000),
            "watchdog_step": TaskCost(0, 50),
        }
    )
    out = auto_assign_lookaheads(sched, cm, max_in_flight=5)
    assert out["start_input_dist"] >= 2, (
        f"NCCL chain dominates default chain — start_input_dist must "
        f"go beyond la=1. Got {out['start_input_dist']}."
    )


def test_auto_la_respects_max_in_flight_cap() -> None:
    sched = _hstu_like_schedule(prefetch=True)
    # Force every off-default task to want a HUGE la — capped to budget.
    cm = CostModel({t.name: TaskCost(0, 999_999) for t in sched.all_tasks()})
    out = auto_assign_lookaheads(sched, cm, max_in_flight=3)
    # Cap = max_in_flight - 1 = 2.
    for name, la in out.items():
        assert la <= 2, (name, la)


def test_auto_la_never_shrinks_below_existing_la() -> None:
    """Caller authored la must be honored as a floor. We're allowed to
    bump it up but not below."""
    tasks: List[Task] = [
        _T("h2d", stream="memcpy", la=4, reads=("batch_cpu",), writes=("batch_gpu",)),
        _T(
            "start_input_dist",
            stream="data_dist",
            la=3,
            reads=("batch_gpu",),
            nccl=True,
        ),
        _T("forward", stream="default", la=0, reads=("batch_gpu",)),
    ]
    sched = Schedule(
        stages=(Stage(tasks=tuple(tasks)),),
        stream_slots=("default", "memcpy", "data_dist"),
    )
    cm = CostModel(
        {
            "h2d": TaskCost(0, 100),
            "start_input_dist": TaskCost(0, 100),
            "forward": TaskCost(0, 1000),
        }
    )
    out = auto_assign_lookaheads(sched, cm, max_in_flight=10)
    assert out["h2d"] >= 4
    assert out["start_input_dist"] >= 3


def test_auto_la_propagates_producer_la_for_slot_reads() -> None:
    """If consumer's la is bumped, producer must also be bumped to a
    la at least as high so the ring slot can hold the producer's data
    when the consumer reads."""
    tasks: List[Task] = [
        _T(
            "producer",
            stream="memcpy",
            la=1,
            reads=("batch_cpu",),
            writes=("batch_gpu",),
        ),
        # Consumer is on a NCCL-bound stream with huge cost so the
        # auto-scheduler will push it higher, which must drag producer up.
        _T("consumer", stream="data_dist", la=2, reads=("batch_gpu",), nccl=True),
        _T("forward", stream="default", la=0, reads=("batch_gpu",)),
    ]
    sched = Schedule(
        stages=(Stage(tasks=tuple(tasks)),),
        stream_slots=("default", "memcpy", "data_dist"),
    )
    cm = CostModel(
        {
            # NCCL queue (just consumer here) >> default chain → push consumer.
            "producer": TaskCost(0, 100),
            "consumer": TaskCost(0, 50_000),
            "forward": TaskCost(0, 1_000),
        }
    )
    out = auto_assign_lookaheads(sched, cm, max_in_flight=10)
    # consumer pushed to >= 2.
    assert out["consumer"] >= 2
    # producer must be at least consumer's la (slot read invariant).
    assert out["producer"] >= out["consumer"], (out["producer"], out["consumer"])


def test_auto_la_max_in_flight_below_one_rejected() -> None:
    sched = _hstu_like_schedule()
    cm = _uniform_cost_model(sched)
    with pytest.raises(ValueError, match="max_in_flight"):
        auto_assign_lookaheads(sched, cm, max_in_flight=0)


def test_auto_la_existing_la_zero_off_default_gets_at_least_one() -> None:
    """An off-default-stream task currently at la=0 (unusual but legal)
    must be pushed to la>=1 by the auto-scheduler — la=0 means
    same-progress execution which the engine forbids for ring-stored
    consumers."""
    tasks: List[Task] = [
        _T(
            "zero_la_offdefault",
            stream="memcpy",
            la=0,
            reads=("batch_cpu",),
            writes=("batch_gpu",),
        ),
        _T("forward", stream="default", la=0, reads=("batch_gpu",)),
    ]
    sched = Schedule(
        stages=(Stage(tasks=tuple(tasks)),), stream_slots=("default", "memcpy")
    )
    cm = CostModel(
        {
            "zero_la_offdefault": TaskCost(0, 50),
            "forward": TaskCost(0, 1000),
        }
    )
    out = auto_assign_lookaheads(sched, cm, max_in_flight=5)
    assert out["zero_la_offdefault"] >= 1
    # forward stays at 0.
    assert out["forward"] == 0


# --------------------------------------------------------------------
# Sanity: the auto-scheduler doesn't reorder forward / backward / opt
# (bit-exact contract — see feedback_no_async_optimizer.md)
# --------------------------------------------------------------------


def test_auto_la_never_moves_optimizer_off_default_or_to_nonzero_la() -> None:
    """Hard guard against accidentally proposing async/stale-grad
    optimizer setup. optimizer_step must keep stream='default' la=0."""
    sched = _hstu_like_schedule(prefetch=True)
    cm = _uniform_cost_model(sched)
    out = auto_assign_lookaheads(sched, cm, max_in_flight=10)
    assert out["optimizer_step"] == 0
    # Sanity check forward/backward also stay at la=0.
    assert out["forward"] == 0
    assert out["backward"] == 0
    assert out["finalize_model_grads"] == 0


# --------------------------------------------------------------------
# Hard contract: bit-exact guard rails (added after codex CRITICAL #1)
# --------------------------------------------------------------------


def test_auto_la_rejects_optimizer_step_nonzero_lookahead_input() -> None:
    """A user-authored optimizer_step at la>0 is async-style and must
    be rejected before any la assignment runs."""
    tasks: List[Task] = [
        _T("forward", stream="default", la=0, reads=("batch_gpu",), writes=("losses",)),
        _T(
            "backward",
            stream="default",
            la=0,
            reads=("losses",),
            writes=("local_loss_sum",),
            depends_on=("forward",),
        ),
        _T("finalize_model_grads", stream="default", la=0, depends_on=("backward",)),
        _T(
            "optimizer_step",
            stream="default",
            la=1,  # ← BAD
            depends_on=("finalize_model_grads",),
            writes=("step_result",),
        ),
        _T("h2d", stream="memcpy", la=2, reads=("batch_cpu",), writes=("batch_gpu",)),
    ]
    sched = Schedule(
        stages=(Stage(tasks=tuple(tasks)),), stream_slots=("default", "memcpy")
    )
    cm = CostModel({t.name: TaskCost(0, 1000) for t in tasks})
    with pytest.raises(ValueError, match="optimizer_step.*lookahead=0"):
        auto_assign_lookaheads(sched, cm, max_in_flight=5)


def test_auto_la_rejects_optimizer_step_nondefault_stream_input() -> None:
    tasks: List[Task] = [
        _T("forward", stream="default", la=0, reads=("batch_gpu",), writes=("losses",)),
        _T(
            "backward",
            stream="default",
            la=0,
            reads=("losses",),
            writes=("local_loss_sum",),
            depends_on=("forward",),
        ),
        _T("finalize_model_grads", stream="default", la=0, depends_on=("backward",)),
        _T(
            "optimizer_step",
            stream="opt_side",
            la=0,  # ← BAD
            depends_on=("finalize_model_grads",),
            writes=("step_result",),
        ),
        _T("h2d", stream="memcpy", la=2, reads=("batch_cpu",), writes=("batch_gpu",)),
    ]
    sched = Schedule(
        stages=(Stage(tasks=tuple(tasks)),),
        stream_slots=("default", "opt_side", "memcpy"),
    )
    cm = CostModel({t.name: TaskCost(0, 1000) for t in tasks})
    with pytest.raises(ValueError, match="optimizer_step.*default"):
        auto_assign_lookaheads(sched, cm, max_in_flight=5)


def test_auto_la_propagation_cannot_bump_default_chain_through_consumer() -> None:
    """If an off-default consumer's la is bumped past a default-stream
    producer (via reads or depends_on), propagation must NOT silently
    bump the default-stream producer — it should raise."""
    tasks: List[Task] = [
        # forward writes a slot that an off-default task reads and
        # demands a high la — the engine cannot bump forward's la, so
        # this must raise rather than silently break forward.
        _T("h2d", stream="memcpy", la=2, reads=("batch_cpu",), writes=("batch_gpu",)),
        _T(
            "forward",
            stream="default",
            la=0,
            reads=("batch_gpu",),
            writes=("model_output",),
        ),
        _T(
            "backward",
            stream="default",
            la=0,
            reads=("model_output",),
            writes=("local_loss_sum",),
            depends_on=("forward",),
        ),
        _T("finalize_model_grads", stream="default", la=0, depends_on=("backward",)),
        _T(
            "optimizer_step",
            stream="default",
            la=0,
            depends_on=("finalize_model_grads",),
            writes=("step_result",),
        ),
        # Off-default reader of model_output that wants high la.
        _T(
            "post_process", stream="data_dist", la=2, reads=("model_output",), nccl=True
        ),
    ]
    sched = Schedule(
        stages=(Stage(tasks=tuple(tasks)),),
        stream_slots=("default", "memcpy", "data_dist"),
    )
    cm = CostModel(
        {
            "h2d": TaskCost(0, 100),
            "forward": TaskCost(0, 1000),
            "backward": TaskCost(0, 500),
            "finalize_model_grads": TaskCost(0, 100),
            "optimizer_step": TaskCost(0, 1000),
            "post_process": TaskCost(0, 50_000),  # forces high la
        }
    )
    with pytest.raises(ValueError, match="frozen producer"):
        auto_assign_lookaheads(sched, cm, max_in_flight=5)


def test_auto_la_floor_above_cap_rejected() -> None:
    """If a user-authored la exceeds (max_in_flight - 1), the
    scheduler must raise rather than silently shrink it (codex MAJOR #4)."""
    tasks: List[Task] = [
        _T(
            "h2d",
            stream="memcpy",
            la=4,  # floor = 4
            reads=("batch_cpu",),
            writes=("batch_gpu",),
        ),
        _T("forward", stream="default", la=0, reads=("batch_gpu",)),
    ]
    sched = Schedule(
        stages=(Stage(tasks=tuple(tasks)),), stream_slots=("default", "memcpy")
    )
    cm = CostModel({"h2d": TaskCost(0, 100), "forward": TaskCost(0, 100)})
    # max_in_flight=3 → cap=2 < floor=4 — must raise.
    with pytest.raises(ValueError, match="exceeds max_in_flight"):
        auto_assign_lookaheads(sched, cm, max_in_flight=3)


def test_auto_la_max_in_flight_one_offdefault_conflict_rejected() -> None:
    """max_in_flight=1 is allowed only when no off-default task needs
    la>=1. The scheduler must reject it when an off-default task's
    floor was 1."""
    tasks: List[Task] = [
        _T(
            "h2d",
            stream="memcpy",
            la=1,  # ← needs la>=1, but cap=0
            reads=("batch_cpu",),
            writes=("batch_gpu",),
        ),
        _T("forward", stream="default", la=0, reads=("batch_gpu",)),
    ]
    sched = Schedule(
        stages=(Stage(tasks=tuple(tasks)),), stream_slots=("default", "memcpy")
    )
    cm = CostModel({"h2d": TaskCost(0, 100), "forward": TaskCost(0, 100)})
    with pytest.raises(ValueError, match="exceeds max_in_flight"):
        auto_assign_lookaheads(sched, cm, max_in_flight=1)


def test_auto_la_cross_iter_depends_on_caps_consumer_la() -> None:
    """If a consumer authored ``cross_iter_depends_on=((P, -N),)``,
    the scheduler must NOT bump consumer.la — doing so would shift
    ``slot_offset = consumer.la + neg_offset`` out of range."""
    tasks: List[Task] = [
        _T("p", stream="memcpy", la=2, reads=("batch_cpu",), writes=("p_out",)),
        # cross_iter consumer at la=1 with neg_offset=-1 → slot_offset=0.
        # If scheduler bumps c.la to 2, slot_offset=1 — wrong slot,
        # consumer reads stale or wrong-iter data.
        _T(
            "c", stream="data_dist", la=1, nccl=True, cross_iter_depends_on=(("p", -1),)
        ),
        _T("forward", stream="default", la=0, reads=("p_out",)),
    ]
    sched = Schedule(
        stages=(Stage(tasks=tuple(tasks)),),
        stream_slots=("default", "memcpy", "data_dist"),
    )
    cm = CostModel(
        {
            "p": TaskCost(0, 100),
            "c": TaskCost(0, 50_000),  # huge — would otherwise bump
            "forward": TaskCost(0, 100),
        }
    )
    out = auto_assign_lookaheads(sched, cm, max_in_flight=5)
    # c stays at its authored la=1 even though cost suggests higher.
    assert out["c"] == 1


def test_auto_la_multiple_writers_same_slot_name_different_offsets() -> None:
    """When two different tasks both write the same slot name (at
    different offsets), the scheduler must consider all of them when
    propagating la for a reader."""
    tasks: List[Task] = [
        _T(
            "writer_high",
            stream="memcpy",
            la=3,
            reads=("batch_cpu",),
            writes=("batch_gpu",),
        ),
        # Second writer of "batch_gpu" at a different offset (legal as
        # long as the (slot.name, batch_offset) pair is unique). For
        # this test we use a different stream so deps inference does
        # not reject for name-level multi-stream.
        _T(
            "writer_low",
            stream="memcpy",
            la=2,
            reads=("batch_cpu",),
            writes=("batch_gpu",),
        ),
        _T("reader", stream="data_dist", la=2, nccl=True, reads=("batch_gpu",)),
        _T("forward", stream="default", la=0, reads=("batch_gpu",)),
    ]
    sched = Schedule(
        stages=(Stage(tasks=tuple(tasks)),),
        stream_slots=("default", "memcpy", "data_dist"),
    )
    cm = CostModel(
        {
            "writer_high": TaskCost(0, 100),
            "writer_low": TaskCost(0, 100),
            "reader": TaskCost(0, 50_000),  # forces high la
            "forward": TaskCost(0, 100),
        }
    )
    out = auto_assign_lookaheads(sched, cm, max_in_flight=10)
    # Both writers must rise to (or stay above) the reader's la.
    assert out["writer_high"] >= out["reader"]
    assert out["writer_low"] >= out["reader"]


def test_auto_la_no_default_stream_tasks_policy() -> None:
    """When there are no default-stream tasks, critical path = 0 and
    every off-default task simply gets la=floor (no bumping needed)."""
    tasks: List[Task] = [
        _T("h2d", stream="memcpy", la=2, reads=("batch_cpu",), writes=("batch_gpu",)),
        _T("post", stream="data_dist", la=1, reads=("batch_gpu",), nccl=True),
    ]
    sched = Schedule(
        stages=(Stage(tasks=tuple(tasks)),), stream_slots=("memcpy", "data_dist")
    )
    cm = CostModel({"h2d": TaskCost(0, 100), "post": TaskCost(0, 999)})
    out = auto_assign_lookaheads(sched, cm, max_in_flight=5)
    assert out["h2d"] == 2
    # post stays at its authored la — when cp_us=0, the algorithm
    # falls back to need=1 and floor=1, both clamped by cap.
    assert out["post"] >= 1


def test_auto_la_empty_string_stream_normalized_to_default() -> None:
    """``stream=""`` is treated identically to ``stream=None`` and to
    ``stream="default"``. A bit-exact task authored with ``stream=""``
    must NOT bypass the default-stream guard (codex MAJOR from re-review)."""
    tasks: List[Task] = [
        # optimizer_step authored with empty-string stream — must be
        # accepted (treated as default) since the engine maps ""→default.
        _T("forward", stream="default", la=0, reads=("batch_gpu",)),
        _T(
            "backward",
            stream="default",
            la=0,
            reads=("losses",),
            depends_on=("forward",),
        ),
        _T("finalize_model_grads", stream="default", la=0, depends_on=("backward",)),
        _T(
            "optimizer_step",
            stream="",
            la=0,  # ← empty string
            depends_on=("finalize_model_grads",),
            writes=("step_result",),
        ),
        _T("h2d", stream="memcpy", la=2, reads=("batch_cpu",), writes=("batch_gpu",)),
    ]
    sched = Schedule(
        stages=(Stage(tasks=tuple(tasks)),), stream_slots=("default", "memcpy")
    )
    cm = CostModel({t.name: TaskCost(0, 1000) for t in tasks})
    # Should NOT raise — "" is normalized to default.
    out = auto_assign_lookaheads(sched, cm, max_in_flight=5)
    assert out["optimizer_step"] == 0


def test_auto_la_empty_string_stream_for_offdefault_task_still_treated_default() -> (
    None
):
    """A non-bit-exact task with ``stream=""`` is treated as on
    default — its la is NOT bumped (default-stream tasks are frozen)."""
    tasks: List[Task] = [
        _T("zero_grad_empty", stream="", la=0),
        _T("forward", stream="default", la=0, reads=("batch_gpu",)),
        _T("h2d", stream="memcpy", la=2, reads=("batch_cpu",), writes=("batch_gpu",)),
    ]
    sched = Schedule(
        stages=(Stage(tasks=tuple(tasks)),), stream_slots=("default", "memcpy")
    )
    cm = CostModel({t.name: TaskCost(0, 1000) for t in tasks})
    out = auto_assign_lookaheads(sched, cm, max_in_flight=5)
    # Empty-stream task is treated as default-stream, frozen at floor.
    assert out["zero_grad_empty"] == 0


def test_auto_la_cross_iter_cap_binding_below_consumer_raises() -> None:
    """Codex MAJOR (re-review #2): when a producer's cross_iter cap
    forces it BELOW the consumer's required la, propagation must
    raise rather than silently leave the slot read invariant broken."""
    tasks: List[Task] = [
        # producer authored cross_iter at la=1 — its cap is 1.
        _T(
            "p",
            stream="memcpy",
            la=1,
            reads=("batch_cpu",),
            writes=("p_out",),
            cross_iter_depends_on=(("seed", -1),),
        ),
        # We need a "seed" task referenced by cross_iter — it's just a
        # dummy that exists in the schedule.
        _T("seed", stream="memcpy", la=2, reads=("batch_cpu",), writes=("seed_out",)),
        # consumer reads p_out — its la would be bumped high.
        _T("c", stream="data_dist", la=2, reads=("p_out",), nccl=True),
        _T("forward", stream="default", la=0, reads=("p_out",)),
    ]
    sched = Schedule(
        stages=(Stage(tasks=tuple(tasks)),),
        stream_slots=("default", "memcpy", "data_dist"),
    )
    cm = CostModel(
        {
            "p": TaskCost(0, 100),
            "seed": TaskCost(0, 100),
            "c": TaskCost(0, 50_000),  # forces high la
            "forward": TaskCost(0, 100),
        }
    )
    with pytest.raises(ValueError, match="cross-iter cap"):
        auto_assign_lookaheads(sched, cm, max_in_flight=5)


def test_auto_la_default_bit_exact_tasks_exported_from_package() -> None:
    """``DEFAULT_BIT_EXACT_TASKS`` should be importable from the
    autosched package (codex NIT from re-review)."""
    from commons.pipeline.engine.autosched import DEFAULT_BIT_EXACT_TASKS

    assert "optimizer_step" in DEFAULT_BIT_EXACT_TASKS
    assert "forward" in DEFAULT_BIT_EXACT_TASKS
    assert "backward" in DEFAULT_BIT_EXACT_TASKS
    assert "finalize_model_grads" in DEFAULT_BIT_EXACT_TASKS


def test_auto_la_all_zero_costs_policy() -> None:
    """All zero costs → every off-default task is recommended at
    la=max(floor, 1) (no NCCL/CP-driven bumping). Default-stream
    tasks stay at floor."""
    sched = _hstu_like_schedule(prefetch=True)
    cm = CostModel({t.name: TaskCost(0, 0) for t in sched.all_tasks()})
    out = auto_assign_lookaheads(sched, cm, max_in_flight=5)
    for t in sched.all_tasks():
        if (t.stream or "default") == "default":
            assert out[t.name] == t.batch_offset, t.name
        else:
            assert out[t.name] >= t.batch_offset, t.name
            assert out[t.name] <= 4  # cap = 5-1 = 4
