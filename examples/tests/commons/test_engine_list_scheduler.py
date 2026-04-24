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

"""V9 — `schedule_tasks()` critical-path list scheduler."""

import pytest
from commons.pipeline.engine import DataSlot, Task
from commons.pipeline.engine.autosched import CostModel, schedule_tasks


def _noop(ctx):
    return None


def _costs(**kwargs) -> CostModel:
    """Shorthand: {name: cpu_us_and_gpu_us} → CostModel."""
    return CostModel.from_dict(
        {name: {"cpu_us": 0.0, "gpu_us": float(v)} for name, v in kwargs.items()}
    )


def test_no_tasks_rejected() -> None:
    with pytest.raises(ValueError, match="no tasks"):
        schedule_tasks([], CostModel({}), stream_slots=("default",))


def test_duplicate_task_name_rejected() -> None:
    a = Task.from_fn(name="dup", fn=_noop, stream="default")
    b = Task.from_fn(name="dup", fn=_noop, stream="default")
    with pytest.raises(ValueError, match="Duplicate task name"):
        schedule_tasks([a, b], _costs(dup=1.0), stream_slots=("default",))


def test_linear_chain_preserved() -> None:
    """Pure linear dep chain A → B → C must produce [A, B, C] regardless
    of input order, since the slot DAG forces it."""
    a = Task.from_fn(name="a", fn=_noop, writes=(DataSlot("x"),), stream="default")
    b = Task.from_fn(
        name="b",
        fn=_noop,
        reads=(DataSlot("x"),),
        writes=(DataSlot("y"),),
        stream="default",
    )
    c = Task.from_fn(name="c", fn=_noop, reads=(DataSlot("y"),), stream="default")
    schedule = schedule_tasks(
        [c, a, b],  # deliberately out of order
        _costs(a=10, b=10, c=10),
        stream_slots=("default",),
    )
    names = [t.name for t in schedule.stages[0].tasks]
    assert names == ["a", "b", "c"]


def test_critical_path_prioritized() -> None:
    """Two independent chains sharing no slots: longer chain runs
    first. Schedule:
        A → B → C   (costs 100, 100, 100 — total 300)
        D → E       (costs 10, 10 — total 20)
    With two independent roots, scheduler's priority queue should
    pick A first (cp=300), then D (cp=20), etc. We don't enforce a
    specific ordering between the two chains globally, but A must
    come before D in the output because A's CP is strictly larger
    and ties break by name (A < D)."""
    a = Task.from_fn(name="a", fn=_noop, writes=(DataSlot("x_a"),), stream="default")
    b = Task.from_fn(
        name="b",
        fn=_noop,
        reads=(DataSlot("x_a"),),
        writes=(DataSlot("y_a"),),
        stream="default",
    )
    c = Task.from_fn(name="c", fn=_noop, reads=(DataSlot("y_a"),), stream="default")
    d = Task.from_fn(name="d", fn=_noop, writes=(DataSlot("x_d"),), stream="default")
    e = Task.from_fn(
        name="e",
        fn=_noop,
        reads=(DataSlot("x_d"),),
        stream="default",
    )
    cm = _costs(a=100, b=100, c=100, d=10, e=10)
    schedule = schedule_tasks([e, d, c, b, a], cm, stream_slots=("default",))
    names = [t.name for t in schedule.stages[0].tasks]
    # A must precede D (higher CP length).
    assert names.index("a") < names.index("d")
    # Within each chain, order follows deps.
    assert names.index("a") < names.index("b") < names.index("c")
    assert names.index("d") < names.index("e")


def test_cycle_in_depends_on_rejected() -> None:
    """Two tasks that reference each other via depends_on form a
    cycle — the topo sort prefix can't place them."""
    a = Task.from_fn(name="a", fn=_noop, depends_on=("b",), stream="default")
    b = Task.from_fn(name="b", fn=_noop, depends_on=("a",), stream="default")
    with pytest.raises(ValueError, match="cycle"):
        schedule_tasks([a, b], _costs(a=1, b=1), stream_slots=("default",))


def test_depends_on_edges_honored() -> None:
    """Pure `depends_on` chain with no slots."""
    a = Task.from_fn(name="a", fn=_noop, stream="default")
    b = Task.from_fn(name="b", fn=_noop, depends_on=("a",), stream="default")
    c = Task.from_fn(name="c", fn=_noop, depends_on=("b",), stream="default")
    schedule = schedule_tasks(
        [c, b, a], _costs(a=5, b=5, c=5), stream_slots=("default",)
    )
    names = [t.name for t in schedule.stages[0].tasks]
    assert names == ["a", "b", "c"]


def test_output_passes_validator() -> None:
    """The scheduler should emit schedules that pass the V5 validator
    (it calls `validate()` at the end). Exercise with a realistic
    prefetch-style set."""
    from commons.pipeline.engine.autosched import validate

    h2d = Task.from_fn(
        name="h2d",
        fn=_noop,
        reads=(DataSlot("batch_cpu", batch_offset=1),),
        writes=(DataSlot("batch_gpu", batch_offset=1),),
        stream="memcpy",
        batch_offset=1,
    )
    zero_grad = Task.from_fn(name="zero_grad", fn=_noop, stream="default")
    forward = Task.from_fn(
        name="forward",
        fn=_noop,
        reads=(DataSlot("batch_gpu"),),
        writes=(DataSlot("step_result"),),
        depends_on=("zero_grad",),
        stream="default",
    )
    backward = Task.from_fn(
        name="backward",
        fn=_noop,
        reads=(DataSlot("step_result"),),
        depends_on=("forward",),
        stream="default",
    )
    optimizer = Task.from_fn(
        name="optimizer",
        fn=_noop,
        depends_on=("backward",),
        stream="default",
    )
    cm = _costs(h2d=10, zero_grad=1, forward=100, backward=100, optimizer=5)
    schedule = schedule_tasks(
        [optimizer, backward, forward, zero_grad, h2d],  # reversed
        cm,
        stream_slots=("default", "memcpy"),
    )
    # Validator sign-off already happens inside schedule_tasks;
    # call once more here for explicit assertion.
    validate(schedule)
    names = [t.name for t in schedule.stages[0].tasks]
    # zero_grad → forward → backward → optimizer chain
    assert names.index("zero_grad") < names.index("forward")
    assert names.index("forward") < names.index("backward")
    assert names.index("backward") < names.index("optimizer")


def test_deterministic_tiebreak_on_ties() -> None:
    """Two independent tasks with identical CP must come out in a
    deterministic order (sorted by name) every time."""
    a = Task.from_fn(name="a", fn=_noop, stream="default")
    b = Task.from_fn(name="b", fn=_noop, stream="default")
    cm = _costs(a=1, b=1)
    s1 = schedule_tasks([a, b], cm, stream_slots=("default",))
    s2 = schedule_tasks([b, a], cm, stream_slots=("default",))
    n1 = [t.name for t in s1.stages[0].tasks]
    n2 = [t.name for t in s2.stages[0].tasks]
    assert n1 == n2 == ["a", "b"]


def test_scheduler_rejects_unknown_task_in_depends_on() -> None:
    a = Task.from_fn(
        name="a",
        fn=_noop,
        depends_on=("nonexistent",),
        stream="default",
    )
    with pytest.raises(ValueError, match="not in the input task list"):
        schedule_tasks([a], _costs(a=1), stream_slots=("default",))


def test_diamond_dag_schedule_preserves_join_order() -> None:
    """Diamond / fan-in shape: root writes x; two middle tasks
    (b, c) both read x. A join task d reads outputs from both b
    and c. Canonical shape that surfaces indegree-tracking bugs in
    list schedulers.

        root (writes x)
         /        \\
       b(reads x, writes y_b)   c(reads x, writes y_c)
         \\        /
         d (reads y_b and y_c)
    """
    root = Task.from_fn(
        name="root",
        fn=_noop,
        writes=(DataSlot("x"),),
        stream="default",
    )
    b = Task.from_fn(
        name="b",
        fn=_noop,
        reads=(DataSlot("x"),),
        writes=(DataSlot("y_b"),),
        stream="default",
    )
    c = Task.from_fn(
        name="c",
        fn=_noop,
        reads=(DataSlot("x"),),
        writes=(DataSlot("y_c"),),
        stream="default",
    )
    d = Task.from_fn(
        name="d",
        fn=_noop,
        reads=(DataSlot("y_b"), DataSlot("y_c")),
        stream="default",
    )
    schedule = schedule_tasks(
        [d, c, b, root],  # reversed from logical order
        _costs(root=10, b=10, c=10, d=10),
        stream_slots=("default",),
    )
    names = [t.name for t in schedule.stages[0].tasks]
    # root must be first; d must be last; {b, c} somewhere in between.
    assert names[0] == "root"
    assert names[-1] == "d"
    assert set(names[1:3]) == {"b", "c"}
    # b must come before d; c must come before d.
    assert names.index("b") < names.index("d")
    assert names.index("c") < names.index("d")


def test_scheduled_schedule_executes_end_to_end() -> None:
    """End-to-end integration: construct a `SchedulablePipeline` from
    a scheduler-produced `Schedule`, drive `progress(...)` to
    exhaustion, confirm the engine consumes the scheduled schedule
    without crashing and produces the declared step_result. Covers
    the gap between static validation (V5) and actual runtime
    execution (V2-V4)."""
    import torch
    from commons.pipeline.engine import SchedulablePipeline, StreamPool

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _forward(ctx):
        x = ctx.slots["batch_cpu"]
        # Simple scalar: sum the batch elements.
        ctx.slots.set("step_result", x.sum())

    def _extra(ctx):
        # Runs after forward via depends_on, just to exercise a
        # non-trivial DAG shape.
        pass

    forward = Task.from_fn(
        name="forward",
        fn=_forward,
        reads=(DataSlot("batch_cpu"),),
        writes=(DataSlot("step_result"),),
        stream="default",
    )
    extra = Task.from_fn(
        name="extra",
        fn=_extra,
        depends_on=("forward",),
        stream="default",
    )

    # Deliberately feed in reverse — scheduler must reorder.
    schedule = schedule_tasks(
        [extra, forward],
        _costs(forward=50, extra=5),
        stream_slots=("default",),
    )
    names = [t.name for t in schedule.stages[0].tasks]
    assert names == ["forward", "extra"]

    # Now actually run the pipeline from the scheduled schedule.
    pool = StreamPool(
        {
            "default": torch.cuda.default_stream(device)
            if device.type == "cuda"
            else None
        }
    )
    pipe = SchedulablePipeline(schedule, pool)

    batches = [torch.tensor([1.0, 2.0, 3.0], device=device) for _ in range(3)]
    results = []
    it = iter(batches)
    while True:
        try:
            r = pipe.progress(it)
        except StopIteration:
            break
        results.append(r)

    assert len(results) == 3
    # Each result is the batch.sum() = 6.0
    for i, r in enumerate(results):
        assert r is not None
        assert r.item() == 6.0, f"iter {i}: got {r.item()}, expected 6.0"
