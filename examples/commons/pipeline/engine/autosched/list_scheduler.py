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

"""Critical-path list scheduler for Problem #1 (SPEC §4.3).

Inputs:
  - A bag of `Task`s (user-authored or preset-generated).
  - A `CostModel` with per-task timing.
  - `stream_slots` — the declared stream inventory.

Output: a `Schedule` (stages + stream bindings) that respects all
slot + `depends_on` edges AND is valid per SPEC §4.2.

v1 algorithm: topological + priority-based list scheduling.

  - Build the intra-iter dependency DAG (slot-writer → slot-reader,
    `depends_on` edges).
  - Compute each task's **critical-path length** (remaining time
    to the deepest leaf, using `CostModel.total_us`).
  - Greedy ready-queue: at each step, pick the highest-priority
    ready task and append it to the schedule.
  - All tasks go in a single `Stage` (V1 semantics — stages are
    organizational, not required for scheduling correctness).

This is a straightforward offline heuristic. It minimizes critical
path length under the given costs; not optimal for all workloads
(e.g., multi-stream concurrency isn't modeled beyond edge ordering).
Ship the heuristic, measure, iterate — don't add ILP / simulated
annealing unless the simple version demonstrably underperforms.
"""

from typing import Dict, List, Sequence, Tuple

from ..schedule import Schedule, Stage
from ..task import DataSlot, Task
from .cost_model import CostModel
from .validator import validate

__all__ = ["schedule_tasks"]


def schedule_tasks(
    tasks: Sequence[Task],
    cost_model: CostModel,
    stream_slots: Tuple[str, ...],
) -> Schedule:
    """Arrange `tasks` into a valid `Schedule` minimizing critical
    path length under `cost_model`.

    The returned Schedule:
      - Contains one `Stage` with all tasks in priority-sorted order
        (higher critical-path length runs earlier).
      - Has `stream_slots` exactly as passed.
      - Preserves each task's declared `stream` — the v1 scheduler
        does NOT rebind streams (that would require modeling
        per-stream resource contention; punt to a later slice).

    Raises `ScheduleValidationError` if the resulting Schedule
    violates SPEC §4.2.
    """
    if not tasks:
        raise ValueError("schedule_tasks() called with no tasks")

    task_by_name: Dict[str, Task] = {}
    for task in tasks:
        if task.name in task_by_name:
            raise ValueError(
                f"Duplicate task name {task.name!r} in input — cannot "
                f"schedule (SPEC §4.2 rule 1)."
            )
        task_by_name[task.name] = task

    # Build the intra-iter dependency DAG: edges predecessor → successor.
    adjacency_out: Dict[str, List[str]] = {t.name: [] for t in tasks}
    indegree: Dict[str, int] = {t.name: 0 for t in tasks}

    # Slot edges: writer → reader, intra-iter (same batch_offset).
    slot_writer: Dict[DataSlot, Task] = {}
    for task in tasks:
        for slot in task.writes:
            slot_writer[slot] = task
    for consumer in tasks:
        for read_slot in consumer.reads:
            producer = slot_writer.get(read_slot)
            if producer is None:
                continue  # cross-iter or engine-populated; no intra-iter edge
            if producer.name == consumer.name:
                continue  # self-loops on a slot are impossible (rule 4)
            adjacency_out[producer.name].append(consumer.name)
            indegree[consumer.name] += 1

    # depends_on edges.
    for consumer in tasks:
        for dep_name in consumer.depends_on:
            if dep_name not in task_by_name:
                raise ValueError(
                    f"Task {consumer.name!r}.depends_on references "
                    f"{dep_name!r} which is not in the input task list."
                )
            adjacency_out[dep_name].append(consumer.name)
            indegree[consumer.name] += 1

    # Critical-path length per task: longest path (weighted by
    # cost_model.total_us) from this task to any leaf.
    # Compute via reverse topological order.
    reverse_topo = _reverse_topo_order(tasks, adjacency_out, indegree)
    cp_length: Dict[str, float] = {}
    for name in reverse_topo:
        successor_cp = [cp_length[s] for s in adjacency_out[name] if s in cp_length]
        own_cost = cost_model.get(name).total_us
        cp_length[name] = own_cost + (max(successor_cp) if successor_cp else 0.0)

    # List scheduling: ready queue ordered by critical-path length
    # (descending). Ties broken by task name for determinism.
    ordered: List[Task] = []
    ready: List[str] = [name for name, deg in indegree.items() if deg == 0]
    remaining_indegree = dict(indegree)
    while ready:
        # Pick highest-CP task; deterministic tiebreaker on name.
        ready.sort(key=lambda n: (-cp_length[n], n))
        picked = ready.pop(0)
        ordered.append(task_by_name[picked])
        for succ in adjacency_out[picked]:
            remaining_indegree[succ] -= 1
            if remaining_indegree[succ] == 0:
                ready.append(succ)

    if len(ordered) != len(tasks):
        raise ValueError(
            "List scheduler produced fewer tasks than input — the "
            "intra-iter DAG has a cycle. Expected SPEC §4.2 rule 7 "
            "to reject this upstream, but the input wasn't validated "
            "first."
        )

    schedule = Schedule(
        stages=(Stage(tasks=tuple(ordered)),),
        stream_slots=stream_slots,
    )
    # Consolidate: run the full V5 validator on the output. Catches
    # any rule violation introduced by scheduler bugs.
    validate(schedule)
    return schedule


def _reverse_topo_order(
    tasks: Sequence[Task],
    adjacency_out: Dict[str, List[str]],
    indegree: Dict[str, int],
) -> List[str]:
    """Kahn's algorithm, then reverse. Returns task names in reverse
    topological order (leaves first). Used for critical-path
    calculation which propagates from leaves to roots."""
    working_indegree = dict(indegree)
    queue: List[str] = [n for n, d in working_indegree.items() if d == 0]
    topo: List[str] = []
    while queue:
        n = queue.pop(0)
        topo.append(n)
        for succ in adjacency_out[n]:
            working_indegree[succ] -= 1
            if working_indegree[succ] == 0:
                queue.append(succ)
    if len(topo) != len(tasks):
        raise ValueError("Cannot topologically sort: input task DAG has a cycle.")
    return list(reversed(topo))
