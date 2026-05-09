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

"""Fire-order auto-scheduler — derive per-task ``lookahead`` and within-
thread fire order from a resource-conflict overlap matrix and a cost
model, with the explicit goal of hiding NCCL/IO behind the longest
default-stream compute chain.

The contract is bit-exact: the scheduler may only reassign
``lookahead`` (and equivalently the topo-sort tie-break) within the
DAG; it never replaces ``optimizer_step`` with a stale/async version
nor moves it to a non-default stream where the next iteration's
``forward`` could read pre-update parameters. See
``feedback_no_async_optimizer.md``.

## What this module produces

Given a `Schedule` plus a `CostModel` (per-task GPU/CPU duration
estimates), :func:`auto_assign_lookaheads` returns a
``dict[task_name -> recommended_lookahead]``. The caller can rebuild
new ``Task`` objects with the suggested la and feed them back to the
engine — no in-place mutation of the existing schedule.

## Resource model

A task contends for any of these resources during its execution:

  * ``stream`` — a single CUDA stream (FIFO, exclusive)
  * ``nccl_comm`` — a NCCL communicator (serializes across streams
    when sharing the same comm)
  * ``pcie`` — host↔device transfer bandwidth (h2d / UVM prefetch)

Two tasks can run **concurrently across different batches** iff they
share none of {stream, nccl_comm, pcie}. This produces the overlap
matrix ``O[i][j] ∈ {1, 0, "S", "N", "P"}`` where the non-zero
characters identify which resource forced serialization.

## Algorithm

1. Build the resource label per task from ``Task.stream``, optional
   ``Task.nccl`` flag, and a heuristic ``pcie`` set (``h2d``,
   ``prefetch_embeddings``).
2. Build the overlap matrix and the *default-stream critical path*
   (sum of GPU durations for all default-stream tasks at lookahead=0).
3. For each non-default-stream task with a lookahead currently set
   above 0, pick a *minimum* lookahead such that the task's GPU work
   completes before its consumer on the critical path begins —
   i.e. ``la = max(1, ceil(gpu_us / critical_path_us))``.
4. Lookahead never exceeds the user-supplied ``max_in_flight``
   constraint (memory budget) and must remain consistent with the
   existing reads/writes / depends_on DAG (we never lower a task's
   la below its dependency's la requirement).

The algorithm is intentionally conservative: it only *increases* la,
never decreases. A user can dial ``max_in_flight`` up to expand the
search space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from ..schedule import Schedule
from ..task import Task
from .cost_model import CostModel

__all__ = [
    "TaskResource",
    "auto_assign_lookaheads",
    "compute_overlap_matrix",
    "default_stream_critical_path_us",
    "describe_overlap_matrix",
    "task_resources",
    "DEFAULT_BIT_EXACT_TASKS",
]


# Tasks that MUST stay on the default stream at lookahead=0 — moving
# them would break bit-exact convergence (see
# ``feedback_no_async_optimizer.md``). We enforce this as a hard check
# in :func:`auto_assign_lookaheads`.
DEFAULT_BIT_EXACT_TASKS: FrozenSet[str] = frozenset(
    {
        "optimizer_step",
        "finalize_model_grads",
        "backward",
        "forward",
    }
)


# ----------------------------------------------------------------------
# Resource model
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TaskResource:
    """Set of resources a task contends for."""

    stream: str
    nccl_comm: Optional[str]  # None = not a NCCL collective
    pcie: bool  # True for h2d / UVM prefetch / similar host↔device transfer

    def conflicts_with(self, other: "TaskResource") -> Optional[str]:
        """Return the resource label that forces serialization, or
        ``None`` if the two tasks can overlap.

        Resolution priority: stream > nccl_comm > pcie. (Stream tightens
        scheduling first; NCCL comm is the next coarse barrier; PCIe is
        last because it's a bandwidth share, not a hard exclusive lock,
        but for the purposes of the matrix we still mark it.)
        """
        if self.stream == other.stream:
            return "stream"
        if (
            self.nccl_comm is not None
            and other.nccl_comm is not None
            and self.nccl_comm == other.nccl_comm
        ):
            return "nccl"
        if self.pcie and other.pcie:
            return "pcie"
        return None


# Heuristic: which task names are PCIe-bound (host↔device transfer).
# We accept overrides; this default mirrors the HSTU pipeline tasks
# that talk to host memory or UVM-backed DynamicEmb cache.
_DEFAULT_PCIE_TASKS: FrozenSet[str] = frozenset({"h2d", "prefetch_embeddings"})

# Heuristic: which streams use the data-parallel NCCL communicator.
# We map every NCCL-enabled task to a single "dp" comm by default —
# this is the right model for HSTU at TP=1 (shuffle, input_dist, gtok,
# DDP grads, finalize). When the schedule exposes per-task NCCL comm
# names, the caller can pass ``nccl_comm_of`` to refine.
_DEFAULT_NCCL_COMM_FOR_NCCL_TASKS = "dp"


def _normalize_stream(stream: Optional[str], default_stream: str) -> str:
    """``Task.stream`` is Optional[str]; both ``None`` and ``""`` mean
    "no explicit stream" and should map to ``default_stream``. Use this
    helper at every site that compares a task's stream to the default —
    avoids falsy-string bypasses (codex MAJOR).
    """
    if stream is None or stream == "":
        return default_stream
    return stream


def task_resources(
    tasks: Sequence[Task],
    *,
    pcie_tasks: Optional[FrozenSet[str]] = None,
    nccl_comm_of: Optional[Mapping[str, str]] = None,
) -> Dict[str, TaskResource]:
    """Return ``{task_name: TaskResource}`` extracted from ``tasks``.

    ``Task.nccl=True`` is mapped to the DP communicator by default;
    callers can pass ``nccl_comm_of`` to override per-task. Tasks not
    listed in ``pcie_tasks`` (default: ``h2d``, ``prefetch_embeddings``)
    are treated as not PCIe-bound.
    """
    if pcie_tasks is None:
        pcie_tasks = _DEFAULT_PCIE_TASKS
    out: Dict[str, TaskResource] = {}
    for t in tasks:
        nccl_comm: Optional[str] = None
        if nccl_comm_of is not None and t.name in nccl_comm_of:
            nccl_comm = nccl_comm_of[t.name]
        elif getattr(t, "nccl", False):
            nccl_comm = _DEFAULT_NCCL_COMM_FOR_NCCL_TASKS
        out[t.name] = TaskResource(
            stream=t.stream or "default",
            nccl_comm=nccl_comm,
            pcie=t.name in pcie_tasks,
        )
    return out


# ----------------------------------------------------------------------
# Overlap matrix
# ----------------------------------------------------------------------


def compute_overlap_matrix(
    tasks: Sequence[Task],
    resources: Optional[Mapping[str, TaskResource]] = None,
) -> Dict[Tuple[str, str], str]:
    """Return ``{(task_i, task_j): label}`` for every ordered pair of
    distinct task names.

    ``label`` is:
      * ``"ok"`` — tasks can run concurrently across batches
      * ``"stream"`` / ``"nccl"`` / ``"pcie"`` — serialized on that
        shared resource
      * ``"self"`` — the diagonal (same task)

    The matrix is symmetric: ``M[(a,b)] == M[(b,a)]``.
    """
    if resources is None:
        resources = task_resources(tasks)
    names = [t.name for t in tasks]
    out: Dict[Tuple[str, str], str] = {}
    for a in names:
        for b in names:
            if a == b:
                out[(a, b)] = "self"
                continue
            label = resources[a].conflicts_with(resources[b])
            out[(a, b)] = label if label is not None else "ok"
    return out


def describe_overlap_matrix(
    tasks: Sequence[Task],
    resources: Optional[Mapping[str, TaskResource]] = None,
) -> str:
    """Render the overlap matrix as an ASCII grid for human inspection.

    Useful for debugging or for embedding in a benchmark report.
    """
    if resources is None:
        resources = task_resources(tasks)
    matrix = compute_overlap_matrix(tasks, resources)
    names = [t.name for t in tasks]
    legend = {"self": "-", "ok": ".", "stream": "S", "nccl": "N", "pcie": "P"}
    width = max(len(n) for n in names)
    lines: List[str] = []
    header = " " * (width + 2) + " ".join(f"{i:>2d}" for i in range(len(names)))
    lines.append(header)
    for i, a in enumerate(names):
        row = [f"{i:>2d} {a:<{width}}"]
        for b in names:
            row.append(f"{legend.get(matrix[(a, b)], '?'):>2}")
        lines.append(" ".join(row))
    lines.append("")
    lines.append(
        "legend: .=ok, S=same stream, N=same NCCL comm, P=PCIe contended, -=self"
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Critical path on the default stream
# ----------------------------------------------------------------------


def default_stream_critical_path_us(
    tasks: Sequence[Task],
    cost_model: CostModel,
    *,
    default_stream: str = "default",
) -> float:
    """Sum of GPU durations of all tasks on the default stream that
    sit at the smallest existing lookahead (the "current iter" chain).

    This is the wall-time floor that NCCL/IO must hide behind. We use
    the smallest lookahead present on the default stream (typically
    0) so we capture the chain that defines step time.
    """
    default_stream_tasks = [
        t
        for t in tasks
        if _normalize_stream(t.stream, default_stream) == default_stream
    ]
    if not default_stream_tasks:
        return 0.0
    min_la = min(t.batch_offset for t in default_stream_tasks)
    chain = [t for t in default_stream_tasks if t.batch_offset == min_la]
    return sum(cost_model.get(t.name).gpu_us for t in chain)


# ----------------------------------------------------------------------
# Lookahead assignment
# ----------------------------------------------------------------------


def auto_assign_lookaheads(
    schedule: Schedule,
    cost_model: CostModel,
    *,
    max_in_flight: int = 5,
    pcie_tasks: Optional[FrozenSet[str]] = None,
    nccl_comm_of: Optional[Mapping[str, str]] = None,
    default_stream: str = "default",
    bit_exact_tasks: FrozenSet[str] = DEFAULT_BIT_EXACT_TASKS,
) -> Dict[str, int]:
    """Return ``{task_name: recommended_lookahead}``.

    The returned mapping is intended to be passed to a Schedule
    rebuilder (e.g. the HSTU pipeline factory) — this function does
    NOT mutate the input ``schedule``.

    Rules:

    1. Tasks on the default stream keep their existing lookahead. We
       never reorder the compute chain (fwd → bwd → opt) because doing
       so would either violate the DAG or break bit-exactness.
    2. Tasks NOT on the default stream get a recommended la chosen so
       their GPU work completes before the default-stream chain
       reaches a point that needs the data.
    3. We cap la by ``max_in_flight - 1`` (because the ring needs
       ``max_offset + 1`` slots).
    4. We never shrink la below the existing value (callers may have
       authored an explicit la that's needed for slot routing).

    The algorithm is greedy and uses a single global critical-path
    estimate; it does not simulate the steady-state interleave. That
    is intentional — the search space we expose is intentionally
    conservative and bit-exact-safe.
    """
    if max_in_flight < 1:
        raise ValueError(f"max_in_flight must be >= 1, got {max_in_flight}")

    tasks = list(schedule.all_tasks())
    resources = task_resources(tasks, pcie_tasks=pcie_tasks, nccl_comm_of=nccl_comm_of)
    cp_us = default_stream_critical_path_us(
        tasks, cost_model, default_stream=default_stream
    )

    # Hard-guard bit-exact tasks (CRITICAL #1 from codex review). Any
    # task in the bit-exact set must already be on the default stream
    # at lookahead=0 in the input schedule; otherwise the recommended
    # mapping cannot honor the bit-exact contract.
    #
    # ``Task.stream`` is an Optional[str] but the engine treats both
    # ``None`` and ``""`` as "no explicit stream"; the executor maps
    # both to the same default-stream context. We use the module-level
    # ``_normalize_stream`` helper at every site to avoid a falsy-
    # string bypass (codex MAJOR).
    for t in tasks:
        if t.name in bit_exact_tasks:
            if _normalize_stream(t.stream, default_stream) != default_stream:
                raise ValueError(
                    f"bit-exact task {t.name!r} must be on the "
                    f"{default_stream!r} stream, got {t.stream!r}. "
                    f"Async / off-default optimizer is forbidden."
                )
            if t.batch_offset != 0:
                raise ValueError(
                    f"bit-exact task {t.name!r} must have lookahead=0, "
                    f"got {t.batch_offset}. Async / stale-grad mode is "
                    f"forbidden."
                )

    # Index tasks for cheap dependency checks.
    by_name: Dict[str, Task] = {t.name: t for t in tasks}

    # Slot writers (slot.name → producers list) for read/write dep
    # traversal. The engine permits multiple writers at different
    # offsets when the (name, offset) pair is unique; we track every
    # writer so propagation does not bump only the most recently
    # declared one.  (codex MAJOR #3.)
    writers_by_slot_name: Dict[str, List[str]] = {}
    for t in tasks:
        for slot in t.writes:
            writers_by_slot_name.setdefault(slot.name, []).append(t.name)

    # Cross-iter constraints: each (consumer.name) → list of
    # (-neg_offset) requirements.  When we want to bump a consumer's
    # lookahead, we need ``consumer.la + neg_offset >= 0`` (codex
    # CRITICAL #2). Keep a per-task upper-bound floor.
    cross_iter_la_cap: Dict[str, int] = {t.name: max_in_flight - 1 for t in tasks}
    for t in tasks:
        for _producer, neg_offset in getattr(t, "cross_iter_depends_on", ()) or ():
            # consumer.la + neg_offset >= 0  →  consumer.la >= -neg_offset
            # but ALSO consumer.la <= future-max so that the engine's
            # slot_offset = consumer.la + neg_offset still lands in a
            # ring slot.  We treat this as: consumer.la must equal
            # exactly the user-authored value; do not bump.
            cross_iter_la_cap[t.name] = min(cross_iter_la_cap[t.name], t.batch_offset)

    out: Dict[str, int] = {t.name: t.batch_offset for t in tasks}

    cap = max_in_flight - 1

    # Pre-flight: any task whose authored la already exceeds cap is a
    # configuration error. We never silently shrink (codex MAJOR #4).
    for t in tasks:
        if t.batch_offset > cap:
            raise ValueError(
                f"Task {t.name!r} has lookahead={t.batch_offset} which "
                f"exceeds max_in_flight-1={cap}. Increase max_in_flight."
            )

    # Compute task la in topological order (producer before consumer)
    # so producer la is finalized when consumer's bound is computed.
    for t in tasks:
        # Default-stream tasks AND bit-exact tasks stay put — never
        # propagated up either (see propagation guard below).
        if _normalize_stream(t.stream, default_stream) == default_stream:
            continue
        if t.name in bit_exact_tasks:
            continue

        # Existing la is the floor.
        floor = t.batch_offset

        # If GPU work is bigger than one critical-path window, push
        # la higher so the data is ready when the consumer reaches it.
        gpu_us = cost_model.get(t.name).gpu_us
        if cp_us > 0 and gpu_us > cp_us:
            need = max(1, math.ceil(gpu_us / cp_us))
        else:
            need = 1  # always at least 1 ahead for an off-default stream

        # NCCL DP comm tasks all serialize: account for the cumulative
        # NCCL queue ahead of this task so its head-of-line completes
        # before the consumer reads.
        if resources[t.name].nccl_comm is not None:
            queue_us = _cumulative_nccl_queue_us(t, tasks, resources, cost_model)
            if cp_us > 0:
                need = max(need, math.ceil(queue_us / cp_us))

        # Honor predecessor la (every writer of a slot we read must
        # have la >= our recommended; we conservatively use the
        # highest existing la across all writers, since the ring slot
        # at our offset will hold one of them after rotation).
        for read_slot in t.reads:
            for producer_name in writers_by_slot_name.get(read_slot.name, ()):
                if producer_name != t.name:
                    need = max(need, out[producer_name])

        # depends_on: producer.la must be >= consumer.la for same-batch
        # ordering. We don't shrink consumer's la here, only ensure we
        # don't recommend below the producer's existing la.
        for dep_name in t.depends_on or ():
            if dep_name in by_name:
                need = max(need, out[dep_name])

        # Cross-iter cap (CRITICAL #2): if user authored
        # ``cross_iter_depends_on`` we cannot bump the consumer's
        # lookahead because that would shift slot_offset out of range.
        cap_for_t = min(cap, cross_iter_la_cap[t.name])
        recommended = max(floor, min(need, cap_for_t))
        out[t.name] = recommended

    # Producers of slots that someone else reads must have la at least
    # equal to the highest reader's la (otherwise the ring slot can't
    # hold the producer's value when the reader gets there). Walk
    # consumers and bump producers if needed.  Bit-exact tasks and
    # default-stream tasks are never bumped — they remain at their
    # authored la regardless of any consumer's recommendation.
    def _is_frozen(name: str) -> bool:
        if name in bit_exact_tasks:
            return True
        prod = by_name.get(name)
        return (
            prod is not None
            and _normalize_stream(prod.stream, default_stream) == default_stream
        )

    changed = True
    while changed:
        changed = False
        for t in tasks:
            for read_slot in t.reads:
                for producer in writers_by_slot_name.get(read_slot.name, ()):
                    if producer == t.name:
                        continue
                    if _is_frozen(producer):
                        # Producer is frozen — if consumer's la is
                        # higher, we have a hard inconsistency the
                        # caller must fix (cannot bump producer up).
                        if out[producer] < out[t.name]:
                            raise ValueError(
                                f"Auto-scheduler inconsistency: "
                                f"consumer {t.name!r} la={out[t.name]} > "
                                f"frozen producer {producer!r} la="
                                f"{out[producer]}. Reduce consumer la or "
                                f"author producer at higher la."
                            )
                        continue
                    if out[producer] < out[t.name]:
                        target = out[t.name]
                        new_la = min(target, cross_iter_la_cap[producer])
                        if new_la < target:
                            # Cross-iter cap binds below what the
                            # consumer needs — would leave consumer's
                            # la > producer's la, breaking ring slot
                            # invariant. Surface to caller (codex
                            # MAJOR #2 from second review).
                            raise ValueError(
                                f"Auto-scheduler inconsistency: "
                                f"consumer {t.name!r} la={target} requires "
                                f"producer {producer!r} at la>={target}, "
                                f"but producer's cross-iter cap is "
                                f"{cross_iter_la_cap[producer]}. Reduce "
                                f"consumer's required la or relax the "
                                f"producer's cross_iter_depends_on."
                            )
                        if new_la == out[producer]:
                            continue
                        out[producer] = new_la
                        changed = True
            for dep_name in t.depends_on or ():
                if dep_name not in by_name:
                    continue
                if _is_frozen(dep_name):
                    if out[dep_name] < out[t.name]:
                        raise ValueError(
                            f"Auto-scheduler inconsistency: "
                            f"consumer {t.name!r} la={out[t.name]} > "
                            f"frozen depends_on producer {dep_name!r} "
                            f"la={out[dep_name]}."
                        )
                    continue
                if out[dep_name] < out[t.name]:
                    target = out[t.name]
                    new_la = min(target, cross_iter_la_cap[dep_name])
                    if new_la < target:
                        raise ValueError(
                            f"Auto-scheduler inconsistency: "
                            f"consumer {t.name!r} la={target} requires "
                            f"depends_on producer {dep_name!r} at "
                            f"la>={target}, but cross-iter cap is "
                            f"{cross_iter_la_cap[dep_name]}."
                        )
                    if new_la == out[dep_name]:
                        continue
                    out[dep_name] = new_la
                    changed = True

    return out


def _cumulative_nccl_queue_us(
    task: Task,
    all_tasks: Sequence[Task],
    resources: Mapping[str, TaskResource],
    cost_model: CostModel,
) -> float:
    """Sum of GPU durations of every other NCCL task on the same
    comm — they serialize even on different streams."""
    my_comm = resources[task.name].nccl_comm
    if my_comm is None:
        return 0.0
    total = 0.0
    for t in all_tasks:
        if t.name == task.name:
            total += cost_model.get(t.name).gpu_us
            continue
        r = resources.get(t.name)
        if r is None or r.nccl_comm != my_comm:
            continue
        total += cost_model.get(t.name).gpu_us
    return total
