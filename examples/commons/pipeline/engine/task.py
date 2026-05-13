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

"""Task and DataSlot: the primitive units of the pipeline engine."""

from dataclasses import dataclass
from enum import IntFlag
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Union

__all__ = [
    "DataSlot",
    "SameProgressSyncSide",
    "Task",
]


# A read or write entry can be authored as a bare slot name (string)
# or as an explicit ``DataSlot(name, batch_offset)``. String entries
# are normalized to ``DataSlot(name, task.batch_offset)`` at
# construction time: the user's per-batch DAG describes slots by name,
# and the slot's batch_offset is implied by the owning task's offset.
# Existing imperative call sites that pass explicit DataSlot objects
# continue to work unchanged.
SlotRef = Union[str, "DataSlot"]


def _normalize_slot_refs(
    refs: Optional[Iterable[SlotRef]], batch_offset: int
) -> Tuple["DataSlot", ...]:
    if refs is None:
        return ()
    out = []
    for r in refs:
        if isinstance(r, DataSlot):
            out.append(r)
        elif isinstance(r, str):
            out.append(DataSlot(r, batch_offset))
        else:
            raise TypeError(
                f"reads/writes entries must be str or DataSlot, got "
                f"{type(r).__name__}: {r!r}"
            )
    return tuple(out)


# Dependency fields are intentionally separate:
#
#   ``depends_on=("X", ...)``
#       Same-batch logical dependency. The consumer waits for X to have
#       processed the same user batch, even if X did so in an earlier
#       ``progress()`` call because it has a larger lookahead.
#
#   ``cross_iter_depends_on=("X", ("Y", -2), ...)``
#       Different-batch dependency. A bare name is shorthand for
#       ``("name", -1)``; tuple form waits for an older batch by N
#       iterations. Zero or positive offsets are rejected.
#
#   ``same_progress_sync=("X", ("Y", SameProgressSyncSide.GPU), ...)``
#       Current-progress dependency. The consumer waits for X's work in
#       this exact ``progress()`` call, independent of the two tasks'
#       lookahead values. Use it for stream coherency on shared state
#       that is not represented by ``reads`` / ``writes``. Each edge can
#       specify whether it is enforced on CPU, GPU, or both.


class SameProgressSyncSide(IntFlag):
    """Where a ``same_progress_sync`` edge is enforced.

    ``CPU`` drives topo/ticket/thread ordering. ``GPU`` inserts
    cross-stream waits; without ``CPU``, the producer must be host-
    ordered by another path before the consumer enqueues the wait.
    """

    NONE = 0
    CPU = 1
    GPU = 2
    BOTH = CPU | GPU


@dataclass(frozen=True)
class _SameProgressSync:
    """One ``same_progress_sync`` edge.

    Bare task names are accepted by ``Task`` and normalized to this
    shape with ``sides=BOTH``.
    """

    task: str
    sides: SameProgressSyncSide = SameProgressSyncSide.BOTH

    def __post_init__(self) -> None:
        if not isinstance(self.task, str):
            raise TypeError(
                "same_progress_sync task name must be a string; "
                f"got {type(self.task).__name__}: {self.task!r}"
            )
        if not self.task:
            raise ValueError("same_progress_sync task name must be non-empty")
        object.__setattr__(
            self,
            "sides",
            _normalize_same_progress_sync_side(
                self.sides, field_name=f"same_progress_sync.{self.task}.sides"
            ),
        )


def _validate_bare_name_refs(
    refs: Optional[Iterable[str]],
    field_name: str,
) -> Tuple[str, ...]:
    """Validate a tuple-of-bare-task-names field (``depends_on``)."""
    if refs is None:
        return ()
    out: list = []
    for r in refs:
        if isinstance(r, str):
            out.append(r)
        else:
            raise TypeError(
                f"{field_name} entries must be bare task-name strings; "
                f"got {type(r).__name__}: {r!r}."
            )
    return tuple(out)


_SameProgressSyncRef = Union[
    str,
    _SameProgressSync,
    Tuple[str, Union[SameProgressSyncSide, int]],
    Mapping[str, Any],
]


def _validate_cross_iter_depends_on(
    refs: Optional[Iterable[Union[str, Tuple[str, int]]]],
) -> Tuple[Tuple[str, int], ...]:
    """Validate ``cross_iter_depends_on=`` entries.

    Two equivalent author syntaxes:
      - Bare task name ``"X"`` is shorthand for ``("X", -1)`` —
        "wait for X from 1 iter ago" (the most common case).
      - Tuple ``("X", -N)`` for N != 1 — "wait for X from N iters ago".

    Both forms are normalized to the ``(name, neg_int)`` internal
    representation. Positive / zero offsets are rejected.
    """
    if refs is None:
        return ()
    out: list = []
    for r in refs:
        if isinstance(r, str):
            # Bare-name shorthand: (name, -1)
            out.append((r, -1))
            continue
        if not (isinstance(r, tuple) and len(r) == 2):
            raise TypeError(
                f"cross_iter_depends_on entries must be a bare task "
                f"name (shorthand for N=1) or a (name, -N) tuple; "
                f"got {type(r).__name__}: {r!r}."
            )
        name, offset = r
        if not isinstance(name, str) or not isinstance(offset, int):
            raise TypeError(
                f"cross_iter_depends_on tuple entries must be (str, "
                f"int), got ({type(name).__name__}, "
                f"{type(offset).__name__}): {r!r}"
            )
        if offset >= 0:
            raise ValueError(
                f"cross_iter_depends_on=({name!r}, {offset}): offset "
                f"must be negative ('wait for X from N iters ago'). "
                f"For same-batch waits use ``depends_on``; for "
                f"current-progress waits use ``same_progress_sync``."
            )
        out.append((name, offset))
    return tuple(out)


def _normalize_same_progress_sync_side(
    value: Any,
    *,
    field_name: str,
) -> SameProgressSyncSide:
    if value is None:
        return SameProgressSyncSide.BOTH
    if isinstance(value, int) and not isinstance(value, bool) and value < 0:
        raise ValueError(f"{field_name}={value} contains unknown bits")
    try:
        sides = SameProgressSyncSide(value)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"{field_name} must be a SameProgressSyncSide IntFlag or "
            f"compatible int, got {type(value).__name__}: {value!r}"
        ) from e
    valid_mask = int(SameProgressSyncSide.BOTH)
    if int(sides) & ~valid_mask:
        raise ValueError(
            f"{field_name}={int(sides)} contains unknown bits; "
            f"valid bits are CPU={int(SameProgressSyncSide.CPU)} and "
            f"GPU={int(SameProgressSyncSide.GPU)}."
        )
    return sides


def _normalize_same_progress_sync_ref(
    ref: _SameProgressSyncRef,
    *,
    field_name: str,
) -> _SameProgressSync:
    if isinstance(ref, _SameProgressSync):
        return ref
    if isinstance(ref, str):
        return _SameProgressSync(ref)
    if isinstance(ref, tuple) and len(ref) == 2:
        name, sides = ref
        if not isinstance(name, str):
            raise TypeError(
                f"{field_name} tuple entries must be (str, side), got "
                f"({type(name).__name__}, {type(sides).__name__}): {ref!r}"
            )
        return _SameProgressSync(
            name,
            _normalize_same_progress_sync_side(
                sides, field_name=f"{field_name}.{name}.sides"
            ),
        )
    if isinstance(ref, Mapping):
        unknown = set(ref) - {"task", "name", "sides"}
        if unknown:
            raise ValueError(f"Unknown {field_name} field(s): {sorted(unknown)}")
        has_task = "task" in ref
        has_name = "name" in ref
        if has_task == has_name:
            raise ValueError(
                f"{field_name} object must contain exactly one of task/name"
            )
        name = ref["task"] if has_task else ref["name"]
        if not isinstance(name, str):
            raise TypeError(
                f"{field_name}.task must be a string, got "
                f"{type(name).__name__}: {name!r}"
            )
        return _SameProgressSync(
            name,
            _normalize_same_progress_sync_side(
                ref.get("sides", SameProgressSyncSide.BOTH),
                field_name=f"{field_name}.{name}.sides",
            ),
        )
    raise TypeError(
        f"{field_name} entries must be task names or (task, side) edges; "
        f"got {type(ref).__name__}: {ref!r}."
    )


def _validate_same_progress_sync_refs(
    refs: Optional[Iterable[_SameProgressSyncRef]],
    field_name: str,
) -> Tuple[_SameProgressSync, ...]:
    if refs is None or refs is False:
        return ()
    if isinstance(refs, (str, _SameProgressSync, Mapping)):
        return (_normalize_same_progress_sync_ref(refs, field_name=field_name),)

    out: list = []
    for ref in refs:
        out.append(_normalize_same_progress_sync_ref(ref, field_name=field_name))
    return tuple(out)


def _same_progress_sync_edges(
    task,
    *,
    side: Optional[SameProgressSyncSide] = None,
) -> Tuple[_SameProgressSync, ...]:
    edges = getattr(task, "same_progress_sync", ())
    if side is None:
        return tuple(edges)
    return tuple(edge for edge in edges if edge.sides & side)


def _same_progress_sync_names(task) -> Tuple[str, ...]:
    return tuple(edge.task for edge in _same_progress_sync_edges(task))


class DataSlot:
    """Opaque named handle for inter-task data flow.

    Identity is `(name, batch_offset)`. Two DataSlots compare equal iff
    both fields match.
    """

    __slots__ = ("name", "batch_offset")

    def __init__(self, name: str, batch_offset: int = 0) -> None:
        if batch_offset < 0:
            raise ValueError(
                f"DataSlot.batch_offset must be >= 0, got {batch_offset} "
                f"for slot '{name}'."
            )
        self.name = name
        self.batch_offset = batch_offset

    def __repr__(self) -> str:
        return f"DataSlot({self.name!r}, batch_offset={self.batch_offset})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataSlot):
            return NotImplemented
        return self.name == other.name and self.batch_offset == other.batch_offset

    def __hash__(self) -> int:
        return hash((self.name, self.batch_offset))


class Task:
    """Schedulable unit of work.

    Tasks can be authored either by subclassing and overriding
    ``init(ctx)`` / ``run(ctx)``, or by wrapping a callable with
    ``Task.from_fn(...)``.

    Cross-task data should flow through ``ctx.slots`` and be declared
    with ``reads`` / ``writes`` so the engine can infer ordering. Shared
    state outside slots is allowed, but if two tasks can touch it
    concurrently, authors must add an explicit dependency or colocate
    those tasks via ``thread_map``.

    Use ``lookahead=`` for cross-iteration positioning. Internally the
    engine stores the same value as ``batch_offset``.
    """

    # Defaults so subclasses can override as class attributes.
    name: str = ""
    stream: str = "default"
    batch_offset: int = 0
    reads: Tuple[DataSlot, ...] = ()
    writes: Tuple[DataSlot, ...] = ()
    depends_on: Tuple[str, ...] = ()
    cross_iter_depends_on: Tuple[Tuple[str, int], ...] = ()
    same_progress_sync: Tuple[_SameProgressSync, ...] = ()
    nvtx_tag: Optional[str] = None
    nccl: bool = False

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        stream: Optional[str] = None,
        lookahead: Optional[int] = None,
        reads: Optional[Iterable[SlotRef]] = None,
        writes: Optional[Iterable[SlotRef]] = None,
        depends_on: Optional[Iterable[str]] = None,
        cross_iter_depends_on: Optional[Iterable[Union[str, Tuple[str, int]]]] = None,
        same_progress_sync: Optional[Iterable[_SameProgressSyncRef]] = None,
        nvtx_tag: Optional[str] = None,
        nccl: Optional[bool] = None,
    ) -> None:
        # Instance fields override class defaults when provided.
        if name is not None:
            self.name = name
        if stream is not None:
            self.stream = stream
        # ``lookahead`` is the user-facing name for the cross-iter
        # offset. The engine still stores it as ``self.batch_offset``
        # internally so existing code (deps.py, executor.py,
        # autosched.validator) continues to read ``task.batch_offset``.
        # The public ``Task.lookahead`` property below returns the same
        # value.
        if lookahead is not None:
            self.batch_offset = lookahead
        raw_reads = reads if reads is not None else self.reads
        raw_writes = writes if writes is not None else self.writes
        raw_depends_on = depends_on if depends_on is not None else self.depends_on
        raw_cross_iter_depends_on = (
            cross_iter_depends_on
            if cross_iter_depends_on is not None
            else self.cross_iter_depends_on
        )
        raw_same_progress_sync = (
            same_progress_sync
            if same_progress_sync is not None
            else self.same_progress_sync
        )

        # Normalize unconditionally: covers both constructor kwargs
        # and subclass class attributes. Without this, a subclass that
        # writes ``reads = ("x",)`` or ``cross_iter_depends_on = ("X",)``
        # would retain the raw class-attribute tuple because the
        # constructor's kwarg branch only fires when an explicit kwarg
        # is passed.
        self.reads = _normalize_slot_refs(raw_reads, self.batch_offset)
        self.writes = _normalize_slot_refs(raw_writes, self.batch_offset)
        self.depends_on = _validate_bare_name_refs(raw_depends_on, "depends_on")
        self.cross_iter_depends_on = _validate_cross_iter_depends_on(
            raw_cross_iter_depends_on
        )
        self.same_progress_sync = _validate_same_progress_sync_refs(
            raw_same_progress_sync, "same_progress_sync"
        )

        # Sanity: a single producer name appearing in multiple of
        # the three dependency fields is almost certainly an
        # authoring mistake — they express different semantics.
        within_names = set(self.depends_on)
        cross_names = {n for (n, _) in self.cross_iter_depends_on}
        sync_names = set(_same_progress_sync_names(self))
        for a_name, a_set, b_name, b_set in [
            ("depends_on", within_names, "cross_iter_depends_on", cross_names),
            ("depends_on", within_names, "same_progress_sync", sync_names),
            ("cross_iter_depends_on", cross_names, "same_progress_sync", sync_names),
        ]:
            overlap = a_set & b_set
            if overlap:
                raise ValueError(
                    f"Task {self.name!r} lists producer name(s) "
                    f"{sorted(overlap)!r} in both ``{a_name}`` and "
                    f"``{b_name}``. Pick one semantic per producer."
                )
        if nvtx_tag is not None:
            self.nvtx_tag = nvtx_tag
        if nccl is not None:
            self.nccl = nccl

        if not self.name:
            raise ValueError(
                "Task.name is required (either via subclass attribute "
                "or constructor argument)."
            )
        if self.batch_offset < 0:
            raise ValueError(
                f"Task '{self.name}'.batch_offset must be >= 0, got "
                f"{self.batch_offset}."
            )

    @property
    def lookahead(self) -> int:
        """User-facing name for ``batch_offset``.

        Read-only — set via the constructor. The engine still keys all
        internal data structures on ``batch_offset``; this property is
        the user-facing alias.
        """
        return self.batch_offset

    def init(self, ctx) -> None:
        """One-time setup called when the pipeline is built.

        Subclasses override to cache module references, allocate
        scratch buffers, register hooks, etc. Default no-op.
        """
        return None

    def run(self, ctx) -> None:
        """Per-iteration workload. Subclasses MUST override (or use
        `Task.from_fn` to supply a callable)."""
        raise NotImplementedError(
            f"Task '{self.name}' did not override run(). "
            f"Either subclass Task and override run(ctx), or construct "
            f"via Task.from_fn(name, fn, ...)."
        )

    @classmethod
    def from_fn(
        cls,
        name: str,
        fn: Callable,
        *,
        stream: str = "default",
        lookahead: Optional[int] = None,
        reads: Iterable[SlotRef] = (),
        writes: Iterable[SlotRef] = (),
        depends_on: Iterable[str] = (),
        cross_iter_depends_on: Iterable[Union[str, Tuple[str, int]]] = (),
        same_progress_sync: Iterable[_SameProgressSyncRef] = (),
        nvtx_tag: Optional[str] = None,
        nccl: bool = False,
    ) -> "Task":
        """Factory for the lambda-style authoring form.

        ``reads`` / ``writes`` may be a tuple of bare slot names
        (tagged with the task's lookahead) or explicit ``DataSlot``
        objects. Use ``depends_on`` for same-batch task ordering and
        ``cross_iter_depends_on`` for explicit previous-batch waits.

        ``lookahead`` is the public name for the internal
        ``batch_offset``.
        """
        return _FnTask(
            fn=fn,
            name=name,
            stream=stream,
            lookahead=lookahead,
            reads=reads,
            writes=writes,
            depends_on=depends_on,
            cross_iter_depends_on=cross_iter_depends_on,
            same_progress_sync=same_progress_sync,
            nvtx_tag=nvtx_tag,
            nccl=nccl,
        )


class _FnTask(Task):
    """Internal: wraps a callable in a Task. Produced by `Task.from_fn`."""

    def __init__(self, fn: Callable, **kwargs) -> None:
        super().__init__(**kwargs)
        self._fn = fn

    def run(self, ctx) -> None:
        self._fn(ctx)
