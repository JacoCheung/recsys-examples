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

"""Task and DataSlot — the two primitive units of the engine.

See SPEC §4.1 for the field semantics and §4.2 for how Tasks compose
into a Schedule.
"""

from typing import Callable, Iterable, Optional, Tuple, Union

__all__ = ["DataSlot", "Task"]


# A read or write entry can be authored as a bare slot name (string)
# or as an explicit ``DataSlot(name, batch_offset)``. Per SPEC_p4 v2,
# string entries are normalized to ``DataSlot(name, task.batch_offset)``
# at construction time — the user's per-batch DAG describes slots by
# name, and the slot's batch_offset is implied by the owning task's
# offset. Existing imperative call sites that pass explicit DataSlot
# objects continue to work unchanged.
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


# `depends_on` entries: SPEC_p4 v2 §5 allows
#   - bare task name "X"      → within-iter ordering, equivalent to ("X", 0)
#   - tuple ("X", -N)         → cross-iter pure-control dependency, this
#                               task must wait for `X` from `N` iters ago.
# Positive offsets are rejected. The constructor splits the user-provided
# entries into two engine-internal tuples: ``depends_on`` (within-iter,
# bare names) and ``cross_iter_depends_on`` (list of (name, -N)).
DependsOnRef = Union[str, Tuple[str, int]]


def _normalize_depends_on(
    refs: Optional[Iterable[DependsOnRef]],
) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, int], ...]]:
    """Split user-facing depends_on into (within_iter, cross_iter)."""
    if refs is None:
        return (), ()
    within: list = []
    cross: list = []
    for r in refs:
        if isinstance(r, str):
            within.append(r)
        elif isinstance(r, tuple) and len(r) == 2:
            name, offset = r
            if not isinstance(name, str) or not isinstance(offset, int):
                raise TypeError(
                    f"depends_on tuple entries must be (str, int), got "
                    f"({type(name).__name__}, {type(offset).__name__}): {r!r}"
                )
            if offset > 0:
                raise ValueError(
                    f"depends_on tuple ({name!r}, {offset}): positive "
                    f"iteration offset is not allowed. Use negative for "
                    f"cross-iter ('wait for X from N iters ago'); use 0 "
                    f"or the bare string {name!r} for within-iter."
                )
            if offset == 0:
                within.append(name)
            else:
                cross.append((name, offset))
        else:
            raise TypeError(
                f"depends_on entries must be str or (str, int) tuple, got "
                f"{type(r).__name__}: {r!r}"
            )

    # SPEC_p4 v2 §5 strictness: a single producer name cannot appear
    # both as within-iter and as cross-iter on the same consumer. If
    # the user truly wants two independent edges they should use
    # distinct producer names; mixing the two forms on the same name
    # is treated as a likely mistake.
    overlap = set(within) & {n for (n, _) in cross}
    if overlap:
        raise ValueError(
            f"depends_on lists the same producer name(s) "
            f"{sorted(overlap)!r} both as within-iter (bare string or "
            f"(name, 0)) and as cross-iter ((name, -N)). Pick one "
            f"semantic per producer; if you really want two ordering "
            f"edges to that producer, give them distinct names."
        )

    return tuple(within), tuple(cross)


class DataSlot:
    """Opaque named handle for inter-task data flow.

    Identity is `(name, batch_offset)`. Two DataSlots compare equal iff
    both fields match. In V1 (`in_flight_batches=1`) only
    `batch_offset=0` is meaningful; V4 generalizes to N>1.
    """

    __slots__ = ("name", "batch_offset")

    def __init__(self, name: str, batch_offset: int = 0) -> None:
        if batch_offset < 0:
            raise ValueError(
                f"DataSlot.batch_offset must be >= 0, got {batch_offset} "
                f"for slot '{name}' (SPEC §4.2 rule 2)."
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

    Two authoring forms:

    (a) **Subclass** — override `init(ctx)` and `run(ctx)` for stateful
        tasks (HugeCTR `Scheduleable` analog):

            class MyTask(Task):
                name = "my_task"
                stream = "default"
                def run(self, ctx): ...

    (b) **`Task.from_fn(name, fn, ...)`** — wrap a callable for
        stateless / one-off tasks (HugeCTR `StreamContextScheduleable`
        analog).

    Field semantics: see SPEC §4.1. V1 exercises `name`, `fn`,
    `stream`, `reads`, `writes`. `priority`, `absolute_stream`,
    `batch_offset`, `depends_on`, `nvtx_tag` are accepted and stored
    but not yet exercised by the engine (lands in V2+).

    Per SPEC_p4 v2, the user-facing field name for cross-iter
    positioning is ``lookahead``; ``batch_offset`` is the internal
    engine alias. Both are accepted at construction time, and the
    public read-only ``Task.lookahead`` property returns
    ``self.batch_offset``. If both are passed and they disagree, the
    constructor raises ``ValueError``. New code should prefer
    ``lookahead``; ``batch_offset`` is kept for the imperative
    `Schedule` API and existing call sites.
    """

    # Defaults so subclasses can override as class attributes.
    name: str = ""
    stream: str = "default"
    priority: int = 0
    absolute_stream: bool = False
    batch_offset: int = 0
    reads: Tuple[DataSlot, ...] = ()
    writes: Tuple[DataSlot, ...] = ()
    depends_on: Tuple[str, ...] = ()
    cross_iter_depends_on: Tuple[Tuple[str, int], ...] = ()
    nvtx_tag: Optional[str] = None
    nccl: bool = False

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        stream: Optional[str] = None,
        priority: Optional[int] = None,
        absolute_stream: Optional[bool] = None,
        lookahead: Optional[int] = None,
        reads: Optional[Tuple[DataSlot, ...]] = None,
        writes: Optional[Tuple[DataSlot, ...]] = None,
        depends_on: Optional[Iterable[DependsOnRef]] = None,
        nvtx_tag: Optional[str] = None,
        nccl: Optional[bool] = None,
    ) -> None:
        # Instance fields override class defaults when provided.
        if name is not None:
            self.name = name
        if stream is not None:
            self.stream = stream
        if priority is not None:
            self.priority = priority
        if absolute_stream is not None:
            self.absolute_stream = absolute_stream
        # SPEC_p4 v2 §5: ``lookahead`` is the user-facing name for the
        # cross-iter offset. The engine still stores it as
        # ``self.batch_offset`` internally so existing code
        # (deps.py, executor.py, autosched.validator) continues to read
        # ``task.batch_offset``. The public ``Task.lookahead`` property
        # below returns the same value. The legacy ``batch_offset=``
        # constructor keyword was removed in Phase C — call sites must
        # use ``lookahead=`` now.
        if lookahead is not None:
            self.batch_offset = lookahead
        # `reads` / `writes` accept either explicit DataSlot objects
        # (legacy imperative API) or bare slot names; bare names are
        # auto-tagged with `self.batch_offset` so the user-facing
        # SPEC_p4 v2 form `reads=("foo", "bar")` works without the user
        # restating the offset.
        if reads is not None:
            self.reads = _normalize_slot_refs(reads, self.batch_offset)
        if writes is not None:
            self.writes = _normalize_slot_refs(writes, self.batch_offset)
        if depends_on is not None:
            # Accept SPEC_p4 v2 union form: bare names are within-iter,
            # ("X", -N) tuples are cross-iter pure-control. Split into
            # the two engine-internal tuples; existing engine code
            # (deps.py, executor.py) reads `self.depends_on` for the
            # within-iter set as before.
            within, cross = _normalize_depends_on(depends_on)
            self.depends_on = within
            if cross:
                self.cross_iter_depends_on = cross
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
                f"{self.batch_offset} (SPEC §4.2 rule 2)."
            )

    @property
    def lookahead(self) -> int:
        """SPEC_p4 v2 user-facing name for ``batch_offset``.

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
        priority: int = 0,
        absolute_stream: bool = False,
        lookahead: Optional[int] = None,
        reads: Iterable[SlotRef] = (),
        writes: Iterable[SlotRef] = (),
        depends_on: Iterable[DependsOnRef] = (),
        nvtx_tag: Optional[str] = None,
        nccl: bool = False,
    ) -> "Task":
        """Factory for the lambda-style authoring form.

        ``reads`` and ``writes`` may be a tuple of bare slot names
        (auto-tagged with the task's lookahead) or explicit ``DataSlot``
        objects (legacy imperative form). Returns a ``Task`` instance
        whose ``run(ctx)`` dispatches to ``fn(ctx)``.

        SPEC_p4 v2 §7 Phase C removed the legacy ``batch_offset=``
        keyword — call sites must use ``lookahead=``.
        """
        return _FnTask(
            fn=fn,
            name=name,
            stream=stream,
            priority=priority,
            absolute_stream=absolute_stream,
            lookahead=lookahead,
            reads=reads,
            writes=writes,
            depends_on=depends_on,
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
