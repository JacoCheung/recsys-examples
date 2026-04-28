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


# Three distinct user-facing fields express three distinct ordering
# semantics, framed in terms of the user's logical batch flow (not
# engine implementation details). They are NOT interchangeable, and
# each is validated separately:
#
#   ``depends_on=("X", "Y", ...)``                          (Tuple[str, ...])
#       **same-batch logical dependency**: "this task processing
#       batch K must wait for X to have processed batch K". The
#       engine maps this to a ring-rotated wait_event lookup at
#       slot offset=consumer.batch_offset, finding X's event from
#       whichever progress() call X processed batch K in (which may
#       be the current progress, if X.lookahead == self.lookahead,
#       or an earlier progress, if X.lookahead > self.lookahead and
#       the ring has rotated X's event into the consumer's slot).
#       Future-read (X.lookahead < self.lookahead) is rejected at
#       construction time.
#
#   ``cross_iter_depends_on=("X", ("Y", -2), ...)``
#       **different-batch logical dependency**: "this task processing
#       batch K must wait for X to have processed batch K-N". Two
#       equivalent author syntaxes:
#         - Bare name ``"X"`` is shorthand for ``("X", -1)`` — the
#           common "wait for prev batch's X" case.
#         - Tuple ``("X", -N)`` for N != 1 — wait for X from N batches
#           ago. Positive / zero offsets are rejected (use
#           ``depends_on`` for same-progress wait).
#       Engine reads X's event from a ring-history slot N iters
#       earlier than same-batch staging would give. Currently rare in
#       HSTU (no task uses it); reserved for future authors who need
#       to depend on a strictly earlier batch's output, e.g. streaming
#       statistics.
#
#   ``same_progress_sync=("X", "Y", ...)``                          (Tuple[str, ...])
#       **same-progress wait**: "this task must wait for X's work in
#       *this same* progress() call to finish, regardless of which
#       batches X and self are processing."
#
#       Engine contract:
#         - Adds a topological-sort edge X → self (so X is guaranteed
#           to fire before self within the progress).
#         - For threaded execution, adds a CPU-side ``threading.Event``
#           wait (cross-thread case only).
#         - For cross-stream, emits ``wait_event`` at slot offset=
#           ``producer.batch_offset`` (where X recorded its event in
#           *this* progress). Same-stream relies on CUDA stream FIFO.
#
#       Three known uses:
#
#       (a) Stream coherency on out-of-slot mutable state. When
#           consumer and producer touch a shared object that is NOT
#           declared in any reads/writes slot (e.g. dynamicemb cache,
#           a torchrec module attribute, a global counter) and they
#           run on different streams, the engine cannot infer a
#           dataflow edge. ``same_progress_sync`` makes the wait
#           explicit. Mirrors the legacy
#           ``default_stream.wait_stream(prefetch_stream)`` pattern.
#           HSTU prefetch variant uses this for
#           ``backward.same_progress_sync=("prefetch_embeddings",)``
#           — drains prefetch's GPU work before backward / before
#           the next progress kicks off.
#
#       (b) Explicit form of the Δ=0 cross_iter_depends_on. When the
#           user's mental model is "this la=c task waits for that
#           la=p task's current-progress output" with c = p + N for
#           some positive N, they may write either
#           ``same_progress_sync=(X,)`` (clearer when the intent is
#           explicitly same-progress) OR
#           ``cross_iter_depends_on=((X, -N),)`` with the batch-flow
#           phrasing — the engine auto-promotes the latter to the
#           same mechanical contract (same topo edge, same slot
#           lookup, same CPU edge). See SPEC_cross_iter_delta0_autoconvert.md.
#           Concrete example: la=1 forward wants to use post-update
#           weights from la=0 update in the same progress → either
#           ``forward.same_progress_sync=("update",)`` or
#           ``forward.cross_iter_depends_on=(("update", -1),)`` works.
#           See ``test_engine_same_progress_sync_correctness.py``
#           (parametrized over both forms).
#
#       (c) Producer/consumer in same progress on different batches
#           with NO sharing of out-of-slot state, but where the
#           consumer wants the producer's current-progress event for
#           coherent measurement / logging / metric aggregation. (Less
#           common; aux statistics tasks etc.)
#
#       Direction convention: ``X.same_progress_sync=()`` is empty;
#       declarations live on the consumer side. ``X.la`` and
#       ``consumer.la`` may differ — the field is la-agnostic.


def _validate_bare_name_refs(
    refs: Optional[Iterable[str]],
    field_name: str,
) -> Tuple[str, ...]:
    """Validate a tuple-of-bare-task-names field (``depends_on`` /
    ``same_progress_sync``)."""
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
            # Bare-name shorthand → (name, -1)
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
                f"For same-progress wait, use bare name in "
                f"``depends_on`` instead."
            )
        out.append((name, offset))
    return tuple(out)


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

    --------------------------------------------------------------------
    Communication discipline (read this before writing a new task!)
    --------------------------------------------------------------------
    The first-class channel for cross-task data is ``ctx.slots[name]``,
    declared in ``reads`` / ``writes`` so the engine's DAG can see it.

    Tasks MAY also share other state — module instance attributes
    (e.g. ``module.forward._context``), ``PipelineState`` fields,
    global Python singletons. Sharing alone is fine; what kills you
    is **concurrent** read/write. The engine inserts CPU-side
    ``threading.Event`` waits ONLY for edges it can see: ``reads`` /
    ``writes`` on the same slot, or an explicit ``depends_on``. So:
    if two tasks touch the same out-of-slot object AND the engine has
    no DAG edge between them (different slots, no ``depends_on``) —
    including **cross-iter pipelined** overlap (iter K of one task
    racing iter K+1 of another) — the author must colocate them on
    one thread via ``thread_map``. Within a single thread, sequential
    execution serializes them automatically.

    Convention only — no runtime check. Past offender: torchrec's
    ``set_context()`` was called from ``start_input_dist`` and
    ``forward``, both writing shared ``module.forward._context``.
    Same-iter ordering was implicit (``forward`` reads the
    ``torchrec_ctx`` slot that ``start_input_dist`` writes), but the
    cross-iter pipelined edge — iter K's ``forward`` overlapping iter
    K+1's ``start_input_dist`` — had no shared slot. The colocation
    was once enforced by a now-deleted runtime check; the underlying
    mutation was removed entirely in the PostProc cleanup.
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
    same_progress_sync: Tuple[str, ...] = ()
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
        depends_on: Optional[Iterable[str]] = None,
        cross_iter_depends_on: Optional[Iterable[Union[str, Tuple[str, int]]]] = None,
        same_progress_sync: Optional[Iterable[str]] = None,
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
        # Stash raw kwarg values onto self before normalization so the
        # normalize step below uses the kwarg if provided, otherwise
        # whatever the subclass set as a class attribute.
        if depends_on is not None:
            self.depends_on = depends_on  # type: ignore[assignment]
        if cross_iter_depends_on is not None:
            self.cross_iter_depends_on = cross_iter_depends_on  # type: ignore[assignment]
        if same_progress_sync is not None:
            self.same_progress_sync = same_progress_sync  # type: ignore[assignment]

        # Normalize unconditionally — covers both constructor kwargs
        # AND subclass class attributes. Without this, a subclass that
        # writes ``cross_iter_depends_on = ("X",)`` (bare-name shorthand)
        # would never get normalized into ``(("X", -1),)`` because the
        # constructor's kwarg branch only fires when an explicit kwarg
        # is passed.
        self.depends_on = _validate_bare_name_refs(self.depends_on, "depends_on")
        self.cross_iter_depends_on = _validate_cross_iter_depends_on(
            self.cross_iter_depends_on
        )
        self.same_progress_sync = _validate_bare_name_refs(
            self.same_progress_sync, "same_progress_sync"
        )

        # Sanity: a single producer name appearing in multiple of
        # the three dependency fields is almost certainly an
        # authoring mistake — they express different semantics.
        within_names = set(self.depends_on)
        cross_names = {n for (n, _) in self.cross_iter_depends_on}
        sync_names = set(self.same_progress_sync)
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
        depends_on: Iterable[str] = (),
        cross_iter_depends_on: Iterable[Union[str, Tuple[str, int]]] = (),
        same_progress_sync: Iterable[str] = (),
        nvtx_tag: Optional[str] = None,
        nccl: bool = False,
    ) -> "Task":
        """Factory for the lambda-style authoring form.

        ``reads`` / ``writes`` may be a tuple of bare slot names
        (auto-tagged with the task's lookahead) or explicit ``DataSlot``
        objects.

        ``depends_on`` is a tuple of bare task names — each producer's
        completion event is awaited *in this same progress() call*
        (same-progress wait).

        ``cross_iter_depends_on`` is a tuple of ``(name, -N)`` pairs —
        each names a producer whose output from N iters ago is
        consumed; engine emits a wait_event on the ring slot at
        offset=producer.lookahead+(-N).

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
