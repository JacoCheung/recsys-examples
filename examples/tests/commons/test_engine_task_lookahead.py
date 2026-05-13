# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``Task.lookahead`` public API."""

import pytest
from commons.pipeline.engine.task import DataSlot, SameProgressSyncSide, Task


def test_task_default_lookahead_is_zero() -> None:
    t = Task("x")
    assert t.batch_offset == 0
    assert t.lookahead == 0


def test_task_lookahead_kw_sets_batch_offset() -> None:
    t = Task("x", lookahead=2)
    assert t.batch_offset == 2
    assert t.lookahead == 2


def test_task_batch_offset_kw_removed() -> None:
    """The old ``batch_offset=`` keyword is not accepted."""
    with pytest.raises(TypeError, match="batch_offset"):
        Task("x", batch_offset=3)  # type: ignore[call-arg]


def test_task_negative_lookahead_rejected() -> None:
    with pytest.raises(ValueError, match="batch_offset must be >= 0"):
        Task("x", lookahead=-1)


def test_task_lookahead_is_read_only() -> None:
    t = Task("x", lookahead=2)
    with pytest.raises(AttributeError):
        t.lookahead = 3  # type: ignore[misc]


def test_from_fn_lookahead_kw() -> None:
    t = Task.from_fn("x", fn=lambda ctx: None, lookahead=2)
    assert t.batch_offset == 2
    assert t.lookahead == 2


def test_from_fn_batch_offset_kw_removed() -> None:
    """Task.from_fn also rejects the old keyword."""
    with pytest.raises(TypeError, match="batch_offset"):
        Task.from_fn("x", fn=lambda ctx: None, batch_offset=2)  # type: ignore[call-arg]


# ---------------------------------------------------------------------
# Slot ref normalization (str shorthand vs explicit DataSlot)
# ---------------------------------------------------------------------


def test_str_reads_writes_get_task_offset() -> None:
    """Bare slot names get task.batch_offset automatically."""
    t = Task("x", lookahead=2, reads=("foo", "bar"), writes=("out",))
    assert t.reads == (DataSlot("foo", 2), DataSlot("bar", 2))
    assert t.writes == (DataSlot("out", 2),)


def test_explicit_dataslot_reads_writes_preserved() -> None:
    """Imperative API still works — explicit DataSlot objects pass
    through unchanged regardless of task.batch_offset."""
    t = Task(
        "x",
        lookahead=0,
        reads=(DataSlot("foo", 5),),
        writes=(DataSlot("out", 7),),
    )
    assert t.reads == (DataSlot("foo", 5),)
    assert t.writes == (DataSlot("out", 7),)


def test_invalid_slot_ref_type_rejected() -> None:
    with pytest.raises(TypeError, match="reads/writes entries must be str or DataSlot"):
        Task("x", lookahead=0, reads=(123,))  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# depends_on (bare names, same-progress) and
# cross_iter_depends_on ((name, -N), ring-rotated) — separate fields
# ---------------------------------------------------------------------


def test_depends_on_bare_names_only() -> None:
    """depends_on accepts bare task names."""
    t = Task("x", depends_on=("a", "b"))
    assert t.depends_on == ("a", "b")
    assert t.cross_iter_depends_on == ()


def test_cross_iter_depends_on_tuple_form() -> None:
    """``cross_iter_depends_on`` accepts (name, -N) tuples."""
    t = Task("x", cross_iter_depends_on=(("optimizer_step", -1),))
    assert t.depends_on == ()
    assert t.cross_iter_depends_on == (("optimizer_step", -1),)


def test_cross_iter_depends_on_bare_name_shorthand() -> None:
    """Bare cross-iter dependency names mean previous iteration."""
    t = Task("x", cross_iter_depends_on=("optimizer_step",))
    assert t.cross_iter_depends_on == (("optimizer_step", -1),)


def test_cross_iter_zero_offset_rejected() -> None:
    """offset must be strictly negative — 0 means same-progress, use
    bare name in ``depends_on`` instead."""
    with pytest.raises(ValueError, match="offset must be negative"):
        Task("x", cross_iter_depends_on=(("future", 0),))


def test_cross_iter_invalid_tuple_shape_rejected() -> None:
    with pytest.raises(TypeError, match="cross_iter_depends_on entries"):
        Task(
            "x",
            cross_iter_depends_on=(("only_one_field",),),  # type: ignore[arg-type]
        )


def test_cross_iter_invalid_tuple_types_rejected() -> None:
    with pytest.raises(TypeError, match="cross_iter_depends_on tuple entries"):
        Task("x", cross_iter_depends_on=((123, -1),))  # type: ignore[arg-type]


def test_depends_on_non_str_rejected() -> None:
    """``depends_on`` is bare-names-only — tuples / non-strings rejected."""
    with pytest.raises(TypeError, match="depends_on entries"):
        Task("x", depends_on=(("optimizer_step", -1),))  # type: ignore[arg-type]


def test_from_fn_both_fields() -> None:
    t = Task.from_fn(
        "forward",
        fn=lambda ctx: None,
        lookahead=0,
        depends_on=("backward",),
        cross_iter_depends_on=(("optimizer_step", -1),),
    )
    assert t.depends_on == ("backward",)
    assert t.cross_iter_depends_on == (("optimizer_step", -1),)


def test_same_name_in_both_fields_rejected() -> None:
    """A producer name in BOTH ``depends_on`` and
    ``cross_iter_depends_on`` is almost certainly a mistake — the two
    fields express incompatible semantics (same-progress vs ring-
    rotated). If the user really wants two ordering edges to that
    producer, give them distinct names."""
    with pytest.raises(
        ValueError, match="both ``depends_on``.*``cross_iter_depends_on``"
    ):
        Task("x", depends_on=("X",), cross_iter_depends_on=(("X", -1),))


# ---------------------------------------------------------------------
# same_progress_sync field validation
# ---------------------------------------------------------------------
#
# ``same_progress_sync=("X",)`` is the third dependency field — a
# bare-name-only same-progress GPU/stream coherency wait, NOT a logical
# data-flow edge. Validation mirrors ``depends_on`` (bare names only)
# but the engine treats it differently downstream.


def test_same_progress_sync_bare_names_accepted() -> None:
    """``same_progress_sync`` accepts bare task names (the only valid
    authoring form)."""
    t = Task("x", same_progress_sync=("a", "b"))
    assert t.same_progress_sync == ("a", "b")
    assert t.same_progress_sync_sides == SameProgressSyncSide.BOTH
    assert t.depends_on == ()
    assert t.cross_iter_depends_on == ()


def test_same_progress_sync_sides_accepts_intflag() -> None:
    t = Task(
        "x",
        same_progress_sync=("a",),
        same_progress_sync_sides=SameProgressSyncSide.CPU,
    )

    assert t.same_progress_sync_sides == SameProgressSyncSide.CPU


def test_same_progress_sync_sides_rejects_unknown_bits() -> None:
    with pytest.raises(ValueError, match="unknown bits"):
        Task("x", same_progress_sync=("a",), same_progress_sync_sides=4)


def test_same_progress_sync_tuple_form_rejected() -> None:
    """``same_progress_sync`` is bare-names-only — ``("X", -1)``
    tuples (the cross_iter_depends_on form) are not accepted."""
    with pytest.raises(TypeError, match="same_progress_sync entries"):
        Task("x", same_progress_sync=(("X", -1),))  # type: ignore[arg-type]


def test_same_progress_sync_overlap_with_depends_on_rejected() -> None:
    """A producer name appearing in BOTH ``same_progress_sync`` and
    ``depends_on`` is an authoring mistake — pick one semantic."""
    with pytest.raises(
        ValueError,
        match="both ``depends_on``.*``same_progress_sync``",
    ):
        Task("x", depends_on=("X",), same_progress_sync=("X",))


def test_same_progress_sync_overlap_with_cross_iter_rejected() -> None:
    """A producer name appearing in BOTH ``same_progress_sync`` and
    ``cross_iter_depends_on`` is an authoring mistake — pick one
    semantic."""
    with pytest.raises(
        ValueError,
        match="both ``cross_iter_depends_on``.*``same_progress_sync``",
    ):
        Task(
            "x",
            cross_iter_depends_on=(("X", -1),),
            same_progress_sync=("X",),
        )


def test_from_fn_same_progress_sync_kw() -> None:
    """``Task.from_fn(... same_progress_sync=("X",))`` works and
    normalizes the same way as the subclass form."""
    t = Task.from_fn(
        "x",
        fn=lambda ctx: None,
        same_progress_sync=("X",),
        same_progress_sync_sides=SameProgressSyncSide.GPU,
    )
    assert t.same_progress_sync == ("X",)
    assert t.same_progress_sync_sides == SameProgressSyncSide.GPU


# ---------------------------------------------------------------------
# Subclass class-attr shorthand normalization
# ---------------------------------------------------------------------
#
# Per task.py docstring: ``Task.__init__`` runs the validator
# unconditionally on ``self.{depends_on, cross_iter_depends_on,
# same_progress_sync}`` so subclass class attributes are normalized
# even when no constructor kwarg is passed. Without this, a subclass
# with ``cross_iter_depends_on = ("X",)`` (bare-name shorthand) would
# never reach ``_validate_cross_iter_depends_on`` and the engine would
# trip on the un-normalized tuple.


def test_subclass_cross_iter_class_attr_bare_name_normalized() -> None:
    """Subclass with bare-name class-attr ``cross_iter_depends_on``
    has its instance-level field normalized to ``(("X", -1),)``."""

    class _SubA(Task):
        name = "sub_a"
        cross_iter_depends_on = ("optimizer",)  # type: ignore[assignment]

        def run(self, ctx) -> None:
            pass

    t = _SubA()
    assert t.cross_iter_depends_on == (("optimizer", -1),)


def test_subclass_read_write_class_attr_bare_names_normalized() -> None:
    """Subclass class attributes should support the same bare-slot
    shorthand as constructor kwargs and Task.from_fn."""

    class _SubSlots(Task):
        name = "sub_slots"
        batch_offset = 2
        reads = ("input",)  # type: ignore[assignment]
        writes = ("output",)  # type: ignore[assignment]

        def run(self, ctx) -> None:
            pass

    t = _SubSlots()
    assert t.reads == (DataSlot("input", batch_offset=2),)
    assert t.writes == (DataSlot("output", batch_offset=2),)


def test_subclass_class_attr_overridden_by_constructor_kwarg() -> None:
    """When both class-attr default and constructor kwarg are present,
    the kwarg wins; the result is still normalized."""

    class _SubD(Task):
        name = "sub_d"
        cross_iter_depends_on = ("default_producer",)  # type: ignore[assignment]

        def run(self, ctx) -> None:
            pass

    # Constructor passes a different producer — kwarg wins, and the
    # bare-name shorthand still gets normalized to (name, -1).
    t = _SubD(cross_iter_depends_on=("override_producer",))
    assert t.cross_iter_depends_on == (("override_producer", -1),)
