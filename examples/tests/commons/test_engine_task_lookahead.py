# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SPEC_p4 v2 ``Task.lookahead`` user-facing API.

``lookahead`` is the only public keyword for the cross-iter offset on
``Task`` / ``Task.from_fn``; the engine still stores it internally as
``task.batch_offset``. SPEC_p4 v2 §7 Phase C removed the legacy
``batch_offset=`` keyword on the user-facing constructors.

This module verifies:

* ``Task(lookahead=N)`` populates ``task.batch_offset`` and the
  read-only ``task.lookahead`` property.
* The legacy ``batch_offset=`` keyword is gone — passing it raises
  ``TypeError`` (Python kw-only-arg machinery).
* ``Task.from_fn`` accepts ``lookahead=`` only.
* ``reads``/``writes`` accept bare slot names (auto-tagged with
  ``task.batch_offset``) and explicit ``DataSlot`` objects.
"""

import pytest
from commons.pipeline.engine.task import DataSlot, Task


def test_task_default_lookahead_is_zero() -> None:
    t = Task("x")
    assert t.batch_offset == 0
    assert t.lookahead == 0


def test_task_lookahead_kw_sets_batch_offset() -> None:
    t = Task("x", lookahead=2)
    assert t.batch_offset == 2
    assert t.lookahead == 2


def test_task_batch_offset_kw_removed() -> None:
    """SPEC_p4 v2 §7 Phase C removed ``batch_offset=`` from
    ``Task.__init__``. Python's kw-only-arg machinery raises
    ``TypeError`` when passed."""
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
    """``Task.from_fn(batch_offset=...)`` is also gone."""
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


def test_mixed_str_and_dataslot_reads() -> None:
    """str entries auto-tagged, DataSlot entries passed through —
    side by side in the same tuple."""
    t = Task(
        "x",
        lookahead=2,
        reads=("foo", DataSlot("bar", 9)),
    )
    assert t.reads == (DataSlot("foo", 2), DataSlot("bar", 9))


def test_invalid_slot_ref_type_rejected() -> None:
    with pytest.raises(TypeError, match="reads/writes entries must be str or DataSlot"):
        Task("x", lookahead=0, reads=(123,))  # type: ignore[arg-type]


def test_from_fn_str_shorthand() -> None:
    t = Task.from_fn("x", fn=lambda ctx: None, lookahead=1, reads=("foo",))
    assert t.reads == (DataSlot("foo", 1),)
    assert t.lookahead == 1


# ---------------------------------------------------------------------
# depends_on: cross-iter (name, -N) syntax
# ---------------------------------------------------------------------


def test_depends_on_str_is_within_iter() -> None:
    """Bare task name → within-iter, kept on .depends_on."""
    t = Task("x", depends_on=("a", "b"))
    assert t.depends_on == ("a", "b")
    assert t.cross_iter_depends_on == ()


def test_depends_on_tuple_zero_offset_is_within_iter() -> None:
    """('X', 0) is shorthand for bare 'X' — both go to .depends_on."""
    t = Task("x", depends_on=(("a", 0), "b"))
    assert t.depends_on == ("a", "b")
    assert t.cross_iter_depends_on == ()


def test_depends_on_negative_offset_goes_to_cross_iter() -> None:
    """('X', -N) → cross-iter, kept on .cross_iter_depends_on."""
    t = Task("x", depends_on=(("optimizer_step", -1),))
    assert t.depends_on == ()
    assert t.cross_iter_depends_on == (("optimizer_step", -1),)


def test_depends_on_mixed_split_correctly() -> None:
    t = Task(
        "x",
        depends_on=("backward", ("optimizer_step", -1), ("h2d", -2)),
    )
    assert t.depends_on == ("backward",)
    assert t.cross_iter_depends_on == (("optimizer_step", -1), ("h2d", -2))


def test_depends_on_positive_offset_rejected() -> None:
    with pytest.raises(ValueError, match="positive iteration offset"):
        Task("x", depends_on=(("future", 1),))


def test_depends_on_invalid_tuple_shape_rejected() -> None:
    with pytest.raises(TypeError, match="depends_on entries"):
        Task("x", depends_on=(("only_one_field",),))  # type: ignore[arg-type]


def test_depends_on_invalid_tuple_types_rejected() -> None:
    with pytest.raises(TypeError, match="depends_on tuple entries"):
        Task("x", depends_on=((123, -1),))  # type: ignore[arg-type]


def test_depends_on_invalid_entry_type_rejected() -> None:
    with pytest.raises(TypeError, match="depends_on entries"):
        Task("x", depends_on=(123,))  # type: ignore[arg-type]


def test_from_fn_cross_iter_depends_on() -> None:
    t = Task.from_fn(
        "forward",
        fn=lambda ctx: None,
        lookahead=0,
        depends_on=("backward", ("optimizer_step", -1)),
    )
    assert t.depends_on == ("backward",)
    assert t.cross_iter_depends_on == (("optimizer_step", -1),)


def test_depends_on_same_name_within_and_cross_rejected() -> None:
    """SPEC_p4 v2 §5: same producer name cannot appear both
    within-iter and cross-iter."""
    with pytest.raises(ValueError, match="both as within-iter.*cross-iter"):
        Task("x", depends_on=("X", ("X", -1)))


def test_depends_on_same_name_tuple_zero_and_cross_rejected() -> None:
    """`('X', 0)` is the tuple form of within-iter — also rejected
    when paired with `('X', -N)` cross-iter."""
    with pytest.raises(ValueError, match="both as within-iter.*cross-iter"):
        Task("x", depends_on=(("X", 0), ("X", -1)))
