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

from typing import Callable, Optional, Tuple

__all__ = ["DataSlot", "Task"]


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
    nvtx_tag: Optional[str] = None

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        stream: Optional[str] = None,
        priority: Optional[int] = None,
        absolute_stream: Optional[bool] = None,
        batch_offset: Optional[int] = None,
        reads: Optional[Tuple[DataSlot, ...]] = None,
        writes: Optional[Tuple[DataSlot, ...]] = None,
        depends_on: Optional[Tuple[str, ...]] = None,
        nvtx_tag: Optional[str] = None,
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
        if batch_offset is not None:
            self.batch_offset = batch_offset
        if reads is not None:
            self.reads = reads
        if writes is not None:
            self.writes = writes
        if depends_on is not None:
            self.depends_on = depends_on
        if nvtx_tag is not None:
            self.nvtx_tag = nvtx_tag

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
        batch_offset: int = 0,
        reads: Tuple[DataSlot, ...] = (),
        writes: Tuple[DataSlot, ...] = (),
        depends_on: Tuple[str, ...] = (),
        nvtx_tag: Optional[str] = None,
    ) -> "Task":
        """Factory for the lambda-style authoring form.

        Returns a `Task` instance whose `run(ctx)` dispatches to `fn(ctx)`.
        """
        return _FnTask(
            fn=fn,
            name=name,
            stream=stream,
            priority=priority,
            absolute_stream=absolute_stream,
            batch_offset=batch_offset,
            reads=reads,
            writes=writes,
            depends_on=depends_on,
            nvtx_tag=nvtx_tag,
        )


class _FnTask(Task):
    """Internal: wraps a callable in a Task. Produced by `Task.from_fn`."""

    def __init__(self, fn: Callable, **kwargs) -> None:
        super().__init__(**kwargs)
        self._fn = fn

    def run(self, ctx) -> None:
        self._fn(ctx)
