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

"""Schedule + Stage — the declarative plan consumed by
`SchedulablePipeline`.

A `Schedule` is a tuple of `Stage`s plus a declared stream inventory.
Each `Stage` groups Tasks that share a dependency frontier for
visual/organizational purposes; cross-stream waits are inserted by the
engine from slot/`depends_on` edges regardless of stage boundaries
(SPEC §4.2). `in_flight_batches` is a derived property — never
authored.
"""

from dataclasses import dataclass
from typing import Tuple

from .task import Task

__all__ = ["Stage", "Schedule"]


@dataclass(frozen=True)
class Stage:
    """Ordered group of Tasks. Within-stage declaration order = CPU
    submission order for same-stream tasks."""

    tasks: Tuple[Task, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tasks, tuple):
            # dataclass allows accidental lists; normalize-or-fail.
            raise TypeError(
                f"Stage.tasks must be a tuple, got {type(self.tasks).__name__}. "
                f"Use Stage(tasks=(task_a, task_b, ...))."
            )


@dataclass(frozen=True)
class Schedule:
    """Full pipeline plan.

    Fields:
        stages: ordered tuple of Stages.
        stream_slots: tuple of stream names that the Schedule's Tasks
            may reference. Must contain every `task.stream` across all
            tasks. Validator enforces.

    `in_flight_batches` is computed from `max(task.batch_offset) + 1`
    across all tasks (SPEC §4.2 rule 4). Never authored.
    """

    stages: Tuple[Stage, ...]
    stream_slots: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stages, tuple):
            raise TypeError(
                f"Schedule.stages must be a tuple of Stage, got "
                f"{type(self.stages).__name__}."
            )
        if not isinstance(self.stream_slots, tuple):
            raise TypeError(
                f"Schedule.stream_slots must be a tuple of str, got "
                f"{type(self.stream_slots).__name__}."
            )
        if not self.stream_slots:
            raise ValueError(
                "Schedule.stream_slots must declare at least one stream "
                "(typically 'default')."
            )

    @property
    def in_flight_batches(self) -> int:
        """Derived from task batch_offsets (SPEC §4.2 rule 4)."""
        offsets = [task.batch_offset for stage in self.stages for task in stage.tasks]
        return max(offsets, default=0) + 1

    def all_tasks(self) -> Tuple[Task, ...]:
        """Flat iteration over every task in declaration order."""
        return tuple(task for stage in self.stages for task in stage.tasks)
