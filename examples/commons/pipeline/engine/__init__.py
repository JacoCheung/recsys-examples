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

"""Schedulable pipeline engine (Problem #1).

Framework-agnostic: this package must never import torchrec, megatron,
fbgemm_gpu, or commons.distributed.* — enforced by
examples/tests/commons/test_engine_import_hygiene.py.
"""

from .autosched import ScheduleValidationError
from .context import TaskContext
from .executor import SequentialExecutor, ThreadedExecutor
from .pipeline import SchedulablePipeline
from .schedule import Schedule, Stage
from .streams import StreamPool
from .task import DataSlot, Task

__all__ = [
    "DataSlot",
    "Schedule",
    "SchedulablePipeline",
    "ScheduleValidationError",
    "SequentialExecutor",
    "Stage",
    "StreamPool",
    "Task",
    "TaskContext",
    "ThreadedExecutor",
]
