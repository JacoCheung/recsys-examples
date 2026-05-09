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

"""Auto-scheduler package — V5 validator, V9 cost model + list scheduler."""

from .cost_model import CostModel, CostProfiler, TaskCost
from .fire_order import (
    DEFAULT_BIT_EXACT_TASKS,
    TaskResource,
    auto_assign_lookaheads,
    compute_overlap_matrix,
    default_stream_critical_path_us,
    describe_overlap_matrix,
    task_resources,
)
from .list_scheduler import schedule_tasks
from .validator import ScheduleValidationError, validate

__all__ = [
    "CostModel",
    "CostProfiler",
    "DEFAULT_BIT_EXACT_TASKS",
    "ScheduleValidationError",
    "TaskCost",
    "TaskResource",
    "auto_assign_lookaheads",
    "compute_overlap_matrix",
    "default_stream_critical_path_us",
    "describe_overlap_matrix",
    "schedule_tasks",
    "task_resources",
    "validate",
]
