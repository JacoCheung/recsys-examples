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

"""HSTU pipeline adapter — Problem #2.

Ports the HSTU-specific training pipeline onto the Problem #1
schedulable engine without touching the legacy pipeline files.

Public API:
    HSTUPipeline — the adapter class with `progress()` matching legacy.
    HSTUPipelineFactory — registry for named pipeline variants.
"""

from .factory import HSTUPipelineFactory
from .pipeline import (
    HSTU_DEFAULT_THREAD_MAP,
    HSTU_THREAD_MAP_PRESETS,
    HSTUPipeline,
    resolve_hstu_thread_map_variant,
)

__all__ = [
    "HSTUPipeline",
    "HSTUPipelineFactory",
    "HSTU_DEFAULT_THREAD_MAP",
    "HSTU_THREAD_MAP_PRESETS",
    "resolve_hstu_thread_map_variant",
]
