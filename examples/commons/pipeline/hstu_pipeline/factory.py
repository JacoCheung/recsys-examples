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

"""Factory registry for HSTUPipeline variants."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .pipeline import HSTUPipeline

__all__ = ["HSTUPipelineFactory"]


def _make_sparse_dist(**kwargs: Any) -> HSTUPipeline:
    return HSTUPipeline(prefetch=False, **kwargs)


def _make_prefetch_sparse_dist(**kwargs: Any) -> HSTUPipeline:
    return HSTUPipeline(prefetch=True, **kwargs)


class HSTUPipelineFactory:
    """Registry for HSTU pipeline variants.

    Pre-registered at module import:

      ``"hstu_sparse_dist"``          — non-prefetch variant
      ``"hstu_prefetch_sparse_dist"`` — prefetch variant

    Extra custom variants can be registered via ``register()``.
    """

    _registry: Dict[str, Callable[..., HSTUPipeline]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., HSTUPipeline]) -> None:
        if name in cls._registry:
            raise ValueError(
                f"Pipeline '{name}' already registered in HSTUPipelineFactory"
            )
        cls._registry[name] = factory

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> HSTUPipeline:
        if name not in cls._registry:
            raise KeyError(
                f"Unknown HSTU pipeline '{name}'. Registered: "
                f"{sorted(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list(cls) -> list:
        return sorted(cls._registry.keys())


# Pre-registrations
HSTUPipelineFactory.register("hstu_sparse_dist", _make_sparse_dist)
HSTUPipelineFactory.register("hstu_prefetch_sparse_dist", _make_prefetch_sparse_dist)
