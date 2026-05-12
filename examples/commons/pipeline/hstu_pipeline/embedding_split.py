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

"""HSTU embedding-output distribution helpers.

The engine runs ``compute_and_output_dist`` as a task before forward.
Forward wrappers in this module then consume the pre-populated awaitable
from the per-batch TorchRec context.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import torch
from commons.pipeline.utils import (
    PipelinedForward,
    PrefetchPipelinedForward,
    PrefetchTrainPipelineContext,
)

__all__ = [
    "HSTUTrainPipelineContext",
    "HSTUPipelinedForward",
    "HSTUPrefetchPipelinedForward",
    "_compute_and_output_dist_for_module",
]


@dataclass
class HSTUTrainPipelineContext(PrefetchTrainPipelineContext):
    """Context type for HSTU pipeline. Inherits everything
    PrefetchTrainPipelineContext provides (``input_dist_*_requests``,
    ``module_contexts``, ``module_input_post_prefetch``, ...) and adds
    ``embedding_a2a_requests`` so the new ``compute_output_dist`` task
    can stash the output a2a awaitable for the forward task to pick
    up.

    Used by both prefetch and non-prefetch HSTU paths — non-prefetch
    simply leaves ``module_input_post_prefetch`` empty.
    """

    embedding_a2a_requests: Dict[str, Any] = field(default_factory=dict)


class _HSTUPipelinedForwardMixin:
    """Shared ``__call__`` body for HSTU forward wrappers.

    Both ``HSTUPipelinedForward`` (non-prefetch) and
    ``HSTUPrefetchPipelinedForward`` (prefetch) need the same runtime
    behavior — read ``embedding_a2a_requests`` (pre-populated by the
    engine's ``compute_output_dist`` task) and return the awaitable;
    no NCCL submitted here. They differ only in which TorchRec wrapper
    they extend (so they pass the right ``isinstance`` type asserts in
    ``commons/pipeline/utils.py`` — ``_start_data_dist`` accepts any
    of the three stock wrappers, but ``_prefetch_embeddings`` requires
    ``PrefetchPipelinedForward`` specifically).

    Mixin order in the concrete classes places this BEFORE the
    TorchRec base, so ``__call__`` here overrides the inherited
    NCCL-submitting one.
    """

    # pyre-ignore [2, 24]
    def __call__(self, *input, **kwargs):
        assert self._name in self._context.embedding_a2a_requests, (
            f"HSTU forward wrapper: embedding_a2a_requests[{self._name!r}] "
            f"not populated. The engine's compute_output_dist task must "
            f"have run before this forward call."
        )
        awaitable = self._context.embedding_a2a_requests.pop(self._name)
        # Preserve the stock wrapper's producer-stream wait.
        if self._stream is not None:
            torch.get_device_module(self._device).current_stream().wait_stream(
                self._stream
            )
        return awaitable


class HSTUPipelinedForward(_HSTUPipelinedForwardMixin, PipelinedForward):
    """Non-prefetch HSTU forward wrapper. Subclasses ``PipelinedForward``
    so the engine's ``_start_data_dist`` bootstrap accepts it via the
    ``isinstance(forward, (PipelinedForward, ...))`` check.
    """


class HSTUPrefetchPipelinedForward(
    _HSTUPipelinedForwardMixin, PrefetchPipelinedForward
):
    """Prefetch HSTU forward wrapper. Subclasses ``PrefetchPipelinedForward``
    so both ``_start_data_dist`` (any of 3 wrappers) and
    ``_prefetch_embeddings`` (specifically ``PrefetchPipelinedForward``)
    type asserts pass. Constructor signature inherits the prefetch_stream
    kwarg expected by TorchRec's ``_rewrite_model`` in prefetch mode.
    """


def _compute_and_output_dist_for_module(
    module: Any,
    context: HSTUTrainPipelineContext,
    *,
    is_prefetch: bool,
) -> None:
    """Run ``module.compute_and_output_dist(ctx, data)`` and stash the
    returned awaitable into ``context.embedding_a2a_requests``.

    Parameters
    ----------
    module
        A TorchRec ``ShardedModule`` (the original sharded embedding
        module, NOT the wrapped ``HSTUPipelinedForward``).
    context
        Per-batch ``HSTUTrainPipelineContext``.
    is_prefetch
        True when the engine's ``prefetch_embeddings`` task ran
        upstream — pull data + module_ctx from
        ``module_input_post_prefetch`` / ``module_contexts_post_prefetch``.
        False otherwise — wait on
        ``input_dist_tensors_requests[name]`` and pop from
        ``module_contexts``.

    Drain-safe: if upstream dict has no entry for this module's name
    (drain phase, warmup, or upstream task skipped), the function
    silently returns. The forward task's own assert will catch genuine
    misconfiguration.
    """
    name = module.forward.name
    if is_prefetch:
        if (
            name not in context.module_input_post_prefetch
            or name not in context.module_contexts_post_prefetch
        ):
            return
        data = context.module_input_post_prefetch.pop(name)
        ctx = context.module_contexts_post_prefetch.pop(name)
    else:
        if (
            name not in context.input_dist_tensors_requests
            or name not in context.module_contexts
        ):
            return
        request = context.input_dist_tensors_requests.pop(name)
        data = request.wait()
        ctx = context.module_contexts.pop(name)

    awaitable = module.compute_and_output_dist(ctx, data)
    context.embedding_a2a_requests[name] = awaitable
