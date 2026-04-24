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

"""Module-private preset components for `SchedulablePipeline.basic(...)`.

These 5 makers are **not** re-exported from `engine/__init__.py`.
Users who need to compose Tasks directly (T3/T4 adoption) write their
own forward/loss/backward Tasks — the preset API is one function
(`SchedulablePipeline.basic`), not a library of building blocks.

v1 scope (V2): single-stream, no prefetch. `prefetch=True` and
`memcpy_stream=True` branches raise `NotImplementedError` until V4.
"""

from typing import Any, Callable, Optional

import torch

from .task import DataSlot, Task

__all__ = [
    "_make_h2d_task",
    "_make_zero_grad_task",
    "_make_forward_task",
    "_make_backward_task",
    "_make_optimizer_task",
    "_default_loss_extractor",
]


def _to_device(batch: Any, device: torch.device) -> Any:
    """Recursively `.to(device, non_blocking=True)` for tensors in
    common container shapes (tensor / tuple / list / dict). Non-tensor
    leaves pass through unchanged."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, tuple):
        return tuple(_to_device(b, device) for b in batch)
    if isinstance(batch, list):
        return [_to_device(b, device) for b in batch]
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch


def _default_loss_extractor(step_result: Any) -> torch.Tensor:
    """Extract scalar loss from common model-return shapes.

    - `Tensor` → the tensor itself
    - `tuple` / `list` → first element (must be a Tensor)
    - `dict` → the `"loss"` key (must be a Tensor)
    Otherwise raise; user overrides via `loss_fn` kwarg.
    """
    if isinstance(step_result, torch.Tensor):
        return step_result
    if isinstance(step_result, (tuple, list)):
        if len(step_result) == 0:
            raise ValueError(
                f"model returned an empty "
                f"{type(step_result).__name__}; cannot extract loss. "
                f"Pass loss_fn= to customize."
            )
        loss = step_result[0]
        if not isinstance(loss, torch.Tensor):
            raise TypeError(
                f"model return's first element is "
                f"{type(loss).__name__}, not Tensor. "
                f"Pass loss_fn= to customize extraction."
            )
        return loss
    if isinstance(step_result, dict):
        if "loss" not in step_result:
            raise KeyError(
                f"model returned a dict without a 'loss' key; got keys "
                f"{list(step_result.keys())}. Pass loss_fn= to customize."
            )
        loss = step_result["loss"]
        if not isinstance(loss, torch.Tensor):
            raise TypeError(
                f"model return's 'loss' value is "
                f"{type(loss).__name__}, not Tensor. "
                f"Pass loss_fn= to customize extraction."
            )
        return loss
    raise TypeError(
        f"Cannot extract loss from {type(step_result).__name__}; "
        f"pass loss_fn= to customize."
    )


def _make_h2d_task(
    device: torch.device,
    *,
    stream: str = "default",
    batch_offset: int = 0,
) -> Task:
    """Reads `'batch_cpu'` (engine-populated), writes `'batch_gpu'`
    moved to `device` non-blocking."""

    def _fn(ctx) -> None:
        batch_gpu = _to_device(ctx.slots["batch_cpu"], device)
        ctx.slots.set("batch_gpu", batch_gpu)

    return Task.from_fn(
        name="h2d",
        fn=_fn,
        reads=(DataSlot("batch_cpu", batch_offset=batch_offset),),
        writes=(DataSlot("batch_gpu", batch_offset=batch_offset),),
        stream=stream,
        batch_offset=batch_offset,
    )


def _make_zero_grad_task(
    optimizer: torch.optim.Optimizer,
    *,
    stream: str = "default",
) -> Task:
    """`optimizer.zero_grad(set_to_none=True)`. No slots; runs first
    via declaration order within its stage."""

    def _fn(ctx) -> None:
        optimizer.zero_grad(set_to_none=True)

    return Task.from_fn(name="zero_grad", fn=_fn, stream=stream)


def _make_forward_task(
    model: torch.nn.Module,
    *,
    forward_fn: Optional[Callable[[torch.nn.Module, Any], Any]] = None,
    loss_fn: Optional[Callable[[Any], torch.Tensor]] = None,
    stream: str = "default",
) -> Task:
    """Reads `'batch_gpu'`, runs `forward_fn(model, batch_gpu)` (defaults
    to `model(batch_gpu)`), writes `'loss'` (scalar via `loss_fn` /
    auto-extractor) and `'step_result'` (verbatim return, passed to
    `pipe.progress()` caller)."""

    _forward_fn = forward_fn if forward_fn is not None else (lambda m, b: m(b))
    _loss_fn = loss_fn if loss_fn is not None else _default_loss_extractor

    def _fn(ctx) -> None:
        result = _forward_fn(model, ctx.slots["batch_gpu"])
        loss = _loss_fn(result)
        ctx.slots.set("loss", loss)
        ctx.slots.set("step_result", result)

    return Task.from_fn(
        name="forward",
        fn=_fn,
        reads=(DataSlot("batch_gpu"),),
        writes=(DataSlot("loss"), DataSlot("step_result")),
        depends_on=("zero_grad",),
        stream=stream,
    )


def _make_backward_task(
    *,
    backward_fn: Optional[Callable[[torch.Tensor], None]] = None,
    stream: str = "default",
) -> Task:
    """Reads `'loss'`, calls `backward_fn(loss)` (defaults to
    `loss.backward()`). Writes no slot — pure ordering source for the
    optimizer task via `depends_on=("backward",)`."""

    _backward_fn = (
        backward_fn if backward_fn is not None else (lambda loss: loss.backward())
    )

    def _fn(ctx) -> None:
        _backward_fn(ctx.slots["loss"])

    return Task.from_fn(
        name="backward",
        fn=_fn,
        reads=(DataSlot("loss"),),
        stream=stream,
    )


def _make_optimizer_task(
    optimizer: torch.optim.Optimizer,
    *,
    optimizer_step_fn: Optional[Callable[[], None]] = None,
    stream: str = "default",
) -> Task:
    """Runs `optimizer_step_fn()` (defaults to `optimizer.step`).
    `depends_on=('backward',)` encodes the pure-ordering edge (SPEC
    §4.2 rule 6)."""

    _step_fn = optimizer_step_fn if optimizer_step_fn is not None else optimizer.step

    def _fn(ctx) -> None:
        _step_fn()

    return Task.from_fn(
        name="optimizer",
        fn=_fn,
        depends_on=("backward",),
        stream=stream,
    )
