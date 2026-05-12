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

"""Named CUDA-stream registry.

Maps user-declared stream names (`"default"`, `"memcpy"`, `"comm"`, ...)
to `torch.cuda.Stream` objects. Task bodies reach streams through
`ctx.stream_pool.use(name)` rather than holding Stream objects directly
— this keeps schedules serialization-friendly.
"""

from contextlib import contextmanager
from typing import Iterator, Mapping, Optional, Tuple

import torch


class StreamPool:
    """Named stream registry.

    ``get(name)`` returns the raw registered value. ``use(name)`` is the
    runtime context and resolves ``None`` to the anchor device's default
    CUDA stream when CUDA is available.
    """

    def __init__(self, streams: Mapping[str, Optional[torch.cuda.Stream]]) -> None:
        if "default" not in streams:
            raise ValueError(
                "StreamPool must declare a 'default' stream slot "
                "(even if mapped to None). Got streams="
                f"{list(streams.keys())}"
            )
        self._streams: dict = dict(streams)

        # Anchor device for `None → default_stream(device)` resolution
        # at wait-insertion time. Picks the device of the first
        # concrete stream in the pool; falls back to the current CUDA
        # device if no concrete streams are declared and CUDA is
        # available. `None` on pure CPU hosts.
        self._anchor_device: Optional[torch.device] = None
        for stream in self._streams.values():
            if stream is not None:
                self._anchor_device = stream.device
                break
        if self._anchor_device is None and torch.cuda.is_available():
            self._anchor_device = torch.device(f"cuda:{torch.cuda.current_device()}")

    @property
    def anchor_device(self) -> Optional[torch.device]:
        """Device used to anchor `None → default_stream(device)`
        resolution. Callers that need a concrete stream for a `None`
        slot should use `torch.cuda.default_stream(anchor_device)`.
        `None` only on pure-CPU hosts."""
        return self._anchor_device

    def get(self, name: str) -> Optional[torch.cuda.Stream]:
        """Return the stream slot's raw value — either a
        `torch.cuda.Stream` or `None`. Does NOT resolve `None` to
        `default_stream()`; that's the caller's decision based on
        whether it needs a concrete stream (e.g. `wait_stream`) or is
        fine with a no-op (e.g. `torch.cuda.stream(None)`)."""
        if name not in self._streams:
            raise KeyError(
                f"Stream slot '{name}' not declared in pool. "
                f"Declared slots: {list(self._streams.keys())}."
            )
        return self._streams[name]

    @contextmanager
    def use(self, name: str) -> Iterator[None]:
        """Set the named stream as current for the task body."""
        stream = self.get(name)
        if stream is None and self._anchor_device is not None:
            stream = torch.cuda.default_stream(self._anchor_device)
        with torch.cuda.stream(stream):
            yield

    def names(self) -> Tuple[str, ...]:
        return tuple(self._streams.keys())
