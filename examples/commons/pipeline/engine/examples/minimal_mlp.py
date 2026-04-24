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

"""T2 adoption demo — preset + prefetch overlap.

Single-GPU MLP training with:
  - H2D on `memcpy` stream (batch N+1's upload overlaps batch N's
    forward/backward/optimizer on `default` stream)
  - `SchedulablePipeline.basic(prefetch=True, memcpy_stream=True)`

Diff vs vanilla loop: still ≤ 8 lines (same as T1 — prefetch is a
kwarg flip, no loop body change). The overlap happens inside the
engine's BatchRing + cross-stream wait_stream auto-insertion.

Run:
    python examples/commons/pipeline/engine/examples/minimal_mlp.py

On CUDA hosts you should see wall-clock improvement over the T1
(non-prefetch) path for large-enough batches. On CPU, this degrades
to a single-stream loop (no-op equivalence — still correct).
"""

import time
from typing import List

import torch
from commons.pipeline.engine import SchedulablePipeline

_BATCH = 64
_IN = 256
_OUT = 32
_STEPS = 50


def _make_data(device: torch.device) -> List[torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    return [
        torch.randn(_BATCH, _IN, device=device, generator=gen) for _ in range(_STEPS)
    ]


def _run(pipe: SchedulablePipeline, data: List[torch.Tensor]) -> float:
    """Drive pipe through data; return wall-clock seconds."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    it = iter(data)
    while True:
        try:
            pipe.progress(it)
        except StopIteration:
            break
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.perf_counter() - t0


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- T1: vanilla preset ---
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    model = torch.nn.Linear(_IN, _OUT).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    pipe_t1 = SchedulablePipeline.basic(model, opt, loss_fn=lambda out: out.sum())

    # Move data off-device to make H2D meaningful for T2.
    cpu_data = [b.cpu() for b in _make_data(device)]
    t1_secs = _run(pipe_t1, [b.to(device) for b in cpu_data])
    print(f"T1 (no prefetch):    {t1_secs * 1000:.2f} ms for {_STEPS} steps")

    # --- T2: preset + prefetch + memcpy stream ---
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    model2 = torch.nn.Linear(_IN, _OUT).to(device)
    opt2 = torch.optim.SGD(model2.parameters(), lr=1e-3)
    pipe_t2 = SchedulablePipeline.basic(
        model2,
        opt2,
        loss_fn=lambda out: out.sum(),
        prefetch=True,
        memcpy_stream=(device.type == "cuda"),
    )

    # T2 consumes CPU tensors — the engine's h2d task moves them
    # to GPU on `memcpy` stream.
    t2_secs = _run(pipe_t2, cpu_data)
    print(f"T2 (prefetch+memcpy): {t2_secs * 1000:.2f} ms for {_STEPS} steps")

    # Numerical parity sanity: final params should match under
    # identical init + data (H2D overlap doesn't change math).
    for (_, p1), (_, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.allclose(
            p1, p2, atol=1e-5, rtol=0
        ), "T1 vs T2 param divergence — prefetch broke math"
    print("T1 vs T2 numerical parity confirmed (atol=1e-5)")


if __name__ == "__main__":
    main()
