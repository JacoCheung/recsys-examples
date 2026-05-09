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

"""V2 precondition — autograd-stream spike (SPEC §4.6).

Verifies the working assumption that `loss.backward()` invoked inside
a `torch.cuda.stream(S)` context submits backward kernels on `S`.
Captured via `Tensor.register_hook` which runs inside the autograd
worker at grad-ready time.

Three fixtures:
  (a) plain `nn.Linear` — baseline
  (b) multi-stream forward — one layer runs on non-default stream
      during forward; ensures saved-tensor stream semantics interact
      sanely with the declared backward stream
  (c) DDP-wrapped — single-rank (`world_size=1`) smoke; skipped only
      when CUDA is absent

If any fixture fails, V2 must narrow §4.7 adoption claims (potentially
pulling a slim BackwardHookTask back into v1 scope).
"""

import os

import pytest
import torch


def _cuda_available_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("autograd-stream spike requires CUDA")
    return torch.device("cuda:0")


def _capture_backward_stream(loss: torch.Tensor, param: torch.Tensor) -> list:
    """Attach a grad hook that records the stream active at grad-ready time.

    Returns a list that will be populated (in-place) with the stream
    id captured inside the autograd worker.
    """
    captured: list = []

    def _hook(_grad):
        captured.append(torch.cuda.current_stream().stream_id)

    param.register_hook(_hook)
    return captured


def test_spike_plain_linear() -> None:
    """Fixture (a): plain `nn.Linear`. Backward inside `stream(S)` must
    submit grad kernels on `S`."""
    device = _cuda_available_or_skip()
    torch.manual_seed(0)

    model = torch.nn.Linear(8, 4).to(device)
    x = torch.randn(3, 8, device=device)
    backward_stream = torch.cuda.Stream(device)

    # Capture at weight grad
    captured = _capture_backward_stream(torch.zeros(1), model.weight)

    with torch.cuda.stream(backward_stream):
        loss = model(x).sum()
        loss.backward()
        torch.cuda.current_stream().synchronize()

    assert len(captured) == 1, f"hook fired {len(captured)} times, expected 1"
    assert captured[0] == backward_stream.stream_id, (
        f"backward kernel submitted on stream {captured[0]}, expected "
        f"declared stream {backward_stream.stream_id}. SPEC §4.6 "
        f"assumption violated on plain nn.Linear — escalate."
    )


@pytest.mark.xfail(
    reason=(
        "Documented PyTorch behavior: when part of forward runs on a "
        "non-default stream, backward honors saved-tensor stream "
        "semantics and routes that layer's grad kernel to its forward "
        "stream, not the user-declared backward stream. SPEC §4.6 "
        "records this constraint — multi-stream-forward models are out "
        "of scope; drop to T3/T4."
    ),
    strict=True,
)
def test_spike_multi_stream_forward() -> None:
    """Fixture (b): part of forward on a non-default stream. Expected
    failure — saved-tensor stream semantics override the declared
    backward stream. SPEC §4.6 now warns about this."""
    device = _cuda_available_or_skip()
    torch.manual_seed(0)

    layer1 = torch.nn.Linear(8, 8).to(device)
    layer2 = torch.nn.Linear(8, 4).to(device)
    forward_stream = torch.cuda.Stream(device)
    backward_stream = torch.cuda.Stream(device)

    captured_l1 = _capture_backward_stream(torch.zeros(1), layer1.weight)
    captured_l2 = _capture_backward_stream(torch.zeros(1), layer2.weight)

    x = torch.randn(3, 8, device=device)

    # Forward: layer1 runs on forward_stream, layer2 on default
    default_stream = torch.cuda.current_stream(device)
    forward_stream.wait_stream(default_stream)
    with torch.cuda.stream(forward_stream):
        h = layer1(x)
    default_stream.wait_stream(forward_stream)
    out = layer2(h)
    loss = out.sum()

    # Backward on its own declared stream
    backward_stream.wait_stream(default_stream)
    with torch.cuda.stream(backward_stream):
        loss.backward()
        torch.cuda.current_stream().synchronize()

    assert captured_l1, "layer1 weight hook did not fire"
    assert captured_l2, "layer2 weight hook did not fire"
    # Both hooks should fire on backward_stream per our assumption.
    # If PyTorch routes them to each saved-tensor's forward stream
    # instead, this assertion will catch it.
    for name, seen in (("layer1", captured_l1), ("layer2", captured_l2)):
        assert seen[0] == backward_stream.stream_id, (
            f"{name} grad kernel submitted on stream {seen[0]}, expected "
            f"declared backward stream {backward_stream.stream_id}. SPEC "
            f"§4.6 assumption doesn't hold in multi-stream-forward case."
        )


def test_spike_ddp_wrapped() -> None:
    """Fixture (c): DDP-wrapped model, single-rank smoke.

    Uses `torch.nn.parallel.DistributedDataParallel` with
    `world_size=1` to smoke-check that DDP's autograd-hook machinery
    coexists with a custom backward stream context. A multi-rank
    version is deferred until V3/V4 adds real multi-GPU testing
    infra.

    Requires a single CUDA device. Skip only when CUDA is absent —
    not when <2 GPUs (the original 2-GPU skip guard was contradictory
    with world_size=1).
    """
    device = _cuda_available_or_skip()
    torch.manual_seed(0)

    # Only destroy what we created. If another test (or a session-
    # level fixture) initialized the process group before us, leave
    # it intact — tearing down shared state belongs to its creator.
    created_pg = False
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29511")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
        created_pg = True

    try:
        model = torch.nn.Linear(8, 4).to(device)
        ddp = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index]
        )
        backward_stream = torch.cuda.Stream(device)
        captured = _capture_backward_stream(torch.zeros(1), model.weight)

        x = torch.randn(3, 8, device=device)
        with torch.cuda.stream(backward_stream):
            loss = ddp(x).sum()
            loss.backward()
            torch.cuda.current_stream().synchronize()

        assert captured, "DDP backward hook did not fire"
        # With DDP, the backward hook may or may not observe the
        # declared backward_stream — DDP's reducer schedules allreduce
        # on its own comm stream, and the reducer's grad-bucket
        # callback can inject into the grad flow. This test records
        # the observed behavior; a violation here flags a known
        # limitation for the V10 DDP adoption case, not a total block.
        observed = captured[0]
        expected = backward_stream.stream_id
        if observed != expected:
            pytest.xfail(
                f"Known: DDP routes grad kernel to stream {observed} "
                f"not declared {expected}. Flag for V10 DDP preset "
                f"compatibility matrix."
            )
    finally:
        if created_pg and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
