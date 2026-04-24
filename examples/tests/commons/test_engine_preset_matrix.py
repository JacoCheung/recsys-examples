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

"""V10 — preset compatibility matrix.

SPEC §4.7 T2 claims the `forward_fn` / `loss_fn` / `backward_fn` /
`optimizer_step_fn` escape kwargs cover the 4 canonical realistic
training-loop shapes:

  (a) vanilla                              — ≤ 8-line diff
  (b) AMP + GradScaler                     — ≤ 15-line diff
  (c) gradient clipping                    — ≤ 15-line diff
  (d) LR scheduler `.step()` after optim   — ≤ 15-line diff

Each scenario runs the engine version AND a hand-written non-engine
reference on the same seed + data; final params must match within
`atol=1e-5`. If any scenario can't hit its bar, the README
explicitly moves it to T3/T4 — the test is the gate.
"""

from typing import List

import pytest
import torch
from commons.pipeline.engine import SchedulablePipeline

_STEPS = 20
_BATCH = 8
_IN = 16
_OUT = 4
_SEED = 42


def _seeded_init(device: torch.device, seed: int = _SEED):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = torch.nn.Linear(_IN, _OUT).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    return model, opt


def _make_batches(device: torch.device, seed: int = _SEED + 1):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return [
        torch.randn(_BATCH, _IN, device=device, generator=gen) for _ in range(_STEPS)
    ]


def _compare_params(
    model_ref: torch.nn.Module,
    model_eng: torch.nn.Module,
    scenario: str,
) -> None:
    for (_, p_ref), (_, p_eng) in zip(
        model_ref.named_parameters(), model_eng.named_parameters()
    ):
        assert torch.allclose(p_ref, p_eng, atol=1e-5, rtol=0), (
            f"[{scenario}] engine params diverged from non-engine "
            f"reference after {_STEPS} steps. "
            f"max_abs_diff={(p_ref - p_eng).abs().max().item():.6g}"
        )


def _drive_engine(pipe: SchedulablePipeline, batches: List[torch.Tensor]) -> None:
    it = iter(batches)
    while True:
        try:
            pipe.progress(it)
        except StopIteration:
            break


# ----------------------------------------------------------------------
# (a) vanilla
# ----------------------------------------------------------------------


def test_preset_matrix_a_vanilla() -> None:
    """Vanilla `model(batch) → loss → backward → step`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reference
    ref_model, ref_opt = _seeded_init(device)
    ref_batches = _make_batches(device)
    for b in ref_batches:
        ref_opt.zero_grad(set_to_none=True)
        loss = ref_model(b).sum()
        loss.backward()
        ref_opt.step()

    # Engine
    eng_model, eng_opt = _seeded_init(device)
    pipe = SchedulablePipeline.basic(eng_model, eng_opt, loss_fn=lambda out: out.sum())
    _drive_engine(pipe, _make_batches(device))

    _compare_params(ref_model, eng_model, "a-vanilla")


# ----------------------------------------------------------------------
# (b) AMP + GradScaler
# ----------------------------------------------------------------------


def test_preset_matrix_b_amp_gradscaler() -> None:
    """AMP autocast on forward, GradScaler wrapping backward, and
    `scaler.step` + `scaler.update` replacing `optimizer.step`."""
    if not torch.cuda.is_available():
        pytest.skip("AMP+GradScaler path tested only on CUDA")
    device = torch.device("cuda")

    from torch.amp import GradScaler, autocast

    # Reference
    ref_model, ref_opt = _seeded_init(device)
    ref_scaler = GradScaler("cuda")
    ref_batches = _make_batches(device)
    for b in ref_batches:
        ref_opt.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=torch.float16):
            out = ref_model(b)
            loss = out.sum()
        ref_scaler.scale(loss).backward()
        ref_scaler.step(ref_opt)
        ref_scaler.update()

    # Engine
    eng_model, eng_opt = _seeded_init(device)
    eng_scaler = GradScaler("cuda")

    def _forward_fn(model, batch):
        with autocast("cuda", dtype=torch.float16):
            return model(batch)

    def _backward_fn(loss):
        eng_scaler.scale(loss).backward()

    def _optimizer_step_fn():
        eng_scaler.step(eng_opt)
        eng_scaler.update()

    pipe = SchedulablePipeline.basic(
        eng_model,
        eng_opt,
        loss_fn=lambda out: out.sum(),
        forward_fn=_forward_fn,
        backward_fn=_backward_fn,
        optimizer_step_fn=_optimizer_step_fn,
    )
    _drive_engine(pipe, _make_batches(device))

    _compare_params(ref_model, eng_model, "b-amp-gradscaler")


# ----------------------------------------------------------------------
# (c) gradient clipping
# ----------------------------------------------------------------------


def test_preset_matrix_c_gradient_clipping() -> None:
    """`clip_grad_norm_` between backward and optimizer.step, via
    `optimizer_step_fn` escape kwarg."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MAX_NORM = 1.0

    # Reference
    ref_model, ref_opt = _seeded_init(device)
    ref_batches = _make_batches(device)
    for b in ref_batches:
        ref_opt.zero_grad(set_to_none=True)
        loss = ref_model(b).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ref_model.parameters(), max_norm=MAX_NORM)
        ref_opt.step()

    # Engine
    eng_model, eng_opt = _seeded_init(device)

    def _optimizer_step_fn():
        torch.nn.utils.clip_grad_norm_(eng_model.parameters(), max_norm=MAX_NORM)
        eng_opt.step()

    pipe = SchedulablePipeline.basic(
        eng_model,
        eng_opt,
        loss_fn=lambda out: out.sum(),
        optimizer_step_fn=_optimizer_step_fn,
    )
    _drive_engine(pipe, _make_batches(device))

    _compare_params(ref_model, eng_model, "c-gradient-clipping")


# ----------------------------------------------------------------------
# (d) LR scheduler
# ----------------------------------------------------------------------


def test_preset_matrix_d_lr_scheduler() -> None:
    """`scheduler.step()` immediately after `optimizer.step()`, via
    `optimizer_step_fn`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reference
    ref_model, ref_opt = _seeded_init(device)
    ref_scheduler = torch.optim.lr_scheduler.StepLR(ref_opt, step_size=5, gamma=0.5)
    ref_batches = _make_batches(device)
    for b in ref_batches:
        ref_opt.zero_grad(set_to_none=True)
        loss = ref_model(b).sum()
        loss.backward()
        ref_opt.step()
        ref_scheduler.step()

    # Engine
    eng_model, eng_opt = _seeded_init(device)
    eng_scheduler = torch.optim.lr_scheduler.StepLR(eng_opt, step_size=5, gamma=0.5)

    def _optimizer_step_fn():
        eng_opt.step()
        eng_scheduler.step()

    pipe = SchedulablePipeline.basic(
        eng_model,
        eng_opt,
        loss_fn=lambda out: out.sum(),
        optimizer_step_fn=_optimizer_step_fn,
    )
    _drive_engine(pipe, _make_batches(device))

    _compare_params(ref_model, eng_model, "d-lr-scheduler")

    # Also verify the scheduler advanced the same number of times.
    assert eng_scheduler.last_epoch == ref_scheduler.last_epoch, (
        f"LR scheduler `last_epoch` desync: engine={eng_scheduler.last_epoch}, "
        f"ref={ref_scheduler.last_epoch}"
    )
