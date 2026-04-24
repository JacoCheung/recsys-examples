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

"""V6 — Determinism harness.

SPEC §10 acceptance: "Same schedule + same seed on two runs →
bit-identical loss every step". This is a regression harness on top
of V1-V4: no new engine code, just asserts two full runs under
identical conditions produce byte-equal tensors.

Uses `torch.equal` (bit-identical), not `torch.allclose`. Covers:
  - V1/V2 single-stream train loop via `SchedulablePipeline.basic`
  - V3 two-stream hand-written schedule
  - V4 prefetch (in_flight_batches=2)
  - Per-parameter grad byte equality after one backward
"""

import os
from typing import Callable, List, Tuple

import torch

# CUDA deterministic-algorithms flag. Without this, some kernels
# (reductions, scatter/index ops) can be non-deterministic even
# with full seed control. This module-level setup gives the
# `torch.equal` assertions real teeth on CUDA.
#
# `warn_only=True`: if a specific op lacks a deterministic
# implementation, warn instead of hard-fail — the tests in this
# file don't exercise such ops, but leaves room for future
# additions without silent breakage.
#
# `CUBLAS_WORKSPACE_CONFIG=:4096:8` required by torch docs for
# deterministic cuBLAS on CUDA ≥ 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)

from commons.pipeline.engine import (
    DataSlot,
    SchedulablePipeline,
    Schedule,
    Stage,
    StreamPool,
    Task,
)

_IN = 8
_OUT = 4
_BATCH = 4
_STEPS = 20
_SEED = 42


def _set_all_seeds(seed: int) -> None:
    """Set every RNG that could leak entropy into this test."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_model_opt_batches(
    device: torch.device,
    seed: int = _SEED,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, List[torch.Tensor]]:
    """Identical initialization given the same seed + device."""
    _set_all_seeds(seed)
    model = torch.nn.Linear(_IN, _OUT).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 1)
    batches = [
        torch.randn(_BATCH, _IN, device=device, generator=gen) for _ in range(_STEPS)
    ]
    return model, opt, batches


def _params_snapshot(model: torch.nn.Module) -> List[torch.Tensor]:
    """Return a list of parameter tensors detached + cloned (so they
    can be compared after further training mutates in place)."""
    return [p.detach().clone() for p in model.parameters()]


def _grads_snapshot(model: torch.nn.Module) -> List[torch.Tensor]:
    """Snapshot `.grad` for each parameter (None → zero tensor to
    avoid comparison asymmetry)."""
    return [
        (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for p in model.parameters()
    ]


def _run_train_loop(
    make_pipe: Callable[[torch.nn.Module, torch.optim.Optimizer], SchedulablePipeline],
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Drive `_STEPS` iterations through a freshly-constructed pipe.
    Returns (per-step step_result snapshots, final parameter snapshot)."""
    model, opt, batches = _make_model_opt_batches(device)
    pipe = make_pipe(model, opt)
    step_results: List[torch.Tensor] = []
    it = iter(batches)
    while True:
        try:
            r = pipe.progress(it)
        except StopIteration:
            break
        step_results.append(r.detach().clone())
    return step_results, _params_snapshot(model)


# ----------------------------------------------------------------------
# V1/V2 — single-stream, `SchedulablePipeline.basic`
# ----------------------------------------------------------------------


def test_determinism_basic_single_stream() -> None:
    """Two runs via `SchedulablePipeline.basic(...)` with identical
    seed/data/model → bit-identical step_results and final params."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make(model, opt):
        return SchedulablePipeline.basic(model, opt, loss_fn=lambda o: o.sum())

    results_a, params_a = _run_train_loop(_make, device)
    results_b, params_b = _run_train_loop(_make, device)

    assert len(results_a) == len(results_b) == _STEPS
    for i, (a, b) in enumerate(zip(results_a, results_b)):
        assert torch.equal(a, b), (
            f"step {i}: step_result diverged between runs. "
            f"Non-determinism leaked somewhere in the pipeline."
        )
    for i, (pa, pb) in enumerate(zip(params_a, params_b)):
        assert torch.equal(pa, pb), f"Parameter {i} diverged after {_STEPS} steps."


# ----------------------------------------------------------------------
# V3 — two-stream hand-written schedule
# ----------------------------------------------------------------------


def _make_two_stream_pipe(
    model: torch.nn.Module, opt: torch.optim.Optimizer
) -> SchedulablePipeline:
    """H2D on `memcpy`, compute on `default`. Same math, different
    stream placement."""
    device = next(model.parameters()).device

    def _h2d(ctx):
        ctx.slots.set("batch_gpu", ctx.slots["batch_cpu"].to(device))

    def _forward(ctx):
        out = model(ctx.slots["batch_gpu"])
        ctx.slots.set("loss", out.sum())
        ctx.slots.set("step_result", out)

    def _backward(ctx):
        ctx.slots["loss"].backward()

    def _opt_step(ctx):
        opt.step()

    def _zero_grad(ctx):
        opt.zero_grad(set_to_none=True)

    tasks = (
        Task.from_fn(
            name="h2d",
            fn=_h2d,
            reads=(DataSlot("batch_cpu"),),
            writes=(DataSlot("batch_gpu"),),
            stream="memcpy" if device.type == "cuda" else "default",
        ),
        Task.from_fn(name="zero_grad", fn=_zero_grad, stream="default"),
        Task.from_fn(
            name="forward",
            fn=_forward,
            reads=(DataSlot("batch_gpu"),),
            writes=(DataSlot("loss"), DataSlot("step_result")),
            depends_on=("zero_grad",),
            stream="default",
        ),
        Task.from_fn(
            name="backward",
            fn=_backward,
            reads=(DataSlot("loss"),),
            stream="default",
        ),
        Task.from_fn(
            name="optimizer",
            fn=_opt_step,
            depends_on=("backward",),
            stream="default",
        ),
    )
    if device.type == "cuda":
        pool = StreamPool(
            {
                "default": torch.cuda.default_stream(device),
                "memcpy": torch.cuda.Stream(device),
            }
        )
        slots: tuple[str, ...] = ("default", "memcpy")
    else:
        pool = StreamPool({"default": None})
        slots = ("default",)
    schedule = Schedule(stages=(Stage(tasks=tasks),), stream_slots=slots)
    return SchedulablePipeline(schedule, pool)


def test_determinism_two_stream_v3_schedule() -> None:
    """Two runs of a two-stream hand-written schedule must produce
    bit-identical results. Cross-stream `wait_stream` auto-insertion
    must not introduce non-determinism."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_a, params_a = _run_train_loop(_make_two_stream_pipe, device)
    results_b, params_b = _run_train_loop(_make_two_stream_pipe, device)

    for i, (a, b) in enumerate(zip(results_a, results_b)):
        assert torch.equal(a, b), f"two-stream step {i}: step_result diverged"
    for i, (pa, pb) in enumerate(zip(params_a, params_b)):
        assert torch.equal(pa, pb), f"two-stream param {i} diverged"


def test_determinism_two_stream_matches_single_stream_reference() -> None:
    """Stronger correctness check: the two-stream schedule must
    produce bit-identical outputs to the single-stream `basic(...)`
    schedule on the same seed + same batches. This confirms that
    cross-stream `wait_stream` auto-insertion gives the SAME math,
    not merely the REPEATABLE-but-wrong math.

    A regression where `wait_stream` is missing would desync the
    consumer kernels on `default` from the producer on `memcpy` and
    the two-stream outputs would drift from the single-stream
    reference even if two-stream→two-stream remained repeatable.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _single(model, opt):
        return SchedulablePipeline.basic(model, opt, loss_fn=lambda o: o.sum())

    single_results, single_params = _run_train_loop(_single, device)
    two_results, two_params = _run_train_loop(_make_two_stream_pipe, device)

    assert len(single_results) == len(two_results) == _STEPS
    for i, (a, b) in enumerate(zip(single_results, two_results)):
        assert torch.equal(a, b), (
            f"two-stream vs single-stream diverged at step {i}. "
            f"Cross-stream wait_stream ordering may have broken "
            f"the math, producing repeatable-but-wrong output."
        )
    for i, (pa, pb) in enumerate(zip(single_params, two_params)):
        assert torch.equal(pa, pb), (
            f"two-stream vs single-stream: param {i} diverged after " f"{_STEPS} steps."
        )


# ----------------------------------------------------------------------
# V4 — prefetch (in_flight_batches=2)
# ----------------------------------------------------------------------


def test_determinism_v4_prefetch_preset() -> None:
    """Two runs with `SchedulablePipeline.basic(..., prefetch=True,
    memcpy_stream=True)`. Prefill/drain + ring-advance must not
    introduce non-determinism."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make(model, opt):
        return SchedulablePipeline.basic(
            model,
            opt,
            loss_fn=lambda o: o.sum(),
            prefetch=True,
            memcpy_stream=(device.type == "cuda"),
        )

    results_a, params_a = _run_train_loop(_make, device)
    results_b, params_b = _run_train_loop(_make, device)

    assert len(results_a) == _STEPS
    for i, (a, b) in enumerate(zip(results_a, results_b)):
        assert torch.equal(a, b), f"prefetch step {i}: step_result diverged"
    for i, (pa, pb) in enumerate(zip(params_a, params_b)):
        assert torch.equal(pa, pb), f"prefetch param {i} diverged"


# ----------------------------------------------------------------------
# Per-parameter grad byte equality after one backward
# ----------------------------------------------------------------------


def test_determinism_grad_byte_equal_after_one_backward() -> None:
    """Run exactly one progress() call on two identically-seeded
    models via `SchedulablePipeline.basic`, then compare `.grad`
    tensors byte-for-byte."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make(model, opt):
        return SchedulablePipeline.basic(model, opt, loss_fn=lambda o: o.sum())

    grads_runs: List[List[torch.Tensor]] = []
    for _ in range(2):
        model, opt, batches = _make_model_opt_batches(device)
        pipe = _make(model, opt)
        pipe.progress(iter(batches))  # one step
        grads_runs.append(_grads_snapshot(model))

    grads_a, grads_b = grads_runs
    for i, (ga, gb) in enumerate(zip(grads_a, grads_b)):
        assert torch.equal(ga, gb), (
            f"grad of parameter {i} diverged between two runs of "
            f"the same schedule + seed. Non-deterministic autograd "
            f"or non-deterministic pipeline ordering."
        )


# ----------------------------------------------------------------------
# Smoke: different seeds → different results (sanity)
# ----------------------------------------------------------------------


def test_sanity_different_seeds_produce_different_results() -> None:
    """Bottom-line sanity: `torch.equal` is actually discriminating.

    Different seeds → different model inits → different step_results.
    If this test fails, the `torch.equal`-based assertions above are
    always-passing no-ops, masking any real non-determinism regression.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make(model, opt):
        return SchedulablePipeline.basic(model, opt, loss_fn=lambda o: o.sum())

    model_a, opt_a, batches_a = _make_model_opt_batches(device, seed=_SEED)
    model_b, opt_b, batches_b = _make_model_opt_batches(device, seed=_SEED + 1)

    pipe_a = _make(model_a, opt_a)
    pipe_b = _make(model_b, opt_b)

    r_a = pipe_a.progress(iter(batches_a))
    r_b = pipe_b.progress(iter(batches_b))

    assert not torch.equal(r_a, r_b), (
        "Different seeds produced bit-identical step_result — "
        "torch.equal is not discriminating. The determinism "
        "assertions above would always pass regardless of regressions."
    )


def test_sanity_fresh_pipes_dont_share_engine_state() -> None:
    """Second-order sanity: two freshly-constructed `SchedulablePipeline`
    instances on the SAME seed + same batches must both individually
    produce bit-identical output.

    **Intentional**: both `_make_model_opt_batches(device)` calls use
    the default seed `_SEED`. This is NOT a copy-paste bug — the
    whole point is that running the SAME seed twice should give the
    SAME output. If any engine-internal class-level state, cache,
    or RNG side-effect bled from pipe_a into pipe_b, their outputs
    would drift despite the seed being identical.

    Construction order: build pipe_a, drive it to exhaustion; THEN
    build pipe_b from a fresh seeded setup; compare per-step outputs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make(model, opt):
        return SchedulablePipeline.basic(model, opt, loss_fn=lambda o: o.sum())

    # Run A fully, then run B with a fresh seeded setup.
    model_a, opt_a, batches_a = _make_model_opt_batches(device)
    pipe_a = _make(model_a, opt_a)
    results_a = []
    it_a = iter(batches_a)
    while True:
        try:
            results_a.append(pipe_a.progress(it_a).detach().clone())
        except StopIteration:
            break

    # Pipe B: freshly seeded, identical to A's setup.
    model_b, opt_b, batches_b = _make_model_opt_batches(device)
    pipe_b = _make(model_b, opt_b)
    results_b = []
    it_b = iter(batches_b)
    while True:
        try:
            results_b.append(pipe_b.progress(it_b).detach().clone())
        except StopIteration:
            break

    assert len(results_a) == len(results_b) == _STEPS
    for i, (a, b) in enumerate(zip(results_a, results_b)):
        assert torch.equal(a, b), (
            f"step {i}: pipe B (built after pipe A finished) "
            f"produced different output from pipe A with the same "
            f"seed. Engine-internal shared state is leaking between "
            f"instances."
        )
