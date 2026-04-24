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

"""V2 — full train loop via `SchedulablePipeline.basic(...)`.

Acceptance (SPEC §4.7 T1, plan V2):
  - 20 steps of nn.Linear regression via the engine
  - Loss decreases overall with acceptable minibatch noise
    (enforced as ≥15/19 adjacent pairs strictly decreasing)
  - Final parameters match a hand-written non-pipelined reference
    loop within atol=1e-5 given identical seeds
  - The preset API is `SchedulablePipeline.basic(model, optimizer)`
    (classmethod)
"""


import pytest
import torch
from commons.pipeline.engine import SchedulablePipeline

_STEPS = 20
_BATCH = 16
_IN = 10
_OUT = 1


def _reference_loop(model, optimizer, batches):
    """Hand-written non-pipelined reference for parity comparison."""
    losses = []
    for x in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().clone())
    return losses


def _make_model_and_opt(device: torch.device, seed: int):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = torch.nn.Linear(_IN, _OUT).to(device)
    # SGD with no momentum for clean deterministic comparison.
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    return model, optimizer


def _make_batches(device: torch.device, seed: int):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return [
        torch.randn(_BATCH, _IN, device=device, generator=gen) for _ in range(_STEPS)
    ]


def test_full_train_loop_matches_reference() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reference run
    ref_model, ref_opt = _make_model_and_opt(device, seed=42)
    ref_batches = _make_batches(device, seed=123)
    _reference_loop(ref_model, ref_opt, ref_batches)

    # Engine run — separate model + optimizer with identical init.
    # nn.Linear(10, 1) returns shape (batch, 1); the reference loop
    # sums it into a scalar loss. Match via `loss_fn`.
    eng_model, eng_opt = _make_model_and_opt(device, seed=42)
    eng_batches = _make_batches(device, seed=123)
    pipe = SchedulablePipeline.basic(eng_model, eng_opt, loss_fn=lambda out: out.sum())
    eng_losses = []
    batch_iter = iter(eng_batches)
    for _ in range(_STEPS):
        result = pipe.progress(batch_iter)
        assert result is not None, "progress() must return step_result"
        # step_result is the raw model return (Tensor(batch, 1));
        # match the reference loss computation.
        eng_losses.append(result.detach().sum().clone())

    # Loss monotonically decreases (SGD on quadratic → yes)
    losses_as_floats = [l.item() for l in eng_losses]
    decreased = sum(1 for a, b in zip(losses_as_floats, losses_as_floats[1:]) if b < a)
    # Allow 1-2 non-monotonic steps due to minibatch noise; require
    # >=15/19 to be monotonically decreasing.
    assert decreased >= 15, (
        f"Expected loss to decrease monotonically across 20 steps; "
        f"only {decreased}/19 adjacent pairs decreased. "
        f"Losses: {losses_as_floats}"
    )

    # Final parameters match the reference exactly — same seed, same
    # data, same optimizer, deterministic single-stream forward.
    for (name, p_ref), (_, p_eng) in zip(
        ref_model.named_parameters(), eng_model.named_parameters()
    ):
        assert torch.allclose(p_ref, p_eng, atol=1e-5, rtol=0), (
            f"Parameter '{name}' diverged from reference after "
            f"{_STEPS} steps:\n  ref_mean={p_ref.mean().item():.6g} "
            f"eng_mean={p_eng.mean().item():.6g}\n"
            f"  max_abs_diff={(p_ref - p_eng).abs().max().item():.6g}"
        )


def test_basic_with_custom_forward_fn() -> None:
    """T2 escape hatch: `forward_fn` lets the user wrap model(batch)
    in autocast / custom context. V2 tests the kwarg is wired through
    but uses a trivial passthrough to keep the test deterministic."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = torch.nn.Linear(_IN, _OUT).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    captured = {"called": 0}

    def _forward(m, b):
        captured["called"] += 1
        return m(b)

    pipe = SchedulablePipeline.basic(
        model, opt, forward_fn=_forward, loss_fn=lambda out: out.sum()
    )
    x = torch.randn(4, _IN, device=device)
    pipe.progress(iter([x]))
    assert captured["called"] == 1


def test_basic_with_custom_backward_fn() -> None:
    """T2 escape hatch: `backward_fn` lets the user wrap the backward
    call (e.g. for GradScaler). V2 uses a trivial passthrough."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = torch.nn.Linear(_IN, _OUT).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    captured = {"called": 0}

    def _backward(loss):
        captured["called"] += 1
        loss.backward()

    pipe = SchedulablePipeline.basic(
        model, opt, backward_fn=_backward, loss_fn=lambda out: out.sum()
    )
    x = torch.randn(4, _IN, device=device)
    pipe.progress(iter([x]))
    assert captured["called"] == 1


def test_basic_with_custom_optimizer_step_fn() -> None:
    """T2 escape hatch: `optimizer_step_fn` replaces `optimizer.step()`
    with arbitrary body (clip + scheduler.step + etc.)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = torch.nn.Linear(_IN, _OUT).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    captured = {"called": 0}

    def _step():
        captured["called"] += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

    pipe = SchedulablePipeline.basic(
        model, opt, optimizer_step_fn=_step, loss_fn=lambda out: out.sum()
    )
    x = torch.randn(4, _IN, device=device)
    pipe.progress(iter([x]))
    assert captured["called"] == 1


def test_step_equivalent_to_progress_single_batch() -> None:
    """`pipe.step(batch)` must be observably equivalent to
    `pipe.progress(iter([batch]))`. Both paths: same model state
    update, same returned step_result."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Two identically-seeded models; one trained via step(), one via progress().
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    model_step = torch.nn.Linear(_IN, _OUT).to(device)
    opt_step = torch.optim.SGD(model_step.parameters(), lr=1e-2)

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    model_prog = torch.nn.Linear(_IN, _OUT).to(device)
    opt_prog = torch.optim.SGD(model_prog.parameters(), lr=1e-2)

    pipe_step = SchedulablePipeline.basic(
        model_step, opt_step, loss_fn=lambda out: out.sum()
    )
    pipe_prog = SchedulablePipeline.basic(
        model_prog, opt_prog, loss_fn=lambda out: out.sum()
    )

    # Verify initial params match (sanity)
    for (_, p_s), (_, p_p) in zip(
        model_step.named_parameters(), model_prog.named_parameters()
    ):
        assert torch.equal(p_s, p_p), "seeded models should init identically"

    x = torch.randn(4, _IN, device=device)

    r_step = pipe_step.step(x)
    r_prog = pipe_prog.progress(iter([x]))

    # Return values agree
    assert torch.equal(r_step, r_prog), (
        "step(batch) and progress(iter([batch])) must return equal "
        "step_result tensors"
    )

    # Model params updated identically (same gradients, same SGD step)
    for (_, p_s), (_, p_p) in zip(
        model_step.named_parameters(), model_prog.named_parameters()
    ):
        assert torch.equal(p_s, p_p), (
            "step() and progress(iter([.])) must produce identical " "parameter updates"
        )


def test_basic_device_kwarg_overrides_parameter_inference() -> None:
    """`device=` kwarg must take precedence over parameter-based
    inference.

    Proof: place the model on CUDA, then pass `device=cpu`. Default
    inference would pick CUDA (from parameters); the kwarg should
    override to CPU. Observable: `pipe._stream_pool.get("default")`
    returns `None` (StreamPool's CPU sentinel) when kwarg is honored,
    or a CUDA `Stream` object when silently ignored.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to distinguish param-inferred stream vs CPU override")

    # Model lives on CUDA — default inference WOULD pick CUDA.
    model = torch.nn.Linear(_IN, _OUT).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Explicit override to CPU
    pipe = SchedulablePipeline.basic(
        model, opt, device=torch.device("cpu"), loss_fn=lambda out: out.sum()
    )

    default_slot = pipe._stream_pool.get("default")
    assert default_slot is None, (
        f"`device=cpu` kwarg was ignored: StreamPool['default'] = "
        f"{default_slot!r}, expected None (CPU sentinel). Parameter-"
        f"based inference leaked through despite explicit override."
    )


def test_basic_prefetch_true_constructs_in_v4() -> None:
    """V4 enables `prefetch=True` — must no longer raise. Full
    semantic parity between prefetch and non-prefetch is covered by
    `test_engine_multi_batch.py::test_preset_prefetch_parity`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Linear(_IN, _OUT).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    pipe = SchedulablePipeline.basic(model, opt, prefetch=True, memcpy_stream=True)
    # in_flight_batches is derived from h2d task's batch_offset=1
    assert pipe._schedule.in_flight_batches == 2


def test_duplicate_task_names_rejected() -> None:
    """SPEC §4.2 rule 1: unique task names. Pipeline ctor must reject
    duplicates."""
    from commons.pipeline.engine import Schedule, Stage, StreamPool, Task

    t1 = Task.from_fn(name="dup", fn=lambda ctx: None, stream="default")
    t2 = Task.from_fn(name="dup", fn=lambda ctx: None, stream="default")
    schedule = Schedule(
        stages=(Stage(tasks=(t1, t2)),),
        stream_slots=("default",),
    )
    pool = StreamPool({"default": None})
    with pytest.raises(ValueError, match="Duplicate task name"):
        SchedulablePipeline(schedule, pool)


def test_depends_on_unresolved_rejected() -> None:
    """SPEC §4.2 rule 6: depends_on must resolve to an earlier task."""
    from commons.pipeline.engine import Schedule, Stage, StreamPool, Task

    t = Task.from_fn(
        name="t",
        fn=lambda ctx: None,
        stream="default",
        depends_on=("no_such_task",),
    )
    schedule = Schedule(
        stages=(Stage(tasks=(t,)),),
        stream_slots=("default",),
    )
    pool = StreamPool({"default": None})
    with pytest.raises(ValueError, match="depends_on references 'no_such_task'"):
        SchedulablePipeline(schedule, pool)
