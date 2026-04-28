# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end correctness for ``same_progress_sync`` and the Δ=0
``cross_iter_depends_on`` auto-promotion to the same mechanical
contract (per ``SPEC_cross_iter_delta0_autoconvert.md``).

User's 1:1 example:
    fwd.lookahead = 1, stream="default"
    bwd.lookahead = 1, stream="default", depends_on=("fwd",)
    update.lookahead = 0, stream="default", depends_on=("bwd",)
    fwd.same_progress_sync=("update",)            # form A
        — or equivalently —
    fwd.cross_iter_depends_on=(("update",-1),)    # form B (Δ=0, auto-promoted)

Necessity: pre-spec code rejected form B at construction. The
parametrized arm over form B would fail to even construct without the
auto-convert. The form-A arm would produce wrong logits if topological
ordering of update→fwd were lost (fwd would read pre-step weights).

The check is an invariant: per-iter fwd outputs and final params match
the manual la=1-pipelined SGD baseline exactly (atol=1e-5).
"""
import copy
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn
from commons.pipeline.engine import (
    SchedulablePipeline,
    Schedule,
    Stage,
    StreamPool,
    Task,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required: this test exercises real CUDA streams via the engine",
)


_LR = 1.0
_N_PROGRESS = 6


def _make_init_model_and_data(device):
    """Same init weights and data sequence for both runs."""
    torch.manual_seed(0)
    model = nn.Linear(2, 2, bias=False).to(device)
    torch.manual_seed(42)
    data = [
        (
            torch.randn(4, 2, device=device),
            torch.randn(4, 2, device=device),
        )
        for _ in range(_N_PROGRESS + 4)
    ]
    return model, data


def _build_pipe(model, opt, logit_log, *, sync_form: str):
    def fwd_fn(ctx):
        x, _y = ctx.slots["batch_cpu"]
        out = model(x)
        ctx.slots.set("logit", out)
        logit_log.append(out.detach().cpu().clone())

    def bwd_fn(ctx):
        out = ctx.slots["logit"]
        _x, y = ctx.slots["batch_cpu"]
        opt.zero_grad()
        loss = ((out - y) ** 2).mean()
        loss.backward()

    def update_fn(ctx):
        opt.step()

    fwd_kwargs: Dict[str, Any]
    if sync_form == "same_progress_sync":
        fwd_kwargs = dict(same_progress_sync=("update",))
    elif sync_form == "cross_iter":
        fwd_kwargs = dict(cross_iter_depends_on=(("update", -1),))
    else:
        raise ValueError(f"unknown sync_form: {sync_form!r}")

    fwd = Task.from_fn(
        "fwd",
        fwd_fn,
        lookahead=1,
        stream="default",
        reads=("batch_cpu",),
        writes=("logit",),
        **fwd_kwargs,
    )
    bwd = Task.from_fn(
        "bwd",
        bwd_fn,
        lookahead=1,
        stream="default",
        reads=("logit", "batch_cpu"),
        depends_on=("fwd",),
    )
    update_task = Task.from_fn(
        "update",
        update_fn,
        lookahead=0,
        stream="default",
        depends_on=("bwd",),
    )

    schedule = Schedule(
        stages=(Stage(tasks=(fwd, bwd, update_task)),),
        stream_slots=("default",),
    )
    pool = StreamPool(
        {"default": torch.cuda.default_stream(torch.cuda.current_device())}
    )
    return SchedulablePipeline(schedule, pool)


def _run_engine(init_model, data, *, sync_form: str):
    model = copy.deepcopy(init_model)
    opt = torch.optim.SGD(model.parameters(), lr=_LR)
    logit_log: list = []
    pipe = _build_pipe(model, opt, logit_log, sync_form=sync_form)

    batch_iter = iter(data)
    for _ in range(_N_PROGRESS):
        pipe.progress(batch_iter)

    return logit_log, model


def _run_manual_baseline(init_model, data):
    """la=1-pipelined SGD by hand. Mirrors the engine's op order:
    prefill (fwd+bwd on B_0, no update); then per iteration: step()
    applies prev grad, fwd reads post-update weights, bwd refills grad.
    """
    model = copy.deepcopy(init_model)
    opt = torch.optim.SGD(model.parameters(), lr=_LR)
    logits = []

    x0, y0 = data[0]
    out0 = model(x0)
    logits.append(out0.detach().cpu().clone())
    opt.zero_grad()
    loss0 = ((out0 - y0) ** 2).mean()
    loss0.backward()

    for n in range(_N_PROGRESS):
        opt.step()
        x, y = data[n + 1]
        out = model(x)
        logits.append(out.detach().cpu().clone())
        opt.zero_grad()
        loss = ((out - y) ** 2).mean()
        loss.backward()

    return logits, model


@pytest.mark.parametrize(
    "sync_form",
    ["same_progress_sync", "cross_iter"],
    ids=["explicit_same_progress_sync", "cross_iter_delta_zero_autopromoted"],
)
def test_engine_logits_match_manual_baseline(sync_form: str) -> None:
    """End-to-end: engine matches manual la=1-pipelined SGD on every
    fwd output and on final params (atol=1e-5).

    Covers both equivalent forms — explicit ``same_progress_sync`` and
    Δ=0 ``cross_iter_depends_on`` (auto-promoted).
    """
    device = torch.device("cuda")
    init_model, data = _make_init_model_and_data(device)

    engine_logits, engine_model = _run_engine(init_model, data, sync_form=sync_form)
    baseline_logits, baseline_model = _run_manual_baseline(init_model, data)

    expected = _N_PROGRESS + 1
    assert len(engine_logits) == expected
    assert len(baseline_logits) == expected

    for i, (b, e) in enumerate(zip(baseline_logits, engine_logits)):
        max_delta = (b - e).abs().max().item()
        assert torch.allclose(b, e, atol=1e-5, rtol=0), (
            f"iter {i}: logit mismatch (max |Δ|={max_delta:.6g}).\n"
            f"  baseline = {b.flatten().tolist()}\n"
            f"  engine   = {e.flatten().tolist()}"
        )

    for (n_b, p_b), (_, p_e) in zip(
        baseline_model.named_parameters(),
        engine_model.named_parameters(),
    ):
        max_delta = (p_b - p_e).abs().max().item()
        assert torch.allclose(
            p_b, p_e, atol=1e-5, rtol=0
        ), f"param '{n_b}': max |Δw| = {max_delta:.6g}"
