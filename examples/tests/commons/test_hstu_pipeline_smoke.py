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

"""HSTU adapter smoke tests."""

import pytest


def test_import_hstu_pipeline() -> None:
    from commons.pipeline.hstu_pipeline import HSTUPipeline, HSTUPipelineFactory

    assert HSTUPipeline is not None
    assert HSTUPipelineFactory is not None


def test_factory_pre_registers_both_variants() -> None:
    from commons.pipeline.hstu_pipeline import HSTUPipelineFactory

    registered = HSTUPipelineFactory.list()
    assert "hstu_sparse_dist" in registered
    assert "hstu_prefetch_sparse_dist" in registered


def test_factory_unknown_name_raises() -> None:
    from commons.pipeline.hstu_pipeline import HSTUPipelineFactory

    with pytest.raises(KeyError, match="Unknown HSTU pipeline"):
        HSTUPipelineFactory.create("no_such_pipeline")


def test_factory_duplicate_register_rejected() -> None:
    from commons.pipeline.hstu_pipeline import HSTUPipelineFactory

    with pytest.raises(ValueError, match="already registered"):
        HSTUPipelineFactory.register(
            "hstu_sparse_dist", lambda **kw: None  # type: ignore[arg-type]
        )


def _make_dummy_state():
    """Build PipelineState without requiring Megatron runtime setup."""
    import torch
    from commons.pipeline.hstu_pipeline.tasks import PipelineState

    return PipelineState(
        model=torch.nn.Linear(4, 4),
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
        device=torch.device("cpu"),
    )


def test_h2d_task_fields() -> None:
    from commons.pipeline.hstu_pipeline.tasks import make_h2d_task

    state = _make_dummy_state()
    t = make_h2d_task(state, lookahead=2)
    assert t.name == "h2d"
    assert t.stream == "memcpy"
    assert t.batch_offset == 2
    assert any(s.name == "batch_cpu" for s in t.reads)
    write_names = {s.name for s in t.writes}
    assert write_names == {"batch_gpu", "torchrec_ctx"}


def test_nccl_tasks_declaration_order() -> None:
    """nccl=True tasks keep the expected submission order."""
    from commons.pipeline.hstu_pipeline.pipeline import HSTUPipeline

    class _NonIdentityShuffler:
        def shuffle(self, batch, pg):
            return batch

    pipe = HSTUPipeline(
        model=__import__("torch").nn.Linear(4, 4),
        optimizer=__import__("torch").optim.SGD(
            [__import__("torch").nn.Parameter(__import__("torch").zeros(1))],
            lr=0.1,
        ),
        device=__import__("torch").device("cpu"),
        prefetch=False,
        prefetch_depth=1,
        batch_shuffler=_NonIdentityShuffler(),
    )
    schedule, _ = pipe._build_schedule()

    nccl_order = [
        t.name
        for stage in schedule.stages
        for t in stage.tasks
        if getattr(t, "nccl", False)
    ]
    expected = [
        "start_shuffle",
        "finish_shuffle",
        "start_input_dist",
        "wait_input_dist",
        "global_tokens_allreduce",
        "compute_output_dist",
        "backward",
        "finalize_model_grads",
    ]
    assert nccl_order == expected, (
        f"NCCL submission order mismatch.\n"
        f"  expected: {expected}\n"
        f"  actual:   {nccl_order}"
    )


def test_identity_shuffler_skips_shuffle_nccl() -> None:
    """Identity shuffler does not mark shuffle tasks as NCCL."""
    from commons.pipeline.hstu_pipeline.tasks import (
        make_finish_shuffle_task,
        make_start_shuffle_task,
    )

    state = _make_dummy_state()  # identity by default
    state.is_identity_shuffler = True
    assert make_start_shuffle_task(state, lookahead=1).nccl is False
    assert make_finish_shuffle_task(state, lookahead=1).nccl is False


# ----------------------------------------------------------------------
# Schedule construction — validates via engine's validator
# ----------------------------------------------------------------------


def _make_noop_pipeline(prefetch: bool, prefetch_depth: int = 1):
    """Construct HSTUPipeline without calling _rewrite_model."""
    import torch
    from commons.pipeline.hstu_pipeline import HSTUPipeline

    pipe = HSTUPipeline(
        model=torch.nn.Linear(4, 4),
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
        device=torch.device("cpu"),
        prefetch=prefetch,
        prefetch_depth=prefetch_depth,
    )
    return pipe


def test_schedule_construction_deep_queue(monkeypatch) -> None:
    """prefetch_depth expands the in-flight queue."""
    monkeypatch.delenv("HSTU_LA_DEPTH", raising=False)
    pipe = _make_noop_pipeline(prefetch=False, prefetch_depth=3)
    schedule, pool = pipe._build_schedule()
    from commons.pipeline.engine.autosched.validator import validate

    validate(schedule, pool)
    # in_flight_batches = max_offset + 1 = (depth+1) + 1 = depth + 2 = 5
    assert schedule.in_flight_batches == 5


def test_default_depth_one_has_three_batches(monkeypatch) -> None:
    """Default depth keeps both variants at 3 in-flight batches."""
    monkeypatch.delenv("HSTU_LA_DEPTH", raising=False)
    p_np = _make_noop_pipeline(prefetch=False, prefetch_depth=1)
    s_np, pool_np = p_np._build_schedule()
    from commons.pipeline.engine.autosched.validator import validate

    validate(s_np, pool_np)
    assert s_np.in_flight_batches == 3

    p_pf = _make_noop_pipeline(prefetch=True, prefetch_depth=1)
    s_pf, pool_pf = p_pf._build_schedule()
    validate(s_pf, pool_pf)
    assert (
        s_pf.in_flight_batches == 3
    ), "Prefetch variant must also be 3 in-flight to fit dynamicemb cache"


def test_hstu_la_depth_env_override(monkeypatch) -> None:
    """Benchmark sweeps can override the public prefetch_depth layout
    with a named lookahead profile."""
    monkeypatch.setenv("HSTU_LA_DEPTH", "6")
    pipe = _make_noop_pipeline(prefetch=True, prefetch_depth=1)
    schedule, _ = pipe._build_schedule()
    assert schedule.in_flight_batches == 6


def test_cuda_mem_watchdog_task_is_env_gated(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_MEM_WATCHDOG", raising=False)
    pipe = _make_noop_pipeline(prefetch=True, prefetch_depth=1)
    schedule, _ = pipe._build_schedule()
    names = {t.name for stage in schedule.stages for t in stage.tasks}
    assert "watchdog_step" not in names

    monkeypatch.setenv("CUDA_MEM_WATCHDOG", "1")
    pipe = _make_noop_pipeline(prefetch=True, prefetch_depth=1)
    schedule, _ = pipe._build_schedule()
    names = {t.name for stage in schedule.stages for t in stage.tasks}
    assert "watchdog_step" in names


def test_no_v0_context_references_in_hstu_pipeline() -> None:
    """Executable HSTU adapter code must stay on v1 contexts."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[3]
    pkg = root / "examples" / "commons" / "pipeline" / "hstu_pipeline"
    offenders: list = []
    pattern = re.compile(r"(?<!#\s)\bversion\s*=\s*0\b")
    for py in pkg.rglob("*.py"):
        text = py.read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue  # comment line
            if pattern.search(line):
                offenders.append(f"{py.name}:{lineno}: {stripped}")
    assert (
        not offenders
    ), "hstu_pipeline must be v1-only; found v0 references:\n  " + "\n  ".join(
        offenders
    )


def test_create_torchrec_ctx_rejects_v0() -> None:
    """create_torchrec_ctx rejects contexts that are not v1."""
    import torch
    from commons.pipeline.hstu_pipeline.tasks import PipelineState

    class _V0Ctx:
        def __init__(self, index=0, version=1):
            self.index = index
            self.version = 0  # malicious: ignore kwarg, hardcode v0

    state = PipelineState(
        model=torch.nn.Linear(4, 4),
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
        device=torch.device("cpu"),
        torchrec_context_type=_V0Ctx,
    )
    with pytest.raises(AssertionError, match="forbids v0"):
        state.create_torchrec_ctx()


def test_default_thread_map_covers_every_task_name() -> None:
    """Default thread_map covers every task emitted by the schedule."""
    from commons.pipeline.hstu_pipeline.pipeline import HSTU_DEFAULT_THREAD_MAP

    # Prefetch variant has the widest task set (includes prefetch_embeddings)
    p = _make_noop_pipeline(prefetch=True, prefetch_depth=1)
    schedule, _ = p._build_schedule()
    task_names = {t.name for stage in schedule.stages for t in stage.tasks}

    missing = task_names - set(HSTU_DEFAULT_THREAD_MAP)
    assert not missing, (
        f"HSTU_DEFAULT_THREAD_MAP is missing entries for: {sorted(missing)}. "
        f"Add them to the map in hstu_pipeline/pipeline.py."
    )

    optional = {"watchdog_step"}
    extra = set(HSTU_DEFAULT_THREAD_MAP) - task_names
    assert extra <= optional, (
        "HSTU_DEFAULT_THREAD_MAP has unexpected entries for tasks not "
        f"present in the default schedule: {sorted(extra - optional)}"
    )


def test_default_thread_map_has_two_threads() -> None:
    """Default thread_map keeps at least two worker groups."""
    from commons.pipeline.hstu_pipeline.pipeline import HSTU_DEFAULT_THREAD_MAP

    threads = set(HSTU_DEFAULT_THREAD_MAP.values())
    assert len(threads) >= 2, (
        f"Default thread map uses only {threads} thread(s); need ≥ 2 "
        f"for real parallelism."
    )


def test_default_threaded_is_true() -> None:
    """HSTUPipeline defaults to threaded execution."""
    import torch
    from commons.pipeline.hstu_pipeline import HSTUPipeline
    from commons.pipeline.hstu_pipeline.pipeline import HSTU_DEFAULT_THREAD_MAP

    pipe = HSTUPipeline(
        model=torch.nn.Linear(4, 4),
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
        device=torch.device("cpu"),
    )
    assert pipe._threaded is True
    assert pipe._thread_map == HSTU_DEFAULT_THREAD_MAP


def test_prefetch_depth_validation() -> None:
    import torch
    from commons.pipeline.hstu_pipeline import HSTUPipeline

    with pytest.raises(ValueError, match="prefetch_depth must be >= 1"):
        HSTUPipeline(
            model=torch.nn.Linear(4, 4),
            optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
            device=torch.device("cpu"),
            prefetch_depth=0,
        )


# ----------------------------------------------------------------------
# Integration with pretrain_gr_ranking.py
# ----------------------------------------------------------------------


def test_hstu_pipeline_attach_updates_model_fwd_default_path() -> None:
    """attach(new_model) updates the default forward target."""
    import torch
    from commons.pipeline.hstu_pipeline import HSTUPipeline

    model = torch.nn.Linear(4, 4)
    pipe = HSTUPipeline(
        model=model,
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
        device=torch.device("cpu"),
    )
    assert pipe._state.model is model
    assert pipe._state.model_fwd is model

    new_model = torch.nn.Linear(4, 4)
    pipe.attach(new_model)
    assert pipe._state.model is new_model
    assert pipe._state.model_fwd is new_model, (
        "attach() must re-route model_fwd to the new model when no "
        "custom_model_fwd was supplied"
    )


def test_hstu_pipeline_attach_preserves_custom_model_fwd() -> None:
    """attach(new_model) preserves user-supplied custom forward."""
    import torch
    from commons.pipeline.hstu_pipeline import HSTUPipeline

    model = torch.nn.Linear(4, 4)
    captured = {}

    def custom_fwd(batch):
        captured["called"] = True
        return torch.zeros(1), None

    pipe = HSTUPipeline(
        model=model,
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
        device=torch.device("cpu"),
        custom_model_fwd=custom_fwd,
    )
    assert pipe._state.model_fwd is custom_fwd

    new_model = torch.nn.Linear(4, 4)
    pipe.attach(new_model)
    assert pipe._state.model is new_model
    assert (
        pipe._state.model_fwd is custom_fwd
    ), "attach() must NOT overwrite a user-supplied custom forward"


def test_hstu_pipeline_exposes_underscore_model_attribute() -> None:
    """HSTUPipeline exposes _model for existing trainer code."""
    import torch
    from commons.pipeline.hstu_pipeline import HSTUPipeline

    model = torch.nn.Linear(4, 4)
    pipe = HSTUPipeline(
        model=model,
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
        device=torch.device("cpu"),
    )
    assert (
        pipe._model is model
    ), "HSTUPipeline._model must be the same object passed at construction"

    # detach() returns the bare model; attach(new_model) should
    # re-route _model to the new module.
    new_model = torch.nn.Linear(4, 4)
    pipe.attach(new_model)
    assert pipe._model is new_model


def test_pretrain_gr_ranking_backend_env_var_routing() -> None:
    """Backend env switch accepts known values and rejects typos."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3]
    script = root / "examples" / "hstu" / "training" / "pretrain_gr_ranking.py"
    text = script.read_text()
    assert (
        "RECSYS_PIPELINE_BACKEND" in text
    ), "pretrain_gr_ranking.py must read env var RECSYS_PIPELINE_BACKEND"
    assert (
        "HSTUPipelineFactory" in text
    ), "pretrain_gr_ranking.py must offer the new HSTU backend"
    # Must reject typos rather than silently fall through.
    assert "must be 'legacy' or 'new'" in text
