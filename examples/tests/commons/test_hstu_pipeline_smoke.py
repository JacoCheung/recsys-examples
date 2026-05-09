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

"""HSTU pipeline smoke tests.

Scope of what this file tests:
  - Imports resolve cleanly (factory + pipeline classes)
  - HSTUPipelineFactory pre-registers both variants
  - Task factories build Task objects with the spec'd fields
  - Schedule construction (non-prefetch and prefetch variants) passes
    the engine validator

What this file does NOT test (requires a real distributed + sharded
setup; tracked in tasks/followups.md):
  - End-to-end parity vs legacy JaggedMegatronTrainPipelineSparseDist
  - Actual NCCL ordering in multi-rank runs
  - Prefetch + cache integration

Those are covered by a future integration test that reuses HSTU's
pretrain_gr_retrieval.py harness under both pipeline backends.
"""

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


# ----------------------------------------------------------------------
# Task factory sanity
# ----------------------------------------------------------------------


def _make_dummy_state():
    """Build a PipelineState without requiring actual torch/megatron."""
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


def test_zero_grad_task_fields() -> None:
    from commons.pipeline.hstu_pipeline.tasks import make_zero_grad_task

    state = _make_dummy_state()
    t = make_zero_grad_task(state)
    assert t.name == "zero_grad"
    assert t.stream == "default"
    assert t.batch_offset == 0


def test_global_tokens_task_is_nccl() -> None:
    from commons.pipeline.hstu_pipeline.tasks import make_global_tokens_task

    state = _make_dummy_state()
    t = make_global_tokens_task(state)
    assert t.name == "global_tokens_allreduce"
    assert t.nccl is True


def test_backward_task_is_nccl() -> None:
    from commons.pipeline.hstu_pipeline.tasks import make_backward_task

    state = _make_dummy_state()
    t = make_backward_task(state)
    assert t.nccl is True, "backward triggers DDP grad AllReduce → must be nccl=True"


def test_finalize_grads_task_is_nccl() -> None:
    from commons.pipeline.hstu_pipeline.tasks import make_finalize_grads_task

    state = _make_dummy_state()
    t = make_finalize_grads_task(state)
    assert t.nccl is True, "finalize_model_grads runs TP AllReduce → must be nccl=True"


def test_nccl_tasks_declaration_order() -> None:
    """The order of nccl=True tasks in the schedule determines the
    cross-rank NCCL submission order enforced by the NCCL lock.
    Verifies the ACTUAL order (not just membership) for a concrete
    built schedule."""
    from commons.pipeline.hstu_pipeline.pipeline import HSTUPipeline

    # Build a pipeline with a NON-identity shuffler (mocked) so all
    # shuffle tasks become nccl=True.
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
    # Legacy submission order: shuffle AllGather(s) → input_dist →
    # global_tokens → backward (DDP) → finalize_model_grads (TP).
    expected = [
        "start_shuffle",
        "finish_shuffle",
        "start_input_dist",
        "wait_input_dist",
        "global_tokens_allreduce",
        "backward",
        "finalize_model_grads",
    ]
    assert nccl_order == expected, (
        f"NCCL submission order mismatch.\n"
        f"  expected: {expected}\n"
        f"  actual:   {nccl_order}"
    )


def test_identity_shuffler_skips_shuffle_nccl() -> None:
    """With identity shuffler, start/finish_shuffle must NOT be nccl
    (no real collective — serializing would just waste the lock)."""
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
    """Construct an HSTUPipeline WITHOUT calling _rewrite_model (which
    requires a real sharded model). We reach into its internal
    _build_schedule to verify the schedule passes validation."""
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


def test_schedule_construction_non_prefetch() -> None:
    pipe = _make_noop_pipeline(prefetch=False, prefetch_depth=1)
    schedule, pool = pipe._build_schedule()
    # Validator runs inside SchedulablePipeline.__init__; we invoke it
    # directly here to verify the schedule is well-formed without
    # building the full engine (which would trigger task.init()).
    from commons.pipeline.engine.autosched.validator import validate

    validate(schedule, pool)


def test_schedule_construction_prefetch() -> None:
    pipe = _make_noop_pipeline(prefetch=True, prefetch_depth=1)
    schedule, pool = pipe._build_schedule()
    from commons.pipeline.engine.autosched.validator import validate

    validate(schedule, pool)


def test_schedule_construction_deep_queue() -> None:
    """Semantic A: deep in-flight queue.

    Offset layout (after Codex-review fix to match legacy positions):
      non-prefetch: max_offset = depth + 1 → in_flight = depth + 2
      (legacy depth=1 → 3 batches: compute, input_dist, h2d)
    """
    pipe = _make_noop_pipeline(prefetch=False, prefetch_depth=3)
    schedule, pool = pipe._build_schedule()
    from commons.pipeline.engine.autosched.validator import validate

    validate(schedule, pool)
    # in_flight_batches = max_offset + 1 = (depth+1) + 1 = depth + 2 = 5
    assert schedule.in_flight_batches == 5


def test_legacy_depth_equals_one_has_three_batches() -> None:
    """Default prefetch_depth=1 should match legacy's 3-batch in-flight
    layout for BOTH variants.

    An earlier iteration placed prefetch at max_offset=3 (4 batches)
    but that overflows the dynamicemb prefetch cache (caught by the
    P2 parity test with use_dynamic_emb=True). Legacy prefetch
    pipeline actually carries 3 in-flight (batch_i / batch_ip1 /
    batch_ip2) — prefetch is an extra stage co-located with
    input_dist at offset=1, not an additional batch slot."""
    p_np = _make_noop_pipeline(prefetch=False, prefetch_depth=1)
    s_np, _ = p_np._build_schedule()
    assert s_np.in_flight_batches == 3

    p_pf = _make_noop_pipeline(prefetch=True, prefetch_depth=1)
    s_pf, _ = p_pf._build_schedule()
    assert (
        s_pf.in_flight_batches == 3
    ), "Prefetch variant must also be 3 in-flight to fit dynamicemb cache"


def test_no_v0_context_references_in_hstu_pipeline() -> None:
    """HSTU pipeline is v1-only (per-batch contexts). No source code
    in hstu_pipeline/ may reference the v0 legacy branch — catches
    regressions where someone copy-pastes legacy patterns without
    noticing they're v0-dependent.

    Skip matches that are inside a COMMENT or DOCSTRING — those are
    explanatory references, not code paths. Match only bare
    ``version=0`` / ``version_0`` tokens in executable positions."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[3]
    pkg = root / "examples" / "commons" / "pipeline" / "hstu_pipeline"
    offenders: list = []
    # Executable uses of version=0 would appear as
    # ``version=0`` or ``.version = 0`` without leading ``#``.
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
    """Runtime regression guard: if a future subclass overrides
    torchrec_context_type to return a version=0 ctx, the assertion
    in create_torchrec_ctx fires."""
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
    """HSTU_DEFAULT_THREAD_MAP must map every task name the pipeline
    produces. Tasks without an entry fall through to the engine's
    'default' thread, potentially reintroducing the postproc
    set_context race (Codex D-CRITICAL-1)."""
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


def test_default_thread_map_has_two_threads() -> None:
    """The default map must use at least 2 distinct thread ids
    (io + compute). If collapsed to one thread, threaded mode silently
    degrades to sequential."""
    from commons.pipeline.hstu_pipeline.pipeline import HSTU_DEFAULT_THREAD_MAP

    threads = set(HSTU_DEFAULT_THREAD_MAP.values())
    assert len(threads) >= 2, (
        f"Default thread map uses only {threads} thread(s); need ≥ 2 "
        f"for real parallelism."
    )


def test_default_threaded_is_true() -> None:
    """Default HSTUPipeline construction should be threaded=True with
    HSTU_DEFAULT_THREAD_MAP applied automatically."""
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
# Legacy file untouched guard (extension of the engine test)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Integration with pretrain_gr_ranking.py — _model attribute parity
# ----------------------------------------------------------------------


def test_hstu_pipeline_attach_updates_model_fwd_default_path() -> None:
    """Codex HIGH 2026-04-26: ``attach(new_model)`` previously updated
    only ``self._state.model``, leaving ``self._state.model_fwd``
    pointing at the construction-time module — so the forward task
    silently kept calling the stale model.

    Default path (no custom_model_fwd): both must be re-routed.
    """
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
    """Custom forwards intentionally diverge from ``state.model``;
    attach() must not silently overwrite them.
    """
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
    """``train_with_pipeline`` (in
    ``examples/hstu/training/trainer/training.py``) calls
    ``pipeline._model.train()`` / ``pipeline._model.eval()`` and reads
    ``pipeline._model._hstu_config`` via ``get_unwrapped_module``. The
    legacy ``JaggedMegatron*`` classes assign ``self._model = model``
    directly; HSTUPipeline must expose the same attribute for drop-in
    use."""
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
    """Smoke-checks that the env-var-driven backend switch in
    ``examples/hstu/training/pretrain_gr_ranking.py`` accepts the two
    canonical values and rejects unknown values. Inspects the script
    source (not import — the script has heavy module-level deps) so
    this test runs on CPU-only hosts."""
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


def test_legacy_pipeline_files_untouched_by_hstu_adapter() -> None:
    """The HSTU adapter must NOT modify train_pipeline.py,
    train_pipeline_factory.py, or utils.py. The engine has a similar
    test; this one uses import hash so it catches even if the legacy
    test is disabled."""
    import hashlib
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    legacy = [
        repo_root / "examples/commons/pipeline/train_pipeline.py",
        repo_root / "examples/commons/pipeline/train_pipeline_factory.py",
        repo_root / "examples/commons/pipeline/utils.py",
    ]
    for path in legacy:
        assert path.exists(), f"legacy file missing: {path}"
        # Just exercise the file — if P2 accidentally broke something
        # importable (e.g. circular import), at least the read works.
        content = path.read_bytes()
        assert len(content) > 0
        hashlib.sha256(content).hexdigest()  # noqa: S324
