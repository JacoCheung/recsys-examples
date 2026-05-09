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

"""Multi-threaded HSTU pipeline: correctness under concurrency.

Multi-threaded tests are intrinsically flaky-prone — a race can hide
90% of the time. These tests combat that with three techniques:

  1. Repetition — every parity check runs N iterations in a loop.
     A single flake fails the test.
  2. Long-run drift check — 50+ steady-state steps on a single fresh
     pipeline to catch slow-building divergences (e.g. cache leaks,
     context-reuse bugs).
  3. Thread utilization probe — asserts multiple CPU threads actually
     execute tasks (so the test isn't accidentally single-threaded).
  4. Cross-mode equivalence — threaded=True and threaded=False are
     both compared against the SAME NonePipeline baseline. This
     catches any logic that's true in one mode but not the other.
"""

import os
import shutil
import threading
from typing import Set

import commons.checkpoint as checkpoint
import commons.utils as init
import pytest
import torch
import torch.distributed as dist
from commons.distributed.finalize_model_grads import finalize_model_grads
from commons.pipeline.hstu_pipeline import HSTUPipeline
from commons.pipeline.hstu_pipeline.pipeline import HSTU_DEFAULT_THREAD_MAP
from commons.pipeline.train_pipeline import JaggedMegatronTrainNonePipeline
from commons.utils.distributed_utils import collective_assert
from test_utils import create_model

_CKPT_DIR = "./gr_checkpoint_hstu_threaded"


def _setup_two_models(
    *,
    seed: int,
    contextual_feature_names,
    max_num_candidates,
    optimizer_type_str,
    dtype,
    use_dynamic_emb,
    pipeline_type,
    num_batches: int,
):
    """Build baseline + pipelined models, warm up baseline, checkpoint
    baseline weights into pipelined so both start bit-identical."""
    model, dense_optimizer, history_batches = create_model(
        task_type="ranking",
        contextual_feature_names=contextual_feature_names,
        max_num_candidates=max_num_candidates,
        optimizer_type_str=optimizer_type_str,
        use_dynamic_emb=use_dynamic_emb,
        pipeline_type="none",
        dtype=dtype,
        seed=seed,
        num_batches=num_batches,
        replicate_batches=True,
    )
    pipelined_model, pipelined_dense_optimizer, _ = create_model(
        task_type="ranking",
        contextual_feature_names=contextual_feature_names,
        max_num_candidates=max_num_candidates,
        optimizer_type_str=optimizer_type_str,
        dtype=dtype,
        use_dynamic_emb=use_dynamic_emb,
        pipeline_type=pipeline_type,
        seed=seed,
        num_batches=num_batches,
        replicate_batches=True,
    )

    # Warm up baseline to realistic weights
    for batch in history_batches:
        model.module.zero_grad_buffer()
        dense_optimizer.zero_grad()
        loss, _ = model(batch)
        collective_assert(not torch.isnan(loss).any(), "loss has nan")
        loss.sum().backward()
        finalize_model_grads([model.module], None)
        dense_optimizer.step()

    # Checkpoint roundtrip
    if dist.get_rank() == 0 and os.path.exists(_CKPT_DIR):
        shutil.rmtree(_CKPT_DIR)
    dist.barrier(device_ids=[torch.cuda.current_device()])
    if dist.get_rank() == 0:
        os.makedirs(_CKPT_DIR, exist_ok=True)
    dist.barrier(device_ids=[torch.cuda.current_device()])

    checkpoint.save(_CKPT_DIR, model, dense_optimizer=dense_optimizer)
    checkpoint.load(
        _CKPT_DIR,
        pipelined_model,
        dense_optimizer=pipelined_dense_optimizer,
    )
    dist.barrier(device_ids=[torch.cuda.current_device()])
    if dist.get_rank() == 0:
        shutil.rmtree(_CKPT_DIR)

    return (
        model,
        dense_optimizer,
        pipelined_model,
        pipelined_dense_optimizer,
        history_batches,
    )


def _drive_parity(
    *,
    baseline_model,
    baseline_optimizer,
    hstu_pipeline,
    history_batches,
    num_steps: int,
    label: str,
):
    """Drive both pipelines step-by-step and assert per-step equality.

    Raises on the first divergence with a descriptive message including
    ``label`` so failing iterations in a repetition loop are distinguishable.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    baseline = JaggedMegatronTrainNonePipeline(
        baseline_model, baseline_optimizer, device=device
    )
    iter_base = iter(history_batches)
    iter_hstu = iter(history_batches)

    for step in range(num_steps):
        base_loss, _, (_, base_logits, _, _) = baseline.progress(iter_base)
        hstu_loss, _, (_, hstu_logits, _, _) = hstu_pipeline.progress(iter_hstu)
        collective_assert(
            torch.allclose(hstu_loss, base_loss, atol=1e-4),
            f"[{label}] step {step} reporting_loss mismatch: "
            f"base={base_loss.item():.6f} hstu={hstu_loss.item():.6f}",
        )
        collective_assert(
            torch.allclose(hstu_logits, base_logits, atol=1e-4),
            f"[{label}] step {step} logits mismatch",
        )


# ----------------------------------------------------------------------
# 1. Threaded parity — native variant (no dynamic_emb)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("contextual_feature_names", [[]])
def test_threaded_parity_native(contextual_feature_names):
    """Basic threaded=True parity — should behave identically to
    threaded=False and to legacy NonePipeline."""
    init.initialize_distributed()
    init.initialize_model_parallel(1)

    (
        model,
        dense_optimizer,
        pipelined_model,
        pipelined_dense_optimizer,
        history_batches,
    ) = _setup_two_models(
        seed=1234,
        contextual_feature_names=contextual_feature_names,
        max_num_candidates=0,
        optimizer_type_str="sgd",
        dtype=torch.bfloat16,
        use_dynamic_emb=False,
        pipeline_type="native",
        num_batches=20,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    hstu = HSTUPipeline(
        pipelined_model,
        pipelined_dense_optimizer,
        device=device,
        prefetch=False,
        threaded=True,  # explicit for documentation
    )
    try:
        _drive_parity(
            baseline_model=model,
            baseline_optimizer=dense_optimizer,
            hstu_pipeline=hstu,
            history_batches=history_batches,
            num_steps=10,
            label="threaded-native",
        )
    finally:
        hstu.shutdown()

    init.destroy_global_state()


# ----------------------------------------------------------------------
# 2. Threaded parity — prefetch variant (no dynamic_emb)
# ----------------------------------------------------------------------


def test_threaded_parity_prefetch():
    """Prefetch variant under threaded=True. Exercises the extra
    prefetch stage + pre-backward NCCL safety barrier under concurrent
    submission."""
    init.initialize_distributed()
    init.initialize_model_parallel(1)

    (
        model,
        dense_optimizer,
        pipelined_model,
        pipelined_dense_optimizer,
        history_batches,
    ) = _setup_two_models(
        seed=2345,
        contextual_feature_names=[],
        max_num_candidates=0,
        optimizer_type_str="sgd",
        dtype=torch.bfloat16,
        use_dynamic_emb=False,
        pipeline_type="prefetch",
        num_batches=20,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    hstu = HSTUPipeline(
        pipelined_model,
        pipelined_dense_optimizer,
        device=device,
        prefetch=True,
        threaded=True,
    )
    try:
        _drive_parity(
            baseline_model=model,
            baseline_optimizer=dense_optimizer,
            hstu_pipeline=hstu,
            history_batches=history_batches,
            num_steps=10,
            label="threaded-prefetch",
        )
    finally:
        hstu.shutdown()

    init.destroy_global_state()


# ----------------------------------------------------------------------
# 3. Anti-flake: N repetitions of the SAME parity check
# ----------------------------------------------------------------------


@pytest.mark.parametrize("num_reps", [5])
def test_threaded_parity_repeated(num_reps):
    """Run the native-threaded parity check ``num_reps`` times in a
    row. A single flaky pass counts as test failure. This catches
    races that only surface under specific thread scheduling.

    Each rep uses a FRESH HSTUPipeline (new thread pool, fresh
    BatchRing) and a FRESH pair of models (bit-identical start)."""
    init.initialize_distributed()
    init.initialize_model_parallel(1)

    for rep in range(num_reps):
        (
            model,
            dense_optimizer,
            pipelined_model,
            pipelined_dense_optimizer,
            history_batches,
        ) = _setup_two_models(
            seed=1234 + rep,  # different seed each rep widens coverage
            contextual_feature_names=[],
            max_num_candidates=0,
            optimizer_type_str="sgd",
            dtype=torch.bfloat16,
            use_dynamic_emb=False,
            pipeline_type="native",
            num_batches=20,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        hstu = HSTUPipeline(
            pipelined_model,
            pipelined_dense_optimizer,
            device=device,
            prefetch=False,
            threaded=True,
        )
        try:
            _drive_parity(
                baseline_model=model,
                baseline_optimizer=dense_optimizer,
                hstu_pipeline=hstu,
                history_batches=history_batches,
                num_steps=10,
                label=f"rep-{rep}",
            )
        finally:
            hstu.shutdown()

    init.destroy_global_state()


# ----------------------------------------------------------------------
# 4. Long-run drift check
# ----------------------------------------------------------------------


def test_threaded_long_run_no_drift():
    """Run 100 steady-state steps under threaded=True. Catches slow-
    building divergences (cache leaks, context-reuse bugs) that a
    short run wouldn't surface."""
    init.initialize_distributed()
    init.initialize_model_parallel(1)

    (
        model,
        dense_optimizer,
        pipelined_model,
        pipelined_dense_optimizer,
        history_batches,
    ) = _setup_two_models(
        seed=9876,
        contextual_feature_names=[],
        max_num_candidates=0,
        optimizer_type_str="sgd",
        dtype=torch.bfloat16,
        use_dynamic_emb=False,
        pipeline_type="native",
        num_batches=110,  # 110 batches for 100-step compare
    )

    device = torch.device("cuda", torch.cuda.current_device())
    hstu = HSTUPipeline(
        pipelined_model,
        pipelined_dense_optimizer,
        device=device,
        prefetch=False,
        threaded=True,
    )
    try:
        _drive_parity(
            baseline_model=model,
            baseline_optimizer=dense_optimizer,
            hstu_pipeline=hstu,
            history_batches=history_batches,
            num_steps=100,
            label="long-run-100",
        )
    finally:
        hstu.shutdown()

    init.destroy_global_state()


# ----------------------------------------------------------------------
# 5. Cross-mode equivalence: threaded vs sequential vs baseline
# ----------------------------------------------------------------------


def test_threaded_matches_sequential_matches_baseline():
    """Two HSTUPipeline instances, one threaded=True one threaded=False,
    both driven in parallel with the NonePipeline baseline. All three
    must agree per step."""
    init.initialize_distributed()
    init.initialize_model_parallel(1)

    (
        model,
        dense_optimizer,
        pipelined_model_thr,
        pipelined_optim_thr,
        history_batches,
    ) = _setup_two_models(
        seed=5555,
        contextual_feature_names=[],
        max_num_candidates=0,
        optimizer_type_str="sgd",
        dtype=torch.bfloat16,
        use_dynamic_emb=False,
        pipeline_type="native",
        num_batches=25,
    )
    # Build a 3rd model with the same seed → same init → then load
    # the baseline checkpoint. Equivalent to the pipelined_model_thr
    # init but a distinct object (so HSTUPipeline threaded=False owns it).
    _, _, pipelined_model_seq, pipelined_optim_seq, _ = _setup_two_models(
        seed=5555,
        contextual_feature_names=[],
        max_num_candidates=0,
        optimizer_type_str="sgd",
        dtype=torch.bfloat16,
        use_dynamic_emb=False,
        pipeline_type="native",
        num_batches=25,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    baseline = JaggedMegatronTrainNonePipeline(model, dense_optimizer, device=device)
    thr = HSTUPipeline(
        pipelined_model_thr,
        pipelined_optim_thr,
        device=device,
        prefetch=False,
        threaded=True,
    )
    seq = HSTUPipeline(
        pipelined_model_seq,
        pipelined_optim_seq,
        device=device,
        prefetch=False,
        threaded=False,
    )

    iter_base = iter(history_batches)
    iter_thr = iter(history_batches)
    iter_seq = iter(history_batches)

    try:
        for step in range(10):
            base_loss, _, (_, base_logits, _, _) = baseline.progress(iter_base)
            thr_loss, _, (_, thr_logits, _, _) = thr.progress(iter_thr)
            seq_loss, _, (_, seq_logits, _, _) = seq.progress(iter_seq)

            # Threaded vs Sequential
            collective_assert(
                torch.allclose(thr_loss, seq_loss, atol=1e-5),
                f"step {step}: threaded ({thr_loss.item():.6f}) != "
                f"sequential ({seq_loss.item():.6f})",
            )
            collective_assert(
                torch.allclose(thr_logits, seq_logits, atol=1e-5),
                f"step {step}: threaded vs sequential logits mismatch",
            )
            # Threaded vs Baseline
            collective_assert(
                torch.allclose(thr_loss, base_loss, atol=1e-4),
                f"step {step}: threaded ({thr_loss.item():.6f}) != "
                f"baseline ({base_loss.item():.6f})",
            )
            collective_assert(
                torch.allclose(thr_logits, base_logits, atol=1e-4),
                f"step {step}: threaded vs baseline logits mismatch",
            )
    finally:
        thr.shutdown()
        seq.shutdown()

    init.destroy_global_state()


# ----------------------------------------------------------------------
# 6. Thread utilization probe
# ----------------------------------------------------------------------


def test_threaded_actually_uses_multiple_threads():
    """Assert that under threaded=True, tasks actually run on ≥ 2
    distinct worker threads (io + compute). If misconfigured, all
    tasks could end up on one thread and the "threaded" mode silently
    degrades to sequential."""
    init.initialize_distributed()
    init.initialize_model_parallel(1)

    (
        _,
        _,
        pipelined_model,
        pipelined_optim,
        history_batches,
    ) = _setup_two_models(
        seed=7777,
        contextual_feature_names=[],
        max_num_candidates=0,
        optimizer_type_str="sgd",
        dtype=torch.bfloat16,
        use_dynamic_emb=False,
        pipeline_type="native",
        num_batches=15,
    )

    # Monkey-patch each task's underlying fn to record the thread
    # that ran it. We wrap from OUTSIDE by capturing thread id when
    # the task's _fn executes.
    seen_threads: Set[str] = set()
    seen_lock = threading.Lock()

    device = torch.device("cuda", torch.cuda.current_device())
    hstu = HSTUPipeline(
        pipelined_model,
        pipelined_optim,
        device=device,
        prefetch=False,
        threaded=True,
    )
    # Drive one iter to build the schedule lazily
    hstu.progress(iter(history_batches))

    # Now peek at the engine's schedule and wrap each task body.
    schedule = hstu._pipe._schedule
    for stage in schedule.stages:
        for task in stage.tasks:
            orig_run = task.run
            task_name = task.name

            def _wrapped_run(ctx, _orig=orig_run, _n=task_name):
                with seen_lock:
                    seen_threads.add(threading.current_thread().name)
                return _orig(ctx)

            task.run = _wrapped_run

    # Drive 5 more iters to accumulate thread samples.
    try:
        for _ in range(5):
            hstu.progress(iter(history_batches))
    except StopIteration:
        pass

    try:
        hstu.shutdown()
    except Exception:
        pass

    assert (
        len(seen_threads) >= 2
    ), f"Expected ≥ 2 worker threads, only saw: {seen_threads}"

    init.destroy_global_state()


# ----------------------------------------------------------------------
# 7. Scheduler-perturbation stress test (Codex E-LOW)
# ----------------------------------------------------------------------


def test_threaded_stress_random_task_delays():
    """Inject random short delays at the START of every task body, so
    consecutive runs exercise different CPU thread interleavings.
    Parity must hold regardless of scheduling order.

    Without this, the plain repeated test only varies data seeds — the
    thread pool reuses workers, so the same tasks tend to land on the
    same threads run-to-run and race paths stay unexplored. Adding a
    random 0–3 ms delay per task widens the interleaving window.

    Implementation: monkey-patch ``Task.from_fn`` before building the
    pipeline so every task it creates has delays from the very first
    ``progress()`` call. This avoids priming the pipeline first
    (which would advance state relative to the baseline).
    """
    import random
    import time

    from commons.pipeline.engine import Task

    init.initialize_distributed()
    init.initialize_model_parallel(1)

    (
        model,
        dense_optimizer,
        pipelined_model,
        pipelined_dense_optimizer,
        history_batches,
    ) = _setup_two_models(
        seed=0xABCD,
        contextual_feature_names=[],
        max_num_candidates=0,
        optimizer_type_str="sgd",
        dtype=torch.bfloat16,
        use_dynamic_emb=False,
        pipeline_type="native",
        num_batches=30,
    )

    device = torch.device("cuda", torch.cuda.current_device())

    # Monkey-patch Task.from_fn so every task built during pipeline
    # construction gets a per-invocation random delay.
    orig_from_fn = Task.from_fn
    rng = random.Random(42 + dist.get_rank())

    @classmethod
    def _from_fn_with_delay(cls, name, fn, **kwargs):
        def _delayed(ctx):
            # 0–3 ms random delay BEFORE the task body runs.
            time.sleep(rng.uniform(0, 0.003))
            return fn(ctx)

        return orig_from_fn(name, _delayed, **kwargs)

    Task.from_fn = _from_fn_with_delay
    try:
        hstu = HSTUPipeline(
            pipelined_model,
            pipelined_dense_optimizer,
            device=device,
            prefetch=False,
            threaded=True,
        )
        try:
            _drive_parity(
                baseline_model=model,
                baseline_optimizer=dense_optimizer,
                hstu_pipeline=hstu,
                history_batches=history_batches,
                num_steps=10,
                label="stress-delays",
            )
        finally:
            hstu.shutdown()
    finally:
        Task.from_fn = orig_from_fn

    init.destroy_global_state()


# ----------------------------------------------------------------------
# 8. Custom thread_map split rejected (Codex C-LOW regression guard)
# ----------------------------------------------------------------------


def test_threaded_custom_bad_thread_map_rejected():
    """A user-supplied thread_map that splits start_input_dist and
    forward across threads must be rejected at construction time,
    not let silent corruption reach training."""
    init.initialize_distributed()
    init.initialize_model_parallel(1)

    device = torch.device("cuda", torch.cuda.current_device())
    (
        _,
        _,
        pipelined_model,
        pipelined_optim,
        history_batches,
    ) = _setup_two_models(
        seed=4321,
        contextual_feature_names=[],
        max_num_candidates=0,
        optimizer_type_str="sgd",
        dtype=torch.bfloat16,
        use_dynamic_emb=False,
        pipeline_type="native",
        num_batches=15,
    )
    bad_map = {
        "start_input_dist": "alpha",
        "forward": "beta",  # different from alpha → race
    }
    hstu = HSTUPipeline(
        pipelined_model,
        pipelined_optim,
        device=device,
        prefetch=False,
        threaded=True,
        thread_map=bad_map,
    )
    with pytest.raises(ValueError, match="set_context-mutating"):
        # lazy init on first progress — validator fires inside
        hstu.progress(iter(history_batches))

    init.destroy_global_state()


# ----------------------------------------------------------------------
# 7. Default thread_map sanity
# ----------------------------------------------------------------------


def test_default_thread_map_covers_every_task_name():
    """HSTU_DEFAULT_THREAD_MAP must map every task name that the
    pipeline ever produces. A task with no entry silently falls
    through to the engine's 'default' thread, which may reintroduce
    the postproc race. This test is the regression guard."""
    # Build a dummy pipeline (no real distributed init needed) just to
    # enumerate its task names.
    import torch
    from commons.pipeline.hstu_pipeline import HSTUPipeline

    # Use CPU device for this schema-only check
    pipe = HSTUPipeline(
        model=torch.nn.Linear(4, 4),
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
        device=torch.device("cpu"),
        prefetch=True,  # widest task set (includes prefetch_embeddings)
        threaded=True,
    )
    schedule, _ = pipe._build_schedule()
    task_names = {t.name for stage in schedule.stages for t in stage.tasks}

    covered = set(HSTU_DEFAULT_THREAD_MAP.keys())
    missing = task_names - covered
    assert not missing, (
        f"HSTU_DEFAULT_THREAD_MAP is missing entries for: {missing}. "
        f"Tasks without an explicit map entry fall through to 'default' "
        f"thread, which may reintroduce the postproc set_context race."
    )
