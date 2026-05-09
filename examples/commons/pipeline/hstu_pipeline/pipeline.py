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

"""HSTUPipeline — adapter that drives the Problem #1 SchedulablePipeline
engine using torchrec's ``_rewrite_model`` + ``PipelinedForward`` for
the HSTU training scenario.

Lazy initialization: ``_rewrite_model`` needs a peek at the first
batch for FX tracing. We defer engine construction to the first
``progress()`` call, matching the legacy lazy-fill behavior.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional, Tuple

import torch
from commons.pipeline.engine import (
    SchedulablePipeline,
    Schedule,
    Stage,
    StreamPool,
    ThreadedExecutor,
)

from .tasks import (
    PipelineState,
    make_backward_task,
    make_finalize_grads_task,
    make_finish_shuffle_task,
    make_forward_task,
    make_global_tokens_task,
    make_h2d_task,
    make_nccl_safety_barrier_task,
    make_optimizer_step_task,
    make_prefetch_task,
    make_start_input_dist_task,
    make_start_shuffle_task,
    make_wait_input_dist_task,
    make_watchdog_task,
    make_zero_grad_task,
)

__all__ = ["HSTUPipeline", "HSTU_DEFAULT_THREAD_MAP"]


# Tasks that mutate shared pipelined-module/postproc state via
# ``module.forward.set_context(...)`` or ``postproc.set_context(...)``.
# These MUST be co-located on a single worker thread — otherwise two
# threads can non-atomically override each other's context pointer and
# produce silent corruption (Codex D-CRITICAL-1 root cause).
#
# Currently the HSTU tasks with this property are:
#   - ``start_input_dist``  — temporarily sets postproc context during
#                              ``_start_data_dist`` (see
#                              ``tasks.make_start_input_dist_task``).
#   - ``forward``           — sets both module.forward + postproc
#                              context via ``state.set_module_context``.
#
# Keep this tuple in sync with task implementations.
_SET_CONTEXT_TASKS: tuple = ("start_input_dist", "forward")


# ----------------------------------------------------------------------
# Safe thread assignment for HSTU
# ----------------------------------------------------------------------
#
# HSTU tasks are split into 2 CPU threads:
#
#   "io"      — pure data movement: h2d, start_shuffle, finish_shuffle.
#               These touch only their own slot's tensor + the batch
#               shuffler (no shared mutable module state).
#   "compute" — everything else: start_input_dist, wait_input_dist,
#               prefetch, zero_grad, global_tokens, forward,
#               backward, finalize, optimizer.
#               These may mutate shared module state
#               (``module.forward._context`` / ``postproc._context``)
#               so they MUST all live on one thread to avoid the
#               set_context race (Codex D-CRITICAL-1).
#
# Cross-thread data dependencies use threading.Event (engine handles
# this). Cross-rank NCCL ordering uses the engine's ``_NcclOrderedLock``.
#
# Users can override via ``thread_map=...`` kwarg if they know their
# model's set_context path is thread-safe (e.g. no pipelined_postprocs).
HSTU_DEFAULT_THREAD_MAP: dict = {
    # io thread
    "h2d": "io",
    "start_shuffle": "io",
    "finish_shuffle": "io",
    # compute thread
    "start_input_dist": "compute",
    "wait_input_dist": "compute",
    "prefetch_embeddings": "compute",
    "zero_grad": "compute",
    "global_tokens_allreduce": "compute",
    "nccl_safety_barrier": "compute",
    "forward": "compute",
    "backward": "compute",
    "finalize_model_grads": "compute",
    "optimizer_step": "compute",
    "watchdog_step": "compute",
}


class HSTUPipeline:
    """Drop-in replacement for ``JaggedMegatronTrainPipelineSparseDist``
    / ``JaggedMegatronPrefetchTrainPipelineSparseDist`` built on the
    schedulable engine.

    Preserves the legacy ``progress()`` return signature:
    ``(local_loss_sum.detach(), global_tokens, model_output)``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        prefetch: bool = False,
        prefetch_depth: int = 1,
        batch_shuffler: Any = None,
        assert_nan_loss: bool = False,
        apply_jit: bool = False,
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[Callable[[Any], Tuple[torch.Tensor, Any]]] = None,
        # Default: multi-threaded with HSTU_DEFAULT_THREAD_MAP (io +
        # compute, two threads). The default map pins all set_context
        # call-sites onto the same thread, avoiding the race Codex
        # D-CRITICAL-1 flagged. Pass ``threaded=False`` to fall back
        # to Sequential, or override ``thread_map=`` to customize.
        threaded: bool = True,
        thread_map: Any = None,
    ) -> None:
        if prefetch_depth < 1:
            raise ValueError(f"prefetch_depth must be >= 1, got {prefetch_depth}")
        if device.type != "cuda":
            # Non-CUDA path is for smoke tests only; HSTU features
            # (shuffler, NCCL, autograd hooks) require CUDA.
            pass

        # Resolve default thread_map when threaded=True and user didn't
        # pass one. Users can pass a dict / callable / "by_stream" /
        # "per_task" to override.
        if threaded and thread_map is None:
            thread_map = HSTU_DEFAULT_THREAD_MAP

        self._prefetch = prefetch
        self._prefetch_depth = prefetch_depth
        self._apply_jit = apply_jit
        self._pipeline_postproc = pipeline_postproc
        self._threaded = threaded
        self._thread_map = thread_map

        # Default shuffler is identity (no-op) — matches legacy default.
        if batch_shuffler is None:
            from commons.distributed.batch_shuffler import IdentityBalancedBatchShuffler

            batch_shuffler = IdentityBalancedBatchShuffler()

        # Build shared state for task closures.
        self._state = PipelineState(
            model=model,
            optimizer=optimizer,
            device=device,
            batch_shuffler=batch_shuffler,
            is_identity_shuffler=self._is_identity(batch_shuffler),
            model_fwd=custom_model_fwd if custom_model_fwd else model,
            assert_nan_loss=assert_nan_loss,
        )

        # Lazily constructed on first progress() call.
        self._pipe: Optional[SchedulablePipeline] = None
        self._original_forwards: list = []
        self._original_kjt_dist_forwards: list = []
        self._model_attached: bool = True

    @staticmethod
    def _is_identity(shuffler: Any) -> bool:
        try:
            from commons.distributed.batch_shuffler import IdentityBalancedBatchShuffler

            return isinstance(shuffler, IdentityBalancedBatchShuffler)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Engine construction (lazy)
    # ------------------------------------------------------------------

    def _build_schedule(self) -> Tuple[Schedule, StreamPool]:
        """Construct the Schedule + StreamPool based on variant.

        Offset layout — **both variants carry 3 batches in flight at
        depth=1**, matching legacy JaggedMegatron (non-prefetch:
        train_pipeline.py:735-740; prefetch: train_pipeline.py:862+):

          non-prefetch (depth=1): max_offset=2
              h2d@2, input_dist@1, compute@0
          prefetch     (depth=1): max_offset=2
              h2d@2, input_dist@1, prefetch@1, compute@0

        Earlier layouts pushed prefetch to max_offset=3 (4 in-flight
        batches) which overflows the dynamicemb prefetch cache
        (outstanding keys > capacity). Legacy's prefetch pipeline
        keeps 3 in-flight (batch_i / batch_ip1 / batch_ip2), so we
        match that.

        depth=K adds K-1 buffer slots between input_dist and compute:
              max_offset = K+1 for both variants.
        """
        depth = self._prefetch_depth
        h2d_offset = depth + 1  # = 2 for depth=1
        input_dist_offset = depth  # = 1 for depth=1
        prefetch_offset = 1 if self._prefetch else None

        tasks = [
            make_h2d_task(self._state, batch_offset=h2d_offset),
            make_start_shuffle_task(self._state, batch_offset=h2d_offset),
            make_finish_shuffle_task(self._state, batch_offset=h2d_offset),
            make_start_input_dist_task(self._state, batch_offset=input_dist_offset),
            make_wait_input_dist_task(self._state, batch_offset=input_dist_offset),
        ]
        if self._prefetch:
            tasks.append(make_prefetch_task(self._state, batch_offset=prefetch_offset))
        tasks.extend(
            [
                make_zero_grad_task(self._state),
                make_global_tokens_task(self._state),
                make_nccl_safety_barrier_task(self._state),
                make_forward_task(self._state),
                make_backward_task(self._state),
                make_finalize_grads_task(self._state),
                make_optimizer_step_task(self._state),
                make_watchdog_task(),
            ]
        )

        stream_slots = ("default", "memcpy", "data_dist")
        if self._prefetch:
            stream_slots = stream_slots + ("prefetch",)

        schedule = Schedule(
            stages=(Stage(tasks=tuple(tasks)),), stream_slots=stream_slots
        )

        device = self._state.device
        pool_dict: dict = {
            "default": (
                torch.cuda.default_stream(device) if device.type == "cuda" else None
            ),
            "memcpy": (
                torch.cuda.Stream(device, priority=-1)
                if device.type == "cuda"
                else None
            ),
            "data_dist": (
                torch.cuda.Stream(device, priority=-1)
                if device.type == "cuda"
                else None
            ),
        }
        if self._prefetch:
            pool_dict["prefetch"] = (
                torch.cuda.Stream(device, priority=-1)
                if device.type == "cuda"
                else None
            )
        return schedule, StreamPool(pool_dict)

    def _rewrite_model(
        self, peek_batch: Any, data_dist_stream: Any, default_stream: Any
    ):
        """Call torchrec's _rewrite_model once, using the SAME streams
        that will later be installed in the engine's StreamPool, and
        return the **real per-batch context** that was used for the
        monkey-patch bootstrap.

        Previously this context was abandoned after ``_override_input_dist_forwards``
        — leaking dynamicemb cache reservations when caching is on
        (prefetch + use_dynamic_emb combo). The context is now
        returned to the caller, which seeds it into the engine's
        ring as the peek batch's real ctx so the batch flows through
        the pipeline normally and its cache entries get consumed by
        forward.
        """
        from commons.pipeline.utils import (
            PipelinedForward,
            PrefetchPipelinedForward,
            PrefetchTrainPipelineContext,
            TrainPipelineContext,
            _override_input_dist_forwards,
        )
        from commons.pipeline.utils import _rewrite_model as torchrec_rewrite_model
        from commons.pipeline.utils import _start_data_dist

        # Seed context type on state for per-batch ctx factory.
        self._state.torchrec_context_type = (
            PrefetchTrainPipelineContext if self._prefetch else TrainPipelineContext
        )

        # Real per-batch context for the peek batch (will be seeded
        # into the engine ring). Use state.create_torchrec_ctx so the
        # index counter stays consistent with subsequent batches.
        peek_ctx = self._state.create_torchrec_ctx()

        pipelined_forward_type = (
            PrefetchPipelinedForward if self._prefetch else PipelinedForward
        )

        (
            pipelined_modules,
            self._state.model,
            self._original_forwards,
            pipelined_postprocs,
            _names,
        ) = torchrec_rewrite_model(
            model=self._state.model,
            context=peek_ctx,
            dist_stream=data_dist_stream,
            default_stream=default_stream,
            batch=peek_batch,
            apply_jit=self._apply_jit,
            pipelined_forward=pipelined_forward_type,
            pipeline_postproc=self._pipeline_postproc,
        )
        self._state.pipelined_modules = pipelined_modules
        self._state.pipelined_postprocs = pipelined_postprocs

        # Bootstrap the input_dist. This populates peek_ctx's
        # splits_requests AND ensures the module's _input_dists
        # attribute exists so KJTAllToAllForward monkeypatch can
        # install. The ctx is REAL — it's seeded into the engine ring
        # below so the peek batch goes through forward/backward like
        # any other batch and its cache entries get consumed.
        _start_data_dist(pipelined_modules, peek_batch, peek_ctx)
        self._original_kjt_dist_forwards = _override_input_dist_forwards(
            pipelined_modules
        )
        return peek_ctx

    def _validate_set_context_colocation(self, schedule) -> None:
        """Refuse a thread_map that splits set_context-calling tasks
        across different threads.

        ``start_input_dist`` and ``forward`` both write the shared
        ``module.forward._context`` / ``postproc._context`` pointer
        without a lock. Running them on different threads reintroduces
        the Codex D-CRITICAL-1 race. This runtime check fails loudly
        at pipeline-construction time rather than letting silent
        corruption reach the parity oracle.
        """
        if not self._threaded:
            return  # sequential executor — no thread race possible
        from commons.pipeline.engine.executor import _resolve_thread_id

        seen: set = set()
        for stage in schedule.stages:
            for task in stage.tasks:
                if task.name in _SET_CONTEXT_TASKS:
                    seen.add(_resolve_thread_id(task, self._thread_map))
        if len(seen) > 1:
            raise ValueError(
                f"thread_map splits the set_context-mutating tasks "
                f"{_SET_CONTEXT_TASKS} across threads {sorted(seen)}. "
                f"These tasks mutate shared pipelined-module state "
                f"non-atomically — they MUST be co-located on one "
                f"worker thread. Use HSTU_DEFAULT_THREAD_MAP or put "
                f"them on the same thread in your custom map."
            )

    def _ensure_pipe(self, peek_batch_cpu: Any, peek_batch_gpu: Any) -> None:
        if self._pipe is not None:
            return
        # Build StreamPool FIRST so we can pass the final data_dist /
        # default streams into _rewrite_model — Codex flagged that
        # using throwaway streams causes PipelinedForward to capture
        # wrong handles.
        schedule, pool = self._build_schedule()

        # Codex C-LOW: refuse any thread_map that lets set_context
        # calls race across threads. Check at construction time.
        self._validate_set_context_colocation(schedule)

        data_dist_stream = pool.get("data_dist")
        default_stream = pool.get("default")
        # _rewrite_model now runs bootstrap on a REAL per-batch ctx
        # (instead of a throwaway) and returns it. We seed it into
        # the engine ring below so the peek batch flows through the
        # pipeline normally.
        peek_ctx = self._rewrite_model(peek_batch_gpu, data_dist_stream, default_stream)

        executor = (
            ThreadedExecutor(thread_map=self._thread_map) if self._threaded else None
        )
        self._pipe = SchedulablePipeline(schedule, pool, executor=executor)

        # Seed the engine with the peek batch's pre-processed state:
        # batch_cpu / batch_gpu both set (h2d already done), torchrec_ctx
        # set (with in-flight input_dist from _start_data_dist bootstrap),
        # and for identity shuffler shuffled_batch is just batch_gpu.
        # Idempotent guards in h2d/start_shuffle/finish_shuffle/start_input_dist
        # tasks skip the work that's already been done.
        seeded: dict = {
            "batch_cpu": peek_batch_cpu,
            "batch_gpu": peek_batch_gpu,
            "torchrec_ctx": peek_ctx,
        }
        if self._state.is_identity_shuffler:
            # Identity shuffler: shuffled_batch IS the batch_gpu.
            seeded["shuffled_batch"] = peek_batch_gpu
        # (For non-identity shuffler, the engine's start_shuffle +
        # finish_shuffle tasks still need to run on the peek batch.
        # The shuffler's start/finish collectives are idempotent at
        # the NCCL level — they'll happen exactly once per batch.)
        self._pipe.seed_first_batch(seeded)

    # ------------------------------------------------------------------
    # Public API (legacy-matching)
    # ------------------------------------------------------------------

    def progress(
        self, dataloader_iter: Iterator[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Drive one full pipeline iteration.

        On the first call, peeks the first batch to run ``_rewrite_model``,
        then builds the engine. Returns ``(loss, global_tokens, output)``
        matching legacy.
        """
        if self._pipe is None:
            # Peek one batch to drive FX tracing + monkeypatch
            # bootstrap. Unlike the earlier throwaway design, the peek
            # batch is now SEEDED into the engine ring so it flows
            # through the pipeline as the first real batch (no data
            # loss, no dynamicemb cache leak).
            try:
                peek_cpu = next(dataloader_iter)
            except StopIteration as e:
                raise StopIteration(
                    "HSTUPipeline: dataloader was empty on first progress() call"
                ) from e
            # H2D the peek batch on memcpy to mirror legacy
            # copy_batch_to_gpu_and_shuffle so FX sees a GPU batch.
            from commons.pipeline.utils import _to_device

            if self._state.device.type == "cuda":
                memcpy_stream = torch.cuda.Stream(self._state.device, priority=-1)
                with torch.cuda.stream(memcpy_stream):
                    peek_gpu = _to_device(
                        peek_cpu, self._state.device, non_blocking=True
                    )
            else:
                peek_gpu = _to_device(peek_cpu, self._state.device, non_blocking=True)
            self._ensure_pipe(peek_cpu, peek_gpu)

        # Engine does the rest.
        return self._pipe.progress(dataloader_iter)

    def attach(self, model: Optional[torch.nn.Module] = None) -> None:
        """Matches legacy attach() — re-enable the pipeline after detach."""
        if model is not None:
            self._state.model = model
        self._model_attached = True
        # If already attached once, nothing to do; _rewrite_model has
        # already mutated the model in place.

    def detach(self) -> torch.nn.Module:
        """Restore original forwards, return the bare model."""
        if self._state.pipelined_modules:
            from commons.pipeline.utils import _pipeline_detach_model

            _pipeline_detach_model(
                model=self._state.model,
                pipelined_modules=self._state.pipelined_modules,
                original_forwards=self._original_forwards,
                original_kjt_dist_forwards=self._original_kjt_dist_forwards,
                pipelined_postprocs=self._state.pipelined_postprocs,
            )
        self._model_attached = False
        return self._state.model

    def shutdown(self) -> None:
        if self._pipe is not None:
            self._pipe.shutdown()

    def __enter__(self) -> "HSTUPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
