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

__all__ = ["HSTUPipeline"]


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
        # Defaults to Sequential to avoid PipelinedForward/postproc
        # set_context race across threads (Codex D-CRITICAL-1). Users
        # who know their model's set_context calls are thread-safe can
        # opt in via threaded=True.
        threaded: bool = False,
        thread_map: Any = None,
    ) -> None:
        if prefetch_depth < 1:
            raise ValueError(f"prefetch_depth must be >= 1, got {prefetch_depth}")
        if device.type != "cuda":
            # Non-CUDA path is for smoke tests only; HSTU features
            # (shuffler, NCCL, autograd hooks) require CUDA.
            pass

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

        Offset layout (matches legacy JaggedMegatron docstring at
        train_pipeline.py:735-740 — legacy keeps 3 positions for
        non-prefetch, 4 for prefetch):

          non-prefetch (depth=1): max_offset=2
              h2d@2, input_dist@1, compute@0
          prefetch     (depth=1): max_offset=3
              h2d@3, input_dist@2, prefetch@1, compute@0

          depth=K adds K-1 buffer slots between input_dist and compute:
              non-prefetch: max_offset = K+1
              prefetch:     max_offset = K+2
        """
        depth = self._prefetch_depth
        if self._prefetch:
            h2d_offset = depth + 2
            input_dist_offset = depth + 1
            prefetch_offset = 1
        else:
            h2d_offset = depth + 1
            input_dist_offset = depth
            prefetch_offset = None

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
    ) -> None:
        """Call torchrec's _rewrite_model once, using the SAME streams
        that will later be installed in the engine's StreamPool.

        This avoids the stream-mismatch bug where PipelinedForward
        captures one stream but the engine uses another, breaking the
        stream synchronization assumptions inside torchrec.
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

        # Bootstrap context used solely to drive FX tracing +
        # populate _input_dists so the KJT monkeypatch has something
        # to override. The context's in-flight collective is
        # intentionally abandoned (known limitation — see
        # tasks/followups.md entry "Problem #2 bootstrap 1-batch loss").
        bootstrap_ctx = self._state.torchrec_context_type(index=-1, version=1)

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
            context=bootstrap_ctx,
            dist_stream=data_dist_stream,
            default_stream=default_stream,
            batch=peek_batch,
            apply_jit=self._apply_jit,
            pipelined_forward=pipelined_forward_type,
            pipeline_postproc=self._pipeline_postproc,
        )
        self._state.pipelined_modules = pipelined_modules
        self._state.pipelined_postprocs = pipelined_postprocs

        # Bootstrap the input_dist so _input_dists attribute exists,
        # enabling the KJT monkeypatch. The bootstrap ctx is then
        # abandoned (one real batch is lost — known limitation).
        _start_data_dist(pipelined_modules, peek_batch, bootstrap_ctx)
        self._original_kjt_dist_forwards = _override_input_dist_forwards(
            pipelined_modules
        )

    def _ensure_pipe(self, peek_batch: Any) -> None:
        if self._pipe is not None:
            return
        # Build StreamPool FIRST so we can pass the final data_dist /
        # default streams into _rewrite_model — Codex flagged that
        # using throwaway streams causes PipelinedForward to capture
        # wrong handles.
        schedule, pool = self._build_schedule()

        data_dist_stream = pool.get("data_dist")
        default_stream = pool.get("default")
        self._rewrite_model(peek_batch, data_dist_stream, default_stream)

        executor = (
            ThreadedExecutor(thread_map=self._thread_map) if self._threaded else None
        )
        self._pipe = SchedulablePipeline(schedule, pool, executor=executor)

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
            # Peek one batch solely to drive FX tracing +
            # _override_input_dist_forwards bootstrap. The peek batch
            # is then DROPPED — its bootstrap input_dist collective is
            # in-flight on a throwaway context and would conflict with
            # the engine's per-batch contexts (different awaitable
            # types before vs after the monkeypatch). Accepting a
            # 1-batch loss is far simpler than threading the bootstrap
            # context into the engine's ring.
            try:
                peek_batch = next(dataloader_iter)
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
                    peek_batch = _to_device(
                        peek_batch, self._state.device, non_blocking=True
                    )
            else:
                peek_batch = _to_device(
                    peek_batch, self._state.device, non_blocking=True
                )
            self._ensure_pipe(peek_batch)
            # peek_batch and its bootstrap context are intentionally
            # NOT re-injected into dataloader_iter.

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
