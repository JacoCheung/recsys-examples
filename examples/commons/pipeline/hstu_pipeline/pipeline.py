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


# ----------------------------------------------------------------------
# Safe thread assignment for HSTU
# ----------------------------------------------------------------------
#
# HSTU tasks are split into 2 CPU threads:
#
#   "io"      — pure data movement: h2d, start_shuffle, finish_shuffle.
#               These touch only their own slot's tensor + the batch
#               shuffler.
#   "compute" — everything else: start_input_dist, wait_input_dist,
#               prefetch, zero_grad, global_tokens, forward,
#               backward, finalize, optimizer.
#
# Two threads (not many) lets H2D / shuffle on "io" overlap with the
# NCCL + GPU compute work on "compute", without paying the per-task
# pool dispatch overhead of full thread-per-task.
#
# Cross-thread data dependencies use threading.Event (engine handles
# this). Cross-rank NCCL ordering uses the engine's ``_NcclOrderedLock``.
#
# Users can override via ``thread_map=...`` kwarg.
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
        custom_model_fwd: Optional[Callable[[Any], Tuple[torch.Tensor, Any]]] = None,
        # Default: multi-threaded with HSTU_DEFAULT_THREAD_MAP (io +
        # compute, two threads) so H2D / shuffle on "io" overlap NCCL +
        # GPU compute on "compute". Pass ``threaded=False`` to fall back
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

        # Track whether the user supplied a custom forward so that
        # attach(new_model) only re-points model_fwd when it shadows
        # the bare model. Custom forwards intentionally diverge from
        # state.model and must survive a model swap.
        self._has_custom_model_fwd: bool = custom_model_fwd is not None

        # Lazily constructed on first progress() call.
        self._pipe: Optional[SchedulablePipeline] = None
        self._original_forwards: list = []
        self._original_kjt_dist_forwards: list = []
        self._model_attached: bool = True

    @property
    def _model(self) -> torch.nn.Module:
        """Match legacy ``self._model`` so existing training loops
        (e.g. ``train_with_pipeline``) can call ``pipeline._model.train()``
        unchanged. The underlying model lives in ``self._state.model``;
        ``attach()``/``detach()`` keep it in sync.
        """
        return self._state.model

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
        h2d_lookahead = depth + 1  # = 2 for depth=1
        input_dist_lookahead = depth  # = 1 for depth=1
        prefetch_lookahead = 1 if self._prefetch else None

        tasks = [
            make_h2d_task(self._state, lookahead=h2d_lookahead),
            make_start_shuffle_task(self._state, lookahead=h2d_lookahead),
            make_finish_shuffle_task(self._state, lookahead=h2d_lookahead),
            make_start_input_dist_task(self._state, lookahead=input_dist_lookahead),
            make_wait_input_dist_task(self._state, lookahead=input_dist_lookahead),
        ]
        # Note: for the prefetch variant, prefetch_embeddings is
        # declared AFTER forward (not before). Same-iter ordering
        # is: forward consumes prev iter's prefetched data first,
        # THEN prefetch_embeddings adds new keys for next iter.
        # This keeps dynamicemb cache outstanding peak at ~1 batch
        # (matches legacy JaggedMegatronPrefetch progress order at
        # train_pipeline.py:993-997). Reversing the order would
        # peak at ~2-3 batches and overflow cache capacity.
        tasks.extend(
            [
                make_zero_grad_task(self._state),
                make_global_tokens_task(self._state),
                make_nccl_safety_barrier_task(self._state),
                make_forward_task(self._state, prefetch=self._prefetch),
            ]
        )
        if self._prefetch:
            tasks.append(make_prefetch_task(self._state, lookahead=prefetch_lookahead))
        # backward dependency edges:
        #   - depends_on=("zero_grad",) — same-batch logical edge:
        #     backward processing batch K writes model.grad which
        #     zero_grad must have cleared first (also for batch K).
        #     Engine has no reads/writes edge between them (zero_grad
        #     mutates out-of-slot model state), so we declare
        #     explicitly. Without this, a custom thread_map could
        #     reorder them across threads.
        #   - same_progress_sync=("prefetch_embeddings",) (prefetch variant
        #     only) — NOT a logical data-flow edge. backward (la=0,
        #     batch K) and prefetch_embeddings (la=1, batch K+1) run
        #     in the same progress() processing different batches,
        #     but share the dynamicemb cache as global mutable state.
        #     prefetch writes the cache on prefetch_stream; backward
        #     reads it via autograd on default_stream — needs cross-
        #     stream GPU coherency wait. Mirrors legacy
        #     ``default_stream.wait_stream(prefetch_stream)``
        #     (train_pipeline.py:993-997).
        backward_deps = ("zero_grad",)
        backward_same_progress_sync: tuple = ()
        if self._prefetch:
            backward_same_progress_sync = ("prefetch_embeddings",)
        tasks.extend(
            [
                make_backward_task(
                    self._state,
                    depends_on=backward_deps,
                    same_progress_sync=backward_same_progress_sync,
                ),
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
        self,
        peek_batch: Any,
        data_dist_stream: Any,
        default_stream: Any,
        memcpy_stream: Any = None,
    ):
        """Call torchrec's _rewrite_model once, using the SAME streams
        that will later be installed in the engine's StreamPool, and
        return the **real per-batch context** that was used for the
        monkey-patch bootstrap.

        Bootstrap stream discipline (Codex HIGH-1): the peek batch was
        H2D'd on ``memcpy_stream``. Before reading it on
        ``data_dist_stream`` for ``_start_data_dist``, ``data_dist``
        must wait on ``memcpy``. We also enter the ``data_dist`` stream
        context for the bootstrap call so any NCCL it issues lands on
        the right communicator path — same discipline torchrec's own
        ``start_sparse_data_dist`` follows
        (``train_pipeline.py:440 with self._stream_context(self._data_dist_stream):``).
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
            _pipelined_postprocs,
            _names,
        ) = torchrec_rewrite_model(
            model=self._state.model,
            context=peek_ctx,
            dist_stream=data_dist_stream,
            default_stream=default_stream,
            batch=peek_batch,
            apply_jit=self._apply_jit,
            pipelined_forward=pipelined_forward_type,
            pipeline_postproc=False,
        )
        self._state.pipelined_modules = pipelined_modules

        # Bootstrap the input_dist on the engine's data_dist stream
        # AFTER ensuring the peek H2D on memcpy_stream is visible
        # there. Mirrors legacy stream discipline.
        device = self._state.device
        if (
            device.type == "cuda"
            and data_dist_stream is not None
            and memcpy_stream is not None
        ):
            data_dist_stream.wait_stream(memcpy_stream)
            with torch.cuda.stream(data_dist_stream):
                _start_data_dist(pipelined_modules, peek_batch, peek_ctx)
        else:
            _start_data_dist(pipelined_modules, peek_batch, peek_ctx)
        self._original_kjt_dist_forwards = _override_input_dist_forwards(
            pipelined_modules
        )
        return peek_ctx

    def _ensure_pipe(self, peek_batch_cpu: Any) -> None:
        if self._pipe is not None:
            return
        # Build StreamPool FIRST so the peek H2D + bootstrap
        # _start_data_dist run on the engine's actual streams (Codex
        # HIGH-1 fix) — previously H2D used a throwaway stream and
        # bootstrap had no stream context, risking NCCL submission
        # on the wrong stream.
        schedule, pool = self._build_schedule()

        memcpy_stream = pool.get("memcpy")
        data_dist_stream = pool.get("data_dist")
        default_stream = pool.get("default")

        # H2D peek batch on engine's memcpy stream (not throwaway).
        from commons.pipeline.utils import _to_device

        device = self._state.device
        if device.type == "cuda" and memcpy_stream is not None:
            with torch.cuda.stream(memcpy_stream):
                peek_batch_gpu = _to_device(peek_batch_cpu, device, non_blocking=True)
            # record_stream so the seeded tensor stays alive across
            # all consumer streams (default for forward, data_dist
            # for input_dist, prefetch for prefetch_embeddings —
            # the prefetch slot only exists in prefetch mode).
            consumers = [default_stream, data_dist_stream]
            if self._prefetch:
                consumers.append(pool.get("prefetch"))
            for consumer in consumers:
                if consumer is not None:
                    peek_batch_gpu.record_stream(consumer)
        else:
            peek_batch_gpu = _to_device(peek_batch_cpu, device, non_blocking=True)

        # _rewrite_model bootstraps on a REAL per-batch ctx on the
        # engine's data_dist stream (after wait_stream on memcpy).
        peek_ctx = self._rewrite_model(
            peek_batch_gpu,
            data_dist_stream,
            default_stream,
            memcpy_stream=memcpy_stream,
        )

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
        self._pipe._seed_first_batch(seeded)

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
            # bootstrap. The peek batch is SEEDED into the engine
            # ring so it flows through the pipeline as the first
            # real batch (no data loss, no dynamicemb cache leak).
            # H2D + bootstrap stream context happen INSIDE
            # _ensure_pipe on the engine's actual streams (Codex
            # HIGH-1).
            try:
                peek_cpu = next(dataloader_iter)
            except StopIteration as e:
                raise StopIteration(
                    "HSTUPipeline: dataloader was empty on first progress() call"
                ) from e
            self._ensure_pipe(peek_cpu)

        # Engine does the rest.
        return self._pipe.progress(dataloader_iter)

    def attach(self, model: Optional[torch.nn.Module] = None) -> None:
        """Matches legacy ``attach()`` — re-enable the pipeline after a
        prior ``detach()``. After this call the next ``progress()`` will
        rebuild the engine and re-install the pipelined forwards via
        ``_rewrite_model``; ``detach()`` clears ``self._pipe`` and the
        bookkeeping state so this happens automatically (Codex
        B-MEDIUM-1).
        """
        if model is not None:
            self._state.model = model
            # Codex HIGH (2026-04-26): without this line, the forward
            # task would still call the construction-time model. Only
            # mirror when model_fwd was the default (== state.model);
            # custom forwards are intentional and must survive attach.
            if not self._has_custom_model_fwd:
                self._state.model_fwd = model
        self._model_attached = True

    def detach(self) -> torch.nn.Module:
        """Restore the original (non-pipelined) module forwards and
        return the bare model. Also clears HSTUPipeline-internal
        bookkeeping (``self._pipe`` and the pipelined-modules /
        original-forwards lists) so the next ``progress()`` after
        ``attach()`` rebuilds the engine from scratch — i.e.
        ``_rewrite_model`` runs again on the (possibly modified)
        model and a fresh ``SchedulablePipeline`` is constructed.
        Without this reset, an attach/progress sequence after detach
        would still hold references to torn-down sharded module
        forwards and skip ``_rewrite_model`` (Codex B-MEDIUM-1).
        """
        if self._state.pipelined_modules:
            from commons.pipeline.utils import _pipeline_detach_model

            _pipeline_detach_model(
                model=self._state.model,
                pipelined_modules=self._state.pipelined_modules,
                original_forwards=self._original_forwards,
                original_kjt_dist_forwards=self._original_kjt_dist_forwards,
                pipelined_postprocs=[],
            )
        # Reset bookkeeping so next progress() rebuilds the engine.
        # See class docstring + B-MEDIUM-1 follow-up.
        if self._pipe is not None:
            self._pipe.shutdown()
        self._pipe = None
        self._state.pipelined_modules = []
        self._original_forwards = []
        self._original_kjt_dist_forwards = []
        self._model_attached = False
        return self._state.model

    def shutdown(self) -> None:
        if self._pipe is not None:
            self._pipe.shutdown()

    def __enter__(self) -> "HSTUPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
