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

"""HSTU adapter for the schedulable pipeline engine.

The adapter keeps the existing training-loop contract while expressing
H2D, shuffle, input distribution, forward/backward, and optimizer work
as engine tasks. Engine construction is lazy because TorchRec
``_rewrite_model`` needs the first batch for tracing.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

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
    make_compute_output_dist_task,
    make_finalize_grads_task,
    make_finish_shuffle_task,
    make_forward_task,
    make_global_tokens_task,
    make_h2d_task,
    make_optimizer_step_task,
    make_prefetch_task,
    make_start_input_dist_task,
    make_start_shuffle_task,
    make_wait_input_dist_task,
    make_watchdog_task,
    make_zero_grad_task,
)

__all__ = [
    "HSTUPipeline",
    "HSTU_DEFAULT_THREAD_MAP",
    "HSTU_THREAD_MAP_PRESETS",
    "resolve_hstu_thread_map_variant",
]

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Safe thread assignment for HSTU
# ----------------------------------------------------------------------
#
# Default HSTU tasks are split into two CPU threads:
#
#   "io"      — pure data movement: h2d, start_shuffle, finish_shuffle.
#               These touch only their own slot's tensor + the batch
#               shuffler.
#   "compute" — everything else: start_input_dist, wait_input_dist,
#               prefetch, zero_grad, global_tokens, forward,
#               backward, finalize, optimizer.
#
# This overlaps H2D/shuffle with NCCL/compute without per-task thread
# dispatch. The engine handles cross-thread deps and NCCL ordering.
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
    "compute_output_dist": "compute",
    "forward": "compute",
    "backward": "compute",
    "finalize_model_grads": "compute",
    "optimizer_step": "compute",
    # Optional diagnostic task: absent unless CUDA_MEM_WATCHDOG=1.
    # Keep it mapped so enabling the watchdog does not fall through to
    # the engine's generic "default" thread.
    "watchdog_step": "compute",
}


# Named thread-map presets used by benchmark sweeps. Production callers
# normally use ``HSTU_DEFAULT_THREAD_MAP`` unless they explicitly pass a
# preset name or set ``HSTU_THREAD_MAP_VARIANT``.
HSTU_THREAD_MAP_PRESETS: dict = {
    "default": HSTU_DEFAULT_THREAD_MAP,
    "by_stream": "by_stream",
    "per_task": "per_task",
    "io_prefetch_compute": {
        # io
        "h2d": "io",
        "start_shuffle": "io",
        "finish_shuffle": "io",
        # prefetch
        "prefetch_embeddings": "prefetch",
        # compute
        "start_input_dist": "compute",
        "wait_input_dist": "compute",
        "zero_grad": "compute",
        "global_tokens_allreduce": "compute",
        "compute_output_dist": "compute",
        "forward": "compute",
        "backward": "compute",
        "finalize_model_grads": "compute",
        "optimizer_step": "compute",
        "watchdog_step": "compute",
    },
    "io_data_dist_compute": {
        "h2d": "io",
        "start_shuffle": "io",
        "finish_shuffle": "io",
        "start_input_dist": "data_dist",
        "wait_input_dist": "data_dist",
        "prefetch_embeddings": "compute",
        "zero_grad": "compute",
        "global_tokens_allreduce": "compute",
        "compute_output_dist": "compute",
        "forward": "compute",
        "backward": "compute",
        "finalize_model_grads": "compute",
        "optimizer_step": "compute",
        "watchdog_step": "compute",
    },
    "io_data_dist_prefetch_compute": {
        "h2d": "io",
        "start_shuffle": "io",
        "finish_shuffle": "io",
        "start_input_dist": "data_dist",
        "wait_input_dist": "data_dist",
        "prefetch_embeddings": "prefetch",
        "zero_grad": "compute",
        "global_tokens_allreduce": "compute",
        "compute_output_dist": "compute",
        "forward": "compute",
        "backward": "compute",
        "finalize_model_grads": "compute",
        "optimizer_step": "compute",
        "watchdog_step": "compute",
    },
    # Benchmark variant: keep every off-critical-path task on one worker.
    "all_noncritical_on_io": {
        "h2d": "io",
        "start_shuffle": "io",
        "finish_shuffle": "io",
        "start_input_dist": "io",
        "wait_input_dist": "io",
        "prefetch_embeddings": "io",
        "zero_grad": "compute",
        "global_tokens_allreduce": "compute",
        "compute_output_dist": "compute",
        "forward": "compute",
        "backward": "compute",
        "finalize_model_grads": "compute",
        "optimizer_step": "compute",
        "watchdog_step": "compute",
    },
}


def resolve_hstu_thread_map_variant(name: Optional[str]) -> Any:
    """Resolve a named variant from ``HSTU_THREAD_MAP_PRESETS``.

    Returns ``None`` for ``name=None`` (caller falls back to
    ``HSTU_DEFAULT_THREAD_MAP``). Raises ``ValueError`` for unknown
    names so a typo in benchmark config doesn't silently fall back to
    the default and corrupt A/B comparisons.
    """
    if name is None:
        return None
    if name not in HSTU_THREAD_MAP_PRESETS:
        raise ValueError(
            f"Unknown HSTU thread_map variant {name!r}. "
            f"Known: {sorted(HSTU_THREAD_MAP_PRESETS.keys())}"
        )
    return HSTU_THREAD_MAP_PRESETS[name]


class HSTUPipeline:
    """Engine-backed HSTU pipeline adapter.

    ``progress()`` returns ``(local_loss_sum, global_tokens, output)``.
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
        # Defaults to the two-thread HSTU map; callers can pass
        # ``threaded=False`` or a custom ``thread_map``.
        threaded: bool = True,
        thread_map: Any = None,
    ) -> None:
        if prefetch_depth < 1:
            raise ValueError(f"prefetch_depth must be >= 1, got {prefetch_depth}")

        # Resolve the default map or an env-selected benchmark preset.
        if threaded and thread_map is None:
            env_variant = os.environ.get("HSTU_THREAD_MAP_VARIANT")
            if env_variant:
                thread_map = resolve_hstu_thread_map_variant(env_variant)
            else:
                thread_map = HSTU_DEFAULT_THREAD_MAP
        elif (
            threaded
            and isinstance(thread_map, str)
            and thread_map in HSTU_THREAD_MAP_PRESETS
        ):
            # Allow callers to pass a preset name directly.
            thread_map = resolve_hstu_thread_map_variant(thread_map)

        self._prefetch = prefetch
        self._prefetch_depth = prefetch_depth
        self._apply_jit = apply_jit
        self._threaded = threaded
        self._thread_map = thread_map

        # Optional auto-scheduler: a JSON cost model can override
        # off-default lookahead values within the in-flight budget.
        self._autosched_cost_file: Optional[str] = (
            os.environ.get("HSTU_AUTOSCHED_COST_FILE", "").strip() or None
        )
        self._autosched_max_in_flight: int = int(
            os.environ.get("HSTU_AUTOSCHED_MAX_IN_FLIGHT", "5")
        )

        # Default shuffler is identity (no-op).
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
        """Compatibility alias for training loops that access ``_model``."""
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

    def _build_schedule(
        self,
        la_overrides: Optional[Dict[str, int]] = None,
    ) -> Tuple[Schedule, StreamPool]:
        """Construct the HSTU schedule and stream pool.

        ``prefetch_depth=1`` carries three batches in flight
        (compute@0, input_dist/prefetch@1, h2d/shuffle@2). Larger depths
        add buffer slots between input distribution and compute.
        """
        # Benchmark sweeps may override the public depth knob with a
        # named lookahead profile.
        import os as _os

        _depth_env = _os.environ.get("HSTU_LA_DEPTH", "").strip()
        if not _depth_env:
            h2d_lookahead = self._prefetch_depth + 1
            start_shuffle_lookahead = self._prefetch_depth + 1
            finish_shuffle_lookahead = self._prefetch_depth + 1
            start_input_dist_lookahead = self._prefetch_depth
            wait_input_dist_lookahead = self._prefetch_depth
            prefetch_lookahead = 1 if self._prefetch else None
        else:
            try:
                _depth = int(_depth_env)
            except ValueError as e:
                raise ValueError(
                    f"HSTU_LA_DEPTH={_depth_env!r} not supported; expected 3 or 6"
                ) from e
            if _depth == 3:
                h2d_lookahead = 2
                start_shuffle_lookahead = 2
                finish_shuffle_lookahead = 2
                start_input_dist_lookahead = 1
                wait_input_dist_lookahead = 1
                prefetch_lookahead = 1 if self._prefetch else None
            elif _depth == 6:
                h2d_lookahead = 5
                start_shuffle_lookahead = 4
                finish_shuffle_lookahead = 3
                start_input_dist_lookahead = 3
                wait_input_dist_lookahead = 2
                prefetch_lookahead = 1 if self._prefetch else None
            else:
                raise ValueError(
                    f"HSTU_LA_DEPTH={_depth} not supported; expected 3 or 6"
                )

        # Apply auto-scheduler / explicit overrides to off-default work.
        if la_overrides:
            h2d_lookahead = la_overrides.get("h2d", h2d_lookahead)
            start_shuffle_lookahead = la_overrides.get(
                "start_shuffle", start_shuffle_lookahead
            )
            finish_shuffle_lookahead = la_overrides.get(
                "finish_shuffle", finish_shuffle_lookahead
            )
            start_input_dist_lookahead = la_overrides.get(
                "start_input_dist", start_input_dist_lookahead
            )
            wait_input_dist_lookahead = la_overrides.get(
                "wait_input_dist", wait_input_dist_lookahead
            )
            if prefetch_lookahead is not None:
                prefetch_lookahead = la_overrides.get(
                    "prefetch_embeddings", prefetch_lookahead
                )

        # Gate lookahead work behind compute_output_dist so the current
        # critical chain submits before future-batch IO/NCCL.
        critical_gate: tuple = ("compute_output_dist",)
        tasks = [
            make_h2d_task(
                self._state,
                lookahead=h2d_lookahead,
                same_progress_sync=critical_gate,
            ),
            make_start_shuffle_task(
                self._state,
                lookahead=start_shuffle_lookahead,
                same_progress_sync=critical_gate,
            ),
            make_finish_shuffle_task(
                self._state,
                lookahead=finish_shuffle_lookahead,
                same_progress_sync=critical_gate,
            ),
            make_start_input_dist_task(
                self._state,
                lookahead=start_input_dist_lookahead,
                same_progress_sync=critical_gate,
            ),
            make_wait_input_dist_task(
                self._state,
                lookahead=wait_input_dist_lookahead,
                same_progress_sync=critical_gate,
            ),
        ]
        # Keep prefetch after forward in declaration order: forward
        # consumes the prior batch's prefetched data before the next
        # prefetch adds keys, limiting DynamicEmb cache pressure.
        tasks.extend(
            [
                make_zero_grad_task(self._state),
                make_global_tokens_task(self._state),
            ]
        )
        # In prefetch mode, compute_output_dist consumes prefetch output.
        if self._prefetch:
            tasks.append(
                make_prefetch_task(
                    self._state,
                    lookahead=prefetch_lookahead,
                    same_progress_sync=critical_gate,
                )
            )
        # compute_output_dist produces awaitables consumed by forward.
        # The memcpy safety wait is folded into forward's body.
        tasks.extend(
            [
                make_compute_output_dist_task(
                    self._state,
                    prefetch=self._prefetch,
                ),
                make_forward_task(self._state, prefetch=self._prefetch),
            ]
        )
        # zero_grad is model-state ordering; prefetch sync is a
        # same-progress GPU coherency edge for DynamicEmb cache state.
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
            ]
        )
        if os.environ.get("CUDA_MEM_WATCHDOG", "0") == "1":
            tasks.append(make_watchdog_task())

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
        """Rewrite TorchRec modules once on the engine's actual streams."""
        from commons.pipeline.utils import _override_input_dist_forwards
        from commons.pipeline.utils import _rewrite_model as torchrec_rewrite_model
        from commons.pipeline.utils import _start_data_dist

        from .embedding_split import (
            HSTUPipelinedForward,
            HSTUPrefetchPipelinedForward,
            HSTUTrainPipelineContext,
        )

        # One context type covers input_dist, prefetch, module context,
        # and output-a2a awaitable fields.
        self._state.torchrec_context_type = HSTUTrainPipelineContext
        peek_ctx = self._state.create_torchrec_ctx()

        # Swap TorchRec's forward wrapper for one that consumes the
        # awaitable produced by compute_output_dist.
        pipelined_forward_type = (
            HSTUPrefetchPipelinedForward if self._prefetch else HSTUPipelinedForward
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

        # Bootstrap input_dist on data_dist after peek H2D is visible.
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
        # Build StreamPool first so peek setup uses the real streams.
        schedule, pool = self._build_schedule()

        # Rebuild with auto-scheduler lookahead recommendations, if enabled.
        if self._autosched_cost_file:
            from commons.pipeline.engine.autosched import (
                CostModel,
                auto_assign_lookaheads,
            )

            cost_model = CostModel.from_json(self._autosched_cost_file)
            recommended = auto_assign_lookaheads(
                schedule,
                cost_model,
                max_in_flight=self._autosched_max_in_flight,
            )
            current_la = {t.name: t.batch_offset for t in schedule.all_tasks()}
            changed = {
                name: (current_la[name], recommended[name])
                for name in recommended
                if current_la.get(name) != recommended[name]
            }
            if changed:
                logger.info(
                    "Auto-scheduler recommended lookahead overrides: %s",
                    changed,
                )
                schedule, pool = self._build_schedule(la_overrides=recommended)

        memcpy_stream = pool.get("memcpy")
        data_dist_stream = pool.get("data_dist")
        default_stream = pool.get("default")

        # H2D peek batch on the engine memcpy stream.
        from commons.pipeline.utils import _to_device

        device = self._state.device
        if device.type == "cuda" and memcpy_stream is not None:
            with torch.cuda.stream(memcpy_stream):
                peek_batch_gpu = _to_device(peek_batch_cpu, device, non_blocking=True)
            # Keep the seeded tensor alive across all consumer streams.
            consumers = [default_stream, data_dist_stream]
            if self._prefetch:
                consumers.append(pool.get("prefetch"))
            for consumer in consumers:
                if consumer is not None:
                    peek_batch_gpu.record_stream(consumer)
        else:
            peek_batch_gpu = _to_device(peek_batch_cpu, device, non_blocking=True)

        # Shuffle before rewrite so the seeded context and batch
        # describe the same rows.
        if self._state.is_identity_shuffler:
            peek_batch_shuffled = peek_batch_gpu
        else:
            from megatron.core import parallel_state

            shuffle_kwargs: dict = {}
            if device.type == "cuda" and memcpy_stream is not None:
                shuffle_ctx: Any = torch.cuda.stream(memcpy_stream)
            else:
                shuffle_ctx = contextlib.nullcontext()
            with shuffle_ctx:
                peek_batch_shuffled = self._state.batch_shuffler.shuffle(
                    peek_batch_gpu,
                    parallel_state.get_data_parallel_group(),
                    **shuffle_kwargs,
                )
            # Same lifetime guarantee as peek_batch_gpu.
            if device.type == "cuda" and memcpy_stream is not None:
                for consumer in consumers:
                    if consumer is not None and hasattr(
                        peek_batch_shuffled, "record_stream"
                    ):
                        peek_batch_shuffled.record_stream(consumer)

        # Bootstrap rewrite on the shuffled peek batch.
        peek_ctx = self._rewrite_model(
            peek_batch_shuffled,
            data_dist_stream,
            default_stream,
            memcpy_stream=memcpy_stream,
        )

        executor = (
            ThreadedExecutor(thread_map=self._thread_map) if self._threaded else None
        )
        self._pipe = SchedulablePipeline(schedule, pool, executor=executor)

        # Seed the engine so first-iteration setup tasks skip duplicate work.
        seeded: dict = {
            "batch_cpu": peek_batch_cpu,
            "batch_gpu": peek_batch_gpu,
            "shuffled_batch": peek_batch_shuffled,
            "torchrec_ctx": peek_ctx,
        }
        self._pipe._seed_first_batch(seeded)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def progress(
        self, dataloader_iter: Iterator[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Drive one full pipeline iteration.

        On the first call, peeks the first batch to run ``_rewrite_model``,
        then builds the engine. Returns ``(loss, global_tokens, output)``.
        """
        if not self._model_attached:
            self.attach(self._state.model)

        if self._pipe is None:
            # Peek one batch for rewrite/bootstrap, then seed it as the
            # first real in-flight batch.
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
        """Re-enable the pipeline after ``detach()``."""
        if model is not None:
            self._state.model = model
            # Only mirror when model_fwd was the default (== state.model);
            # custom forwards are intentional and must survive attach.
            if not self._has_custom_model_fwd:
                self._state.model_fwd = model
        self._model_attached = True

    def detach(self) -> torch.nn.Module:
        """Restore original module forwards and clear engine state."""
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
