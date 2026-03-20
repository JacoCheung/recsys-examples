#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""SWPipeline-based training pipeline adapter for recsys-examples.

Decomposes one training iteration into 11 tasks and schedules them
through the SWPipeline framework.  The serial baseline (single thread,
single default stream) executes tasks via ``SWPipeline.run_one_serial_iter``.

Tasks (topological order)::

    1.  H2DAndShuffle        – H2D copy + optional KK shuffle
    2.  EmbInputDistStart    – start embedding AllToAll (input_dist)
    3.  EmbInputDistWait     – wait for input_dist awaitables
    4.  EmbPrefetch           – DynamicEmb cache warm-up
    5.  EmbForward            – embedding forward (+ dense, via model())
    6.  DenseForward          – (no-op in serial; reserved for pipelined split)
    7.  LossPostprocess       – loss allreduce
    8.  DenseBackward         – backward (+ embedding, via loss.backward())
    9.  EmbBackward           – (no-op in serial; reserved for pipelined split)
    10. FinalizeGrads         – DDP gradient allreduce
    11. OptimStep             – optimizer.step()

Dependency graph (single iteration)::

    H2DAndShuffle → EmbInputDistStart → EmbInputDistWait → EmbPrefetch
        → EmbForward → DenseForward → LossPostprocess
        → DenseBackward → EmbBackward → FinalizeGrads → OptimStep
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple

import nvtx
import torch
import torch.distributed
from commons.distributed.batch_shuffler import (
    BaseTaskBalancedBatchShuffler,
    IdentityBalancedBatchShuffler,
)
from commons.distributed.finalize_model_grads import finalize_model_grads
from commons.pipeline.sw_pipeline import (
    IterContext,
    PipelinePlan,
    PipelineTask,
    SWPipeline,
    TaskSchedule,
)
from commons.pipeline.utils import (
    In,
    NoOpStream,
    Out,
    PrefetchPipelinedForward,
    PrefetchTrainPipelineContext,
    _override_input_dist_forwards,
    _prefetch_embeddings,
    _rewrite_model,
    _start_data_dist,
    _to_device,
)
from commons.utils.distributed_utils import collective_assert
from megatron.core import parallel_state
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
from torchrec.distributed.model_parallel import ShardedModule


class SWSerialTrainPipeline:
    """Training pipeline adapter backed by SWPipeline with 11 tasks.

    Serial baseline: all tasks share stage 0, default stream, and the main
    thread.  ``SWPipeline.run_one_serial_iter()`` executes them in
    topological (dependency) order — the calling thread IS the execution
    thread.  The only exception is the KK shuffle algorithm, which is
    submitted to a background CPU thread via ``start_shuffle_async``.

    Exposes the same interface as ``JaggedMegatronTrainNonePipeline``
    (``progress(dataloader_iter)`` + ``_model`` attribute).

    Args:
        model:  The model to train (may be wrapped by DistributedModelParallel).
        optimizer:  Dense optimizer.
        device:  Target CUDA device.
        batch_shuffler:  Batch shuffler for load balancing.
        pipeline_depth:  SWPipeline depth (1 = serial, no overlap).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        batch_shuffler: BaseTaskBalancedBatchShuffler = IdentityBalancedBatchShuffler(),
        pipeline_depth: int = 1,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._device = device
        self._batch_shuffler = batch_shuffler
        self._is_identity_shuffler = isinstance(
            batch_shuffler, IdentityBalancedBatchShuffler
        )

        # TorchRec pipeline infrastructure (PipelinedForward, context, …)
        self._pipeline_ctx = PrefetchTrainPipelineContext(version=0)
        self._pipelined_modules: List[ShardedModule] = []
        self._original_forwards: list = []
        self._original_kjt_dist_forwards: list = []
        self._initialized: bool = False

        # Per-iteration result buffer: iter_idx → (reporting_loss, output)
        self._results: Dict[int, Tuple[torch.Tensor, Any]] = {}
        self._iter_count: int = 0

        # Build the 11-task SWPipeline
        self._sw_pipeline: SWPipeline = self._build_pipeline()

    # ------------------------------------------------------------------
    # One-time initialisation (first batch)
    # ------------------------------------------------------------------

    def _ensure_initialized(self, batch_gpu: In) -> None:
        """Rewrite model with PrefetchPipelinedForward, kick off the
        first input_dist, and override KJT dist forwards.

        Must be called with a **GPU** batch so that ``_rewrite_model``
        can inspect the KJT attribute names.
        """
        if self._initialized:
            return

        default_stream = torch.get_device_module(self._device).current_stream()
        (
            self._pipelined_modules,
            self._model,
            self._original_forwards,
            _pipelined_postprocs,
            _non_pipelined,
        ) = _rewrite_model(
            model=self._model,
            context=self._pipeline_ctx,
            dist_stream=None,  # serial — everything on default stream
            default_stream=default_stream,
            batch=batch_gpu,
            pipelined_forward=PrefetchPipelinedForward,
        )

        # First input_dist initialises the internal dist objects so that
        # _override_input_dist_forwards can find them.
        _start_data_dist(self._pipelined_modules, batch_gpu, self._pipeline_ctx)
        self._original_kjt_dist_forwards = _override_input_dist_forwards(
            self._pipelined_modules
        )
        self._initialized = True

    # ------------------------------------------------------------------
    # SWPipeline construction — 11 tasks
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> SWPipeline:
        device = self._device
        device_idx = device.index if device.index is not None else 0
        pipeline_ctx = self._pipeline_ctx
        results = self._results
        # local aliases for frequently used attributes — closures capture
        # ``self`` directly for attributes that may be reassigned at runtime
        # (e.g. ``self._model`` is re-bound by ``_rewrite_model``).

        # ---- 1. H2D and shuffle ----

        def h2d_and_shuffle(ctx: IterContext) -> None:
            with nvtx.annotate("## H2D and shuffle ##"):
                batch_gpu = _to_device(ctx.batch, device, non_blocking=True)
                if not self._is_identity_shuffler:
                    dp_group = parallel_state.get_data_parallel_group()
                    handle = self._batch_shuffler.start_shuffle_async(
                        batch_gpu, dp_group
                    )
                    batch_gpu = self._batch_shuffler.finish_shuffle(
                        batch_gpu, handle, dp_group
                    )
                ctx.batch_gpu = batch_gpu

        # ---- 2. Embedding input dist start ----

        def emb_input_dist_start(ctx: IterContext) -> None:
            with nvtx.annotate("## start_sparse_data_dist ##"):
                if not self._initialized:
                    # First batch: _ensure_initialized already ran _start_data_dist.
                    self._ensure_initialized(ctx.batch_gpu)
                    return
                _start_data_dist(self._pipelined_modules, ctx.batch_gpu, pipeline_ctx)

        # ---- 3. Embedding input dist wait ----

        def emb_input_dist_wait(ctx: IterContext) -> None:
            with nvtx.annotate("## wait_sparse_data_dist ##"):
                pipeline_ctx.module_contexts = (
                    pipeline_ctx.module_contexts_next_batch.copy()
                )
                pipeline_ctx.input_dist_tensors_requests.clear()
                for names, awaitable in pipeline_ctx.fused_splits_awaitables:
                    for name, request in zip(names, awaitable.wait()):
                        pipeline_ctx.input_dist_tensors_requests[name] = request

        # ---- 4. Embedding prefetch ----

        def emb_prefetch(ctx: IterContext) -> None:
            with nvtx.annotate("## sharded_module_prefetch ##"):
                pipeline_ctx.module_input_post_prefetch.clear()
                pipeline_ctx.module_contexts_post_prefetch.clear()

                data_per_module = _prefetch_embeddings(
                    ctx.batch_gpu,
                    pipeline_ctx,
                    self._pipelined_modules,
                    device,
                    NoOpStream,  # no separate stream for serial
                    None,  # data_dist_stream = None
                    None,  # forward_stream = None (serial: same stream)
                )
                for module in self._pipelined_modules:
                    fwd = module.forward
                    data = data_per_module[fwd._name]
                    pipeline_ctx.module_input_post_prefetch[fwd._name] = data
                    pipeline_ctx.module_contexts_post_prefetch[
                        fwd._name
                    ] = pipeline_ctx.module_contexts.pop(fwd._name)

        # ---- 5. Embedding forward ----
        # In serial mode, model(batch) triggers PrefetchPipelinedForward
        # which handles both embedding lookup and dense layers.  A true
        # split into separate embedding / dense forward tasks requires
        # model-level API support and is left for the pipelined variant.

        def embedding_forward(ctx: IterContext) -> None:
            if self._model.training:
                with nvtx.annotate("## zero_grad ##"):
                    if hasattr(self._model.module, "zero_grad_buffer"):
                        self._model.module.zero_grad_buffer()
                    self._optimizer.zero_grad()

            with nvtx.annotate("## forward ##"):
                ctx.losses, ctx.output = self._model(ctx.batch_gpu)

        # ---- 6. Dense forward (no-op in serial) ----

        def dense_forward(ctx: IterContext) -> None:
            pass

        # ---- 7. Loss postprocess ----

        def loss_postprocess(ctx: IterContext) -> None:
            with nvtx.annotate("## loss postprocess ##"):
                collective_assert(
                    not torch.isnan(ctx.losses).any(), "loss has nan value"
                )
                local_tokens = torch.tensor(ctx.losses.size(0), device=device).float()
                ctx.local_loss = torch.cat(
                    [torch.sum(ctx.losses).view(1), local_tokens.view(1)]
                )
                ctx.reporting_loss = ctx.local_loss.clone().detach()
                torch.distributed.all_reduce(
                    ctx.reporting_loss,
                    group=parallel_state.get_data_parallel_group(),
                )
                # Store result immediately (valid for both train and eval).
                results[ctx.iter_idx] = (ctx.reporting_loss, ctx.output)

        # ---- 8. Dense backward ----
        # loss.backward() computes gradients for all parameters (dense +
        # embedding).  Splitting requires torch.autograd.grad with
        # retain_graph; left for the pipelined variant.

        def dense_backward(ctx: IterContext) -> None:
            if not self._model.training:
                return
            with nvtx.annotate("## backward ##"):
                dp_size = parallel_state.get_data_parallel_world_size()
                local_loss_average = ctx.local_loss[0] / ctx.reporting_loss[1] * dp_size
                local_loss_average.backward()

        # ---- 9. Embedding backward (no-op in serial) ----

        def embedding_backward(ctx: IterContext) -> None:
            pass

        # ---- 10. Finalize model grads ----

        def finalize_grads(ctx: IterContext) -> None:
            if not self._model.training:
                return
            with nvtx.annotate("## finalize_model_grads ##"):
                if isinstance(self._model.module, DistributedDataParallel):
                    finalize_model_grads([self._model.module], None)

        # ---- 11. Optimizer step ----

        def optim_step(ctx: IterContext) -> None:
            if not self._model.training:
                return
            with nvtx.annotate("## optimizer step ##"):
                self._optimizer.step()

        # ---- Wrap as PipelineTasks ----

        t_h2d = PipelineTask("H2DAndShuffle", h2d_and_shuffle)
        t_dist_start = PipelineTask("EmbInputDistStart", emb_input_dist_start)
        t_dist_wait = PipelineTask("EmbInputDistWait", emb_input_dist_wait)
        t_prefetch = PipelineTask("EmbPrefetch", emb_prefetch)
        t_emb_fwd = PipelineTask("EmbForward", embedding_forward)
        t_dense_fwd = PipelineTask("DenseForward", dense_forward)
        t_loss = PipelineTask("LossPostprocess", loss_postprocess)
        t_dense_bwd = PipelineTask("DenseBackward", dense_backward)
        t_emb_bwd = PipelineTask("EmbBackward", embedding_backward)
        t_finalize = PipelineTask("FinalizeGrads", finalize_grads)
        t_optim = PipelineTask("OptimStep", optim_step)

        # Serial baseline: all stage 0, default stream, depth 1.
        plan = PipelinePlan(
            schedule={
                t_h2d: TaskSchedule(stage=0),
                t_dist_start: TaskSchedule(stage=0),
                t_dist_wait: TaskSchedule(stage=0),
                t_prefetch: TaskSchedule(stage=0),
                t_emb_fwd: TaskSchedule(stage=0),
                t_dense_fwd: TaskSchedule(stage=0),
                t_loss: TaskSchedule(stage=0),
                t_dense_bwd: TaskSchedule(stage=0),
                t_emb_bwd: TaskSchedule(stage=0),
                t_finalize: TaskSchedule(stage=0),
                t_optim: TaskSchedule(stage=0),
            },
            deps=[
                (t_dist_start, t_h2d),
                (t_dist_wait, t_dist_start),
                (t_prefetch, t_dist_wait),
                (t_emb_fwd, t_prefetch),
                (t_dense_fwd, t_emb_fwd),
                (t_loss, t_dense_fwd),
                (t_dense_bwd, t_loss),
                (t_emb_bwd, t_dense_bwd),
                (t_finalize, t_emb_bwd),
                (t_optim, t_finalize),
            ],
            cross_iter_deps=[],
            pipeline_depth=1,
        )

        return SWPipeline(plan, device=device_idx)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def progress(self, dataloader_iter: Iterator[In]) -> Tuple[torch.Tensor, Out]:
        """Run one training (or eval) step using the 11-task SWPipeline.

        Pulls one batch from *dataloader_iter*, executes all tasks
        serially on the default stream via ``run_one_serial_iter``, and
        returns ``(reporting_loss, model_output)``.

        Raises:
            StopIteration: when the data iterator is exhausted.
        """
        batch = next(dataloader_iter, None)
        if batch is None:
            raise StopIteration

        idx = self._iter_count
        self._sw_pipeline.run_one_serial_iter(batch, idx)
        self._iter_count += 1
        return self._results.pop(idx)


# -- Factory registration ---------------------------------------------

from commons.pipeline.train_pipeline_factory import TrainPipelineFactory  # noqa: E402

TrainPipelineFactory.register("jagged_sw_serial", SWSerialTrainPipeline)
