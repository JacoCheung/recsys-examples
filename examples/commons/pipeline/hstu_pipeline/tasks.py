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

"""HSTU-specific Task factories.

Each factory returns a `commons.pipeline.engine.Task` that performs
one step from the legacy `JaggedMegatronTrainPipelineSparseDist.progress()`
loop. Tasks are stateless closures over a shared ``PipelineState`` that
the HSTUPipeline owns — references to the optimizer, pipelined
modules, shuffler, etc., are captured at schedule-build time.

Slot conventions (see SPEC_p2.md §5.2):

  ``batch_cpu``       engine-populated; raw batch from dataloader
  ``batch_gpu``       H2D'd pre-shuffle batch
  ``shuffled_batch``  post-shuffle batch ready for input_dist
  ``torchrec_ctx``    per-batch ``TrainPipelineContext`` (travels with slot)
  ``global_tokens``   scalar AllReduce result (offset=0)
  ``shuffle_handle``  ShuffleHandle from ``start_shuffle_async`` (optional)
  ``losses``          forward losses tensor (offset=0)
  ``output``          forward secondary output (offset=0)
  ``step_result``     final return tuple (offset=0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import nvtx
import torch
from commons.pipeline.engine import Task


@dataclass
class PipelineState:
    """Shared mutable state referenced by task closures.

    Not per-iteration state — that lives in slots. This holds
    references set once at pipeline construction: the model,
    optimizer, device, the pipelined modules list (written by
    ``_rewrite_model``), and the shuffler.
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device
    pipelined_modules: List[Any] = field(default_factory=list)
    batch_shuffler: Any = None
    is_identity_shuffler: bool = True
    model_fwd: Optional[Callable] = None
    assert_nan_loss: bool = False
    next_ctx_index: int = 0
    # Prefetch variants use PrefetchTrainPipelineContext; non-prefetch
    # uses TrainPipelineContext. Set at HSTUPipeline construction.
    torchrec_context_type: Any = None
    # True when the engine schedule includes the prefetch_embeddings
    # task (i.e. ``HSTUPipeline(prefetch=True)``). Read by
    # ``make_compute_output_dist_task`` to choose where to source the
    # KJT input + module ctx from. Set at HSTUPipeline construction.
    uses_prefetch: bool = False

    def create_torchrec_ctx(self):
        """Factory for a fresh per-batch torchrec context.

        **v1 only** — the v0 (single-shared-context, deprecated) branch
        in `commons/pipeline/utils.py::_start_data_dist` is forbidden
        in HSTUPipeline. v0 was a legacy compat code path that mixes
        cross-batch state in one context object; HSTUPipeline's engine
        treats every batch as having its own context (per-batch slot
        in the BatchRing). Mixing the two semantic models was the root
        cause of the ``prefetch + dynamic_emb`` xfail (see
        tasks/followups.md).
        """
        if self.torchrec_context_type is None:
            from commons.pipeline.utils import TrainPipelineContext

            self.torchrec_context_type = TrainPipelineContext
        ctx = self.torchrec_context_type(index=self.next_ctx_index, version=1)
        # Hard assert: prevent any subclass / future refactor from
        # silently switching to v0.
        assert (
            getattr(ctx, "version", None) == 1
        ), f"HSTUPipeline forbids v0 contexts; got version={ctx.version!r}"
        self.next_ctx_index += 1
        return ctx

    def set_module_context(self, torchrec_ctx) -> None:
        """Point every PipelinedForward at the given per-batch context.
        Matches legacy ``TrainPipelineSparseDist._set_module_context``."""
        for module in self.pipelined_modules:
            module.forward.set_context(torchrec_ctx)


# ----------------------------------------------------------------------
# 1. h2d — copy raw CPU batch to GPU on memcpy stream + create ctx
# ----------------------------------------------------------------------


def make_h2d_task(
    state: PipelineState,
    *,
    lookahead: int,
    stream: str = "memcpy",
    same_progress_sync: tuple = (),
) -> Task:
    """Copies engine-populated ``batch_cpu`` to GPU and stamps a fresh
    ``torchrec_ctx`` into the same slot store."""
    from commons.pipeline.train_pipeline import _to_device

    def _fn(ctx):
        raw = ctx.slots.get("batch_cpu", None)
        if raw is None:
            return  # drain: no new batch
        # Idempotent: if _seed_first_batch already pre-populated
        # BOTH batch_gpu AND torchrec_ctx, skip. Partial seed
        # (one of two set) is rejected — silent corruption otherwise
        # because downstream tasks would read None for the missing
        # field.
        seeded_batch_gpu = ctx.slots.get("batch_gpu", None) is not None
        seeded_ctx = ctx.slots.get("torchrec_ctx", None) is not None
        if seeded_batch_gpu and seeded_ctx:
            return  # full seed — skip
        if seeded_batch_gpu or seeded_ctx:
            raise RuntimeError(
                "h2d task: partial seed detected — exactly one of "
                "{batch_gpu, torchrec_ctx} is pre-populated. Seed "
                "either both or neither via _seed_first_batch."
            )
        with nvtx.annotate("## h2d ##"):
            batch_gpu = _to_device(raw, state.device, non_blocking=True)
        ctx.slots.set("batch_gpu", batch_gpu)
        ctx.slots.set("torchrec_ctx", state.create_torchrec_ctx())

    return Task.from_fn(
        "h2d",
        _fn,
        stream=stream,
        lookahead=lookahead,
        reads=("batch_cpu",),
        writes=(
            "batch_gpu",
            "torchrec_ctx",
        ),
        same_progress_sync=same_progress_sync,
    )


# ----------------------------------------------------------------------
# 2. zero_grad — buffer hook + optimizer.zero_grad, offset=0
# ----------------------------------------------------------------------


def make_zero_grad_task(state: PipelineState) -> Task:
    def _fn(ctx):
        # Legacy gates zero_grad + zero_grad_buffer on model.training.
        if not state.model.training:
            return
        with nvtx.annotate("## zero_grad ##"):
            model_inner = getattr(state.model, "module", state.model)
            if hasattr(model_inner, "zero_grad_buffer"):
                model_inner.zero_grad_buffer()
            state.optimizer.zero_grad()

    return Task.from_fn("zero_grad", _fn, stream="default", lookahead=0)


# ----------------------------------------------------------------------
# 3. global_tokens_allreduce — per-batch token count across DP ranks
# ----------------------------------------------------------------------


def make_global_tokens_task(state: PipelineState) -> Task:
    """NCCL AllReduce on batch.num_loss_tokens() at offset=0.
    Writes result into ``global_tokens`` slot."""

    def _fn(ctx):
        batch_gpu = ctx.slots.get("batch_gpu", None)
        if batch_gpu is None or not state.model.training:
            # drain phase OR eval mode — no AllReduce (legacy guards
            # this behind `if self._model.training`).
            ctx.slots.set("global_tokens", None)
            return
        with nvtx.annotate("## global_tokens ##"):
            tokens = batch_gpu.num_loss_tokens().to(state.device)
            torch.distributed.all_reduce(tokens)
        ctx.slots.set("global_tokens", tokens)

    return Task.from_fn(
        "global_tokens_allreduce",
        _fn,
        stream="default",
        lookahead=0,
        reads=("batch_gpu",),
        writes=("global_tokens",),
        nccl=True,
    )


# ----------------------------------------------------------------------
# 4. start_shuffle — AllGather workloads + submit KK to background thread
# ----------------------------------------------------------------------


def make_start_shuffle_task(
    state: PipelineState,
    *,
    lookahead: int,
    same_progress_sync: tuple = (),
) -> Task:
    """Phase 1 of 2-phase KK shuffler. Runs on memcpy stream.

    Emits an AllGather (NCCL) for non-identity shuffler; identity
    path is a no-op. `nccl` flag is set accordingly so the executor's
    ordered NCCL lock only serializes real collectives.
    """
    from megatron.core import parallel_state

    def _fn(ctx):
        if state.is_identity_shuffler:
            ctx.slots.set("shuffle_handle", None)
            return
        batch_gpu = ctx.slots.get("batch_gpu", None)
        if batch_gpu is None:
            ctx.slots.set("shuffle_handle", None)
            return
        # Idempotent: if _seed_first_batch already provided a shuffled
        # result (shuffled_batch slot set), the peek batch's shuffle
        # is already done. Skip re-issuing the KK AllGather.
        if ctx.slots.get("shuffled_batch", None) is not None:
            ctx.slots.set("shuffle_handle", None)
            return
        with nvtx.annotate("## start_kk_async ##"):
            handle = state.batch_shuffler.start_shuffle_async(
                batch_gpu, parallel_state.get_data_parallel_group()
            )
        ctx.slots.set("shuffle_handle", handle)

    return Task.from_fn(
        "start_shuffle",
        _fn,
        stream="memcpy",
        lookahead=lookahead,
        reads=("batch_gpu",),
        writes=("shuffle_handle",),
        nccl=not state.is_identity_shuffler,
        same_progress_sync=same_progress_sync,
    )


# ----------------------------------------------------------------------
# 5. finish_shuffle — wait KK + AllGather batch + index_select
# ----------------------------------------------------------------------


def make_finish_shuffle_task(
    state: PipelineState,
    *,
    lookahead: int,
    same_progress_sync: tuple = (),
) -> Task:
    from megatron.core import parallel_state

    def _fn(ctx):
        batch_gpu = ctx.slots.get("batch_gpu", None)
        if batch_gpu is None:
            return
        # Idempotent: if _seed_first_batch pre-populated shuffled_batch,
        # this iter's shuffle work has already been done externally.
        if ctx.slots.get("shuffled_batch", None) is not None:
            return
        if state.is_identity_shuffler:
            # Identity path: the shuffled_batch IS the batch_gpu
            ctx.slots.set("shuffled_batch", batch_gpu)
            return
        handle = ctx.slots.get("shuffle_handle", None)
        assert handle is not None, "shuffle_handle missing from slot"
        with nvtx.annotate("## finish_shuffle ##"):
            shuffled = state.batch_shuffler.finish_shuffle(
                batch_gpu, handle, parallel_state.get_data_parallel_group()
            )
        ctx.slots.set("shuffled_batch", shuffled)

    return Task.from_fn(
        "finish_shuffle",
        _fn,
        stream="memcpy",
        lookahead=lookahead,
        reads=(
            "batch_gpu",
            "shuffle_handle",
        ),
        writes=("shuffled_batch",),
        # NCCL only for non-identity shuffler (identity is a no-op).
        nccl=not state.is_identity_shuffler,
        same_progress_sync=same_progress_sync,
    )


# ----------------------------------------------------------------------
# 6. start_input_dist — torchrec splits all_to_all
# ----------------------------------------------------------------------


def make_start_input_dist_task(
    state: PipelineState,
    *,
    lookahead: int,
    same_progress_sync: tuple = (),
) -> Task:
    """Calls torchrec's start_sparse_data_dist. Mutates torchrec_ctx in place."""

    def _fn(ctx):
        shuffled = ctx.slots.get("shuffled_batch", None)
        torchrec_ctx = ctx.slots.get("torchrec_ctx", None)
        if shuffled is None or torchrec_ctx is None:
            return
        # Idempotent: if the ctx already has an in-flight splits
        # request (populated by HSTU bootstrap via _seed_first_batch),
        # skip — re-running _start_data_dist would issue a second
        # collective AND leak dynamicemb cache reservations.
        if (
            torchrec_ctx.input_dist_splits_requests
            or torchrec_ctx.fused_splits_awaitables
            or torchrec_ctx.input_dist_tensors_requests
        ):
            return
        # Import here to keep framework-free invariant for engine/
        from commons.pipeline.utils import _start_data_dist

        with nvtx.annotate(f"## start_input_dist {torchrec_ctx.index} ##"):
            _start_data_dist(state.pipelined_modules, shuffled, torchrec_ctx)

    return Task.from_fn(
        "start_input_dist",
        _fn,
        stream="data_dist",
        lookahead=lookahead,
        reads=(
            "shuffled_batch",
            "torchrec_ctx",
        ),
        # Mutates torchrec_ctx in place; we don't redeclare it as
        # writes since the slot already exists (single-writer rule
        # applies to the slot itself, not mutation).
        nccl=True,
        same_progress_sync=same_progress_sync,
    )


# ----------------------------------------------------------------------
# 7. wait_input_dist — awaits splits, populates tensors_requests
# ----------------------------------------------------------------------


def make_wait_input_dist_task(
    state: PipelineState,
    *,
    lookahead: int,
    same_progress_sync: tuple = (),
) -> Task:
    def _fn(ctx):
        torchrec_ctx = ctx.slots.get("torchrec_ctx", None)
        if torchrec_ctx is None:
            return
        with nvtx.annotate(f"## wait_input_dist {torchrec_ctx.index} ##"):
            for names, awaitable in torchrec_ctx.fused_splits_awaitables:
                for name, request in zip(names, awaitable.wait()):
                    torchrec_ctx.input_dist_tensors_requests[name] = request
            torchrec_ctx.input_dist_splits_requests.clear()
            torchrec_ctx.fused_splits_awaitables.clear()

    return Task.from_fn(
        "wait_input_dist",
        _fn,
        stream="data_dist",
        lookahead=lookahead,
        depends_on=("start_input_dist",),
        # awaitable.wait() triggers the tensors all_to_all NCCL op —
        # must participate in the declaration-order NCCL chain.
        nccl=True,
        same_progress_sync=same_progress_sync,
    )


# ----------------------------------------------------------------------
# 8. prefetch_embeddings (prefetch variant only)
# ----------------------------------------------------------------------


def make_prefetch_task(
    state: PipelineState,
    *,
    lookahead: int,
    same_progress_sync: tuple = (),
) -> Task:
    """Calls ShardedModule.prefetch on the prefetch stream and stores
    the result into the context's ``module_input_post_prefetch`` /
    ``module_contexts_post_prefetch`` — matching legacy ``_prefetch``
    (train_pipeline.py:663-692). Without this post-step,
    ``PrefetchPipelinedForward.__call__`` asserts because its
    required slot is empty."""

    def _fn(ctx):
        shuffled = ctx.slots.get("shuffled_batch", None)
        torchrec_ctx = ctx.slots.get("torchrec_ctx", None)
        if shuffled is None or torchrec_ctx is None:
            return
        from commons.pipeline.utils import _prefetch_embeddings

        # Legacy clears these dicts before populating — mirror exactly.
        torchrec_ctx.module_input_post_prefetch.clear()
        torchrec_ctx.module_contexts_post_prefetch.clear()

        with nvtx.annotate("## prefetch ##"):
            # record_stream so the prefetch stream can safely use the
            # batch tensor (legacy does this too).
            shuffled.record_stream(torch.cuda.current_stream())

            data_per_module = _prefetch_embeddings(
                batch=shuffled,
                context=torchrec_ctx,
                pipelined_modules=state.pipelined_modules,
                device=state.device,
                stream_context=torch.get_device_module(state.device).stream,
                data_dist_stream=ctx.stream_pool.get("data_dist"),
                default_stream=ctx.stream_pool.get("default"),
            )
            # Populate ctx.module_input_post_prefetch + _contexts_
            # (legacy does this in _prefetch after _prefetch_embeddings).
            for sharded_module in state.pipelined_modules:
                fwd = sharded_module.forward
                data = data_per_module[fwd._name]
                torchrec_ctx.module_input_post_prefetch[fwd._name] = data
                torchrec_ctx.module_contexts_post_prefetch[
                    fwd._name
                ] = torchrec_ctx.module_contexts.pop(fwd._name)

    return Task.from_fn(
        "prefetch_embeddings",
        _fn,
        stream="prefetch",
        lookahead=lookahead,
        depends_on=("wait_input_dist",),
        same_progress_sync=same_progress_sync,
    )


# ----------------------------------------------------------------------
# 9. compute_output_dist — embedding lookup + cross-rank output a2a (NCCL)
# ----------------------------------------------------------------------


def make_compute_output_dist_task(state: PipelineState, *, lookahead: int = 0) -> Task:
    """Run ``module.compute_and_output_dist(ctx, data)`` for every
    pipelined module in this batch. Local lookup + cross-rank
    ``all_to_all`` (NCCL) for the output dist; the returned awaitables
    are stashed into ``torchrec_ctx.embedding_a2a_requests`` for the
    forward task (``HSTUPipelinedForward``) to consume.

    Stream: ``default`` — same stream as forward / backward / NCCL
    DDP. Running compute_and_output_dist on the default stream avoids
    a cross-stream ``wait_event`` between this task's a2a awaitable
    and ``forward``'s consumer (both are on default → FIFO orders
    them automatically). NCCL ordering against other DP-comm
    collectives is preserved by ``_NcclOrderedLock`` (this task
    participates as a ticket via ``nccl=True``).

    Branches on ``state.uses_prefetch``: with prefetch, KJT input
    comes from ``module_input_post_prefetch`` (populated by the
    upstream ``prefetch_embeddings`` task); without prefetch, it comes
    from ``input_dist_tensors_requests[name].wait()`` (populated by
    upstream ``wait_input_dist``). See
    ``commons.pipeline.hstu_pipeline.embedding_split._compute_and_output_dist_for_module``.

    NCCL: marked ``nccl=True`` because ``compute_and_output_dist``
    submits an a2a collective on the DP comm — must participate in
    the engine's ``_NcclOrderedLock`` chain for cross-rank submission
    ordering.
    """
    from commons.pipeline.hstu_pipeline.embedding_split import (
        _compute_and_output_dist_for_module,
    )

    def _fn(ctx):
        torchrec_ctx = ctx.slots.get("torchrec_ctx", None)
        if torchrec_ctx is None:
            return
        with nvtx.annotate("## compute_output_dist ##"):
            for module in state.pipelined_modules:
                _compute_and_output_dist_for_module(
                    module,
                    torchrec_ctx,
                    is_prefetch=state.uses_prefetch,
                )

    return Task.from_fn(
        "compute_output_dist",
        _fn,
        stream="default",
        lookahead=lookahead,
        # depends_on declared as bare-name docs — wait_input_dist
        # (la=2/1) and prefetch_embeddings (la=1) are cross-la vs
        # compute_output_dist (la=0), so the engine's same-progress
        # DAG builder filters them out. Cross-iter ordering is
        # satisfied by ring-rotated slot reads (torchrec_ctx@0).
        depends_on=("wait_input_dist", "prefetch_embeddings"),
        nccl=True,
    )


# ----------------------------------------------------------------------
# 10. forward
# ----------------------------------------------------------------------


def make_forward_task(state: PipelineState, *, prefetch: bool = False) -> Task:
    """Sets module context, runs model(batch), writes losses + output.

    If ``prefetch=True``, ``forward`` declares an ordering edge to
    ``prefetch_embeddings`` via bare-name ``depends_on``; the engine
    derives the cross-iter offset from the lookahead diff
    (``prefetch_embeddings.lookahead=1`` vs ``forward.lookahead=0``)
    and emits the appropriate ``wait_event`` so the first iteration
    is structurally synchronized — without the user writing the
    iteration arithmetic.
    """

    def _fn(ctx):
        batch_gpu = ctx.slots.get("batch_gpu", None)
        torchrec_ctx = ctx.slots.get("torchrec_ctx", None)
        if batch_gpu is None or torchrec_ctx is None:
            ctx.slots.set("losses", None)
            ctx.slots.set("output", None)
            return
        # NCCL safety fence (folded in from former nccl_safety_barrier
        # task): default stream waits for memcpy stream's finish_shuffle
        # AllGather before any subsequent default-stream NCCL
        # (backward DDP AllReduce / finalize_model_grads). Without
        # this, GPU-side execution order on the DP comm can diverge
        # across ranks → deadlock. _NcclOrderedLock only guarantees
        # host submission order; this guarantees GPU execution order.
        if not state.is_identity_shuffler:
            memcpy = ctx.stream_pool.get("memcpy")
            if memcpy is not None:
                torch.cuda.current_stream().wait_stream(memcpy)
        shuffled = ctx.slots.get("shuffled_batch", batch_gpu)
        state.set_module_context(torchrec_ctx)
        with nvtx.annotate("## forward ##"):
            losses, output = state.model_fwd(shuffled)
        ctx.slots.set("losses", losses)
        ctx.slots.set("output", output)

    # Explicit depends_on so a custom thread_map can't put forward on a
    # different worker thread and lose ordering. The hard topo edge is
    # ``compute_output_dist`` → ``forward`` (same la=0): forward consumes
    # ``embedding_a2a_requests`` populated by compute_output_dist's
    # ``module.compute_and_output_dist`` calls. ``wait_input_dist`` /
    # ``prefetch_embeddings`` are listed for documentation but are
    # cross-la → no topo edge (handled by ring rotation).
    depends_on: tuple = ("compute_output_dist", "wait_input_dist")
    if prefetch:
        depends_on = ("compute_output_dist", "prefetch_embeddings")

    return Task.from_fn(
        "forward",
        _fn,
        stream="default",
        lookahead=0,
        reads=("batch_gpu", "torchrec_ctx", "shuffled_batch"),
        writes=("losses", "output"),
        depends_on=depends_on,
        # forward does NOT AllReduce; DDP hooks fire during backward.
    )


# ----------------------------------------------------------------------
# (former #10 nccl_safety_barrier task has been folded into make_forward_task —
# default_stream.wait_stream(memcpy) now runs at the start of forward's
# body. Removing the standalone Task slot saves one dispatch + NVTX +
# completion event per iter; NCCL safety semantic is unchanged.)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# 11. backward — (loss * dp_size / global_tokens).backward()
# ----------------------------------------------------------------------


def make_backward_task(
    state: PipelineState,
    *,
    depends_on: tuple = (),
    same_progress_sync: tuple = (),
) -> Task:
    """Build the backward task.

    ``depends_on`` carries same-batch logical dependency edges
    (e.g. zero_grad → backward, both for batch K).

    ``same_progress_sync`` carries same-progress GPU coherency edges
    (e.g. prefetch_embeddings → backward in prefetch variant —
    different batches but shared dynamicemb cache, mirroring legacy
    ``default_stream.wait_stream(prefetch_stream)`` at
    train_pipeline.py:993-997).
    """
    from megatron.core import parallel_state

    def _fn(ctx):
        losses = ctx.slots.get("losses", None)
        global_tokens = ctx.slots.get("global_tokens", None)
        if losses is None or global_tokens is None:
            return
        if not state.model.training:
            return
        with nvtx.annotate("## loss postprocess ##"):
            if state.assert_nan_loss:
                from commons.pipeline.train_pipeline import collective_assert

                collective_assert(not torch.isnan(losses).any(), "loss has nan value")
            local_loss_sum = torch.sum(losses)
        dp_size = parallel_state.get_data_parallel_world_size()
        with nvtx.annotate("## backward ##"):
            (local_loss_sum * dp_size / global_tokens).backward()
        # Cache the detached local_loss_sum for step_result
        ctx.slots.set("local_loss_sum", local_loss_sum.detach())

    return Task.from_fn(
        "backward",
        _fn,
        stream="default",
        lookahead=0,
        reads=("losses", "global_tokens"),
        writes=("local_loss_sum",),
        depends_on=depends_on,
        same_progress_sync=same_progress_sync,
        # DDP backward AllReduce fires inside .backward() — mark NCCL.
        nccl=True,
    )


# ----------------------------------------------------------------------
# 12. finalize_model_grads (Megatron TP)
# ----------------------------------------------------------------------


def make_finalize_grads_task(state: PipelineState) -> Task:
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads

    def _fn(ctx):
        if not state.model.training:
            return
        with nvtx.annotate("## finalize_model_grads ##"):
            if isinstance(state.model.module, DistributedDataParallel):
                finalize_model_grads([state.model.module], None)

    return Task.from_fn(
        "finalize_model_grads",
        _fn,
        stream="default",
        lookahead=0,
        depends_on=("backward",),
        nccl=True,
    )


# ----------------------------------------------------------------------
# 13. optimizer_step — writes step_result
# ----------------------------------------------------------------------


def make_optimizer_step_task(state: PipelineState) -> Task:
    def _fn(ctx):
        if state.model.training:
            with nvtx.annotate("## optimizer ##"):
                state.optimizer.step()
        local_loss = ctx.slots.get("local_loss_sum", None)
        global_tokens = ctx.slots.get("global_tokens", None)
        output = ctx.slots.get("output", None)
        ctx.slots.set("step_result", (local_loss, global_tokens, output))

    return Task.from_fn(
        "optimizer_step",
        _fn,
        stream="default",
        lookahead=0,
        depends_on=("finalize_model_grads",),
        writes=("step_result",),
    )


# ----------------------------------------------------------------------
# 14. watchdog_step (optional no-op)
# ----------------------------------------------------------------------


def make_watchdog_task() -> Task:
    def _fn(ctx):
        try:
            from commons.utils.cuda_mem_watchdog import get_cuda_mem_watchdog

            get_cuda_mem_watchdog().step()
        except Exception:
            # Watchdog is optional; missing module → no-op.
            pass

    return Task.from_fn(
        "watchdog_step",
        _fn,
        stream="default",
        lookahead=0,
        depends_on=("optimizer_step",),
    )
