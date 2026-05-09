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

"""HSTUPipeline — adapter that drives the SchedulablePipeline engine
using torchrec's ``_rewrite_model`` + ``HSTUPipelinedForward``
(see ``embedding_split``) for the HSTU training scenario.

Unlike TorchRec's stock ``PipelinedForward`` / ``PrefetchPipelinedForward``
which inline ``compute_and_output_dist`` (local lookup + cross-rank
output a2a NCCL) inside ``__call__``, this engine breaks that work
out into a dedicated ``compute_output_dist`` task. The task runs on
the **default stream** (same as forward / backward) so the resulting
awaitable is FIFO-ordered with forward's consumer without any
cross-stream wait_event; cross-rank NCCL submission ordering is
maintained by the engine's ``_NcclOrderedLock`` ticket. The
``HSTUPipelinedForward.__call__`` itself becomes a thin pull from
``embedding_a2a_requests`` (no NCCL submitted from inside forward).

Lazy initialization: ``_rewrite_model`` needs a peek at the first
batch for FX tracing. We defer engine construction to the first
``progress()`` call, matching the legacy lazy-fill behavior.
"""

from __future__ import annotations

import contextlib
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
    "compute_output_dist": "compute",
    "forward": "compute",
    "backward": "compute",
    "finalize_model_grads": "compute",
    "optimizer_step": "compute",
    "watchdog_step": "compute",
}


# Named thread-map presets for sweeping schedule variants in benchmarks.
# Selected at construction time via ``thread_map=<name>`` or via the
# env var ``HSTU_THREAD_MAP_VARIANT`` (only consulted when the kwarg is
# left at its ``None`` default and ``threaded=True``). Keeping these
# inside the engine package means a sweep doesn't have to bake variant
# logic into the launcher / training script.
#
# Variant rationale (per HSTU pipeline topology audit):
#   default              — io / compute split (2 threads). Baseline. CPU
#                          waits=0 because io tasks live at lookahead=2
#                          and readers live at lookahead<=1, so no exact
#                          slot match across threads. All compute work
#                          (input_dist + prefetch + fwd + bwd + opt)
#                          serializes on one thread → host enqueue is
#                          the bottleneck.
#   by_stream            — one thread per CUDA stream (4 threads in
#                          prefetch mode: default, memcpy, data_dist,
#                          prefetch). Lets data_dist + prefetch host
#                          enqueue run in parallel with default-stream
#                          fwd/bwd. Costs ~2 cross-thread CPU waits per
#                          progress.
#   per_task             — one thread per task (14 threads). Maximum
#                          parallelism but ~15 cross-thread CPU waits
#                          per progress; pool dispatch overhead likely
#                          dominates.
#   io_prefetch_compute  — 3 threads: io / prefetch / compute. Splits
#                          prefetch_embeddings off the compute thread so
#                          its dynamicemb cache writes can overlap with
#                          backward on the default stream. Adds a
#                          cross-thread CPU edge for forward←prefetch
#                          (next-iter consumer pattern, Δ=1).
#   io_data_dist_compute — 3 threads: io / data_dist / compute. Splits
#                          input_dist off so its NCCL host enqueue can
#                          overlap with compute. Useful when input_dist
#                          is comm-heavy.
#   io_data_dist_prefetch_compute — 4 threads: io / data_dist / prefetch
#                          / compute. Combines the above two splits.
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
    # All six critical-gated (la>0) tasks share a single "io" worker thread
    # so they serialize on one host thread instead of spreading across
    # io/data_dist/prefetch. With ``critical_gate=("compute_output_dist",)``
    # on each of the six, they ALL block on compute_output_dist's
    # threading.Event, then fire one-after-another on the io thread —
    # producing a single, contiguous burst of NCCL submissions that share
    # one ticket-acquire/release cadence and minimize cross-thread
    # interleaving on the NCCL ordered lock.
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
        #
        # Env var ``HSTU_THREAD_MAP_VARIANT`` (only when kwarg is None)
        # selects a named preset from ``HSTU_THREAD_MAP_PRESETS``.
        # Used by the benchmark sweep to A/B schedule variants without
        # rebuilding the pipeline construction call site.
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

        # Auto-scheduler hook: when env var
        # ``HSTU_AUTOSCHED_COST_FILE`` points at a JSON cost model,
        # the engine first builds a default schedule, then runs
        # :func:`auto_assign_lookaheads` against the cost model and
        # rebuilds the schedule with the recommended per-task
        # lookaheads. ``HSTU_AUTOSCHED_MAX_IN_FLIGHT`` (default 5)
        # caps the in-flight batch budget. Bit-exact contract is
        # enforced by the auto-scheduler.
        self._autosched_cost_file: Optional[str] = (
            os.environ.get("HSTU_AUTOSCHED_COST_FILE", "").strip() or None
        )
        self._autosched_max_in_flight: int = int(
            os.environ.get("HSTU_AUTOSCHED_MAX_IN_FLIGHT", "5")
        )

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

    def _build_schedule(
        self,
        la_overrides: Optional[Dict[str, int]] = None,
    ) -> Tuple[Schedule, StreamPool]:
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
        self._prefetch_depth
        # 6-la cascade: every same-thread / host-sync chain pair gets +1
        # la so each downstream "wait" drains a producer that fired 1
        # full iter ago (host syncs collapse to ~µs).
        #
        # Pair                                 Host sync source           Fix
        # h2d → start_shuffle                  tolist() after AG          +1 la
        # start_shuffle → finish_shuffle       KK background wait         +1 la
        # start_input_dist → wait_input_dist   awaitable.wait() splits    +1 la
        # wait_input_dist → prefetch_emb       request.wait() tensor a2a  +1 la
        #
        # 6 in-flight batches; prefetch.la − forward.la is unchanged (=1)
        # so dynamicemb prefetch cache outstanding keys budget is intact.
        # HSTU_LA_DEPTH={3,6} env var selects pipeline depth = max(la)+1.
        #   depth=6 (default) = 6-la cascade as commented above (5/4/3/3/2/1)
        #   depth=3           = round2-era depth+1 plateau cascade (2/2/2/1/1/1)
        # Variant naming: <thread_map>_d{depth} encodes la directly into the
        # variant label, so dispatch + analysis read the depth straight off
        # the experiment name without any "OLD/NEW" preset shorthand.
        import os as _os

        _depth = int(_os.environ.get("HSTU_LA_DEPTH", "6"))
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
            raise ValueError(f"HSTU_LA_DEPTH={_depth} not supported; expected 3 or 6")

        # Apply auto-scheduler / explicit overrides. Only the off-default
        # tasks accept lookahead via factory args; default-stream tasks
        # (zero_grad / forward / backward / opt / etc.) are bit-exact
        # at la=0 and never overridden — see
        # ``commons.pipeline.engine.autosched.fire_order``.
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

        # ── Critical-path gate (full_split_d6 experiment) ──
        # Push every la>0 task ("non-critical": h2d / start_shuffle /
        # finish_shuffle / start_input_dist / wait_input_dist /
        # prefetch_embeddings) behind the la=0 critical chain
        # (zero_grad → global_tokens_allreduce → compute_output_dist
        # → forward → backward → ...) by giving them
        # ``same_progress_sync=("compute_output_dist",)``.
        #
        # NCCL ticket order after this gate (topo序):
        #   t0 global_tokens_allreduce  (compute thread, root)
        #   t1 compute_output_dist      (compute thread, root,
        #                                tie-break by decl after t0)
        #   t2 start_shuffle            (gated, fires after t1)
        #   t3 finish_shuffle
        #   t4 start_input_dist
        #   t5 wait_input_dist
        #   t6 backward (DDP)
        #   t7 finalize_model_grads
        #
        # Deadlock-free: compute_output_dist (t1) only waits for t0
        # (global_tokens_allreduce, same compute thread, no external
        # dep) → fires unconditionally → release t1 + set completion
        # → cascade unblocks. No circular wait between gates and
        # _NcclOrderedLock.
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
            ]
        )
        # Prefetch (optional) must run before compute_output_dist so
        # the latter can read ``module_input_post_prefetch`` instead of
        # falling back to ``input_dist_tensors_requests.wait()``.
        if self._prefetch:
            tasks.append(
                make_prefetch_task(
                    self._state,
                    lookahead=prefetch_lookahead,
                    same_progress_sync=critical_gate,
                )
            )
        # compute_output_dist runs ``module.compute_and_output_dist``
        # for each pipelined module (local lookup + cross-rank output
        # ``all_to_all`` NCCL on data_dist stream). The awaitable is
        # stashed into ``torchrec_ctx.embedding_a2a_requests`` for the
        # forward task to pick up. forward.depends_on=("compute_output_dist",)
        # gives the same-la=0 topo edge that orders the two tasks.
        # nccl_safety_barrier is folded into forward's body —
        # default.wait_stream(memcpy) now runs at the start of forward
        # instead of as a standalone task.
        tasks.extend(
            [
                make_compute_output_dist_task(self._state),
                make_forward_task(self._state, prefetch=self._prefetch),
            ]
        )
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
        from commons.pipeline.utils import _override_input_dist_forwards
        from commons.pipeline.utils import _rewrite_model as torchrec_rewrite_model
        from commons.pipeline.utils import _start_data_dist

        from .embedding_split import (
            HSTUPipelinedForward,
            HSTUPrefetchPipelinedForward,
            HSTUTrainPipelineContext,
        )

        # Seed context type on state for per-batch ctx factory.
        # HSTUTrainPipelineContext extends PrefetchTrainPipelineContext
        # so it has every field both prefetch and non-prefetch paths
        # touch (input_dist_*, module_contexts, module_input_post_prefetch),
        # plus ``embedding_a2a_requests`` for the new
        # ``compute_output_dist`` task to feed forward.
        self._state.torchrec_context_type = HSTUTrainPipelineContext
        # Tell the compute_output_dist factory which side of the dict
        # population to read from.
        self._state.uses_prefetch = self._prefetch

        # Real per-batch context for the peek batch (will be seeded
        # into the engine ring). Use state.create_torchrec_ctx so the
        # index counter stays consistent with subsequent batches.
        peek_ctx = self._state.create_torchrec_ctx()

        # Replace TorchRec's PipelinedForward / PrefetchPipelinedForward
        # (both call ``module.compute_and_output_dist`` inline → NCCL
        # serialized on the default stream during forward) with a thin
        # wrapper that only reads the awaitable populated by the new
        # ``compute_output_dist`` engine task. Two variants because
        # TorchRec's ``_prefetch_embeddings`` helper type-asserts
        # ``isinstance(forward, PrefetchPipelinedForward)`` specifically
        # — the prefetch flavor has to subclass that exact wrapper.
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

        # Auto-scheduler hook: rebuild the schedule with recommended
        # per-task lookaheads if a cost model is supplied via env. This
        # is bit-exact (default-stream tasks frozen at la=0); the
        # scheduler raises ValueError if any author-supplied
        # constraint can't be satisfied.
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
            # Diff vs current schedule for visibility in the log.
            current_la = {t.name: t.batch_offset for t in schedule.all_tasks()}
            changed = {
                name: (current_la[name], recommended[name])
                for name in recommended
                if current_la.get(name) != recommended[name]
            }
            if changed:
                # Engine-internal logger (single line summary so a
                # multi-rank training run doesn't spam the log).
                print(
                    f"[HSTUPipeline] auto-scheduler recommended "
                    f"lookahead overrides: {changed}",
                    flush=True,
                )
                schedule, pool = self._build_schedule(la_overrides=recommended)

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

        # Synchronously run the shuffler on the peek batch so the
        # bootstrap ``_start_data_dist`` (and FX-trace) bind to the
        # SAME post-shuffle batch the engine will see in steady state.
        #
        # Why: ``_rewrite_model`` calls ``_start_data_dist(modules,
        # peek_batch, peek_ctx)`` which populates ``peek_ctx`` with
        # input-dist requests sized for ``peek_batch``. We then seed
        # that ctx + ``shuffled_batch`` into the ring's max_offset
        # slot. If the seeded ``shuffled_batch`` doesn't match the
        # batch the ctx was bound to, ``start_input_dist``'s
        # idempotent guard short-circuits at first iter and the ctx's
        # ``module_input_post_prefetch`` (later set by
        # prefetch_embeddings) ends up sized for a DIFFERENT batch
        # than ``shuffled_batch``. Forward then sees mismatched row
        # counts (item embeddings come from ``module_input_post_prefetch``,
        # contextual features from the actual ``shuffled_batch``) and
        # crashes inside ``hstu_preprocess_embeddings``' ``torch.cat``.
        # Mirror legacy ``_copy_batch_to_gpu_and_shuffle`` which also
        # does the shuffle synchronously on ``_memcpy_stream`` before
        # any input_dist work fires.
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
            # record_stream the shuffled tensor onto every consumer
            # stream too — same lifetime guarantee as peek_batch_gpu.
            if device.type == "cuda" and memcpy_stream is not None:
                for consumer in consumers:
                    if consumer is not None and hasattr(
                        peek_batch_shuffled, "record_stream"
                    ):
                        peek_batch_shuffled.record_stream(consumer)

        # _rewrite_model bootstraps on the SHUFFLED peek so the ctx's
        # input-dist state matches the batch start_input_dist will see
        # in steady state.
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

        # Seed the engine with the peek batch's pre-processed state:
        # batch_cpu / batch_gpu / shuffled_batch / torchrec_ctx are all
        # set so the first-iter h2d / start_shuffle / finish_shuffle /
        # start_input_dist tasks all idempotent-skip; the ctx is bound
        # to ``shuffled_batch`` via the bootstrap _start_data_dist
        # above so prefetch_embeddings + forward see consistent state.
        seeded: dict = {
            "batch_cpu": peek_batch_cpu,
            "batch_gpu": peek_batch_gpu,
            "shuffled_batch": peek_batch_shuffled,
            "torchrec_ctx": peek_ctx,
        }
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
