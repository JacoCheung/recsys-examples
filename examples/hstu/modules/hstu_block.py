# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import dataclasses
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from commons.datasets.hstu_batch import HSTUBatch
from commons.utils.nvtx_op import output_nvtx_hook
from configs.hstu_config import HSTUConfig, HSTULayerType
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from modules.debug.debug_hstu_layer import HSTULayer as DebugHSTULayer
from modules.fused_hstu_layer import FusedHSTULayer
from modules.hstu_processor import HSTUBlockPostprocessor, HSTUBlockPreprocessor
from modules.jagged_data import JaggedData
from modules.native_hstu_layer import HSTULayer as NativeHSTULayer
from torchrec.sparse.jagged_tensor import JaggedTensor


class HSTUBlock(MegatronModule):
    """
    HSTUBlock module. A stack of HSTULayers.

    Args:
        config (HSTUConfig): Configuration for the HSTU block.
    """

    @staticmethod
    def _validate_cp_config(config) -> None:
        """Reject CP combinations that v0 does not support.

        Pulled out of `__init__` so unit tests can call it on a tiny
        duck-typed config without spinning up Megatron `parallel_state`
        for `HSTUBlockPreprocessor` (used by the
        `test_block_cp_rejects_*` regression-guards). Reads only three
        config fields:

          - `context_parallel_size`
          - `hstu_layer_type`
          - `sequence_parallel`

        v0 wires CP only through the NATIVE HSTU layer (uses
        `create_hstu_attention(...)` → `FusedHSTUAttention` which
        routes to `hstu_attn_varlen_cp_func`). The FUSED layer uses
        `fused_hstu_op` (separate triton fusion) and the DEBUG layer
        doesn't yet thread CP plumbing through; both are rejected so
        the failure points at the config rather than at a silent-no-op
        layer call.

        CP + sequence_parallel is also rejected: the SP preprocessor
        scatters `jd.values` along row-dim by tp_size
        (`hstu_processor.py:scatter_to_sequence_parallel_region`) but
        CP dispatch (`apply_dualchunkswap_to_jagged`) indexes
        `jd.values` with global `seqlen_offsets` — shape/index
        mismatch as soon as both are on.
        """
        cp_size = config.context_parallel_size
        if cp_size > 1:
            if config.hstu_layer_type != HSTULayerType.NATIVE:
                raise ValueError(
                    "Context Parallelism (cp_size > 1) is only wired through "
                    f"HSTULayerType.NATIVE in v0; got {config.hstu_layer_type}. "
                    "Switch to NATIVE or set context_parallel_size=1."
                )
            if config.sequence_parallel:
                raise ValueError(
                    "Context Parallelism (cp_size > 1) and sequence_parallel "
                    "are not co-wired in v0. The SP preprocessor scatters "
                    "values along the row dim, but CP dispatch indexes with "
                    "global offsets — they cannot stack. Disable one."
                )

    def __init__(
        self,
        config: HSTUConfig,
    ):
        # Validate CP config first so reject paths fail before any heavy
        # init (HSTUBlockPreprocessor needs Megatron `parallel_state`).
        # Tests call `_validate_cp_config` directly on a small
        # duck-typed config to make the reject branches behavioural.
        self._validate_cp_config(config)

        super().__init__(config=config)
        self._training_dtype = torch.float32
        if self.config.bf16:
            self._training_dtype = torch.bfloat16
        if self.config.fp16:
            self._training_dtype = torch.float16

        self._preprocessor = HSTUBlockPreprocessor(
            config,
            is_inference=config.is_inference,
        )  # sequence parallel is from config
        self._postprocessor = HSTUBlockPostprocessor(
            is_inference=config.is_inference, sequence_parallel=config.sequence_parallel
        )

        self._cp_size: int = config.context_parallel_size
        self._cp_group: Optional[dist.ProcessGroup] = None
        self._cp_global_ranks: Optional[Tuple[int, ...]] = None
        if self._cp_size > 1:
            self._cp_group = parallel_state.get_context_parallel_group()
            self._cp_global_ranks = tuple(
                parallel_state.get_context_parallel_global_ranks()
            )

        HSTULayerImpl = (
            FusedHSTULayer
            if config.hstu_layer_type == HSTULayerType.FUSED
            else DebugHSTULayer
            if config.hstu_layer_type == HSTULayerType.DEBUG
            else NativeHSTULayer
        )
        self._attention_layers = torch.nn.ModuleList(
            [HSTULayerImpl(config) for l in range(self.config.num_layers)]
        )

    @output_nvtx_hook(nvtx_tag="HSTUBlock", hook_key_or_attr_name="values")
    def forward(
        self,
        embeddings: Dict[str, JaggedTensor],
        batch: HSTUBatch,
    ) -> Tuple[JaggedData, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the HSTUBlock.

        Args:
            embeddings (Dict[str, JaggedTensor]): The input embeddings.
            batch (HSTUBatch): The input batch.

        Returns:
            JaggedData: The output jagged data.
        """
        jd = self._preprocessor(embeddings, batch)
        # Capture GLOBAL metadata BEFORE the CP dispatch so the second
        # return-tuple is consistent across cp_size>1 and the legacy path.
        # Downstream callers consume these for loss masking and target
        # extraction; both must reflect the unsharded sample shape.
        seqlen_after_preprocessor = jd.seqlen
        num_contextuals_after_preprocessor = (
            jd.contextual_seqlen
            if jd.contextual_seqlen is not None
            else torch.zeros_like(seqlen_after_preprocessor)
        )
        num_candidates_after_preprocessor = (
            jd.num_candidates
            if jd.num_candidates is not None
            else torch.zeros_like(seqlen_after_preprocessor)
        )

        # Slice 6 T6.4: when CP is active, shard the embedding output to
        # this rank's DualChunkSwap chunks before the layer stack, then
        # gather back to the global shape before the postprocessor.
        # `apply_dualchunkswap_to_jagged` permutes `values` and rebuilds
        # `seqlen` / `seqlen_offsets` / `max_seqlen` for the local layout;
        # heterogeneous-mask metadata (`num_candidates`,
        # `contextual_seqlen`) are per-sample (not per-token) and must
        # ride through the dispatch unchanged so each layer's
        # `FusedHSTUAttention` can pass them to the CP wrapper which
        # builds the per-step `func` tensor (het-mask track, see
        # `docs/cp/het_mask_design.md`).
        cp_active = self._cp_size > 1 and self._cp_group is not None
        local_to_global: Optional[torch.Tensor] = None
        global_jd_template: Optional[JaggedData] = None
        if cp_active:
            assert self._cp_group is not None  # for type-checker
            from context_parallel import (
                apply_dualchunkswap_to_jagged,
                gather_jagged_from_cp_rank,
            )

            # NOTE: we deliberately do NOT call
            # `cp_func_cache_scope_enter()` here anymore. The cache is
            # content-keyed (cu_seqlens / num_contexts / num_targets
            # values + step) and lives across training steps, so a
            # dataset that cycles `num_generated_batches` unique
            # cu_seqlens values pays the build cost only once per
            # unique batch — every subsequent recurrence is a hit.
            # Calling scope_enter() here would WIPE that cross-step
            # cache and cap us at the round-5/6 ratio (~21–27%).

            cp_rank = dist.get_rank(self._cp_group)
            global_jd_template = jd
            jd, local_to_global = apply_dualchunkswap_to_jagged(
                jd, cp_size=self._cp_size, cp_rank=cp_rank
            )
            # The CP wrapper (`hstu_attn_varlen_cp_func`) expects the
            # `max_seqlen_q` argument to be the GLOBAL value — it divides
            # internally by cp_size to get the kernel-local max. The
            # dispatcher rewrites `jd.max_seqlen` to local; restore it to
            # global so the downstream layer call (which forwards
            # `jd.max_seqlen` straight to the wrapper) is correct. Local
            # `seqlen` and `seqlen_offsets` stay as the dispatcher
            # produced them — they describe the local jagged layout that
            # the CP wrapper consumes.
            jd = dataclasses.replace(jd, max_seqlen=global_jd_template.max_seqlen)

        for hstu_layer in self._attention_layers:
            jd = hstu_layer(jd)
        # NOTE: do NOT call `cp_func_cache_scope_exit()` here. The cache
        # must outlive forward so the autograd worker thread (which runs
        # backward) can read what this thread wrote. The next training
        # step's `cp_func_cache_scope_enter` will drop these tensors.

        if cp_active:
            assert global_jd_template is not None
            assert local_to_global is not None
            global_total_tokens = global_jd_template.values.shape[0]
            global_values = gather_jagged_from_cp_rank(
                jd.values,
                local_to_global,
                cp_group=self._cp_group,
                global_total_tokens=global_total_tokens,
            )
            # Rebuild a JaggedData carrying the post-attention values in
            # global shape with the original metadata (which the
            # postprocessor reads to split candidates / contextual prefix
            # / etc.).
            jd = dataclasses.replace(global_jd_template, values=global_values)

        return self._postprocessor(jd), (
            seqlen_after_preprocessor.detach(),
            num_contextuals_after_preprocessor.detach(),
            num_candidates_after_preprocessor.detach(),
        )
