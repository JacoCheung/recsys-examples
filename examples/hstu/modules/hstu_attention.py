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
import abc
from typing import Optional, Sequence, Union

import torch
import torch.distributed as dist
from commons.utils.nvtx_op import output_nvtx_hook
from configs import KernelBackend
from hstu import hstu_attn_varlen_func


class HSTUAttention(torch.nn.Module):
    """
    Base module interface for different HSTUAttention backends.

    """

    @abc.abstractmethod
    def forward(
        self,
        tq: torch.Tensor,  # (T, d)
        tk: torch.Tensor,  # (T, d)
        tv: torch.Tensor,  # (T, d)
        offsets: torch.Tensor,  # (batch_size + 1,)
        max_seqlen: int,
        scaling_seqlen: int,
        target_group_size: int = 1,  # target <=> candidates
        num_candidates: Optional[torch.Tensor] = None,
        num_contextuals: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:  # T, d
        """
        Abstract method for the forward pass of HSTUAttention.

        Args:
            tq (torch.Tensor): Query tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the query.
            tk (torch.Tensor): Key tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the key.
            tv (torch.Tensor): Value tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the value.
            offsets (torch.Tensor): Offsets tensor of shape (batch_size, 1), indicating the start position of each sequence in the batch.
            max_seqlen (int): The maximum sequence length across all batches.
            target_group_size (int): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
            num_candidates (torch.Tensor): Tensor containing the number of candidates for each sequence.
            num_contextuals (int | torch.Tensor | None): The number of contextuals for each sequence, could be a single integer or a tensor of shape (batch_size,) when different sequences have different number of contextuals.
        Returns:
            torch.Tensor: Output tensor of shape (T, d).
        """


class TorchHSTUAttention(HSTUAttention):
    """
    Native HUST implementation. All jagged inputs are padded to the maximum length before computation.

    Args:
        num_heads (int): Number of attention heads.
        attention_dim (int): Dimension of the attention.
        linear_dim (int): Dimension of the linear layer.
        is_causal (bool): Whether the attention is causal.
    """

    def __init__(
        self,
        num_heads: int,
        attention_dim: int,
        linear_dim: int,
        is_causal: bool,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.linear_dim = linear_dim
        self.is_causal = is_causal

    def forward(
        self,
        tq: torch.Tensor,  # (T, d)
        tk: torch.Tensor,  # (T, d)
        tv: torch.Tensor,  # (T, d)
        offsets: torch.Tensor,  # (batch_size + 1,)
        max_seqlen: int,
        scaling_seqlen: int = -1,
        target_group_size: int = 1,  # target == candidates
        num_candidates: Optional[torch.Tensor] = None,
        num_contextuals: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:  # T, d
        """
        Forward pass of the TorchHSTUAttention module.

        Args:
            tq (torch.Tensor): Query tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the query.
            tk (torch.Tensor): Key tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the key.
            tv (torch.Tensor): Value tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the value.
            offsets (torch.Tensor): Offsets tensor of shape (batch_size, 1), indicating the start position of each sequence in the batch.
            max_seqlen (int): The maximum sequence length across all batches.
            scaling_seqlen (int): The sequence length to scale the attention output.
            target_group_size (int): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
            num_candidates (torch.Tensor): Tensor containing the number of candidates for each sequence.
            num_contextuals (int | torch.Tensor | None): The number of contextuals for each sequence, could be a single integer or a tensor of shape (batch_size,) when different sequences have different number of contextuals.
        Returns:
            torch.Tensor: Output tensor of shape (T, d).
        """
        from ops.pt_ops.pt_hstu_attention import pytorch_hstu_mha

        if isinstance(num_contextuals, torch.Tensor):
            num_contextuals = num_contextuals.to(torch.int32)
        elif isinstance(num_contextuals, int):
            num_contextuals = (
                torch.tensor([num_contextuals], dtype=torch.int32, device=tq.device)
                .view(1)
                .expand(offsets.size(0) - 1)
                .contiguous()
            )

        return pytorch_hstu_mha(
            max_seq_len=max_seqlen,
            alpha=1.0 / (self.attention_dim**0.5),
            q=tq.view(-1, self.num_heads, self.attention_dim),
            k=tk.view(-1, self.num_heads, self.attention_dim),
            v=tv.view(-1, self.num_heads, self.linear_dim),
            seq_offsets=offsets,
            num_contextuals=num_contextuals,
            num_targets=num_candidates,
            causal=self.is_causal,
            dropout_pr=0.0,
            training=self.training,
            target_group_size=target_group_size,
            scaling_seqlen=scaling_seqlen,
        ).view(-1, self.num_heads * self.linear_dim)


class TritonHSTUAttention(HSTUAttention):
    """
    Triton-based HUST implementation.

    Args:
        num_heads (int): Number of attention heads.
        attention_dim (int): Dimension of the attention.
        linear_dim (int): Dimension of the linear layer.
        is_causal (bool): Whether the attention is causal.
    """

    def __init__(
        self,
        num_heads: int,
        attention_dim: int,
        linear_dim: int,
        is_causal: bool,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.linear_dim = linear_dim
        self.is_causal = is_causal
        self.enable_tma = (
            True if torch.cuda.get_device_properties(0).major >= 9 else False
        )
        assert is_causal, "TritonHSTUAttention does not support is_causal=False"

    def forward(
        self,
        tq: torch.Tensor,  # (T, d)
        tk: torch.Tensor,  # (T, d)
        tv: torch.Tensor,  # (T, d)
        offsets: torch.Tensor,  # (batch_size + 1,)
        max_seqlen: int,
        scaling_seqlen: int = -1,
        target_group_size: int = 1,  # target == candidates
        num_candidates: Optional[torch.Tensor] = None,
        num_contextuals: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:  # T, d
        """
        Forward pass of the TritonHSTUAttention module.

        Args:
             tq (torch.Tensor): Query tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the query.
            tk (torch.Tensor): Key tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the key.
            tv (torch.Tensor): Value tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the value.
            offsets (torch.Tensor): Offsets tensor of shape (batch_size + 1,), indicating the start position of each sequence in the batch, with a terminal offset at the end.
            max_seqlen (int): The maximum sequence length across all batches.
            scaling_seqlen (int): The sequence length to scale the attention output.
            target_group_size (int): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
            num_candidates (torch.Tensor): Tensor containing the number of candidates for each sequence.
            num_contextuals (int | torch.Tensor | None): The number of contextuals for each sequence, could be a single integer or a tensor of shape (batch_size,) when different sequences have different number of contextuals.
        Returns:
            torch.Tensor: Output tensor of shape (T, d).
        """
        from ops.triton_ops.triton_hstu_attention import (  # type: ignore[attr-defined]
            triton_hstu_mha,
        )

        assert (
            target_group_size == 1
        ), "target_group_size is not supported in TritonHSTUAttention"
        if num_contextuals is None:
            num_contextuals = 0
        assert isinstance(
            num_contextuals, int
        ), "num_contextuals must be an integer in TritonHSTUAttention"
        return triton_hstu_mha(
            N=max_seqlen,
            alpha=1.0 / (self.attention_dim**0.5),
            q=tq.view(-1, self.num_heads, self.attention_dim),
            k=tk.view(-1, self.num_heads, self.attention_dim),
            v=tv.view(-1, self.num_heads, self.linear_dim),
            seq_offsets=offsets,
            num_targets=num_candidates,
            contextual_seq_len=num_contextuals,
            scaling_seqlen=scaling_seqlen,
            enable_tma=self.enable_tma,
        ).view(-1, self.num_heads * self.linear_dim)


# TODO, support packed qkv attention
class FusedHSTUAttention(HSTUAttention):
    """
    Cutlass-based HSTU implementation. Auto-dispatches to SM-specific kernels (Ampere/Hopper)
    via the unified hstu_attn_varlen_func API.

    Args:
        num_heads (int): Number of attention heads.
        attention_dim (int): Dimension of the attention.
        linear_dim (int): Dimension of the linear layer.
        is_causal (bool): Whether the attention is causal.
        cp_group: Optional Megatron / torch.distributed CP process group. When
            provided and `world_size(cp_group) > 1`, the forward dispatches to
            `hstu_attn_varlen_cp_func` (the multi-GPU CP wrapper). When None or
            world_size == 1, behaviour is identical to the pre-CP module.
        cp_global_ranks: Sequence of global ranks belonging to `cp_group` (in
            ring order). Required when `cp_group` is multi-rank.
        cp_stream: Optional secondary CUDA stream for ring P2P; defaults to a
            module-cached stream. See SPEC §2 / Slice 5.
    """

    def __init__(
        self,
        num_heads: int,
        attention_dim: int,
        linear_dim: int,
        is_causal: bool,
        *,
        cp_group: Optional["dist.ProcessGroup"] = None,
        cp_global_ranks: Optional[Sequence[int]] = None,
        cp_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.linear_dim = linear_dim
        self.is_causal = is_causal
        self.cp_group = cp_group
        self.cp_global_ranks = (
            tuple(cp_global_ranks) if cp_global_ranks is not None else None
        )
        self.cp_stream = cp_stream
        assert (
            self.linear_dim == self.attention_dim
        ), "only support linear_dim and attention_dim"
        if self.cp_group is not None and self.cp_global_ranks is None:
            raise ValueError(
                "cp_global_ranks is required when cp_group is provided "
                "(needed for ring P2P routing in hstu_attn_varlen_cp_func)"
            )

    @output_nvtx_hook(nvtx_tag="FusedHSTUAttn")
    def forward(
        self,
        tq: torch.Tensor,  # (T, d)
        tk: torch.Tensor,  # (T, d)
        tv: torch.Tensor,  # (T, d)
        offsets: torch.Tensor,  # (batch_size, 1)
        max_seqlen: int,
        scaling_seqlen: int = -1,
        target_group_size: int = 1,  # target == candidates
        num_candidates: Optional[torch.Tensor] = None,
        num_contextuals: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the FusedHSTUAttention module.

        Args:
            tq (torch.Tensor): Query tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the query.
            tk (torch.Tensor): Key tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the key.
            tv (torch.Tensor): Value tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the value.
            offsets (torch.Tensor): Offsets tensor of shape (batch_size + 1,), indicating the start position of each sequence in the batch, with a terminal offset at the end.
            max_seqlen (int): The maximum sequence length across all batches.
            target_group_size (int): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
            num_candidates (torch.Tensor): Tensor containing the number of candidates for each sequence.
            num_contextuals (int | torch.Tensor | None): The number of contextuals for each sequence, could be a single integer or a tensor of shape (batch_size,) when different sequences have different number of contextuals.

        Returns:
            torch.Tensor: Output tensor.
        """
        assert (
            self.is_causal or num_contextuals is None
        ), "Only causal attention is supported when max_num_contextuals > 0 in cutlass backend"
        if isinstance(num_contextuals, torch.Tensor):
            num_contextuals = num_contextuals.to(torch.int32)
        elif isinstance(num_contextuals, int):
            num_contextuals = (
                torch.tensor([num_contextuals], dtype=torch.int32, device=tq.device)
                .view(1)
                .expand(offsets.size(0) - 1)
                .contiguous()
            )
        if scaling_seqlen == -1:
            scaling_seqlen = max_seqlen

        cp_size = dist.get_world_size(self.cp_group) if self.cp_group is not None else 1
        if cp_size > 1:
            # CP wrapper enforces v0 contract: pure causal, no rab, no
            # heterogeneous mask, head_dim ∈ {32,64,128,256}. Reject
            # incompatible inputs here (rather than inside the wrapper) so
            # the failure points at the module config rather than at a
            # generic GuardError.
            if not self.is_causal:
                raise ValueError(
                    "FusedHSTUAttention with cp_size>1 requires is_causal=True "
                    "(v0 contract; sliding-causal lands in v0.5; see SPEC §2)"
                )
            if (
                num_contextuals is not None
                or (
                    isinstance(num_candidates, torch.Tensor)
                    and num_candidates.numel() > 0
                )
                or target_group_size != 1
            ):
                raise ValueError(
                    "FusedHSTUAttention with cp_size>1 does not support "
                    "heterogeneous mask params (num_contextuals, num_candidates, "
                    "target_group_size>1) — those break DualChunkSwap balanced "
                    "sharding (v0 contract, see SPEC §2). Disable CP for this "
                    "config or wait for the heterogeneous-mask CP extension."
                )
            from context_parallel import hstu_attn_varlen_cp_func

            return hstu_attn_varlen_cp_func(
                q=tq.view(-1, self.num_heads, self.attention_dim),
                k=tk.view(-1, self.num_heads, self.attention_dim),
                v=tv.view(-1, self.num_heads, self.linear_dim),
                cu_seqlens_q=offsets.to(torch.int32),
                cu_seqlens_k=offsets.to(torch.int32),
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                scaling_seqlen=scaling_seqlen,
                num_contexts=None,
                num_targets=None,
                target_group_size=1,
                window_size=(-1, 0),
                alpha=1.0 / (self.attention_dim**0.5),
                cp_group=self.cp_group,
                cp_global_ranks=self.cp_global_ranks,
                cp_stream=self.cp_stream,
            ).view(-1, self.num_heads * self.linear_dim)

        return hstu_attn_varlen_func(
            tq.view(-1, self.num_heads, self.attention_dim),
            tk.view(-1, self.num_heads, self.attention_dim),
            tv.view(-1, self.num_heads, self.linear_dim),
            offsets.to(torch.int32),
            offsets.to(torch.int32),
            None,  # seqused_q
            None,  # seqused_k
            max_seqlen,
            max_seqlen,
            scaling_seqlen,
            num_contextuals,
            num_candidates.to(torch.int32)
            if isinstance(num_candidates, torch.Tensor)
            else None,
            target_group_size=target_group_size,
            window_size=(-1, 0) if self.is_causal else (-1, -1),
            alpha=1.0 / (self.attention_dim**0.5),
        ).view(-1, self.num_heads * self.linear_dim)


def create_hstu_attention(
    kernel_backend: KernelBackend,
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    is_causal: bool,
    *,
    cp_group: Optional["dist.ProcessGroup"] = None,
    cp_global_ranks: Optional[Sequence[int]] = None,
    cp_stream: Optional[torch.cuda.Stream] = None,
) -> HSTUAttention:
    """
    Factory function to create an HSTUAttention module based on the kernel backend.

    Args:
        kernel_backend (KernelBackend): The kernel backend to use.
        num_heads (int): Number of attention heads.
        attention_dim (int): Dimension of the attention.
        linear_dim (int): Dimension of the linear layer.
        is_causal (bool): Whether the attention is causal.
        cp_group, cp_global_ranks, cp_stream: Context-parallel plumbing. When
            `cp_group` is non-None and multi-rank, the CUTLASS path dispatches
            to `hstu_attn_varlen_cp_func`. The Triton and Torch backends
            currently raise on multi-rank CP (no CP wrapper for those paths
            yet — Slice 6+ extension).

    Returns:
        HSTUAttention: The created HSTUAttention module.

    Raises:
        ValueError: If the kernel backend is not supported.
    """
    cp_active = cp_group is not None and dist.get_world_size(cp_group) > 1
    attn: HSTUAttention
    if kernel_backend == KernelBackend.CUTLASS:
        sm_major_version = torch.cuda.get_device_properties(0).major
        if sm_major_version in (8, 9):
            attn = FusedHSTUAttention(
                num_heads,
                attention_dim,
                linear_dim,
                is_causal,
                cp_group=cp_group,
                cp_global_ranks=cp_global_ranks,
                cp_stream=cp_stream,
            )
        else:
            if cp_active:
                raise ValueError(
                    "Context Parallelism currently requires the CUTLASS "
                    "FusedHSTUAttention backend (SM 8/9). The Torch fallback "
                    "does not have a CP wrapper. Use is_causal=True with "
                    "head_dim ∈ {32,64,128,256} on A100/H100/H20."
                )
            print(
                "CUTLASS backend only support H100, H20 and A100/Ada series, fallback to PyTorch backend"
            )
            attn = TorchHSTUAttention(
                num_heads,
                attention_dim,
                linear_dim,
                is_causal,
            )
    elif kernel_backend == KernelBackend.TRITON:
        if cp_active:
            raise ValueError(
                "Context Parallelism is only wired through the CUTLASS "
                "FusedHSTUAttention backend in v0; the Triton backend has no "
                "CP wrapper yet. Switch kernel_backend to 'cutlass'."
            )
        if is_causal:
            attn = TritonHSTUAttention(
                num_heads,
                attention_dim,
                linear_dim,
                is_causal,
            )
        else:
            print(
                "Triton backend does not support is_causal=False, fallback to PyTorch backend"
            )
            attn = TorchHSTUAttention(
                num_heads,
                attention_dim,
                linear_dim,
                is_causal,
            )
    else:
        if cp_active:
            raise ValueError(
                "Context Parallelism is only wired through the CUTLASS "
                "FusedHSTUAttention backend in v0; the Torch backend has no "
                "CP wrapper. Switch kernel_backend to 'cutlass'."
            )
        attn = TorchHSTUAttention(
            num_heads,
            attention_dim,
            linear_dim,
            is_causal,
        )

    from commons.utils.attn_perf_tracker import PRINT_HSTU_PERF

    if PRINT_HSTU_PERF:
        from commons.utils.hooks import register_perf_hooks

        register_perf_hooks(attn, num_heads, attention_dim, is_causal)

    return attn
