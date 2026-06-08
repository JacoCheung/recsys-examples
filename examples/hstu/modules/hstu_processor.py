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
import itertools
import os
from typing import Dict, List, Optional, Tuple, Union, cast

import torch
from commons.datasets.hstu_batch import HSTUBatch
from commons.ops.cuda_ops.JaggedTensorOpFunction import jagged_2D_tensor_concat
from commons.ops.length_to_offsets import length_to_complete_offsets
from commons.ops.triton_ops.triton_jagged import triton_split_2D_jagged
from commons.utils.nvtx_op import output_nvtx_hook
from configs.hstu_config import HSTUConfig
from configs.inference_config import InferenceHSTUConfig
from modules.jagged_data import JaggedData, pad_jd_values, unpad_jd_values
from modules.mlp import MLP
from modules.position_encoder import HSTUPositionalEncoder
from modules.utils import init_mlp_weights_optional_bias
from torchrec.sparse.jagged_tensor import JaggedTensor

try:
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.mappings import (
        gather_from_sequence_parallel_region,
        scatter_to_sequence_parallel_region,
    )

    SUPPORT_TRAINING = True
except ImportError:
    SUPPORT_TRAINING = False


_YAMBDA_MLP_CHUNK_SIZE = int(os.environ.get("YAMBDA_MLP_CHUNK_SIZE", "131072"))


def _chunked_mlp_forward(
    mlp: torch.nn.Module,
    inputs: torch.Tensor,
    output_dim: int,
    chunk_size: int = _YAMBDA_MLP_CHUNK_SIZE,
) -> torch.Tensor:
    if not inputs.is_cuda or inputs.size(0) <= chunk_size:
        return mlp(inputs)
    outputs = []
    for start in range(0, inputs.size(0), chunk_size):
        end = min(start + chunk_size, inputs.size(0))
        outputs.append(mlp(inputs[start:end]))
    return torch.cat(outputs, dim=0)


class _SwishLayerNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = torch.nn.functional.layer_norm(
            x,
            (x.shape[-1],),
            self.weight.to(dtype),
            self.bias.to(dtype),
            self.eps,
        )
        return (x * torch.sigmoid(normalized)).to(dtype)


class _LayerNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return torch.nn.functional.layer_norm(
            x,
            (x.shape[-1],),
            self.weight.to(dtype),
            self.bias.to(dtype),
            self.eps,
        ).to(dtype)


class YambdaActionEmbeddingMLP(torch.nn.Module):
    def __init__(
        self,
        action_embedding_dim: int,
        hidden_dim: int,
        output_embedding_dim: int,
    ) -> None:
        super().__init__()
        self._output_embedding_dim = output_embedding_dim
        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(action_embedding_dim, hidden_dim),
            _SwishLayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, output_embedding_dim),
            _LayerNorm(output_embedding_dim),
        ).apply(init_mlp_weights_optional_bias)

    def forward(self, action_embeddings: torch.Tensor) -> torch.Tensor:
        return _chunked_mlp_forward(
            self._mlp, action_embeddings, self._output_embedding_dim
        )


class YambdaEmbeddingMLP(torch.nn.Module):
    def __init__(
        self,
        input_embedding_dim: int,
        hidden_dim: int,
        output_embedding_dim: int,
    ) -> None:
        super().__init__()
        self._output_embedding_dim = output_embedding_dim
        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(input_embedding_dim, hidden_dim),
            _SwishLayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, output_embedding_dim),
            _LayerNorm(output_embedding_dim),
        ).apply(init_mlp_weights_optional_bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return _chunked_mlp_forward(self._mlp, embeddings, self._output_embedding_dim)

    def forward_concat(self, *embedding_parts: torch.Tensor) -> torch.Tensor:
        if len(embedding_parts) == 0:
            raise ValueError("forward_concat expects at least one tensor")
        rows = embedding_parts[0].size(0)
        if not embedding_parts[0].is_cuda or rows <= _YAMBDA_MLP_CHUNK_SIZE:
            return self.forward(torch.cat(list(embedding_parts), dim=1))
        outputs = []
        for start in range(0, rows, _YAMBDA_MLP_CHUNK_SIZE):
            end = min(start + _YAMBDA_MLP_CHUNK_SIZE, rows)
            chunk = torch.cat([part[start:end] for part in embedding_parts], dim=1)
            outputs.append(self._mlp(chunk))
        return torch.cat(outputs, dim=0)


class YambdaContextualEmbeddingLinear(torch.nn.Module):
    def __init__(
        self,
        num_contextual_features: int,
        input_embedding_dim: int,
        output_embedding_dim: int,
    ) -> None:
        super().__init__()
        self._num_contextual_features = num_contextual_features
        std = (2.0 / float(input_embedding_dim + output_embedding_dim)) ** 0.5
        self._weights = torch.nn.Parameter(
            torch.empty(
                num_contextual_features,
                input_embedding_dim,
                output_embedding_dim,
            ).normal_(0.0, std)
        )
        self._bias = torch.nn.Parameter(
            torch.empty(num_contextual_features, output_embedding_dim).fill_(0.0)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.numel() == 0:
            return embeddings.new_empty((0, self._bias.size(1)))
        contextual = embeddings.view(
            -1,
            self._num_contextual_features,
            embeddings.size(-1),
        )
        transformed = torch.baddbmm(
            self._bias.view(-1, 1, self._bias.size(1)).to(contextual.dtype),
            contextual.transpose(0, 1),
            self._weights.to(contextual.dtype),
        ).transpose(0, 1)
        return transformed.reshape(-1, self._bias.size(1))


class YambdaTimestampLayerNormPostprocessor(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        time_duration_features: Tuple[Tuple[int, int], ...] = (
            (60 * 60, 24),
            (24 * 60 * 60, 7),
        ),
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self._layer_norm = torch.nn.LayerNorm(
            normalized_shape=[embedding_dim],
            eps=eps,
        )
        self.register_buffer(
            "_period_units",
            torch.tensor(
                [f[0] for f in time_duration_features],
                dtype=torch.float32,
            ).view(1, -1),
        )
        self.register_buffer(
            "_units_per_period",
            torch.tensor(
                [f[1] for f in time_duration_features],
                dtype=torch.float32,
            ).view(1, -1),
        )
        self._time_feature_combiner = torch.nn.Linear(
            embedding_dim + 2 * len(time_duration_features),
            embedding_dim,
        ).apply(init_mlp_weights_optional_bias)

    def _concat_time_features(
        self,
        embeddings: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        output_dtype = embeddings.dtype
        units_since_epoch = torch.div(
            timestamps.unsqueeze(-1).float(),
            self._period_units,
            rounding_mode="floor",
        )
        units_elapsed = (
            torch.remainder(units_since_epoch, self._units_per_period)
            / self._units_per_period
            * 2
            * 3.14
        )
        units_elapsed = torch.view_as_real(
            torch.polar(
                torch.ones_like(units_elapsed, dtype=torch.float32),
                units_elapsed.to(torch.float32),
            )
        ).flatten(-2, -1)
        return torch.cat([embeddings, units_elapsed.to(output_dtype)], dim=-1)

    def forward(
        self,
        embeddings: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        combined = self._concat_time_features(embeddings, timestamps)
        postprocessed = self._time_feature_combiner(
            combined.to(self._time_feature_combiner.weight.dtype)
        )
        return self._layer_norm(postprocessed)


class YambdaActionEncoder(torch.nn.Module):
    def __init__(
        self,
        action_embedding_dim: int = 8,
        action_weights: Tuple[int, ...] = (1, 2, 4),
        embedding_init_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "_combined_action_weights",
            torch.tensor(action_weights, dtype=torch.int64),
        )
        self._num_action_types = len(action_weights)
        self._action_embedding_dim = action_embedding_dim
        self._action_embedding_table = torch.nn.Parameter(
            torch.empty((self._num_action_types, action_embedding_dim)).normal_(
                mean=0.0, std=embedding_init_std
            )
        )
        self._target_action_embedding_table = torch.nn.Parameter(
            torch.empty((1, self.output_embedding_dim)).normal_(
                mean=0.0, std=embedding_init_std
            )
        )

    @property
    def output_embedding_dim(self) -> int:
        return self._num_action_types * self._action_embedding_dim

    def forward(
        self,
        action_weights_jt: JaggedTensor,
        num_candidates: torch.Tensor,
        max_history_seqlen: int,
        max_num_candidates: int,
    ) -> torch.Tensor:
        seq_actions = action_weights_jt.values().to(torch.int64)
        exploded_actions = (
            torch.bitwise_and(
                seq_actions.unsqueeze(-1),
                self._combined_action_weights.unsqueeze(0),
            )
            > 0
        )
        history_action_embeddings = (
            exploded_actions.unsqueeze(-1).to(self._action_embedding_table.dtype)
            * self._action_embedding_table.unsqueeze(0)
        ).view(-1, self.output_embedding_dim)

        target_offsets = length_to_complete_offsets(num_candidates).to(
            action_weights_jt.offsets().dtype
        )
        total_targets = int(num_candidates.sum().item())
        target_action_embeddings = self._target_action_embedding_table.tile(
            total_targets,
            1,
        )
        action_embeddings, _ = jagged_2D_tensor_concat(
            [history_action_embeddings, target_action_embeddings],
            [action_weights_jt.offsets(), target_offsets],
            [max_history_seqlen, max_num_candidates],
        )
        return action_embeddings


def hstu_preprocess_embeddings(
    embeddings: Dict[str, JaggedTensor],
    batch: HSTUBatch,
    is_inference: bool,
    item_mlp: Optional[MLP] = None,
    contextual_mlp: Optional[torch.nn.Module] = None,
    yambda_content_embedding_mlp: Optional[YambdaEmbeddingMLP] = None,
    yambda_additional_embedding_mlp: Optional[YambdaEmbeddingMLP] = None,
    action_encoder: Optional[YambdaActionEncoder] = None,
    action_embedding_mlp: Optional[YambdaActionEmbeddingMLP] = None,
    dtype: Optional[torch.dtype] = None,
    scaling_seqlen: int = -1,
) -> JaggedData:
    """
    Preprocesses the embeddings for use in the HSTU architecture.

    This method performs the following steps:
    1. **Interleaving**: If action embeddings are present, interleaves them with item embeddings.
                         During inference, action embeddings are only for the history sequence, and
                         they will be interleaved with item embeddings of the history part, while
                         the embeddings of candidates need no interleaving.
    2. **Concatenation**: Concatenates contextual, item, and action embeddings for each sample,
                          following the order specified in the batch.
                          During inference, we concatenate three parts:
                          1) contextual embeedings,
                          2) interleaved *item & action* history embeddings, and
                          3) (item) candidates embeddings
                          for each sample, following the order specified in the batch.

    Args:
        embeddings (Dict[str, JaggedTensor]): A dictionary of embeddings where each key corresponds to a feature name and the value is a jagged tensor.
        batch (HSTUBatch): The batch of ranking data.
        is_inference (bool): Whether is for inference
        dtype (dtype, optional): The output data type of the embeddings.
    Returns:
        JaggedData: The preprocessed jagged data, ready for further processing in the HSTU architecture.
    """
    sequence_feature_names = getattr(batch, "sequence_feature_names", None)
    has_multi_feature_sequence = (
        sequence_feature_names is not None and len(sequence_feature_names) > 0
    )
    if has_multi_feature_sequence:
        sequence_feature_names = cast(List[str], sequence_feature_names)
        sequence_jts = [embeddings[name] for name in sequence_feature_names]
        item_jt = sequence_jts[0]
        dtype = item_jt.values().dtype if dtype is None else dtype
        if (
            yambda_content_embedding_mlp is not None
            and yambda_additional_embedding_mlp is not None
        ):
            sequence_embeddings = yambda_content_embedding_mlp(
                sequence_jts[0].values().to(dtype)
            )
            sequence_embeddings = (
                sequence_embeddings
                + yambda_additional_embedding_mlp.forward_concat(
                    *[jt.values().to(dtype) for jt in sequence_jts[1:]]
                )
            )
        else:
            sequence_embeddings = torch.cat(
                [jt.values().to(dtype) for jt in sequence_jts], dim=1
            )
        sequence_embeddings_lengths = item_jt.lengths()
        sequence_embeddings_lengths_offsets = item_jt.offsets()
        sequence_max_seqlen = batch.feature_to_max_seqlen[sequence_feature_names[0]]
        if item_mlp is not None and yambda_content_embedding_mlp is None:
            sequence_embeddings = item_mlp(sequence_embeddings)
        if action_encoder is not None and action_embedding_mlp is not None:
            action_weights = getattr(batch, "action_weights", None)
            if action_weights is None:
                raise ValueError(
                    "Yambda action encoder is enabled but batch.action_weights is missing"
                )
            if batch.num_candidates is None:
                raise ValueError(
                    "Yambda action encoder expects ranking batches with num_candidates"
                )
            action_weight_feature_name = getattr(
                batch, "action_weight_feature_name", "action_weight"
            )
            action_embeddings = action_encoder(
                action_weights_jt=action_weights[action_weight_feature_name],
                num_candidates=batch.num_candidates,
                max_history_seqlen=batch.feature_to_max_seqlen[
                    action_weight_feature_name
                ],
                max_num_candidates=batch.max_num_candidates,
            )
            sequence_embeddings = sequence_embeddings + action_embedding_mlp(
                action_embeddings
            ).to(sequence_embeddings.dtype)
        has_interleaved_action = False
    else:
        item_jt = embeddings[batch.item_feature_name]  # history + candidate
        dtype = item_jt.values().dtype if dtype is None else dtype
        sequence_embeddings = item_jt.values().to(dtype)
        sequence_embeddings_lengths = item_jt.lengths()
        sequence_embeddings_lengths_offsets = item_jt.offsets()
        sequence_max_seqlen = batch.feature_to_max_seqlen[batch.item_feature_name]
        has_interleaved_action = batch.action_feature_name is not None

        if batch.action_feature_name is not None:
            action_jt = embeddings[batch.action_feature_name]
            jagged_size = sequence_embeddings.size(0)
            embedding_dim = sequence_embeddings.size(1)

            if not is_inference:
                sequence_embeddings = torch.cat(
                    [sequence_embeddings, action_jt.values().to(dtype)], dim=1
                ).view(2 * jagged_size, embedding_dim)
                sequence_embeddings_lengths = sequence_embeddings_lengths * 2
                sequence_embeddings_lengths_offsets = (
                    sequence_embeddings_lengths_offsets * 2
                )
                sequence_max_seqlen = sequence_max_seqlen * 2
            else:
                # TODO@junyi: We can optimize the concat:
                # 1. use jagged split to get [history_embs, candidate_embs]
                # 2. use cat to interleave the history_embs and history_action_embs part
                # 3. use jagged concat to append the candidate_embs

                action_offsets = action_jt.offsets()
                item_offsets = item_jt.offsets()
                candidates_indptr = (
                    item_offsets[: batch.batch_size] + action_jt.lengths()
                )

                item_embs = item_jt.values().to(dtype)
                action_embs = action_jt.values().to(dtype)
                if not torch.compiler.is_compiling():
                    interleaved_embeddings = [
                        (
                            torch.cat(
                                [
                                    item_embs[
                                        item_offsets[idx]
                                        .item() : candidates_indptr[idx]
                                        .item()
                                    ],
                                    action_embs[
                                        action_offsets[idx]
                                        .item() : action_offsets[idx + 1]
                                        .item()
                                    ],
                                ],
                                dim=1,
                            ).view(-1, embedding_dim),
                            item_embs[
                                candidates_indptr[idx]
                                .item() : item_offsets[idx + 1]
                                .item()
                            ],
                        )
                        for idx in range(batch.batch_size)
                    ]
                    interleaved_embeddings = list(
                        itertools.chain(*interleaved_embeddings)
                    )
                    sequence_embeddings = torch.cat(interleaved_embeddings, dim=0).view(
                        -1, embedding_dim
                    )
                else:
                    interleaved_embeddings = list()
                    for idx in range(batch.batch_size):
                        interleaved_embeddings.append(
                            torch.cat(
                                [
                                    item_embs[
                                        torch.arange(
                                            item_offsets[idx], candidates_indptr[idx]
                                        )
                                    ],
                                    action_embs[
                                        torch.arange(
                                            action_offsets[idx], action_offsets[idx + 1]
                                        )
                                    ],
                                ],
                                dim=1,
                            ).view(-1, embedding_dim)
                        )
                        interleaved_embeddings.append(
                            item_embs[
                                torch.arange(
                                    candidates_indptr[idx], item_offsets[idx + 1]
                                )
                            ]
                        )
                    sequence_embeddings = torch.cat(interleaved_embeddings, dim=0).view(
                        -1, embedding_dim
                    )
                sequence_embeddings_lengths = item_jt.lengths() + action_jt.lengths()
                sequence_embeddings_lengths_offsets = (
                    item_jt.offsets() + action_jt.offsets()
                )
                sequence_max_seqlen += batch.feature_to_max_seqlen[
                    batch.action_feature_name
                ]
            if item_mlp is not None:
                sequence_embeddings = item_mlp(sequence_embeddings)

    if (
        batch.num_candidates is not None
        and batch.action_feature_name is not None
        and not is_inference
    ):
        num_candidates = batch.num_candidates * 2
        max_num_candidates = batch.max_num_candidates * 2
    else:
        num_candidates = batch.num_candidates
        max_num_candidates = batch.max_num_candidates

    sequence_timestamps = None
    sequence_timestamps_kjt = getattr(batch, "sequence_timestamps", None)
    if sequence_timestamps_kjt is not None:
        timestamp_feature_name = getattr(
            batch, "sequence_timestamp_feature_name", "sequence_timestamp"
        )
        sequence_timestamps = (
            sequence_timestamps_kjt[timestamp_feature_name].values().to(torch.int64)
        )

    contextual_max_seqlen = 0
    contextual_seqlen = None
    contextual_seqlen_offsets = None
    if len(batch.contextual_feature_names) > 0:
        contextual_max_seqlens = [
            batch.feature_to_max_seqlen[name] for name in batch.contextual_feature_names
        ]
        contextual_jts = [embeddings[name] for name in batch.contextual_feature_names]
        contextual_jts_values = [jt.values().to(dtype) for jt in contextual_jts]
        contextual_jts_offsets = [jt.offsets() for jt in contextual_jts]

        (contextual_sequence_embeddings, contextual_seqlen) = jagged_2D_tensor_concat(
            contextual_jts_values,
            contextual_jts_offsets,
            contextual_max_seqlens,
        )
        # torch._check_tensor_all(torch.sum(contextual_seqlen, dim=0) != 0, "contextual_seqlen is 0")
        if contextual_mlp is not None:
            contextual_sequence_embeddings = contextual_mlp(
                contextual_sequence_embeddings
            )
        contextual_seqlen_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            contextual_seqlen
        )
        contextual_max_seqlen = max(
            len(batch.contextual_feature_names), sum(contextual_max_seqlens)
        )
        if sequence_timestamps is not None:
            contextual_sequence_timestamps = torch.zeros(
                (contextual_sequence_embeddings.size(0), 1),
                device=sequence_timestamps.device,
                dtype=sequence_timestamps.dtype,
            )
            sequence_timestamps, _ = jagged_2D_tensor_concat(
                [
                    contextual_sequence_timestamps,
                    sequence_timestamps.unsqueeze(-1),
                ],
                [contextual_seqlen_offsets, sequence_embeddings_lengths_offsets],
                [contextual_max_seqlen, sequence_max_seqlen],
            )
            sequence_timestamps = sequence_timestamps.squeeze(-1)
        (
            sequence_embeddings,
            sequence_embeddings_lengths,
        ) = jagged_2D_tensor_concat(
            [contextual_sequence_embeddings, sequence_embeddings],
            [contextual_seqlen_offsets, sequence_embeddings_lengths_offsets],
            [contextual_max_seqlen, sequence_max_seqlen],
        )

        sequence_embeddings_lengths_offsets = (
            torch.ops.fbgemm.asynchronous_complete_cumsum(sequence_embeddings_lengths)
        )
        sequence_max_seqlen = sequence_max_seqlen + contextual_max_seqlen

    # After balanced shuffler, dense tensors (num_candidates) are stripped to
    # actual_batch_size while KJTs retain batch_size entries (see BaseBatch
    # invariants).  Re-pad num_candidates with zeros so it stays aligned with
    # the KJT-derived sequence_embeddings_lengths.
    if num_candidates is not None:
        bs_kjt = sequence_embeddings_lengths.size(0)
        if num_candidates.size(0) < bs_kjt:
            num_candidates = torch.nn.functional.pad(
                num_candidates, (0, bs_kjt - num_candidates.size(0))
            )

    num_candidates_offsets = (
        length_to_complete_offsets(num_candidates).to(torch.int32)
        if num_candidates is not None
        else None
    )
    total_candidates_seq_len = None
    if not is_inference:
        if num_candidates is not None:
            total_candidates_seq_len = num_candidates.sum()
        elif contextual_seqlen is not None:
            total_candidates_seq_len = (
                sequence_embeddings_lengths.sum() - contextual_seqlen.sum()
            )
    elif torch.compiler.is_compiling():
        assert (
            num_candidates is not None
        ), "num_candidates should not be None during inference when compiling"
        total_candidates_seq_len = num_candidates.sum()
    # In eager mode, materialize the 0-d tensor to a Python int here so the
    # downstream `torch.empty((seq_len_a, D), ...)` inside _Split2DJaggedFunction
    # does not trigger an implicit D2H sync at the end of the forward pass.
    # During tracing (torch.export / torch.compile) keep it as a tensor so the
    # dynamic shape is preserved in the graph (see PR #327).
    if total_candidates_seq_len is not None and not torch.compiler.is_compiling():
        total_candidates_seq_len = int(total_candidates_seq_len.item())
    return JaggedData(
        values=sequence_embeddings,
        seqlen=sequence_embeddings_lengths.to(
            torch.int32
        ),  # contextual + history + candidate
        seqlen_offsets=sequence_embeddings_lengths_offsets.to(torch.int32),
        timestamps=sequence_timestamps,
        max_seqlen=sequence_max_seqlen,
        max_num_candidates=max_num_candidates,
        num_candidates=num_candidates.to(torch.int32)
        if num_candidates is not None
        else None,
        num_candidates_offsets=num_candidates_offsets,
        contextual_max_seqlen=contextual_max_seqlen,
        contextual_seqlen=contextual_seqlen.to(torch.int32)
        if contextual_seqlen is not None
        else None,
        contextual_seqlen_offsets=contextual_seqlen_offsets.to(torch.int32)
        if contextual_seqlen_offsets is not None
        else None,
        has_interleaved_action=has_interleaved_action,
        scaling_seqlen=scaling_seqlen,
        total_candidates_seq_len=total_candidates_seq_len,
    )


class HSTUBlockPreprocessor(torch.nn.Module):
    """
    HSTUBlock module. A stack of HSTULayers.

    Args:
        config (HSTUConfig): Configuration for the HSTU block.
    """

    def __init__(
        self,
        config: Union[HSTUConfig, InferenceHSTUConfig],
        is_inference: bool,
    ):
        super().__init__()
        self.config = config
        self._training_dtype = torch.float32
        if config.bf16:
            self._training_dtype = torch.bfloat16
        if config.fp16:
            self._training_dtype = torch.float16
        if isinstance(config, HSTUConfig):
            self._sequence_parallel = config.sequence_parallel
        else:
            self._sequence_parallel = False
        self._tp_size = 1
        if is_inference:
            self._sequence_parallel = False
        if not is_inference and SUPPORT_TRAINING:
            self._tp_size = parallel_state.get_tensor_model_parallel_world_size()

        self._item_mlp = None
        self._contextual_mlp = None
        self._yambda_content_embedding_mlp = None
        self._yambda_additional_embedding_mlp = None
        self._yambda_action_encoder = None
        self._yambda_action_embedding_mlp = None
        if config.hstu_preprocessing_config is not None:
            is_yambda_reference_preprocessor = (
                config.hstu_preprocessing_config.enable_yambda_action_encoder
            )
            if is_yambda_reference_preprocessor:
                preprocessor_hidden_dim = (
                    config.hstu_preprocessing_config.yambda_action_mlp_hidden_dim
                )
                self._yambda_content_embedding_mlp = YambdaEmbeddingMLP(
                    input_embedding_dim=config.hidden_size,
                    hidden_dim=preprocessor_hidden_dim,
                    output_embedding_dim=config.hidden_size,
                )
                self._yambda_additional_embedding_mlp = YambdaEmbeddingMLP(
                    input_embedding_dim=(
                        config.hstu_preprocessing_config.yambda_additional_embedding_dim
                    ),
                    hidden_dim=preprocessor_hidden_dim,
                    output_embedding_dim=config.hidden_size,
                )
                self._contextual_mlp = YambdaContextualEmbeddingLinear(
                    num_contextual_features=(
                        config.hstu_preprocessing_config.yambda_num_contextual_features
                    ),
                    input_embedding_dim=config.hidden_size,
                    output_embedding_dim=config.hidden_size,
                )
            elif config.hstu_preprocessing_config.item_embedding_dim > 0:
                self._item_mlp = MLP(
                    in_size=config.hstu_preprocessing_config.item_embedding_dim,
                    layer_sizes=[config.hidden_size, config.hidden_size],
                    activation="relu",
                    bias=True,
                )
            if config.hstu_preprocessing_config.enable_yambda_action_encoder:
                action_embedding_dim = (
                    config.hstu_preprocessing_config.yambda_action_embedding_dim
                )
                self._yambda_action_encoder = YambdaActionEncoder(
                    action_embedding_dim=action_embedding_dim,
                    action_weights=(1, 2, 4),
                )
                self._yambda_action_embedding_mlp = YambdaActionEmbeddingMLP(
                    action_embedding_dim=3 * action_embedding_dim,
                    hidden_dim=(
                        config.hstu_preprocessing_config.yambda_action_mlp_hidden_dim
                    ),
                    output_embedding_dim=config.hidden_size,
                )
            if (
                not is_yambda_reference_preprocessor
                and config.hstu_preprocessing_config.contextual_embedding_dim > 0
            ):
                self._contextual_mlp = MLP(
                    in_size=config.hstu_preprocessing_config.contextual_embedding_dim,
                    layer_sizes=[config.hidden_size, config.hidden_size],
                    activation="relu",
                    bias=True,
                )

        self._positional_encoder: Optional[HSTUPositionalEncoder] = None
        if config.position_encoding_config is not None:
            self._positional_encoder = HSTUPositionalEncoder(
                num_position_buckets=config.position_encoding_config.num_position_buckets,
                num_time_buckets=config.position_encoding_config.num_time_buckets,
                embedding_dim=config.hidden_size,
                is_inference=is_inference,
                use_time_encoding=config.position_encoding_config.use_time_encoding,
                training_dtype=self._training_dtype,
                static_max_seq_len=config.position_encoding_config.static_max_seq_len,
            )
        self._is_inference = is_inference
        self._dropout_ratio = 0.0
        if not self._is_inference:
            assert isinstance(
                config, HSTUConfig
            ), "Training config should be HSTUConfig"
            self._dropout_ratio = config.hidden_dropout
        self._scaling_seqlen = config.scaling_seqlen

    @output_nvtx_hook(nvtx_tag="HSTUBlock preprocess", hook_key_or_attr_name="values")
    def forward(
        self,
        embeddings: Dict[str, JaggedTensor],
        batch: HSTUBatch,
        seq_start_position: Optional[torch.Tensor] = None,
    ) -> JaggedData:
        """
        Preprocesses the embeddings for use in the HSTU architecture.

        This method performs the following steps:
        1. **Interleaving**: If action embeddings are present, interleaves them with item embeddings.
        2. **Concatenation**: Concatenates contextual, item, and action embeddings for each sample, following the order specified in the batch.
        3. **Padding**: Pads the jagged length of JaggedData to the TP size if sequence parallel is enabled.
        4. **Position Encoding**: Applies position encoding to the concatenated embeddings.

        Args:
            embeddings (Dict[str, JaggedTensor]): A dictionary of embeddings where each key corresponds to a feature name and the value is a jagged tensor.
            batch (HSTUBatch): The batch of ranking data.

        Returns:
            JaggedData: The preprocessed jagged data, ready for further processing in the HSTU architecture.
        """
        device = torch.device("cuda", torch.cuda.current_device())
        batch = batch.to(device)
        # Interleaving & concatenation
        jd = hstu_preprocess_embeddings(
            embeddings,
            batch,
            is_inference=self._is_inference,
            item_mlp=self._item_mlp,
            contextual_mlp=self._contextual_mlp,
            yambda_content_embedding_mlp=self._yambda_content_embedding_mlp,
            yambda_additional_embedding_mlp=self._yambda_additional_embedding_mlp,
            action_encoder=self._yambda_action_encoder,
            action_embedding_mlp=self._yambda_action_embedding_mlp,
            dtype=self._training_dtype,
            scaling_seqlen=self._scaling_seqlen,
        )
        if self._sequence_parallel:
            jd = pad_jd_values(jd, self._tp_size)
        if self._positional_encoder is not None:
            if (
                getattr(self._positional_encoder, "_use_time_encoding", False)
                and jd.timestamps is None
            ):
                raise ValueError(
                    "HSTU time positional encoding is enabled but batch sequence "
                    "timestamps are missing"
                )
            jd.values = self._positional_encoder(
                max_seq_len=jd.max_seqlen,
                seq_lengths=jd.seqlen,
                seq_offsets=jd.seqlen_offsets,
                seq_timestamps=jd.timestamps,
                seq_embeddings=jd.values,
                num_targets=jd.num_candidates,
                seq_start_position=seq_start_position,
                max_contextual_seq_len=jd.contextual_max_seqlen,
            )

        jd.values = torch.nn.functional.dropout(
            jd.values,
            p=self._dropout_ratio,
            training=self.training,
        ).to(self._training_dtype)
        # when sequence parallel is on, we need to scatter the values to the sequence parallel region
        # mcore performs the scatter in embedding: https://github.com/NVIDIA/Megatron-LM/blob/a32ff750191d04713ea1c15dcc65308324681016/megatron/core/tensor_parallel/layers.py#L286-L291
        # but we have to perform interleave and concatenation here.
        if self._sequence_parallel:
            jd.values = scatter_to_sequence_parallel_region(jd.values)
        return jd


class HSTUBlockPostprocessor(torch.nn.Module):
    """
    HSTUBlock module. A stack of HSTULayers.

    Args:
        config (HSTUConfig): Configuration for the HSTU block.
    """

    def __init__(
        self,
        is_inference: bool,
        sequence_parallel: bool = False,
        normalize_output: bool = True,
        timestamp_postprocessor: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self._is_inference = is_inference
        self._sequence_parallel = sequence_parallel
        self._normalize_output = normalize_output
        self._timestamp_postprocessor = timestamp_postprocessor

        if self._is_inference:
            self._sequence_parallel = False

    @output_nvtx_hook(nvtx_tag="HSTUBlock postprocess", hook_key_or_attr_name="values")
    def forward(self, jd: JaggedData) -> JaggedData:
        """
        Postprocess the output from the HSTU architecture.
        1. If max_num_candidates > 0, split and only keep last ``num_candidates`` embeddings as candidates embedding for further processing.
        2, If sequence parallel is on, we need to gather the values back and remove the padding.
        3. Remove action embeddings if present. Only use item embedding for further processing.

        Args:
            jd (JaggedData): The jagged data output from the HSTU architecture that needs further processing.

        Returns:
            JaggedData: The postprocessed jagged data.
        """
        sequence_embeddings: torch.Tensor
        sequence_timestamps: Optional[torch.Tensor] = None
        seqlen_offsets: torch.Tensor
        max_seqlen: int
        # the following compute is duplicated among TP ranks, we need to AG and remove the padding,
        # during backward, the gradients are scattered among TP ranks
        if self._sequence_parallel:
            jd.values = gather_from_sequence_parallel_region(
                jd.values, False
            )  # False -> output grad not RS but S
            jd = unpad_jd_values(jd)
        # Derive seq_len_a/b from total_candidates_seq_len to avoid D2H sync.
        # After SP gather + unpad, values.shape[0] is the true total; precomputed length still valid.
        # total_candidates_seq_len is None for inference (set in hstu_preprocess_embeddings).
        if jd.total_candidates_seq_len is not None:
            total_seq = jd.values.shape[0]
            precomputed_b = jd.total_candidates_seq_len
            precomputed_a = total_seq - jd.total_candidates_seq_len
            if not torch.compiler.is_compiling():
                assert precomputed_a >= 0, (
                    f"precomputed_a is negative ({precomputed_a}): total_seq={total_seq}, "
                    f"total_candidates_seq_len={jd.total_candidates_seq_len}"
                )
        else:
            precomputed_a = None
            precomputed_b = None
        if jd.max_num_candidates > 0:
            seqlen_offsets = jd.num_candidates_offsets
            max_seqlen = jd.max_num_candidates
            _, sequence_embeddings = triton_split_2D_jagged(
                jd.values,
                jd.max_seqlen,
                offsets_a=jd.seqlen_offsets - jd.num_candidates_offsets,
                offsets_b=seqlen_offsets,
                seq_len_a=precomputed_a,
                seq_len_b=precomputed_b,
            )
            if jd.timestamps is not None:
                _, sequence_timestamps_2d = triton_split_2D_jagged(
                    jd.timestamps.unsqueeze(-1),
                    jd.max_seqlen,
                    offsets_a=jd.seqlen_offsets - jd.num_candidates_offsets,
                    offsets_b=seqlen_offsets,
                    seq_len_a=precomputed_a,
                    seq_len_b=precomputed_b,
                )
                sequence_timestamps = sequence_timestamps_2d.squeeze(-1)
        elif jd.contextual_max_seqlen > 0:
            seqlen_offsets = jd.seqlen_offsets - jd.contextual_seqlen_offsets
            max_seqlen = jd.max_seqlen - jd.contextual_max_seqlen
            _, sequence_embeddings = triton_split_2D_jagged(
                jd.values,
                jd.max_seqlen,
                offsets_a=jd.contextual_seqlen_offsets,
                offsets_b=seqlen_offsets,
                seq_len_a=precomputed_a,
                seq_len_b=precomputed_b,
            )
            if jd.timestamps is not None:
                _, sequence_timestamps_2d = triton_split_2D_jagged(
                    jd.timestamps.unsqueeze(-1),
                    jd.max_seqlen,
                    offsets_a=jd.contextual_seqlen_offsets,
                    offsets_b=seqlen_offsets,
                    seq_len_a=precomputed_a,
                    seq_len_b=precomputed_b,
                )
                sequence_timestamps = sequence_timestamps_2d.squeeze(-1)
        else:
            sequence_embeddings = jd.values
            sequence_timestamps = jd.timestamps
            seqlen_offsets = jd.seqlen_offsets
            max_seqlen = jd.max_seqlen

        if jd.has_interleaved_action and not self._is_inference:
            if not torch.compiler.is_compiling():
                sequence_embeddings = sequence_embeddings[0::2, ...]
                if sequence_timestamps is not None:
                    sequence_timestamps = sequence_timestamps[0::2]
            else:
                sequence_embeddings = sequence_embeddings.view(
                    sequence_embeddings.size(0) // 2, 2, -1
                )
                sequence_embeddings = sequence_embeddings[:, 0, ...]
                if sequence_timestamps is not None:
                    sequence_timestamps = sequence_timestamps.view(-1, 2)[:, 0]
            seqlen_offsets = seqlen_offsets // 2
            max_seqlen = max_seqlen // 2

        if self._normalize_output:
            sequence_embeddings = sequence_embeddings / torch.linalg.norm(
                sequence_embeddings, ord=2, dim=-1, keepdim=True
            ).clamp(min=1e-6)
        if self._timestamp_postprocessor is not None:
            if sequence_timestamps is None:
                raise ValueError(
                    "Yambda timestamp postprocessor is enabled but timestamps are missing"
                )
            sequence_embeddings = self._timestamp_postprocessor(
                sequence_embeddings,
                sequence_timestamps,
            )

        return JaggedData(
            values=sequence_embeddings,
            timestamps=sequence_timestamps,
            seqlen=torch.diff(seqlen_offsets).to(jd.seqlen.dtype),
            seqlen_offsets=seqlen_offsets.to(jd.seqlen_offsets.dtype),
            max_seqlen=max_seqlen,
            has_interleaved_action=False,
            scaling_seqlen=jd.scaling_seqlen,
        )
