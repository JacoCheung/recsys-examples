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
from typing import Any, Dict, Optional, Tuple

import torch
from commons.datasets.hstu_batch import HSTUBatch
from commons.distributed.dmp_to_tp import (
    dmp_batch_to_tp,
    jt_dict_grad_scaling_and_allgather,
)
from commons.modules.embedding import ShardedEmbedding
from commons.ops.length_to_offsets import length_to_complete_offsets
from commons.ops.triton_ops.triton_jagged import triton_split_2D_jagged
from configs import HSTUConfig, RankingConfig
from megatron.core import parallel_state
from model.base_model import BaseModel
from modules.hstu_block import HSTUBlock
from modules.hstu_processor import YambdaEmbeddingMLP
from modules.metrics import get_multi_event_metric_module
from modules.mlp import MLP
from modules.multi_task_loss_module import MultiTaskLossModule
from torchrec.sparse.jagged_tensor import JaggedTensor


class RankingGR(BaseModel):
    """
    A class representing the ranking model. Inherits from BaseModel. A ranking model consists of
    a sparse architecture and a dense architecture. A ranking model is able to process multiple labels
    and thus has multiple logit dimensions. Each label is associated with a loss functoin (e.g. BCE, CE).

    Args:
        hstu_config (HSTUConfig): The HSTU configuration.
        task_config (RankingConfig): The ranking task configuration.
    """

    def __init__(
        self,
        hstu_config: HSTUConfig,
        task_config: RankingConfig,
    ):
        super().__init__()
        self._tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._hstu_config = hstu_config
        self._task_config = task_config

        self._embedding_collection = ShardedEmbedding(task_config.embedding_configs)

        self._hstu_block = HSTUBlock(hstu_config)
        self._use_yambda_reference_head = (
            hstu_config.hstu_preprocessing_config is not None
            and hstu_config.hstu_preprocessing_config.enable_yambda_action_encoder
        )
        self._yambda_item_embedding_mlp: Optional[YambdaEmbeddingMLP] = None
        if self._use_yambda_reference_head:
            self._yambda_item_embedding_mlp = YambdaEmbeddingMLP(
                input_embedding_dim=3 * hstu_config.hidden_size,
                hidden_dim=hstu_config.hidden_size,
                output_embedding_dim=hstu_config.hidden_size,
            )
        self._mlp = MLP(
            hstu_config.hidden_size,
            task_config.prediction_head_arch,
            task_config.prediction_head_act_type,
            task_config.prediction_head_bias,
            device=self._device,
        )

        # TODO, make reduction configurable
        self._loss_module = MultiTaskLossModule(
            num_classes=task_config.prediction_head_arch[-1],
            num_tasks=task_config.num_tasks,
            reduction="none",
        )
        self._metric_module = get_multi_event_metric_module(
            num_classes=task_config.prediction_head_arch[-1],
            num_tasks=task_config.num_tasks,
            metric_types=task_config.eval_metrics,
            comm_pg=parallel_state.get_data_parallel_group(with_context_parallel=True),
        )

    def bfloat16(self):
        """
        Convert the model to use bfloat16 precision. Only affects the dense module.

        Returns:
            RankingGR: The model with bfloat16 precision.
        """
        self._hstu_block.bfloat16()
        if self._yambda_item_embedding_mlp is not None:
            self._yambda_item_embedding_mlp.bfloat16()
        self._mlp.bfloat16()
        return self

    def half(self):
        """
        Convert the model to use half precision. Only affects the dense module.

        Returns:
            RankingGR: The model with half precision.
        """
        self._hstu_block.half()
        if self._yambda_item_embedding_mlp is not None:
            self._yambda_item_embedding_mlp.half()
        self._mlp.half()
        return self

    def _get_yambda_candidate_item_embeddings(
        self,
        embeddings: Dict[str, JaggedTensor],
        batch: HSTUBatch,
    ) -> Optional[torch.Tensor]:
        if self._yambda_item_embedding_mlp is None:
            return None
        sequence_feature_names = getattr(batch, "sequence_feature_names", None)
        if sequence_feature_names is None or len(sequence_feature_names) == 0:
            return None
        if batch.num_candidates is None:
            raise ValueError("Yambda reference item tower expects num_candidates")

        num_candidates_offsets = length_to_complete_offsets(batch.num_candidates).to(
            embeddings[sequence_feature_names[0]].offsets().dtype
        )
        total_candidates = batch.num_candidates.sum()
        candidate_embeddings = []
        for name in sequence_feature_names:
            jt = embeddings[name]
            _, candidates = triton_split_2D_jagged(
                jt.values(),
                batch.feature_to_max_seqlen[name],
                offsets_a=jt.offsets() - num_candidates_offsets,
                offsets_b=num_candidates_offsets,
                seq_len_b=total_candidates,
            )
            candidate_embeddings.append(candidates)
        return self._yambda_item_embedding_mlp(torch.cat(candidate_embeddings, dim=1))

    def get_logit_and_labels(
        self, batch: HSTUBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the logits and labels for the batch.

        Args:
            batch (HSTUBatch): The batch of ranking data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The logits and labels.
        """
        # DMP embedding
        embeddings: Dict[str, JaggedTensor] = self._embedding_collection(batch.features)
        # maybe freeze embedding for debugging
        embeddings = self._embedding_collection._maybe_detach(embeddings)
        # For model-parallel embedding, torchrec does gradient division by (tp_size * dp_size). However, we only need to divide by dp size. In such case, we need to scale the gradient by tp_size.
        # But simultaneously, the DP embedding might be scaled by tp_size unintentionally. On the other hand, the DDP will divide the DP embedding gradient by dp_size (allreduce avg).
        # We need to perform allreduce sum across tp ranks after/before the DDP allreduce avg.
        grad_scaling_factor = self._tp_size
        embeddings = jt_dict_grad_scaling_and_allgather(
            embeddings,
            grad_scaling_factor,
            parallel_state.get_tensor_model_parallel_group(),
        )
        batch = dmp_batch_to_tp(batch)
        candidate_item_embeddings = self._get_yambda_candidate_item_embeddings(
            embeddings,
            batch,
        )
        # hidden_states is a JaggedData
        hidden_states_jagged, seqlen_after_preprocessor = self._hstu_block(
            embeddings=embeddings,
            batch=batch,
        )
        hidden_states = hidden_states_jagged.values
        if candidate_item_embeddings is not None:
            hidden_states = hidden_states * candidate_item_embeddings
        logits = self._mlp(hidden_states)
        return logits, seqlen_after_preprocessor, batch.labels.values()

    def forward(  # type: ignore[override]
        self,
        batch: HSTUBatch,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]]:
        """
        Perform the forward pass of the model.

        Args:
            batch (HSTUBatch): The batch of ranking data.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]]: The losses and a tuple of losses, logits, and labels.
        """
        (
            jagged_item_logit,
            seqlen_after_preprocessor,
            labels,
        ) = self.get_logit_and_labels(batch)
        losses = self._loss_module(jagged_item_logit.float(), labels)
        if self._use_yambda_reference_head:
            losses = losses * 0.2
        return losses, (
            losses.detach(),
            jagged_item_logit.detach(),
            labels.detach(),
            seqlen_after_preprocessor,  # used to compute achieved flops/s
        )
