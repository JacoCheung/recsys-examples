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
import os
import warnings

# Ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
import argparse
from typing import List, Union

import commons.utils.initialize as init
import gin
import torch  # pylint: disable-unused-import
from commons.distributed.batch_shuffler_factory import BatchShufflerFactory
from commons.distributed.sharding import make_optimizer_and_shard
from commons.pipeline import TrainPipelineFactory
from commons.utils.logger import print_rank_0
from configs import RankingConfig
from megatron.core import parallel_state, tensor_parallel
from model import get_ranking_model
from modules.metrics import get_multi_event_metric_module
from trainer.training import maybe_load_ckpts, train_with_pipeline
from trainer.utils import (
    create_dynamic_optitons_dict,
    create_embedding_configs,
    create_hstu_config,
    create_optimizer_params,
    get_data_loader,
    get_dataset_and_embedding_args,
    get_embedding_vector_storage_multiplier,
)
from utils import (  # from hstu.utils
    BenchmarkDatasetArgs,
    DatasetArgs,
    EmbeddingArgs,
    NetworkArgs,
    OptimizerArgs,
    RankingArgs,
    TensorModelParallelArgs,
    TrainerArgs,
)


def create_ranking_config(
    dataset_args: Union[DatasetArgs, BenchmarkDatasetArgs],
    network_args: NetworkArgs,
    embedding_args: List[EmbeddingArgs],
) -> RankingConfig:
    ranking_args = RankingArgs()

    return RankingConfig(
        embedding_configs=create_embedding_configs(
            dataset_args, network_args, embedding_args
        ),
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )


def main():
    parser = argparse.ArgumentParser(
        description="HSTU Example Arguments", allow_abbrev=False
    )
    parser.add_argument("--gin-config-file", type=str)
    args = parser.parse_args()
    gin.parse_config_file(args.gin_config_file)
    trainer_args = TrainerArgs()
    dataset_args, embedding_args = get_dataset_and_embedding_args(
        trainer_args.pipeline_type == "prefetch"
    )
    network_args = NetworkArgs()
    optimizer_args = OptimizerArgs()
    tp_args = TensorModelParallelArgs()

    init.initialize_distributed()
    init.initialize_model_parallel(
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size
    )
    init.set_random_seed(trainer_args.seed)
    free_memory, total_memory = torch.cuda.mem_get_info()
    print_rank_0(
        f"distributed env initialization done. Free cuda memory: {free_memory / (1024 ** 2):.2f} MB"
    )
    hstu_config = create_hstu_config(network_args, tp_args)
    task_config = create_ranking_config(dataset_args, network_args, embedding_args)
    # We need to create the dataloader before model initialization in case the dataset is random.
    # In our scenario, the dataset across tp rank should be different. We need to fork the state
    with tensor_parallel.get_cuda_rng_tracker().fork():
        train_dataloader, test_dataloader = get_data_loader(
            "ranking", dataset_args, trainer_args, task_config.num_tasks
        )
    model = get_ranking_model(hstu_config=hstu_config, task_config=task_config)

    dynamic_options_dict = create_dynamic_optitons_dict(
        embedding_args,
        network_args.hidden_size,
        training=True,
        embedding_dim_multiplier=get_embedding_vector_storage_multiplier(
            optimizer_args.optimizer_str
        ),
    )

    optimizer_param = create_optimizer_params(optimizer_args)
    device = torch.device("cuda", torch.cuda.current_device())

    # Pipeline backend: "legacy" (default) uses TrainPipelineFactory's
    # JaggedMegatron* classes; "new" uses HSTUPipeline (schedulable
    # engine). Switched via env var RECSYS_PIPELINE_BACKEND so the gin
    # config + CLI surface stay unchanged.
    pipeline_backend = os.environ.get("RECSYS_PIPELINE_BACKEND", "legacy").lower()
    if pipeline_backend not in ("legacy", "new"):
        raise ValueError(
            f"RECSYS_PIPELINE_BACKEND must be 'legacy' or 'new', "
            f"got {pipeline_backend!r}"
        )

    if pipeline_backend == "new" and trainer_args.pipeline_type == "none":
        # The new HSTU adapter only wraps the sparse-dist variants.
        # Fall back to legacy for the non-pipelined "none" type instead
        # of silently changing semantics.
        print_rank_0(
            "RECSYS_PIPELINE_BACKEND=new but pipeline_type=none; "
            "falling back to legacy non-pipelined path."
        )
        pipeline_backend = "legacy"

    critical_stream = torch.cuda.Stream(device) if pipeline_backend == "new" else None
    model_train, dense_optimizer = make_optimizer_and_shard(
        model,
        config=hstu_config,
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
        dynamicemb_options_dict=dynamic_options_dict,
        pipeline_type=trainer_args.pipeline_type,
        device=device,
        ddp_init_stream=critical_stream,
    )

    stateful_metric_module = get_multi_event_metric_module(
        num_classes=task_config.prediction_head_arch[-1],
        num_tasks=task_config.num_tasks,
        metric_types=task_config.eval_metrics,
        comm_pg=parallel_state.get_data_parallel_group(
            with_context_parallel=True
        ),  # ranks in the same TP group do the same compute
    )

    # Create batch shuffler based on configuration
    # For ranking, we have action interleaved with item
    if trainer_args.enable_balanced_shuffler:
        batch_shuffler = BatchShufflerFactory.create(
            "hstu",
            num_heads=hstu_config.num_attention_heads,
            head_dim=hstu_config.kv_channels,
            action_interleaved=True,
        )
    else:
        batch_shuffler = BatchShufflerFactory.create("identity")

    free_memory, total_memory = torch.cuda.mem_get_info()
    print_rank_0(
        f"model initialization done, start training. Free cuda memory: {free_memory / (1024 ** 2):.2f} MB"
    )

    from commons.utils.dynamicemb_cache_stats import auto_install

    auto_install(model_train)

    maybe_load_ckpts(trainer_args.ckpt_load_dir, model, dense_optimizer)

    if os.environ.get("FILL_DYNAMICEMB_TABLES", "0") == "1":
        from dynamicemb.dump_load import get_dynamic_emb_module

        for dyn_module in get_dynamic_emb_module(model_train):
            if hasattr(dyn_module, "fill_tables"):
                try:
                    dyn_module.fill_tables(load_factor=0.95)
                    print_rank_0(f"fill_tables done for {dyn_module.table_names}")
                except TypeError:
                    pass
        torch.cuda.synchronize()
        torch.distributed.barrier()

    if pipeline_backend == "legacy":
        pipeline_type_map = {
            "prefetch": "jagged_prefetch_sparse_dist",
            "native": "jagged_sparse_dist",
            "none": "jagged_none",
        }
        pipeline_name = pipeline_type_map.get(trainer_args.pipeline_type, "jagged_none")
        pipeline = TrainPipelineFactory.create(
            pipeline_name,
            model=model_train,
            optimizer=dense_optimizer,
            device=device,
            batch_shuffler=batch_shuffler,
        )
    else:
        from commons.pipeline.hstu_pipeline import HSTUPipelineFactory

        pipeline_type_map = {
            "prefetch": "hstu_prefetch_sparse_dist",
            "native": "hstu_sparse_dist",
        }
        pipeline_name = pipeline_type_map[trainer_args.pipeline_type]
        pipeline = HSTUPipelineFactory.create(
            pipeline_name,
            model=model_train,
            optimizer=dense_optimizer,
            device=device,
            batch_shuffler=batch_shuffler,
            critical_stream=critical_stream,
        )
    print_rank_0(
        f"[pipeline] backend={pipeline_backend} "
        f"name={pipeline_name} type={trainer_args.pipeline_type}"
    )

    train_with_pipeline(
        pipeline,
        stateful_metric_module,
        trainer_args,
        train_dataloader,
        test_dataloader,
        dense_optimizer,
    )
    init.destroy_global_state()


if __name__ == "__main__":
    main()
