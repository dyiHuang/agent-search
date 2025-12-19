import os
from argparse import ArgumentParser

import hydra
import ray
from deepspeed.constants import CROSS_RANK, CROSS_SIZE

from ppo_trainer import MegatronDeepSpeedPPOTrainer


def parse_args():
    parser = ArgumentParser(description="DeepSpeed distributed training launch"
                                        " utility that creates multiple distributed"
                                        " processes on a single node")
    parser.add_argument("--node_rank",
                        type=int,
                        default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")

    parser.add_argument("--nnodes",
                        type=int,
                        default=1,
                        help="The nums of the node for multi-node distributed "
                             "training")

    return parser.parse_args()

@hydra.main(config_path='config', config_name='ppo_megatron_trainer', version_base=None)
def main(config):
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    args = parse_args()

    os.environ[CROSS_RANK] = str(args.node_rank)
    os.environ[CROSS_SIZE] = str(args.nnodes)

    trainer = MegatronDeepSpeedPPOTrainer(config)
    trainer.train()


if __name__ == '__main__':
    rank = 0
    num_gpus =4
    # 主进程：初始化Ray，分配4张GPU（支持TP=4）
    ray.init(
        ignore_reinit_error=True,
        local_mode=False,  # 必须关闭local_mode，否则无法多卡TP
        # num_gpus=num_gpus,  # 为Ray集群分配num_gpus张GPU
        # _temp_dir="/tmp/ray-tp4",
        runtime_env={
            "env_vars": {
                # "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),  # 明确指定num_gpus张GPU给Actor
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",  # 明确指定num_gpus张GPU给Actor
                "TRUST_REMOTE_CODE": "True",
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                # 禁用Ray的NCCL干扰torchrun
                # "NCCL_ASYNC_ERROR_HANDLING": "0"
                # 再次清空Actor内的分布式环境变量
            }
        },
        # 关键：允许Ray跨进程共享Actor
        _node_ip_address="127.0.0.1"
    )
    print(f"[Rank {rank}] Ray initialized (master), allocated {num_gpus} GPUs for TP={num_gpus}")
    main()
