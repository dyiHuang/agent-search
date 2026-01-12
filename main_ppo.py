import os
from argparse import ArgumentParser

import hydra
import psutil
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


def set_child_process_affinity():
    """
    在 torchrun 子进程内设置 CPU 亲和性
    Args:
        bind_core_list: 可选，指定总核心列表（如 "0-15" 或 [0,1,2,3]）
    """
    # 1. 获取当前子进程的 local_rank 和本地总进程数
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # torchrun 自动设置 LOCAL_RANK
    num_local_procs = int(os.environ.get("WORLD_SIZE", 1))  # 本地总进程数

    # 2. 确定总核心列表
    total_physical_cores = psutil.cpu_count(logical=False)
    core_list = list(range(total_physical_cores))
    total_cores = len(core_list)

    # 3. 计算当前子进程应绑定的核心列表
    cores_per_rank = total_cores // num_local_procs
    assert cores_per_rank >= 1, "每个进程至少需分配 1 个核心"
    start_idx = cores_per_rank * local_rank
    end_idx = cores_per_rank * (local_rank + 1)
    core_list_for_rank = core_list[start_idx:end_idx]  # 当前子进程的核心列表

    # 4. 调用 os.sched_setaffinity 设置亲和性（pid=0 表示当前进程）
    os.sched_setaffinity(0, core_list_for_rank)
    os.environ['OMP_NUM_THREADS'] = f"{len(core_list_for_rank)}"
    print(f"子进程 local_rank={local_rank} 已绑定 CPU 核心: {core_list_for_rank}")


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

    set_child_process_affinity()

    trainer = MegatronDeepSpeedPPOTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
