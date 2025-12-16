import os
from argparse import ArgumentParser

import hydra
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
    main()
