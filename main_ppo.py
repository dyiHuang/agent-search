
import hydra
from ppo_trainer import MegatronDeepSpeedPPOTrainer
from utils import rotary_pos_emb_patch


@hydra.main(config_path='config', config_name='ppo_megatron_trainer', version_base=None)
def main(config):
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    rotary_pos_emb_patch.apply_patch()

    trainer = MegatronDeepSpeedPPOTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
