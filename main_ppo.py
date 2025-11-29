
import hydra
from ppo_trainer import MegatronDeepSpeedPPOTrainer

@hydra.main(config_path='config', config_name='ppo_megatron_trainer', version_base=None)
def main(config):
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    trainer = MegatronDeepSpeedPPOTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
