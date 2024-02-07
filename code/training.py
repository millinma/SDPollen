import hydra
from omegaconf import DictConfig, OmegaConf
import os
from training import ModularTaskTrainer
from helpers import RunFilter
from pathlib import Path
import yaml


@hydra.main(version_base=None, config_path="../conf", config_name="grid_search")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    working_dir = os.getcwd()
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # ? Skip Run if it already exists or if it is not a valid run
    if RunFilter(cfg, working_dir, output_dir).skip():
        return

    # ? Save Config to output directory
    cfg_path = os.path.join(output_dir, ".hydra", "config.yaml")
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    OmegaConf.save(cfg, cfg_path)

    # ? Only launch training if slurm is not used
    trainer = ModularTaskTrainer(
        cfg=cfg,
        output_directory=output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
