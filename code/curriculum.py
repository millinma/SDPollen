import hydra
from omegaconf import DictConfig, OmegaConf
import os
from curriculums import CurriculumScoreManager


@hydra.main(version_base=None, config_path="../conf", config_name="curriculum")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    output_directory = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    cs = CurriculumScoreManager(cfg, output_directory)
    configs, runs = cs.preprocess()

    for config, run in zip(configs, runs):
        cs.run(config, run)

    cs.postprocess(cfg.curriculum.scoring.id)


if __name__ == "__main__":
    main()
