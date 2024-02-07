import os
import hydra
import shutil
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental.callbacks import Callback


class SaveGridSearchConfigCallback(Callback):
    def on_multirun_start(self, config: DictConfig, **kwargs) -> None:
        self.results_dir = config.hydra.sweep.dir

        os.makedirs(self.results_dir, exist_ok=True)
        self.output_file_path = os.path.join(self.results_dir, "config.yaml")

        grid_params = config.hydra.sweeper.params.keys()
        self.grid_params = [k.replace("+", "") for k in grid_params]

        if os.path.exists(os.path.join(self.results_dir, "config.yaml")):
            return

        base_params = {k: v for k, v in config.items() if k != "hydra"}
        initial_config = OmegaConf.create(
            {**base_params, **{param: [] for param in self.grid_params}})
        with open(self.output_file_path, "w") as f:
            OmegaConf.save(config=initial_config, f=f)

    def on_job_start(self, config: DictConfig, **kwargs) -> None:
        if not hasattr(self, "output_file_path"):
            return

        with open(self.output_file_path, "r") as f:
            existing_config = OmegaConf.load(f)
        current_params = {k: v for k, v in config.items() if k != "hydra"}
        for param, value in current_params.items():
            if param in self.grid_params:
                if value not in existing_config[param]:
                    existing_config[param].append(value)
            else:
                existing_config[param] = value
        with open(self.output_file_path, "w") as f:
            OmegaConf.save(config=existing_config, f=f)


class CurriculumScoreConfigCallback(Callback):
    def on_multirun_start(self, config: DictConfig, **kwargs) -> None:
        pass


# ? Initialize hydra if not already initialized for all modules
def global_hydra_init():
    if not GlobalHydra.instance().is_initialized():
        hydra.initialize(version_base=None, config_path="conf")
