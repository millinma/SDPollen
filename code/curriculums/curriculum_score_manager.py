from omegaconf import DictConfig, OmegaConf
from typing import Tuple
import os
import shutil
import pandas as pd

from helpers import global_hydra_init
from postprocessing.postprocessing_utils import load_yaml, save_yaml
from .scoring import CURRICULUM_SCORE_REGISTRY
from .curriculum_plot_utils import CurriculumPlots


class CurriculumScoreManager:
    def __init__(self,
                 cfg: DictConfig,
                 output_directory: str,
                 ) -> None:
        global_hydra_init()
        self.output_directory = output_directory
        self.results_dir = cfg.results_dir
        self.experiment_id = cfg.experiment_id
        self.cfg = cfg
        self.scoring_function = CURRICULUM_SCORE_REGISTRY(
            **cfg.curriculum.scoring,
            output_directory=output_directory,
            results_dir=self.results_dir,
            experiment_id=self.experiment_id
        )

    def preprocess(self) -> Tuple[list, list]:
        configs, runs = self.scoring_function.preprocess()
        self._create_configs(runs, configs)
        self._create_mappings(runs)
        return configs, runs

    def run(self, run_config: DictConfig, run_name: str) -> None:
        scores = os.path.join(
            self.output_directory, run_name, "scores.csv")
        if os.path.exists(scores):
            return
        self.scoring_function.run(self.cfg.copy(), run_config, run_name)

    def postprocess(self, score_id: str, correlation: bool = True) -> None:
        mappings = load_yaml(
            os.path.join(self.output_directory, "mappings.yaml"))
        runs = mappings[score_id]
        self.scoring_function.postprocess(score_id, runs)
        self._visualize_score(score_id)
        if correlation:
            self._correlation_matrix()
            self._correlation_matrix()

    def _create_configs(self, runs: list, configs: list) -> None:
        for run_name, config in zip(runs, configs):
            s = os.path.join(
                self.output_directory, run_name, "score.yaml")
            if os.path.exists(s):
                continue

            os.makedirs(
                os.path.join(self.output_directory, run_name), exist_ok=True
            )
            save_yaml(
                os.path.join(self.output_directory,
                             run_name, "config.yaml"),
                OmegaConf.to_container(config, resolve=True),
            )
            save_yaml(
                os.path.join(self.output_directory, run_name, "score.yaml"),
                OmegaConf.to_container(self.cfg, resolve=True),
            )
        shutil.rmtree(os.path.join(self.output_directory, ".hydra"))

    def _create_mappings(self, runs: list) -> None:
        if not os.path.exists(os.path.join(self.output_directory, "mappings.yaml")):
            mappings = DictConfig({})
        else:
            mappings = DictConfig(load_yaml(
                os.path.join(self.output_directory, "mappings.yaml")))
        mappings[self.cfg.curriculum.scoring.id] = runs
        save_yaml(
            os.path.join(self.output_directory, "mappings.yaml"),
            OmegaConf.to_container(mappings, resolve=True),
        )

    def _visualize_score(self, score_id: str) -> None:
        path = os.path.join(
            self.output_directory, score_id+".csv")
        df = pd.read_csv(path)
        cp = CurriculumPlots(
            output_directory=self.output_directory,
            training_type="",
            **self.cfg.plotting
        )
        cp.plot_score(df, score_id)
        cp.plot_score_balanced(df, score_id)
        cp.plot_scatter_distribution(df, score_id)

    def _correlation_matrix(self) -> None:
        df = pd.DataFrame()
        base_path = os.path.dirname(self.output_directory)
        dirs = [d for d in os.listdir(
            base_path) if os.path.isdir(os.path.join(base_path, d))]
        for score_dir in dirs:
            csv_names = [f for f in os.listdir(os.path.join(
                base_path, score_dir)) if f.endswith(".csv")]
            for csv_name in csv_names:
                score_df = pd.read_csv(os.path.join(
                    base_path, score_dir, csv_name))
                name = csv_name.replace(".csv", "")
                df[name] = score_df["ranks"]
        cp = CurriculumPlots(
            output_directory=base_path,
            training_type="",
            **self.cfg.plotting
        )
        cp.plot_correlation_matrix(df)
        cp.plot_correlation_matrix_custom(df)
