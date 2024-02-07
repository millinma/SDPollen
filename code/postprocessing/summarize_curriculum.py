import os
import pandas as pd
from typing import List
from omegaconf import OmegaConf

from metrics import METRIC_REGISTRY, Metric
from curriculums.curriculum_plot_utils import CurriculumPlots
from .postprocessing_utils import (
    get_run_names,
    get_training_type,
    get_plotting_params
)


class SummarizeCurriculum:
    def __init__(
        self,
        results_dir: str,
        experiment_id: str,
        summary_dir: str = "summary",
        training_dir: str = "training",
        training_type: str = None,
        max_runs_plot: int = None
    ) -> None:
        self.results_dir = results_dir
        self.experiment_id = experiment_id
        self.output_directory = os.path.join(
            self.results_dir,
            self.experiment_id,
            summary_dir
        )
        self.training_directory = os.path.join(
            self.results_dir,
            self.experiment_id,
            training_dir
        )
        os.makedirs(self.output_directory, exist_ok=True)

        run_names = get_run_names(self.training_directory)
        self.run_names = self._filter_curriculum_runs(run_names)
        self.has_curriculums = len(self.run_names) > 0
        if not self.has_curriculums:
            return
        self.training_type = training_type
        if self.training_type is None:
            self.training_type = get_training_type(
                self.training_directory,
                self.run_names
            )
        self.max_runs_plot = max_runs_plot
        self.run_names.sort()
        self.tracking_metric = self._get_tracking_metric(self.run_names[0])

    def summarize(self) -> None:
        self._summarize_pace()

    def plot_pace(self) -> None:
        if not self.has_curriculums:
            return
        path = os.path.join(self.output_directory, "pace.csv")
        df = pd.read_csv(path, index_col="iteration")
        keys = [self._get_best_metric(n) for n in self.run_names]
        reverse = self.tracking_metric.suffix == "max"
        col_order = [
            col for _, col in
            sorted(zip(keys, df.columns), reverse=reverse)
        ]
        df = df[col_order]

        path = os.path.join(self.output_directory, "plots", "curriculum_plots")
        plot_params = get_plotting_params(
            self.training_directory,
            self.run_names[0]
        )
        cp = CurriculumPlots(
            output_directory=path,
            training_type=self.training_type,
            **plot_params
        )
        cp.plot_pace(df.iloc[:, :self.max_runs_plot])

    def _filter_curriculum_runs(self, run_names: List[str]) -> Metric:
        filtered_run_names = []
        for run_name in run_names:
            path = os.path.join(
                self.training_directory,
                run_name,
                "pace.csv"
            )
            if os.path.exists(path):
                filtered_run_names.append(run_name)
        return filtered_run_names

    def _get_tracking_metric(self, run_name: str) -> str:
        path = os.path.join(
            self.training_directory,
            run_name,
            ".hydra",
            "config.yaml"
        )
        cfg = OmegaConf.load(path)
        return METRIC_REGISTRY(**{
            "name": cfg.dataset.tracking_metric,
            "metric": cfg.dataset.tracking_metric
        })

    def _summarize_pace(self) -> None:
        if not self.has_curriculums:
            return
        records = []
        for n in self.run_names:
            path = os.path.join(
                self.training_directory,
                n,
                "pace.csv"
            )
            df = pd.read_csv(path, index_col="iteration")
            records.append(df.rename(columns={"dataset_size": n}))
        path = os.path.join(self.output_directory, "pace.csv")
        df = pd.concat(records, axis=1)
        df = df.sort_values(by="iteration")
        df.to_csv(path)

    def _get_best_metric(self, run_name) -> float:
        path = os.path.join(
            self.training_directory,
            run_name,
            "metrics.csv"
        )
        df = pd.read_csv(path)
        return self.tracking_metric.get_best(df[self.tracking_metric.name])
