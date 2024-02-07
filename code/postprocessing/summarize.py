import os
import shutil
import pandas as pd
from typing import List
from omegaconf import OmegaConf

from helpers import PlotMetrics
from helpers import get_params_to_export
from metrics import METRIC_REGISTRY, Metric
from .postprocessing_utils import (
    load_yaml,
    save_yaml,
    get_run_names,
    get_training_type,
    get_plotting_params,
    get_naming_convention
)


class SummarizeGrid:
    def __init__(
        self,
        results_dir: str,
        experiment_id: str,
        summary_dir: str = "summary",
        training_dir: str = "training",
        clear_old_outputs: bool = True,
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
        if os.path.exists(self.output_directory) and clear_old_outputs:
            shutil.rmtree(self.output_directory)
        os.makedirs(self.output_directory, exist_ok=True)

        self.run_names = get_run_names(self.training_directory)
        self.training_type = training_type
        if self.training_type is None:
            self.training_type = get_training_type(
                self.training_directory,
                self.run_names
            )
        self.max_runs_plot = max_runs_plot
        self.run_names.sort()

    def summarize(self) -> None:
        self._summarize_metrics()
        self._summarize_config()
        self._summarize_best_runs()

    def _summarize_metrics(self) -> None:
        records = []
        for n in self.run_names:
            metrics = self._read_metrics(n)
            test = self._read_test_metrics(n)
            records.append({"run_name": n, **metrics, **test})
        df = pd.DataFrame.from_records(records)
        path = os.path.join(self.output_directory, "metrics.csv")
        df.to_csv(path, index=False)

    def _summarize_config(self) -> None:
        records = []
        for n in self.run_names:
            run_dir = os.path.join(self.training_directory, n)
            config_path = os.path.join(run_dir, ".hydra", "config.yaml")
            config = load_yaml(config_path)
            config = dict(sorted(get_params_to_export(config).items()))
            time = self._read_times(n)
            records.append({"run_name": n, **config, **time})
        df = pd.DataFrame.from_records(records)
        path = os.path.join(self.output_directory, "config.csv")
        df.to_csv(path, index=False)

    def _summarize_best_runs(self) -> None:
        summary_path = os.path.join(self.output_directory, "metrics.csv")
        df = pd.read_csv(summary_path)
        metric_cols = [m for m in df.columns if not m.endswith(".std")]
        metric_cols.remove("run_name")
        metric_cols.remove("best_iteration")
        best_runs = {}
        for metric in metric_cols:
            if "loss" in metric:
                idx = df[metric].idxmin()
            else:
                base_metric = metric.replace("test_", "").replace("train_", "")
                m = METRIC_REGISTRY(
                    **{"name": base_metric, "metric": base_metric})
                idx = m.get_best_pos(df[metric])
            best_runs[metric] = df.loc[idx, "run_name"]
        path = os.path.join(self.output_directory, "best_runs.yaml")
        save_yaml(path, best_runs)

    def _find_metrics_to_plot(self) -> List[str]:
        summary_path = os.path.join(self.output_directory, "metrics.csv")
        df = pd.read_csv(summary_path)
        plot_metrics = [
            m for m in list(df.columns)
            if m in METRIC_REGISTRY.registry_dict.keys()
            or m in ["train_loss", "dev_loss"]
        ]
        return plot_metrics

    def plot_metrics(self) -> None:
        plot_params = get_plotting_params(
            self.training_directory,
            self.run_names[0]
        )
        plotter = PlotMetrics(
            self.output_directory,
            self.training_type,
            **plot_params
        )
        metrics_to_plot = self._find_metrics_to_plot()
        for metric in metrics_to_plot:
            metrics_list = []
            metrics_std_list = []
            for n in self.run_names:
                metrics_path = os.path.join(
                    self.training_directory, n, "metrics.csv")
                df = pd.read_csv(metrics_path, index_col="iteration")
                if metric not in df.columns:
                    continue
                metrics_list.append(df[[metric]].rename(columns={metric: n}))

                std_metric = f"{metric}.std"
                if std_metric in df.columns:
                    metrics_std_list.append(
                        df[[std_metric]].rename(columns={std_metric: n}))

            metrics_df = pd.concat(metrics_list, axis=1)
            metrics_std_df = pd.concat(
                metrics_std_list, axis=1) if metrics_std_list else None
            plotter.plot_metric(
                metrics_df,
                metric,
                metrics_std_df,
                max_runs=self.max_runs_plot
            )

    def plot_aggregated_bars(self) -> None:
        metrics_to_plot = self._find_metrics_to_plot()
        plot_params = get_plotting_params(
            self.training_directory,
            self.run_names[0]
        )
        plotter = PlotMetrics(
            self.output_directory,
            self.training_type,
            **plot_params
        )
        metrics_df = pd.DataFrame()

        # Iterate over each CSV file and build up metrics dataframe
        for n in self.run_names:
            metrics_path = os.path.join(
                self.training_directory, n, "metrics.csv")
            df = pd.read_csv(metrics_path)
            last_row = df.iloc[-1:]
            params = get_naming_convention()
            run_details = n.split("_")
            for i, param in enumerate(params):
                last_row.insert(i, param, run_details[i])
            metrics_df = pd.concat((metrics_df, last_row), ignore_index=True)
        for metric in metrics_to_plot:
            plotter.plot_aggregated_bars(metrics_df, metric)

    def _read_metrics(self, run_name: str) -> dict:
        metrics_path = os.path.join(
            self.training_directory,
            run_name,
            "metrics.csv"
        )
        config_path = os.path.join(
            self.training_directory,
            run_name,
            ".hydra",
            "config.yaml"
        )
        df = pd.read_csv(metrics_path, index_col="iteration")
        cfg = OmegaConf.load(config_path)
        tracking_metric = METRIC_REGISTRY(**{
            "name": cfg.dataset.tracking_metric,
            "metric": cfg.dataset.tracking_metric
        })
        best_iteration = tracking_metric.get_best_pos(
            df[tracking_metric.name])
        metrics_dict = df.loc[best_iteration].to_dict()
        metrics_dict["best_iteration"] = best_iteration
        return metrics_dict

    def _read_test_metrics(self, run_name: str, prefix: str = "test_") -> dict:
        path = os.path.join(
            self.training_directory,
            run_name,
            "_test",
            "test_holistic.yaml"
        )
        test_dict = load_yaml(path)
        test_dict = {prefix+k: v["all"] for k, v in test_dict.items()}
        return test_dict

    def _read_times(self, run_name: str) -> dict:
        path = os.path.join(
            self.training_directory,
            run_name,
            "timer.yaml"
        )
        time_dict = load_yaml(path)
        return {
            "time_train_mean": time_dict["train"]["mean"],
            "time_dev_mean": time_dict["dev"]["mean"],
            "time_test_mean": time_dict["test"]["mean"],
        }
