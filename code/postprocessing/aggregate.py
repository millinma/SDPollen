from .summarize import SummarizeGrid
import os
from collections import defaultdict
import pandas as pd
from helpers import Timer, PlotMetrics, MLFlowLogger
import shutil
from .postprocessing_utils import (
    get_run_names,
    load_yaml,
    save_yaml,
    get_training_type,
    get_plotting_params,
    get_naming_convention
)
from omegaconf import OmegaConf
from metrics import METRIC_REGISTRY


INVALID_AGGREGATIONS = {"dataset", "training"}


class AggregateGrid:
    def __init__(
        self,
        results_dir: str,
        experiment_id: str,
        aggregate_list: list,
        aggregate_prefix: str = "agg",
        training_dir: str = "training",
        max_runs_plot: int = None,
    ) -> None:
        self.aggregate_list = aggregate_list
        self.results_dir = results_dir
        self.experiment_id = experiment_id
        self.aggregate_name = "_".join(
            (aggregate_prefix, *self.aggregate_list))
        self.output_directory = os.path.join(
            self.results_dir,
            self.experiment_id,
            self.aggregate_name
        )
        self.training_directory = os.path.join(
            self.results_dir,
            self.experiment_id,
            training_dir
        )
        self.max_runs_plot = max_runs_plot
        if os.path.exists(self.output_directory):
            shutil.rmtree(self.output_directory)
        os.makedirs(self.output_directory, exist_ok=True)
        self.run_names = get_run_names(self.training_directory)
        self.training_type = get_training_type(
            self.training_directory, self.run_names)
        self.run_names.sort()

    def aggregate(self) -> None:
        aggregated_runs = self._aggregate_run_names(self.aggregate_list)
        for agg_name, run_list in aggregated_runs.items():
            self._aggregate_best(agg_name, run_list)
            self._aggregate_test(agg_name, run_list)
            self._aggregate_config(agg_name, run_list)
            self._aggregate_timer(agg_name, run_list)
            self._aggregate_metrics(agg_name, run_list)
            save_yaml(os.path.join(self.output_directory,
                                   agg_name, "runs.yaml"), run_list)

    def summarize(self) -> None:
        sg = SummarizeGrid(
            results_dir=self.results_dir,
            experiment_id=self.experiment_id,
            training_dir=self.aggregate_name,
            summary_dir=self.aggregate_name,
            clear_old_outputs=False,
            training_type=self.training_type,
            max_runs_plot=self.max_runs_plot
        )
        sg.summarize()
        sg.plot_metrics()

    def _check_if_valid_aggregation(self, over: list) -> None:
        over = set(over)
        if over & INVALID_AGGREGATIONS:
            raise ValueError(
                f"Can't aggregate over {over & INVALID_AGGREGATIONS}")
        naming_convention = set(get_naming_convention())
        if over - naming_convention:
            raise ValueError(
                f"Can't aggregate over {over - naming_convention}")

    def _aggregate_run_names(self, over: list) -> dict:
        self._check_if_valid_aggregation(over)
        parameters = get_naming_convention()
        param_dict = {p: i for i, p in enumerate(parameters)}
        over_idxs = [param_dict[p] for p in over]
        aggregated = defaultdict(list)
        for run_name in self.run_names:
            params = run_name.split("_")
            for idx in over_idxs:
                params[idx] = "#"
            agg_key = "_".join(params)
            aggregated[agg_key].append(run_name)
        return aggregated

    def _aggregate_best(self, agg_name: str, run_list: list):
        os.makedirs(os.path.join(self.output_directory,
                    agg_name, "_best"), exist_ok=True)
        metrics = self._aggregate_yaml(run_list, "_best/dev.yaml", "dev")
        save_yaml(os.path.join(
            self.output_directory, agg_name, "_best", "dev.yaml"), metrics)

    def _aggregate_test(self, agg_name: str, run_list: list):
        path = os.path.join(self.output_directory, agg_name, "_test")
        os.makedirs(path, exist_ok=True)
        metrics = self._aggregate_yaml(
            run_list, "_test/test_holistic.yaml", "test")
        save_yaml(os.path.join(path, "test_holistic.yaml"), metrics)

    def _aggregate_yaml(self, run_list: list, path: str, yaml_type: str):
        assert yaml_type in ["dev", "test"]
        loss_type = "dev_loss" if yaml_type == "dev" else "loss"
        dfs = []
        for run in run_list:
            metrics = load_yaml(os.path.join(
                self.training_directory, run, path))
            dfs.append(pd.DataFrame(metrics))
        df = pd.concat(dfs, keys=run_list, names=["run", "type"])
        df_mean = df.groupby(level="type").mean().reset_index()
        df_std = df.groupby(level="type").std().fillna(0)\
            .reset_index()
        df_std["type"] = df_std["type"].apply(lambda x: f"{x}.std")
        df = pd.concat([df_mean, df_std]).set_index(["type"])
        metrics = df.to_dict()
        metrics[loss_type] = {
            k: v for k, v in metrics[loss_type].items() if "all" in k}
        if yaml_type == "dev":
            metrics["iteration"] = {
                k: v for k, v in metrics["iteration"].items() if "all" in k}
        return metrics

    def _aggregate_config(self, agg_name: str, run_list: list):
        path = os.path.join(self.output_directory, agg_name, ".hydra")
        os.makedirs(path, exist_ok=True)
        run = run_list[0]
        config = OmegaConf.load(os.path.join(
            self.training_directory, run, ".hydra", "config.yaml"))
        for a in self.aggregate_list:
            config[a] = "#"
        save_yaml(
            os.path.join(path, "config.yaml"),
            OmegaConf.to_container(config)
        )

    def _aggregate_timer(self, agg_name: str, run_list: list):
        mean_timer = {
            "train": {"mean_seconds": 0, "total_seconds": 0},
            "dev": {"mean_seconds": 0, "total_seconds": 0},
            "test": {"mean_seconds": 0, "total_seconds": 0},
        }
        for run in run_list:
            timers = load_yaml(os.path.join(
                self.training_directory, run, "timer.yaml"))
            for k, v in timers.items():
                mean_timer[k]["mean_seconds"] += v["mean_seconds"]
                mean_timer[k]["total_seconds"] += v["total_seconds"]
        for k, v in mean_timer.items():
            v["mean_seconds"] /= len(run_list)
            v["mean"] = Timer.pretty_time(v["mean_seconds"])
            v["total_seconds"] /= len(run_list)
            v["total"] = Timer.pretty_time(v["total_seconds"])
        save_yaml(
            os.path.join(self.output_directory, agg_name, "timer.yaml"),
            mean_timer
        )

    def _aggregate_metrics(self, agg_name: str, run_list: list):
        dfs = []
        for run in run_list:
            df = pd.read_csv(
                os.path.join(self.training_directory, run, "metrics.csv"),
                index_col="iteration"
            )
            dfs.append(df)
        cfg = OmegaConf.load(os.path.join(
            self.output_directory, agg_name, ".hydra", "config.yaml"))
        metrics = []
        for m in cfg.dataset.metrics:
            metrics.append(METRIC_REGISTRY(**{"name": m, "metric": m}))
        tracking_metric = METRIC_REGISTRY(**{
            "name": cfg.dataset.tracking_metric,
            "metric": cfg.dataset.tracking_metric
        })

        df = pd.concat(dfs, keys=run_list, names=["run", "iteration"])
        mean_df = df.groupby(level="iteration").mean()
        std_df = df.groupby(level="iteration").std().fillna(0)
        df = mean_df.join(std_df, rsuffix=".std")
        df["iteration"] = df.index
        df.to_csv(os.path.join(self.output_directory,
                  agg_name, "metrics.csv"), index=False)
        df.drop(columns=["iteration"], inplace=True)

        plot_params = get_plotting_params(
            self.training_directory,
            self.run_names[0]
        )
        plotter = PlotMetrics(
            os.path.join(self.output_directory, agg_name),
            self.training_type,
            **plot_params
        )
        plotter.plot_run(df)

        mlflow_logger = MLFlowLogger(
            output_directory=self.output_directory,
            exp_name=self.experiment_id+"."+self.aggregate_name,
            run_name=agg_name,
            metrics=metrics,
            tracking_metric=tracking_metric,
        )
        mlflow_logger.log_params(cfg)
        timers = load_yaml(os.path.join(
            self.output_directory, agg_name, "timer.yaml"))
        mlflow_logger.log_timers({
            "time.train.mean": timers["train"]["mean"],
            "time.dev.mean": timers["dev"]["mean"],
            "time.test.mean": timers["test"]["mean"],
        })
        for iteration in df.index:
            metrics = df.loc[iteration].to_dict()
            metrics = {k: v for k, v in metrics.items()
                       if not k.endswith(".std")}
            mlflow_logger.log_metrics(metrics, iteration)
        test_metrics = load_yaml(os.path.join(
            self.output_directory, agg_name, "_test", "test_holistic.yaml"))
        test_metrics = {"test_"+k: v["all"] for k, v in test_metrics.items()}
        mlflow_logger.log_test_metrics(test_metrics)
        mlflow_logger.log_artifact(os.path.join(
            agg_name, ".hydra", "config.yaml"))
        mlflow_logger.log_artifact(os.path.join(agg_name, "metrics.csv"))
        mlflow_logger.end_run()
