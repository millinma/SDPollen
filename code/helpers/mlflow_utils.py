import mlflow
from mlflow import MlflowClient
import os
from omegaconf import DictConfig
from typing import Union, List


EXPORT_IGNORE_PARAMS = [
    "results_dir",
    "experiment_id",
    "model.dataset",
    "model.base_transform",
    "training.type",
    "training.save_frequency",
    "dataset.metrics",
    "plotting"
]
EXPORT_LOGGING_DEPTH = 2


def get_params_to_export(params: Union[dict, DictConfig], prefix: str = "") -> dict:
    result = {}
    for k, v in params.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if (
            full_key in EXPORT_IGNORE_PARAMS
            or prefix in EXPORT_IGNORE_PARAMS
            or len(full_key.split(".")) > EXPORT_LOGGING_DEPTH
            or prefix.startswith("_")
            or k.startswith("_")
        ):
            continue
        if isinstance(v, (dict, DictConfig)):
            if v.get("id") != "None":
                result[full_key] = v.pop("id")
                result.update(get_params_to_export(v, prefix=full_key))
        else:
            result[full_key] = v
    return result


class MLFlowLogger:
    def __init__(
        self,
        output_directory: str,
        exp_name: str,
        run_name: str,
        metrics: List[callable],
        tracking_metric: callable
    ):
        self.output_directory = output_directory
        self.exp_name = exp_name
        self.run_name = run_name
        self.exp_id = self._get_or_create_experiment()
        mlflow.set_experiment(experiment_id=self.exp_id)
        self.run = self._get_or_create_run()
        self.best_metrics = {
            "train_loss.min": float("inf"),
            "dev_loss.min": float("inf"),
            "best_iteration": 0,
        }
        self.metrics_dict = {}
        for metric in metrics:
            self.best_metrics[f"{metric.name}.{metric.suffix}"] = metric.starting_metric
            self.metrics_dict[metric.name] = metric
        self.tracking_metric = tracking_metric

    def _get_or_create_experiment(self):
        try:
            exp_id = mlflow.create_experiment(name=self.exp_name)
        except:
            exp_id = mlflow.get_experiment_by_name(self.exp_name).experiment_id
        return exp_id

    def _get_or_create_run(self):
        client = MlflowClient()
        runs = mlflow.search_runs(
            experiment_ids=[self.exp_id],
            filter_string=f"tags.mlflow.runName='{self.run_name}'"
        )
        if runs.shape[0] > 0:
            run_id = runs.iloc[0]["run_id"]
            client.delete_run(run_id)
        run = mlflow.start_run(run_name=self.run_name)
        return run

    def log_params(self, params: Union[DictConfig, dict]):
        params = get_params_to_export(params)
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, iteration: int):
        self._update_best_metrics(metrics, iteration)
        self._log_metrics(self.best_metrics)
        self._log_metrics(metrics, iteration)

    def log_test_metrics(self, metrics: dict):
        self._log_metrics(metrics)

    def _update_best_metrics(self, metrics: dict, iteration: int):
        if self.tracking_metric.compare(
            metrics[self.tracking_metric.name],
            self.best_metrics[f"{self.tracking_metric.name}.{self.tracking_metric.suffix}"]
        ):
            self.best_metrics["best_iteration"] = iteration
        for k, v in metrics.items():
            if ".std" in k:
                continue
            if "loss" in k:
                if v < self.best_metrics[k+".min"]:
                    self.best_metrics[k+".min"] = v
            else:
                metric = self.metrics_dict[k]
                if metric.compare(v, self.best_metrics[f"{k}.{metric.suffix}"]):
                    self.best_metrics[f"{k}.{metric.suffix}"] = v

    def _log_metrics(self, metrics: dict, iteration=None):
        mlflow.log_metrics(metrics, step=iteration)

    def log_timers(self, timers: dict):
        mlflow.log_params(timers)

    def log_artifact(self, filename: str, path: str = ""):
        mlflow.log_artifact(os.path.join(
            self.output_directory, path, filename))

    def end_run(self):
        mlflow.end_run()
