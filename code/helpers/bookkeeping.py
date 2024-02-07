import logging
import os
import yaml
from torchinfo import summary
import numpy as np
import sys
import torch
import warnings
from contextlib import contextmanager
from functools import wraps
from metrics import METRIC_REGISTRY


class SuppressStdout:
    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


def _suppress_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper


class Bookkeeping:
    def __init__(self, output_directory, file_handler_path=None) -> None:
        self.output_directory = output_directory
        self.original_stdout = sys.stdout
        # ? Setup Custom Logging
        self.logger = logging.getLogger()
        if not self.logger.hasHandlers():
            self._setup_logger(file_handler_path)

        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(logging.Formatter(
                    "[%(asctime)s][%(levelname)s]\n%(message)s\n"))

    def _setup_logger(self, fp):
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        if fp is not None:
            self.logger.addHandler(logging.FileHandler(fp))
        else:
            self.logger.addHandler(logging.FileHandler(
                os.path.join(self.output_directory, "bookkeeping.log")))

    def log(self, message, level=logging.INFO):
        self.logger.log(level, message)

    def log_to_file(self, message, level=logging.INFO):
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.emit(logging.LogRecord(self.logger.name,
                             level, None, None, message, None, None))

    def create_folder(self, folder_name, path=""):
        os.makedirs(os.path.join(self.output_directory,
                    path, folder_name), exist_ok=True)

    def save_model_summary(self, model, dataset, filename):
        x = np.expand_dims(dataset[0][0], axis=0).shape
        with open(os.path.join(self.output_directory, filename), "w", encoding="utf-8") as f:
            sys.stdout = f
            summary(
                model=model,
                input_size=(x),
                col_names=["input_size", "output_size",
                           "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
            )
            sys.stdout = self.original_stdout

    def save_state(self, obj, filename, path=""):
        p = os.path.join(self.output_directory, path, filename)
        _i = (torch.nn.Module, torch.optim.Optimizer,
              torch.optim.lr_scheduler.LRScheduler)
        if not isinstance(obj, _i):
            raise TypeError(
                f"save_state of type {type(obj)} is not supported.")

        if isinstance(obj, torch.nn.Module):
            obj = obj.cpu()
        torch.save(obj.state_dict(), p)

    def load_state(self, obj, filename, path=""):
        p = os.path.join(self.output_directory, path, filename)
        _i = (torch.nn.Module, torch.optim.Optimizer,
              torch.optim.lr_scheduler.LRScheduler)
        if not isinstance(obj, _i):
            raise TypeError(
                f"load_state of type {type(obj)} is not supported.")

        obj.load_state_dict(torch.load(p))

    @_suppress_warnings
    def save_target_transform(self, target_transform, filename, path=""):
        target_transform.to_yaml(os.path.join(
            self.output_directory, path, filename))

    def save_results_dict(self, results_dict, filename, path=""):
        with open(
            os.path.join(self.output_directory, path, filename),
            "w",
            encoding="utf-8"
        ) as f:
            yaml.dump(results_dict, f)

    def save_results_df(self, results_df, filename, path=""):
        results_df.to_csv(os.path.join(
            self.output_directory, path, filename), index=False)

    def save_results_np(self, results_np, filename, path=""):
        np.save(os.path.join(self.output_directory, path, filename), results_np)

    def save_best_results(self, metrics, filename, tracking_metric="accuracy", path=""):
        best_metrics = {}
        for m in metrics:
            if "loss" in m:
                best_metrics[f"{m}_min"] = metrics[m].min()
            elif m != "iteration":  # TODO: also hacky
                metric = METRIC_REGISTRY(**{"name": m, "metric": m})
                best_metrics[f"{m}_{metric.suffix}"] = metric.get_best(
                    metrics[m])
                if m == tracking_metric:
                    best_metrics["best_iteration"] = metric.get_best_pos(
                        metrics[m])
        with open(
            os.path.join(self.output_directory, path, filename),
            "w",
            encoding="utf-8"
        ) as f:
            yaml.dump(best_metrics, f)
