import audmetric
import numpy as np
import warnings
import functools


METRIC_DICT = {
    "accuracy": {"fn": audmetric.accuracy, "fallback": 0.0},
    "uar": {"fn": audmetric.unweighted_average_recall, "fallback": 0.0},
    "f1": {"fn": audmetric.unweighted_average_fscore, "fallback": 0.0},
    "mse": {"fn": audmetric.mean_squared_error, "fallback": 1e32},
    "mae": {"fn": audmetric.mean_absolute_error, "fallback": 1e32},
    "pcc": {"fn": audmetric.pearson_cc, "fallback": 0.0},
    "ccc": {"fn": audmetric.concordance_cc, "fallback": 0.0}
}


def ignore_runtime_warning(func):
    @functools.wraps(func)
    def wrapper_ignore_warning(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return func(*args, **kwargs)
    return wrapper_ignore_warning


class Metric:
    def __init__(self, metric):
        self._metric = METRIC_DICT[metric]["fn"]
        self.name = metric
        self.fallback = METRIC_DICT[metric]["fallback"]

    @ignore_runtime_warning
    def __call__(self, *args, **kwargs):
        score = self._metric(*args, **kwargs)
        if np.isnan(score):
            return self.fallback
        return score


class Ascending(Metric):
    @property
    def starting_metric(self):
        return -1e32

    @property
    def suffix(self):
        return "max"

    @staticmethod
    def get_best(a):
        return a.max()

    @staticmethod
    def get_best_pos(a):
        return int(a.idxmax())

    @staticmethod
    def compare(a, b):
        return a > b


class Descending(Metric):
    @property
    def starting_metric(self):
        return 1e32

    @property
    def suffix(self):
        return "min"

    @staticmethod
    def get_best(a):
        return a.min()

    @staticmethod
    def get_best_pos(a):
        return int(a.idxmin())

    @staticmethod
    def compare(a, b):
        return a < b
