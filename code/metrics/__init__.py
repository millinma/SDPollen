from .base import Ascending, Descending, Metric
from helpers import BaseRegistry

METRIC_REGISTRY = BaseRegistry({
    "accuracy": Ascending,
    "uar": Ascending,
    "f1": Ascending,
    "mse": Descending,
    "mae": Descending,
    "pcc": Ascending,
    "ccc": Ascending
}, "METRIC_REGISTRY")
