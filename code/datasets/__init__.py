from .pollen_augsburg import Augsburg
from helpers import BaseRegistry

DATASET_REGISTRY = BaseRegistry({
    "Pollen.Augsburg": Augsburg
}, "DATASET_REGISTRY")
