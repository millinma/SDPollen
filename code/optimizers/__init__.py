from torch.optim import Adam, SGD
from .sam import SAM
from helpers import BaseRegistry


OPTIMIZER_REGISTRY = BaseRegistry({
    "Adam": Adam,
    "SGD": SGD,
    "SAM": SAM,
}, "OPTIMIZER_REGISTRY")
