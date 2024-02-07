from torch.optim.lr_scheduler import StepLR
from helpers import BaseRegistry

SCHEDULER_REGISTRY = BaseRegistry({
    "StepLR": StepLR,
}, "SCHEDULER_REGISTRY", required=False)
