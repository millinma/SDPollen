r"""Loss registry with lightweight wrappers.

Wrappers needed to tackle casting to the correct type.
DataLoader automatically casts to the wrong type
see: https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717

"""
from torch.nn import (
    CrossEntropyLoss,
    MSELoss
)
from helpers import BaseRegistry


class _CrossEntropyLoss(CrossEntropyLoss):
    def forward(self, x, y):
        if y.ndim == 1:
            y = y.long()
        return super().forward(x, y)


class _MSELoss(MSELoss):
    def forward(self, x, y):
        # TODO: is there a more elegant way to handle broadcasting
        return super().forward(x.squeeze(-1), y.float())


CRITERION_REGISTRY = BaseRegistry({
    "CrossEntropyLoss": _CrossEntropyLoss,
    "MSELoss": _MSELoss,
}, "CRITERION_REGISTRY")
