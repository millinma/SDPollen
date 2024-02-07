from .augsburg import (
    transform_Base_Augsburg,
    transform_Cnn10_Cnn14_Augsburg
)
from helpers import BaseRegistry
from .utils import wildcard_transform

BASE_TRANSFORMS_REGISTRY = BaseRegistry({
    # general wildcard to transform to tensor
    "None": wildcard_transform,
    # Pollen Augsburg transforms
    "Cnn10_Pollen.Augsburg": transform_Cnn10_Cnn14_Augsburg,
    "Cnn14_Pollen.Augsburg": transform_Cnn10_Cnn14_Augsburg,
    "ResNet50_Pollen.Augsburg": transform_Base_Augsburg,
    "ModifiedEfficientNet_Pollen.Augsburg": transform_Base_Augsburg
}, "BASE_TRANSFORMS_REGISTRY", wildcard=True)
