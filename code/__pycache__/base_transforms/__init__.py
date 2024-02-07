from .cifar_10 import transform_Cnn10_Cnn14_CIFAR10, transform_Base_CIFAR10
from .dcase import (
    transform_ResNet50_ModifiedEfficientNet_DCASE,
    transform_Cnn10_Cnn14_DCASE2016,
    transform_ASTModel_DCASE
)
from .augsburg import (
    transform_Base_Augsburg,
    transform_Cnn10_Cnn14_Augsburg
)
from helpers import BaseRegistry
from .utils import wildcard_transform

BASE_TRANSFORMS_REGISTRY = BaseRegistry({
    # general wildcard to transform to tensor
    "None": wildcard_transform,
    # CIFAR10 transforms
    "Cnn10_CIFAR10": transform_Cnn10_Cnn14_CIFAR10,
    "Cnn14_CIFAR10": transform_Cnn10_Cnn14_CIFAR10,
    "ResNet50_CIFAR10": transform_Base_CIFAR10,
    "ModifiedEfficientNet_CIFAR10": transform_Base_CIFAR10,
    # DCASE2016 transforms
    "Cnn10_DCASE2016": transform_Cnn10_Cnn14_DCASE2016,
    "Cnn14_DCASE2016": transform_Cnn10_Cnn14_DCASE2016,
    "ResNet50_DCASE2016": transform_ResNet50_ModifiedEfficientNet_DCASE,
    "ModifiedEfficientNet_DCASE2016": transform_ResNet50_ModifiedEfficientNet_DCASE,
    "ASTModel_DCASE2016": transform_ASTModel_DCASE,
    # DCASE2020 transforms
    "ResNet50_DCASE2020": transform_ResNet50_ModifiedEfficientNet_DCASE,
    "ModifiedEfficientNet_DCASE2020": transform_ResNet50_ModifiedEfficientNet_DCASE,
    "ASTModel_DCASE2020": transform_ASTModel_DCASE,
    # KIRun.audio transforms
    "ASTModel_KIRun.audio": transform_ASTModel_DCASE,
    # Pollen Augsburg transforms
    "Cnn10_Pollen.Augsburg": transform_Cnn10_Cnn14_Augsburg,
    "Cnn14_Pollen.Augsburg": transform_Cnn10_Cnn14_Augsburg,
    "ResNet50_Pollen.Augsburg": transform_Base_Augsburg,
    "ModifiedEfficientNet_Pollen.Augsburg": transform_Base_Augsburg,
}, "BASE_TRANSFORMS_REGISTRY", wildcard=True)
