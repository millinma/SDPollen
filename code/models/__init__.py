from .resnet_50 import create_ResNet50_model
from .modified_efficientnet import ModifiedEfficientNet
from helpers import BaseRegistry

MODEL_REGISTRY = BaseRegistry({
    "ResNet50": create_ResNet50_model,
    "ModifiedEfficientNet": ModifiedEfficientNet
}, "MODEL_REGISTRY")
