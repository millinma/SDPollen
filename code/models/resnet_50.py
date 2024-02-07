import torch
from torchvision.models import resnet50, ResNet50_Weights


def create_ResNet50_model(output_dim, transfer=False):
    # Load the ResNet50 model
    # ? ResNet50_Weights are the latest available weights for ResNet50
    model = resnet50(weights=ResNet50_Weights if transfer else None)

    # Modify the final fully connected layer for CIFAR10
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, output_dim)
    model.output_dim = output_dim
    return model
