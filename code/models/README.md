### Models

`MODEL_REGISTRY` exports a registry of models. Models can be any class that inherits from `torch.nn.Module`.

#### ModifiedEfficientNet

See [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch).\
Can be transfer learned by specifying `transfer: True` in the model configuration file.
Additionally different model sizes can be chosen by specifying `scaling_type` in the model configuration file (default: `efficientnet-b0`).

#### ResNet50

See [ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html).\
Can be transfer learned by specifying `transfer: True` in the model configuration file.