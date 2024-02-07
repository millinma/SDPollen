from .utils import RGBToGrayscale
from torchvision import transforms


def transform_Cnn10_Cnn14_CIFAR10():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((64, 64), antialias=True),
        RGBToGrayscale(),
    ])


def transform_Base_CIFAR10():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
