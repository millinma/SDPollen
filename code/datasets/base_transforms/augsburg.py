from .utils import RGBToGrayscale  # GrayscaleToRGB
from torchvision import transforms


def transform_Cnn10_Cnn14_Augsburg():
    return transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((64, 64), antialias=True),
        RGBToGrayscale(),
    ])


def transform_Base_Augsburg():
    return transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((64, 64), antialias=True),
    ])
