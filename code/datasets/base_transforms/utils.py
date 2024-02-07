import torch
import numpy as np
from typing import Union
from torchvision import transforms
from audtorch.transforms import Downmix, Upmix


def wildcard_transform(**kwargs):
    return transforms.Compose([NumpyToTensor()])


def _image_to_numpy(image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    image = image.astype(np.float32)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=0)
    return image


def _image_to_tensor(image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    return image


class NumpyToTensor(object):
    def __call__(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _image_to_tensor(data)


class GrayscaleToRGB(object):
    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        image = _image_to_numpy(image)
        image_rgb = Upmix(3, axis=0)(image)
        return _image_to_tensor(image_rgb)


class RGBToGrayscale(object):
    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        image = _image_to_numpy(image)
        image_grs = Downmix(1, axis=0)(image)
        return _image_to_tensor(image_grs)
