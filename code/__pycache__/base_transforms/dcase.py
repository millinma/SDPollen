from .utils import GrayscaleToRGB, RGBToGrayscale
from torchvision import transforms
from transformers import ASTFeatureExtractor
import warnings
from omegaconf import OmegaConf


def transform_ResNet50_ModifiedEfficientNet_DCASE():
    return transforms.Compose([
        GrayscaleToRGB(),
    ])


def transform_Cnn10_Cnn14_DCASE2016():
    return transforms.Compose([
        RGBToGrayscale(),
    ])


def transform_ASTModel_DCASE(fe_transfer=None):
    if fe_transfer is not None:
        feature_extractor = ASTFeatureExtractor.from_pretrained(
            fe_transfer)
    else:
        feature_extractor = ASTFeatureExtractor()
        warnings.warn("ASTFeatureExtractor initialized with default values:\n" +
                      f"{OmegaConf.to_yaml(feature_extractor.__dict__)}")

    def extract_features(signal):
        return feature_extractor(
            signal,
            sampling_rate=16000,
            padding="max_length",
            return_tensors="pt"
        ).input_values[0]

    return transforms.Compose([
        extract_features,
    ])
