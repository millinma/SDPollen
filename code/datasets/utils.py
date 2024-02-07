import audobject
import torch
from os.path import basename, join
import numpy as np
import torchvision
import pandas as pd
import torchaudio
import audiofile


class LabelEncoder(audobject.Object):
    def __init__(self, labels):
        self.labels = sorted(labels)
        codes = range(len(self.labels))
        self.inverse_map = {code: label for code,
                            label in zip(codes, self.labels)}
        self.map = {label: code for code,
                    label in zip(codes, self.labels)}

    def encode(self, x):
        return self.map[x]

    def decode(self, x):
        return self.inverse_map[x]


class CachedDataset(torch.utils.data.Dataset):
    r"""Dataset of cached features. Currentl is
        compatible with Pollen.Augsburg15/31 and 
        DCASE2016/20. 

    Args:
        df: partition dataframe containing labels
        feature_column: column with paths to features
        target_column: column to find labels in (in df)
        data_type: supports image or spectrogram (load method)
        features (optiona)l: dataframe with paths to features
        feature_dir: path to folder with features
        item_transform: indicates if item is transformed
        transform: function used to process features
        target_transform: function used to process labels
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_column: str,
        target_column: str,
        data_type="image",
        features: pd.DataFrame = None,
        feature_dir: str = "",
        item_transform: bool = False,
        transform=None,
        target_transform=None,
        **kwargs
    ):
        self.df = df
        self.features = features
        self.target_column = target_column
        self.feature_column = feature_column
        self.data_type = data_type
        self.transform = transform
        self.target_transform = target_transform
        self.indices = list(self.df.index)
        self.feature_dir = feature_dir
        self.item_transform = item_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        if self.item_transform and self.features is not None:
            index = self.indices[item]
            feature_basename = basename(
                self.features.loc[index, self.feature_column])
            target = self.df[self.target_column].loc[index]
        else:
            feature_basename = self.df.iloc[item][self.feature_column]
            target = self.df.iloc[item][self.target_column]
        data_file = join(self.feature_dir, feature_basename)
        if self.data_type == "image":
            data = torchvision.io.read_image(data_file).float()[:3] / 255
        elif self.data_type == "spectrogram":
            data = np.load(data_file + ".npy")
        if isinstance(self.target_column, list) and len(self.target_column) > 1:
            target = np.array(target.values)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

DATASET_CLASS_REGISTRY = {
    "CachedDataset": CachedDataset
}
