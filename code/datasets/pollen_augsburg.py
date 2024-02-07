import pandas as pd
import os
from typing import List
from .utils import LabelEncoder, DATASET_CLASS_REGISTRY
from .abstract_dataset import AbstractDataset


class Augsburg(AbstractDataset):
    def __init__(self,
                 path: str,
                 metrics: List[str],
                 tracking_metric: str,
                 n_dataset_classes: int,
                 base_transform=None,
                 train_transform=None,
                 dev_transform=None,
                 test_transform=None,
                 target_column="Taxon",
                 seed: int = 42,
                 dataset_type="CachedDataset",
                 ) -> None:

        super().__init__(
            seed=seed,
            task="classification",
            metrics=metrics,
            tracking_metric=tracking_metric,
            target_column=target_column,
            train_transform=train_transform
        )
        self.path = path
        self.dataset_type = dataset_type
        self.dataset_class = DATASET_CLASS_REGISTRY[self.dataset_type]
        self.n_dataset_classes = n_dataset_classes
        if self.n_dataset_classes not in (15, 31):
            print("No valid Pollen Augsburg Dataset selected. \
                Continuing with default Augsburg15.")
            self.n_dataset_classes = 15
        self.train_transform = self._combine_transforms(
            base_transform, train_transform)
        self.dev_transform = self._combine_transforms(
            base_transform, dev_transform)
        self.test_transform = self._combine_transforms(
            base_transform, test_transform)
        self.target_column = target_column
        df_train = self._prepare_dataframes(
            "original_31_traindata.csv", self.n_dataset_classes)
        df_dev = self._prepare_dataframes(
            "original_31_valdata.csv", self.n_dataset_classes)
        df_test = self._prepare_dataframes(
            "original_31_testdata.csv", self.n_dataset_classes)
        self.num_classes = len(df_train[self.target_column].unique())
        self.target_transform = LabelEncoder(
            list(df_train[self.target_column].unique()))

        self.train_dataset = self._create_dataset(
            df_train, self.train_transform)
        self.dev_dataset = self._create_dataset(df_dev, self.dev_transform)
        self.test_dataset = self._create_dataset(df_test, self.test_transform)

        self.df_train = df_train
        self.df_dev = df_dev
        self.df_test = df_test

    def _create_dataset(self, df, transform):
        return self.dataset_class(
            df=df,
            feature_column="Path",
            feature_dir=self.path,
            data_type="image",
            target_column=self.target_column,
            transform=transform,
            target_transform=self.target_transform.encode,
        )

    def _prepare_dataframes(self, file: str, n_dataset_classes: int):
        """
        n_dataset_classes is only defined for 15 and 31 classes
        according to Pollen Paper
        """
        class_file = "original_" + str(n_dataset_classes) + ".names"
        df = pd.read_csv(os.path.join(self.path, file),
                         header=None, names=["Path", self.target_column])
        df[self.target_column] = df["Path"].apply(
            lambda x: x.split("/")[0] if "/" in x else x)
        df.columns = ["Path", self.target_column]
        with open(os.path.join(self.path, class_file), "r") as file:
            class_names = [line.strip() for line in file]
            filtered_df = df[df[self.target_column].isin(class_names)]
        return filtered_df

    def get_datasets(self):
        return self.train_dataset, self.dev_dataset, self.test_dataset

    @property
    def output_dim(self):
        return len(self.target_transform.labels)
