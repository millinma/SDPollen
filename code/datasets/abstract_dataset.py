import torch
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from torchvision import transforms
from torch.utils.data import DataLoader

from metrics import METRIC_REGISTRY


@dataclass
class AbstractDataset(ABC):
    seed: int
    task: str
    metrics: typing.List[str]
    tracking_metric: str
    target_column: str
    train_transform: transforms.Compose
    stratify: typing.List[str] = field(default_factory=lambda: [])

    def __post_init__(self):
        assert self.task in ["classification", "regression"]
        self.metrics = [METRIC_REGISTRY(
            **{"name": m, "metric": m}) for m in self.metrics]
        self.tracking_metric = METRIC_REGISTRY(
            **{"name": self.tracking_metric, "metric": self.tracking_metric})
        if hasattr(self.train_transform, "get_collate_fn"):
            self.get_collate_fn = self.train_transform.get_collate_fn
        else:
            self.get_collate_fn = None

    def _combine_transforms(self, one: transforms.Compose, two: transforms.Compose):
        """Combine two torchvision.transforms.Compose objects into one.

        Args:
            one (transforms.Compose): First set of transforms.
            two (transforms.Compose): Second set of transforms.

        Returns:
            transforms.Compose: Combined set of transforms.
        """
        if one is None and two is None:
            return None
        if one is None:
            return two
        if two is None:
            return one
        return transforms.Compose([one, two])

    def get_loaders(self, batch_size, inference_batch_size):
        if inference_batch_size is None:
            inference_batch_size = batch_size
        g = torch.Generator().manual_seed(self.seed)
        collate_fn = self.get_collate_fn(self) if self.get_collate_fn else None
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            collate_fn=collate_fn
        )
        dev_loader = DataLoader(
            self.dev_dataset, batch_size=inference_batch_size, shuffle=False, generator=g)
        test_loader = DataLoader(
            self.test_dataset, batch_size=inference_batch_size, shuffle=False, generator=g)

        return train_loader, dev_loader, test_loader

    def get_evaluation_data(self):
        return self.df_dev, self.df_test, self.stratify, self.target_transform

    @abstractmethod
    def output_dim(self):
        pass

    @abstractmethod
    def get_datasets(self):
        """Get datasets.

        Returns:
            A tuple or dictionary of PyTorch Dataset objects.
        """
        pass

    def calculate_weight(self, weight_type: str) -> torch.Tensor:
        if self.task != "classification":
            raise ValueError(
                "Weights can only be calculated for classification tasks."
            )
        # ? Currently only balanced weights are supported
        if weight_type != "balanced":
            raise ValueError(
                f"Weight type '{weight_type}' not supported"
            )
        frequency = (
            self.df_train[self.target_column]
            .map(self.target_transform.encode)
            .value_counts()
            .sort_index()
            .values
        )
        weight = torch.tensor(1 / frequency, dtype=torch.float32)
        weight /= weight.sum()
        return weight
