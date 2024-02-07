from abc import ABC, abstractmethod
from omegaconf import DictConfig
import os
from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset

from helpers import set_seed
from models import MODEL_REGISTRY
from datasets import DATASET_REGISTRY
from datasets.abstract_dataset import AbstractDataset
from datasets.base_transforms import BASE_TRANSFORMS_REGISTRY
from postprocessing.postprocessing_utils import load_yaml


class AbstractScore(ABC):
    def __init__(self,
                 output_directory: str,
                 results_dir: str,
                 experiment_id: str,
                 run_name: str,
                 subset: str,
                 reverse_score: bool = False,
                 ) -> None:
        self.output_directory = output_directory
        self.results_dir = results_dir
        self.experiment_id = experiment_id
        self.run_name = run_name
        if subset not in ["train", "dev", "test"]:
            raise ValueError(f"Subset '{subset}' not supported")
        self.subset = subset
        self.reverse_score = reverse_score

    def preprocess(self) -> Tuple[list, list]:
        base_path = os.path.join(self.results_dir, self.experiment_id)
        dirs = [d for d in os.listdir(base_path) if d.startswith("agg_")]
        dirs.append("training")
        for d in dirs:
            if os.path.exists(os.path.join(base_path, d, self.run_name)):
                config = DictConfig(load_yaml(os.path.join(
                    base_path, d, self.run_name, ".hydra", "config.yaml")))
                if d == "training":
                    return [config], [self.run_name]
                runs = load_yaml(os.path.join(
                    base_path, d, self.run_name, "runs.yaml"))
                configs = [DictConfig(load_yaml(os.path.join(
                    base_path, "training", r, ".hydra", "config.yaml"))) for r in runs]
                return configs, runs
        raise ValueError(f"Run {self.run_name} does not exist")

    @abstractmethod
    def run(
        self,
        config: DictConfig,
        run_config: DictConfig,
        run_name: str
    ) -> None:
        ...

    def postprocess(self, score_id: str, runs: list) -> None:
        df = pd.DataFrame()
        labels = None
        decoded = None
        for run in runs:
            scores = pd.read_csv(os.path.join(
                self.output_directory, run, "scores.csv"))
            df[run] = scores["scores"]
            if labels is None:
                labels = scores["labels"]
                decoded = scores["decoded"]
        df["mean"] = df.mean(axis=1)
        df["ranks"] = self._rank_and_normalize(df)
        df["labels"] = labels
        df["decoded"] = decoded
        df[["mean", "ranks", "labels", "decoded"]].to_csv(
            os.path.join(self.output_directory, score_id+".csv"),
            index=False
        )

    def _prepare_data_and_model(
        self,
        cfg: DictConfig
    ) -> Tuple[AbstractDataset, torch.nn.Module]:
        seed = cfg.get("seed", 42)
        set_seed(seed)
        base_transform_args = {"name": cfg.model.name + "_" + cfg.dataset.name}
        model_base_transform_args = cfg.model.pop("base_transform", None)
        if model_base_transform_args:
            base_transform_args.update(model_base_transform_args)
        base_transform = BASE_TRANSFORMS_REGISTRY(**base_transform_args)

        dataset_args = {}
        model_dataset_args = cfg.model.pop("dataset", None)
        if model_dataset_args:
            dataset_args.update(model_dataset_args)

        cfg.criterion = cfg.dataset.pop("criterion", None)
        data: AbstractDataset = DATASET_REGISTRY(
            **cfg.dataset,
            base_transform=base_transform,
            seed=seed,
            **dataset_args,
        )

        cfg.model.pop("pretrained", None)
        cfg.model.output_dim = data.output_dim
        model = MODEL_REGISTRY(**cfg.model)

        return data, model

    def _rank_and_normalize(self, df: pd.DataFrame) -> pd.Series:
        ascending = not self.reverse_score
        ranks = df["mean"].rank(ascending=ascending, method="first")
        ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())
        return ranks

    @staticmethod
    def get_dataset_subset(data: AbstractDataset, subset: str) -> Dataset:
        train, dev, test = data.get_datasets()
        return {"train": train, "dev": dev, "test": test}[subset]
