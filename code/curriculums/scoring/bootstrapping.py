import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from helpers import Timer
from criterions import CRITERION_REGISTRY
from .abstract_score import AbstractScore


class Bootstrapping(AbstractScore):
    def __init__(
        self,
        output_directory: str,
        results_dir: str,
        experiment_id: str,
        run_name: str,
        stop: str = "best",
        subset: str = "train"
    ) -> None:
        super().__init__(
            output_directory=output_directory,
            results_dir=results_dir,
            experiment_id=experiment_id,
            run_name=run_name,
            subset=subset
        )
        self.stop = stop

    def preprocess(self) -> Tuple[list, list]:
        configs, runs = super().preprocess()
        runs = [f"{r}_{self.stop[0]}" for r in runs]
        return configs, runs

    def run(
        self,
        config: DictConfig,
        run_config: DictConfig,
        run_name: str
    ) -> None:
        full_run_name = run_name
        run_name = "_".join(run_name.split("_")[:-1])
        self.DEVICE = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        run_path = os.path.join(self.output_directory, full_run_name)
        forward_timer = Timer(run_path, "model_forward")
        self.disable_progress_bar = not config.get("_progress_bar", False)
        self.batch_size = config.get(
            "batch_size", run_config.get("batch_size", 32))

        criterion_config = run_config.dataset.pop("criterion")
        # This is an ugly hot-fix from the source repo to use the non-balanced 
        # cross-entropy loss.
        criterion_config.id = "CrossEntropyLoss"
        criterion_config.pop("weight")
        data, model = self._prepare_data_and_model(run_config)
        weight_type = criterion_config.get("weight", None)
        if weight_type and data.task == "classification":
            criterion = CRITERION_REGISTRY(**{
                **criterion_config,
                "weight": data.calculate_weight(weight_type)
            })
        else:
            criterion = CRITERION_REGISTRY(**criterion_config)
        criterion.to(self.DEVICE)

        training_dir = os.path.join(
            self.results_dir, self.experiment_id, "training", run_name)
        if self.stop == "best":
            model_checkpoint = os.path.join(
                training_dir, "_best", "model.pth.tar")
        else:
            dirs = os.listdir(training_dir)
            dirs = [d for d in dirs if d.startswith(
                run_config.training.type.lower())]
            dirs = sorted(dirs, key=lambda x: int(x.split("_")[-1]))
            model_checkpoint = os.path.join(
                training_dir, dirs[-1], "model.pth.tar")

        model.load_state_dict(torch.load(model_checkpoint))
        model.eval()
        criterion.reduction = "none"
        dataset = self.get_dataset_subset(data, self.subset)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        forward_timer.start()
        outputs, labels = self._forward_pass(
            model,
            criterion,
            loader,
            full_run_name
        )
        forward_timer.stop()
        df = pd.DataFrame()
        df["scores"] = outputs
        df["labels"] = labels
        df["decoded"] = df["labels"].apply(data.target_transform.decode)
        df.to_csv(os.path.join(self.output_directory,
                  full_run_name, "scores.csv"), index=False)
        forward_timer.save()

    def _forward_pass(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        loader: DataLoader,
        run_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        outputs = torch.zeros(len(loader.dataset))
        labels = torch.zeros(len(loader.dataset))
        model.to(self.DEVICE)
        with torch.no_grad():
            for idx, (x, y) in enumerate(tqdm(
                loader,
                desc=run_name,
                disable=self.disable_progress_bar
            )):
                _lower = idx*self.batch_size
                _upper = min(len(loader.dataset), (idx+1)*self.batch_size)
                x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                outs = model(x)
                loss = criterion(outs, y)
                outputs[_lower:_upper] = loss
                labels[_lower:_upper] = y.cpu()
        return outputs.numpy(), labels.numpy()
