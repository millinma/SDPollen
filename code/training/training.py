import os
import yaml
import torch
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig

from models import MODEL_REGISTRY
from datasets import DATASET_REGISTRY
from optimizers import OPTIMIZER_REGISTRY
from schedulers import SCHEDULER_REGISTRY
from criterions import CRITERION_REGISTRY
from .callback_manager import CallbackManager
from datasets.abstract_dataset import AbstractDataset
from datasets.base_transforms import BASE_TRANSFORMS_REGISTRY
from helpers import set_seed, Bookkeeping, Timer, PlotMetrics, MLFlowLogger


class ModularTaskTrainer:
    def __init__(self,
                 cfg: DictConfig,
                 output_directory: str,
                 experiment_id: str = None,
                 run_name: str = None,
                 ) -> None:
        set_seed(cfg.seed)
        self.output_directory = Path(output_directory)
        self.training = cfg.training
        self.save_frequency = self.training.get("save_frequency", 1)

        # ? Save current requirements.txt
        working_directory = os.getcwd()
        shutil.copyfile(
            os.path.join(working_directory, "requirements.txt"),
            os.path.join(output_directory, "requirements.txt")
        )

        # ? Allow Model to have custom base transforms and dataset options
        # ? with **kwargs, only the relevant parameters will be passed to the dataset
        # ? Load Transforms and Datasets
        base_transform_args = {"name": cfg.model.name + "_" + cfg.dataset.name}
        model_base_transform_args = cfg.model.pop("base_transform", None)
        if model_base_transform_args:
            base_transform_args.update(model_base_transform_args)
        base_transform = BASE_TRANSFORMS_REGISTRY(**base_transform_args)

        # ? Pop criterion to be setup later
        cfg.criterion = cfg.dataset.pop("criterion").copy()

        dataset_args = {}
        model_dataset_args = cfg.model.pop("dataset", None)
        if model_dataset_args:
            dataset_args.update(model_dataset_args)

        
        self.data: AbstractDataset = DATASET_REGISTRY(
            **cfg.dataset,
            base_transform=base_transform,
            seed=cfg.seed,
            **dataset_args,
        )

        # ? Create Bookkeeping and MLFlow Logger
        self.bookkeeping = Bookkeeping(
            output_directory=output_directory,
            file_handler_path=os.path.join(output_directory, "training.log")
        )
        self.mlflow_logger = MLFlowLogger(
            output_directory=self.output_directory,
            exp_name=experiment_id or self.output_directory.parent.parent.name,
            run_name=run_name or self.output_directory.name,
            metrics=self.data.metrics,
            tracking_metric=self.data.tracking_metric
        )
        self.mlflow_logger.log_params(cfg.copy())

        # ? Datasets and Evaluation Data
        self.train_dataset, self.dev_dataset, self.test_dataset = self.data.get_datasets()
        self.df_dev, self.df_test, self.stratify, self.target_transform = self.data.get_evaluation_data()
        self.task = self.data.task

        # ? Misc Training Parameters
        self.DEVICE = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.disable_progress_bar = not cfg.get("_progress_bar", False)

        # ? Load Criterion aligned with Dataset and set up weights if specified
        weight_type = cfg.criterion.get("weight", None)
        if weight_type and self.task == "classification":
            self.criterion = CRITERION_REGISTRY(**{
                **cfg.criterion,
                "weight": self.data.calculate_weight(weight_type)
            })
        else:
            self.criterion = CRITERION_REGISTRY(**cfg.criterion)
        self.criterion.to(self.DEVICE)

        # ? Load Pretrained Model and Optimizer Checkpoints if specified
        model_checkpoint = cfg.model.pop("pretrained", None)
        optimizer_checkpoint = cfg.optimizer.pop("pretrained", None)
        scheduler_checkpoint = cfg.scheduler.pop("pretrained", None)

        # ? Load Model (with dataset output_dim if not set in config)
        cfg.model.output_dim = self.data.output_dim
        self.output_dim = cfg.model.output_dim
        self.model = MODEL_REGISTRY(**cfg.model)
        if model_checkpoint:
            self.model.load_state_dict(torch.load(model_checkpoint))
        self.bookkeeping.save_model_summary(
            self.model,
            self.train_dataset,
            "model_summary.txt"
        )

        # ? Load Optimizer
        self.optimizer = OPTIMIZER_REGISTRY(
            params=self.model.parameters(),
            **cfg.optimizer
        )
        if optimizer_checkpoint:
            self.optimizer.load_state_dict(torch.load(optimizer_checkpoint))

        # ? Load Scheduler
        self.scheduler = SCHEDULER_REGISTRY(
            optimizer=self.optimizer,
            **cfg.scheduler
        )
        if scheduler_checkpoint:
            self.scheduler.load_state_dict(torch.load(scheduler_checkpoint))

        # ? Create Dataloaders
        self._inference_batch_size = cfg.get("_inference_batch_size", None)
        loaders = self.data.get_loaders(
            cfg.batch_size,
            self._inference_batch_size
        )
        self.train_loader, self.dev_loader, self.test_loader = loaders

        # ? Misc Training Parameters
        self.DEVICE = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.disable_progress_bar = not cfg.get("_progress_bar", False)

        # ? Take metrics from dataset and add train/dev loss
        metrics = [m.name for m in self.data.metrics] + \
            ["train_loss", "dev_loss"]
        self.metrics = pd.DataFrame(columns=metrics)
        self.max_dev_metric = self.data.tracking_metric.starting_metric
        self.best_iteration = 1

        # ? Save initial (and best) Model, Optimizer and Encoder State
        self.bookkeeping.create_folder("_initial")
        self.bookkeeping.save_state(self.model, "model.pth.tar", "_initial")
        self.bookkeeping.save_state(
            self.optimizer, "optimizer.pth.tar", "_initial")
        if self.scheduler:
            self.bookkeeping.save_state(
                self.scheduler, "scheduler.pth.tar", "_initial")

        self.bookkeeping.create_folder("_best")
        self.bookkeeping.save_state(self.model, "model.pth.tar", "_best")
        self.bookkeeping.save_state(
            self.optimizer, "optimizer.pth.tar", "_best")
        if self.scheduler:
            self.bookkeeping.save_state(
                self.scheduler, "scheduler.pth.tar", "_best")

        self.bookkeeping.save_target_transform(
            self.target_transform, "target_transform.yaml")

        # ? Create Timers
        self.train_timer = Timer(output_directory, "train")
        self.dev_timer = Timer(output_directory, "dev")
        self.test_timer = Timer(output_directory, "test")

        # ? Create Callback Manager
        self.callback_manager = CallbackManager(self)

        # ? Create Plot Metrics
        self.plot_metrics = PlotMetrics(
            self.output_directory,
            self.training.get("type", None),
            **cfg.plotting
        )

    def train(self):
        self.callback_manager.callback(
            position="cb_on_train_begin",
            trainer=self
        )
        # ? Allow optimizers to have custom step functions
        custom_step = getattr(self.optimizer, "custom_step", False) and callable(
            self.optimizer.custom_step)
        self.train_step_fn = self.optimizer.custom_step if custom_step else self._train_step

        self.training_type = self.training.get("type", None)

        if self.training_type == "Epoch":
            self.train_epochs()
        elif self.training_type == "Step":
            self.train_steps()
        else:
            raise ValueError(
                f"Training type {self.training_type} not supported")

        # ? Score best model on test set
        self.bookkeeping.load_state(
            self.model, "model.pth.tar", "_best")
        self.bookkeeping.load_state(
            self.optimizer, "optimizer.pth.tar", "_best")
        self.model = self.model.to(self.DEVICE)
        self.model.eval()
        self.bookkeeping.create_folder("_test")
        self.test_timer.start()
        test_results = self.evaluate(
            -1,
            "_test",
            self.test_loader,
            self.df_test,
            dev_evaluation=False,
            save_to="test_holistic"
        )
        self.test_timer.stop()
        self.metrics["iteration"] = self.metrics.index
        self.bookkeeping.save_results_df(self.metrics, "metrics.csv")
        self.bookkeeping.save_best_results(
            self.metrics, "best_results.yaml", self.data.tracking_metric.name, "_best")
        self.bookkeeping.log(
            "Best results at {} {}:\n{}".format(
                self.training_type,
                self.best_iteration,
                yaml.dump(self.metrics.loc[self.best_iteration].to_dict())
            )
        )
        self.mlflow_logger.log_test_metrics(test_results)
        self.bookkeeping.log(
            "Test results:\n{}".format(
                yaml.dump(test_results)
            )
        )

        # ? Save Timers
        self.train_timer.save()
        self.dev_timer.save()
        self.test_timer.save()
        self.mlflow_logger.log_timers({
            "time."+t.timer_type+".mean":
                Timer.pretty_time(t.get_mean_seconds()) for t in
                [self.train_timer, self.dev_timer, self.test_timer]
        })

        # ? Plot Metrics
        self.plot_metrics.plot_run(self.metrics)

        self.callback_manager.callback(
            position="cb_on_train_end",
            trainer=self,
            test_results=test_results
        )

        # ? Log Artifacts to MLFlow and end run
        self.mlflow_logger.log_artifact("model_summary.txt")
        self.mlflow_logger.log_artifact("metrics.csv")
        self.mlflow_logger.log_artifact("config.yaml", ".hydra")
        self.mlflow_logger.end_run()

    def train_epochs(self):
        train_loss = []
        self.train_timer.start()
        for epoch in range(1, self.training.num_iterations+1):
            self.callback_manager.callback(
                position="cb_on_iteration_begin",
                trainer=self,
                iteration=epoch
            )
            epoch_folder = f"epoch_{epoch}"
            self.bookkeeping.create_folder(epoch_folder)
            self.model.train()
            self.model.to(self.DEVICE)
            for batch_idx, (data, target) in enumerate(tqdm(
                self.train_loader,
                desc="Train",
                disable=self.disable_progress_bar
            )):
                data, target = data.to(self.DEVICE), target.to(self.DEVICE)
                self.callback_manager.callback(
                    position="cb_on_step_begin",
                    trainer=self,
                    iteration=epoch,
                    batch_idx=batch_idx,

                )
                l = self.train_step_fn(
                    self.model,
                    data,
                    target,
                    self.criterion,
                )
                self.callback_manager.callback(
                    position="cb_on_step_end",
                    trainer=self,
                    iteration=epoch,
                    batch_idx=batch_idx,
                    loss=l
                )
                train_loss.append(l)
            if self.scheduler:
                self.scheduler.step()
            if epoch % self.training.eval_frequency == 0:
                self.train_timer.stop()
                train_loss = sum(train_loss) / len(train_loss)
                self.metrics.loc[epoch, "train_loss"] = train_loss
                self.dev_timer.start()
                self.evaluate(epoch, epoch_folder,
                              self.dev_loader, self.df_dev)
                self.dev_timer.stop()
                self.mlflow_logger.log_metrics(
                    self.metrics.loc[epoch].to_dict(), epoch)
                if epoch < self.training.num_iterations:
                    train_loss = []
                    self.train_timer.start()
                self.callback_manager.callback(
                    position="cb_on_iteration_end",
                    trainer=self,
                    iteration=epoch,
                    metrics=self.metrics.loc[epoch].to_dict()
                )
            self.callback_manager.callback(
                position="cb_on_loader_exhausted",
                trainer=self,
                iteration=epoch
            )

    def train_steps(self):
        pbar = tqdm(
            total=self.training.eval_frequency,
            desc="Train",
            disable=self.disable_progress_bar
        )
        step = 0
        self.callback_manager.callback(
            position="cb_on_iteration_begin",
            trainer=self,
            iteration=step
        )
        self.train_loader_iter = iter(self.train_loader)
        train_loss = []
        self.train_timer.start()
        while step < self.training.num_iterations:
            step += 1
            pbar.update(1)
            self.model.train()
            self.model.to(self.DEVICE)
            try:
                data, target = next(self.train_loader_iter)
            except StopIteration:
                self.callback_manager.callback(
                    position="cb_on_loader_exhausted",
                    trainer=self,
                    iteration=step
                )
                self.train_loader_iter = iter(self.train_loader)
                data, target = next(self.train_loader_iter)
            data, target = data.to(self.DEVICE), target.to(self.DEVICE)
            self.callback_manager.callback(
                position="cb_on_step_begin",
                trainer=self,
                iteration=step,
                batch_idx=step
            )
            l = self.train_step_fn(
                self.model,
                data,
                target,
                self.criterion,
            )
            self.callback_manager.callback(
                position="cb_on_step_end",
                trainer=self,
                iteration=step,
                batch_idx=step,
                loss=l
            )
            train_loss.append(l)
            if self.scheduler:
                self.scheduler.step()
            if step % self.training.eval_frequency == 0:
                self.train_timer.stop()
                step_folder = f"step_{step}"
                self.bookkeeping.create_folder(step_folder)
                train_loss = sum(train_loss) / len(train_loss)
                self.metrics.loc[step, "train_loss"] = train_loss
                self.dev_timer.start()
                self.evaluate(
                    step,
                    step_folder,
                    self.dev_loader,
                    self.df_dev
                )
                self.dev_timer.stop()
                self.mlflow_logger.log_metrics(
                    self.metrics.loc[step].to_dict(), step)
                if step < self.training.num_iterations:
                    train_loss = []
                    pbar.reset()
                    self.train_timer.start()
                self.callback_manager.callback(
                    position="cb_on_iteration_end",
                    trainer=self,
                    iteration=step,
                    metrics=self.metrics.loc[step].to_dict()
                )

    def _train_step(self, model, data, target, criterion):
        self.optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(
        self,
        iteration: int,
        iteration_folder,
        loader,
        df,
        dev_evaluation=True,
        save_to="dev"
    ):
        self.model.eval()
        results = self._evaluate(loader=loader)
        if dev_evaluation:
            results["dev_loss"] = results.pop("loss")
            # TODO: it's a bit ugly to filter like this
            for key in list(set(self.metrics.columns) - set(["train_loss"])):
                self.metrics.loc[iteration, key] = results[key]
        else:
            test_results = {"test_loss": results["loss"]}
            # TODO: another ugly filter
            for key in list(set(self.metrics.columns) - set(["train_loss", "dev_loss"])):
                test_results[f"test_{key}"] = results[key]

        results_df = pd.DataFrame(
            index=df.index,
            data=results["predictions"],
            columns=["predictions"]
        )
        results_df["predictions"] = results_df["predictions"].apply(
            self.target_transform.decode)

        self.bookkeeping.save_results_df(
            results_df.reset_index(),
            "results.csv",
            iteration_folder
        )
        if dev_evaluation:
            self.bookkeeping.log(
                "Dev results at {} {}:\n{}".format(
                    self.training_type,
                    iteration,
                    yaml.dump(self.metrics.loc[iteration].to_dict())
                )
            )

        self.bookkeeping.save_results_np(
            results["targets"], "targets.npy", iteration_folder)
        self.bookkeeping.save_results_np(
            results["predictions"], "predictions.npy", iteration_folder)
        self.bookkeeping.save_results_np(
            results["outputs"], "outputs.npy", iteration_folder)

        logging_results = self._disaggregated_evaluation(
            df=results_df,
            groundtruth=df,
            stratify=self.stratify,
        )
        if dev_evaluation:
            logging_results["dev_loss"] = {"all": results["dev_loss"]}
            logging_results["iteration"] = iteration
        else:
            logging_results["loss"] = {"all": results["loss"]}

        self.bookkeeping.save_results_dict(
            logging_results, save_to+".yaml", iteration_folder)

        if not dev_evaluation:
            return test_results

        if self.data.tracking_metric.compare(
            results[self.data.tracking_metric.name],
            self.max_dev_metric
        ):
            self.max_dev_metric = results[self.data.tracking_metric.name]
            self.best_iteration = iteration
            self.bookkeeping.save_state(self.model, "model.pth.tar", "_best")
            self.bookkeeping.save_state(
                self.optimizer, "optimizer.pth.tar", "_best")
            if self.scheduler:
                self.bookkeeping.save_state(
                    self.scheduler, "scheduler.pth.tar", "_best")

            # ? additionally save all best results
            self.bookkeeping.save_results_df(
                results_df.reset_index(),
                "results.csv",
                "_best"
            )
            self.bookkeeping.save_results_dict(
                logging_results, "dev.yaml", "_best")
            self.bookkeeping.save_results_np(
                results["targets"], "targets.npy", "_best")
            self.bookkeeping.save_results_np(
                results["predictions"], "predictions.npy", "_best")
            self.bookkeeping.save_results_np(
                results["outputs"], "outputs.npy", "_best")

        if iteration % self.save_frequency == 0 or iteration == self.training.num_iterations:
            self.bookkeeping.save_state(
                self.model, "model.pth.tar", iteration_folder)
            self.bookkeeping.save_state(
                self.optimizer, "optimizer.pth.tar", iteration_folder)
            if self.scheduler:
                self.bookkeeping.save_state(
                    self.scheduler, "scheduler.pth.tar", iteration_folder)

    def _evaluate(self, loader):
        outputs = torch.zeros((len(loader.dataset), self.output_dim))
        targets = torch.zeros(len(loader.dataset))
        with torch.no_grad():
            loss = 0
            for index, (features, target) in enumerate(tqdm(
                loader,
                desc="Evaluate",
                disable=self.disable_progress_bar
            )):
                start_index = index * loader.batch_size
                end_index = (index + 1) * loader.batch_size
                if end_index > len(loader.dataset):
                    end_index = len(loader.dataset)
                features = features.to(self.DEVICE)
                output = self.model(features)
                outputs[start_index:end_index, :] = output
                targets[start_index:end_index] = target
                loss += self.criterion(output.to(self.DEVICE),
                                       target.to(self.DEVICE)).cpu().item()
            loss /= (len(loader) + 1)
        targets = targets.numpy()
        outputs = outputs.cpu()
        if self.task == "classification":
            predictions = outputs.argmax(dim=1).numpy()
        elif self.task == "regression":
            predictions = outputs.numpy().squeeze(axis=-1)
        outputs = outputs.numpy()
        results = {
            "targets": targets,
            "predictions": predictions,
            "outputs": outputs,
            "loss": loss,
        }
        for metric in self.data.metrics:
            results[metric.name] = metric(targets, predictions)
        return results

    def _disaggregated_evaluation(self, df, groundtruth, stratify):
        df = df.reindex(groundtruth.index)
        results = {m.name: {} for m in self.data.metrics}
        for metric in self.data.metrics:
            results[metric.name]["all"] = metric(
                groundtruth[self.data.target_column],
                df["predictions"]
            )
            for s in stratify:
                for v in groundtruth[s].unique():
                    idx = groundtruth.loc[groundtruth[s] == v].index
                    results[metric.name][v] = metric(
                        groundtruth.reindex(idx)[self.data.target_column],
                        df.reindex(idx)["predictions"]
                    )
        return results
