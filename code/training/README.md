### Training

Training consists of a `ModularTaskTrainer` that gets initialized with a Hydra configuration.
The `ModularTaskTrainer` then initializes the model, optimizer, scheduler, criterion, dataset, and dataloader based on the configuration and trains the model.
The run results are stored in `${results_dir}/${experiment_id}/training/`.

Additionally, runs are automatically reported to MLflow and can be viewed in the MLflow UI.
If the training is interrupted, the run can be resumed by running the same command again.
Any duplicate runs are automatically rerun if they failed or were interrupted.
All finished runs do not get rerun.

You can additionally specify filters for invalid parameters in the Hydra configuration file.
Specify filters with: `case ! condition` (`&` and `|` are optional chaining operators). These however can not be mixed, so you can not use `&` and `|` in the same filter.

Here are some filter examples that can be used in the Hydra configuration file:

```yaml
# conf/training/grid_search.yaml
_filters:
    - model.name == "Cnn10" ! optimizer.lr > 0.1
    # dont allow runs with Cnn10 model and lr > 0.1
    - optimizer.lr > 0.001 ! batch_size < 32 & model.name == "Cnn10"
    # dont allow runs with lr > 0.001 and batch_size < 32 for Cnn10 model
```

The training functionality for a single run and a grid search is already implemented in `code/training.py`.

### Callbacks

Callbacks can be defined by any instance of the `data`, `model`, `optimizer`, `scheduler`, `criterion` and `curriculum` modules. Any of these modules can define one or more methods following the signatures below. These will be called at the appropriate time during training.

```python
def cb_on_train_begin(self, trainer: ModularTaskTrainer) -> None:
    pass

def cb_on_train_end(self, trainer: ModularTaskTrainer, test_results: dict) -> None:
    pass

def cb_on_iteration_begin(self, trainer: ModularTaskTrainer, iteration: int) -> None:
    pass

def cb_on_iteration_end(self, trainer: ModularTaskTrainer, iteration: int, metrics: dict) -> None:
    pass

def cb_on_loader_exhausted(self, trainer: ModularTaskTrainer, iteration: int) -> None:
    pass

def cb_on_step_begin(self, trainer: ModularTaskTrainer, iteration: int, batch_idx: int) -> None:
    pass

def cb_on_step_end(self, trainer: ModularTaskTrainer, iteration: int, batch_idx: int, loss: float) -> None:
    pass
```
