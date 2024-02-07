# Sample Difficulty for Automatic Pollen Classification

This repository is designed to train models on the [Augsburg15](https://www.sciencedirect.com/science/article/pii/S0048969721040043) pollen classification dataset (available upon request) and to produce likelihood-based difficulty samples.

The corresponding paper has been submitted to the IEEE EMBC2024 conference and is currently under review.

Code base by **Simon David Noel Rampp**. Contributions by **Manuel Milling** and **Andreas Triantafyllopoulos**.

## Installation

This repository is intended to be used with Python `3.10.11` and PyTorch `2.1.0+cu121`.

Create a virtual environment install the requirements:

```bash
python -m venv .venv # and activate it
pip install -r requirements.txt
```

## Paper Reproduction

### Initial Model Training (Section III. A.)

Data has to be placed in `data/`. Configurations of the grid search and paths can be adjusted in `conf/grid_search.yaml`. By default, the paper's grid search is already configured. Results including aggregation over training runs are created in `results/reproduce/`.

The initial grid search can be reproduced by executing:

```python
python code/training.py
```

### Sample Difficulty Estimation (Section III. B.)

Configurations of the grid search and paths can be adjusted in `conf/curriculum.yaml` and have to be consistent with existing training runs. By default, the paper's difficulty estimation is already configured. Results are created in `results/reproduce/curriculum`.

The sample difficulty estimation can be reproduced by executing:

```python
python code/curriculum.py
```

### Easiest and Hardest Sample Selection (Section III. E.)

The easiest and hardest samples are selected based on the estimated difficulty scores. Results are created in `results/reproduce/file_selection`.

The sample selection can be reproduced by executing:

```python
python code/easiest_hardest_file_selection.py
```

## Repository Structure

This part contains a more in-depth discussion of the repository structure. Please also refer to the READMEs in the subfolders.

### Hydra Configuration

This repository uses [Hydra](https://hydra.cc/) for configuration management.
All Hydra configuration files are located in the `conf/` directory.
Scripts like `training.py` use Hydra to load the configuration files.

To run a script with a specific configuration file by executing the following command:

```bash
python code/training.py --config-name=your_config.yaml
```

We use Hydra for grid searches over hyperparameters in the `training.py`.
All parameters can be specified in separate `configuration files` or as primitives (see example below).

### Creating a Configuration File

When creating a configuration file, you can specify the parameters in the following way:

```yaml
# conf/model/ResNet50-Example.yaml
id: ResNet50-Example
name: ResNet50
_private: parameter
# ... other parameters
```

However there are some restrictions to the naming convention of the parameters:

- The `id` parameter should be unique for each configuration file and the same as the name of the configuration file.
- Since the `id` is also a file name, it should not contain any illegal file name characters (e.g. `:`). Additionally, it should not contain any underscores (`_`) or hashes (`#`), since they are used to separate the different parameters in the run names and to indicate aggregated results. Optimally, the `id` should only consist of alphanumeric characters and dashes (`-`).
- The `name` parameter should the same as in the registry (see [Example](#example)).

### Modules

The following sections provide a short overview of the modules and their functionalities.
Their goal is to ease the understanding of the code, but not to be a complete documentation.

- [Criterions](./code/criterions/README.md): Criterions for the models.
- [Curriculums](./code/curriculums/README.md): Curriculums for the datasets.
- [Helpers](./code/helpers/README.md): Helper functions for the modules.
- [Models](./code/models/README.md): Model definitions.
- [Optimizers](./code/optimizers/README.md): Optimizers for the models.
- [Postprocessing](./code/postprocessing/README.md): Postprocessing of the results.
- [Schedulers](./code/schedulers/README.md): Schedulers for the optimizers.
- [Training](./code/training/README.md): Training of the models.

All modules are located in the `code/${module}` directory and used to separate different datasets, models, augmentation strategies, etc...

Each module that is intended to be used in combination with Hydra should provide yaml configuration files in the `conf/` directory.
