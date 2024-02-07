# Sample Difficulty for Automatic Pollen Classification 

This repository is designed to train models on the [Augsburg15](https://www.sciencedirect.com/science/article/pii/S0048969721040043) pollen classification dataset and to produce likelihood-based difficulty samples. The correspoonding paper has been submitted to the IEEE EMBC2024 conference and is currently under review.  


## Usage

This section gives a rough overview for quick usage of the repo. This repo is based on hydra and according configuration files in `conf/` directory. 


### Local Machine Installation

For installation on your local machine (without poetry, just pip i assume), create a virtual environment called `.venv` and install the requirements:

```bash
python -m venv .venv # and activate it
pip install -r requirements.txt
```

If your local machine uses different cuda versions, you can install the correct version of pytorch [here](https://pytorch.org/get-started/locally/). Check if cuda is available and install it manually if necessary.

```bash
python -c "import torch; print(torch.cuda.is_available())" # verify cuda is available
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 #your version here
```

### Training

- Data has to be placed in `data/`.
- Configurations of the grid search and paths can be adjusted in `conf/grid_search.yaml`.
- Run `code/training.py` to run the grid search training.
- Results including aggregation over training runs are created in `results/default_grid/`.

### Sample Difficulty Estimation

- Configurations of the grid search and paths can be adjusted in `conf/curriculum.yaml` and have to be consistent with existing training runs.
- Run `code/curriculum.py` to run the sample difficulty estimation. 
Results will be stored in `results/default_grid/curriculum`.
- To select the estimated easiest and hardest samples run `code/easiest_hardest_file_selection.py`. Files will be stored in `results/default_grid/file_selection`

## Structure

This part contains a more in-depth discussion of the repository structure. Please also refer to the READMEs in the subfolders.

### Hydra Configuration

Optionally, you can use Hydra to create configuration files for your experiments. All Hydra configuration files are located in the `conf/` directory.
Scripts like `training.py` use Hydra to load the configuration files. To run a script with a specific configuration file, use the following command:

```bash
python code/training.py --config-name=your_config.yaml
```

We use Hydra for grid searches over hyperparameters in the `training.py`.
Additionally Hydra can be used to run a single experiment with a specific configuration file.
All parameters can be specified in separate `configuration files` or as primitives (see example below).

### Creating a Configuration File

When creating a configuration file, you can specify the parameters in the following way:

```yaml
# conf/model/Cnn10-Example.yaml
id: Cnn10-Example
name: Cnn10
output_dim: 10
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
- [Datasets](./code/datasets/README.md): Datasets for the models.
- [Helpers](./code/helpers/README.md): Helper functions for the modules.
- [Models](./code/models/README.md): Model definitions.
- [Optimizers](./code/optimizers/README.md): Optimizers for the models.
- [Postprocessing](./code/postprocessing/README.md): Postprocessing of the results.
- [Schedulers](./code/schedulers/README.md): Schedulers for the optimizers.
- [Training](./code/training/README.md): Training of the models.

All modules are located in the `code/${module}` directory and used to separate different datasets, models, augmentation strategies, etc...

Each module that is intended to be used in combination with Hydra should provide yaml configuration files in the `conf/` directory.
