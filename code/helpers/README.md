### Helpers

Any helper functions or classes that are used in multiple modules can be placed in the `code/helpers/` directory.

### Documentation

#### Slurm

Slurm Manager creates a slurm array job which creates separate jobs for each run of the grid search. Additionally it waits for all jobs to finish. The array job is specified in `gpu_scripts/grid_search_array.sh`.
Slurm Training gets called by the array job and runs the training for one run of the grid search.

#### BaseRegistry

`BaseRegistry` is a base class for registries. It allows you to register classes or functions to a registry and retrieve them by name, e.g. for configuration with Hydra.

#### Bookkeeping

The `Bookkeeping` class saves all outputs to `${results_dir}/${experiment_id}/training/${run_name}` and allows you to:

- log messages to console or only to file
- create folders
- save and load models, optimizers, shedulers and encoders
- export a model summary
- save and load results, e.g. as a dictionary, dataframe or numpy array

#### Hydra

Hydra cleanup deletes any grid search runs including their directories that were not completed.
Hydra utils create a global grid search configuration file.

#### MLFlowLogger

`MLFlowLogger` automatically handles the creation as well as the management of experiments and runs.
Additionally, it allows for logging parameters, metrics, artifacts and timers to MLFlow.

#### Plotting

`PlotMetrics` allows for plotting all metrics of one run and one metric of multiple runs.

##### Plotting Configuration

Plots can be styled using hydra configurations.
The `Default` style does not use $\LaTeX$ and is very similar to the default matplotlib / seaborn style.
The `Thesis` style uses $\LaTeX$.
Additional styles can be added by creating a new configuration in `conf/plotting/` and specifying the following parameters:

- `figsize`: The size of the figure in inches as a tuple.
- `latex`: Whether to use $\LaTeX$ for rendering.
- `filetypes`: A list of filetypes to save the figure as.
- `pickle`: Whether to save the figure as a pickle file for later style changes.
- `context`: The context to use for seaborn.
- `palette`: The palette to use for seaborn.
- `replace_none`: Whether to replace `None` values with a compact representation using `~`. Replacements are only applied to the legend and help with saving space when configurations are complex. For example, `Some_Run_None_None_1` becomes `Some_Run~~1`.
- `rcParams`: A dictionary of matplotlib rcParams to override.

##### Plotting with LaTeX

If specified, plots will automatically use $\LaTeX$ for rendering if available on the system.
To set up $\LaTeX$ on the NixOS cluster, append the $\LaTeX$ config to the end of the Nix Packages (`.env/fix.flake`) as follows:

```nix
...
packages = with pkgs; [
  ...
  zlib
  (texlive.combine {        # <-- from here
    inherit (texlive)
    scheme-medium
    collection-latexextra
    cm-super;
  })                        # <-- to here
];
...
```

This will automatically install $\LaTeX$ once the environment is (re-)activated. To verify that $\LaTeX$ is installed, run `latex --version`.

#### Seed

Sets the seed for all random number generators (`random`, `numpy`, `torch`) to a fixed value.

#### Timer

`Timer` allows for timing, retrieving and saving the time of code blocks in different formats.

#### RunFilter

`RunFilter` implements the functionality to filter and skip runs by rules specified in the grid search configuration file (`_filters` list).
Additionally, it checks if the run was already sucessfilly completed and if the run should be skipped.
