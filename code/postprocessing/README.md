### Postprocessing

Postprocessing can be any kind of analysis of the training results or a grid search.
It is run after the training is finished and the results are stored in `${results_dir}/${experiment_id}/${postprocessing_type}` etc..

#### Summarize

Summarizes the metrics and configurations of the runs and plots each metric.

#### Aggregate

Aggregates runs by a specified key (or keys), summarizes the metrics and configurations of the runs and plots each metric (using Summarize).
Additionally, the aggregated runs are logged to MLflow for easy comparison.

Summarize and Aggregate are already combined in the `code/postprocessing.py` script.
For example, a grid search could be summarized and aggregated by `seed` and `optimizer and seed` in a DVC pipeline like this:

```bash
stages:
  ...
  postprocessing:
    cmd: >
      python code/postprocessing.py
      -rd=${results_dir}
      -id=${experiment_id}
      -agg seed
      -agg optimizer seed
    ...
```
