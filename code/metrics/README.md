### Metrics

`METRIC_REGISTRY` exports a registry of metrics. Metrics can be any class or function that takes in the model output and the target as input to their `__call__` function and returns a `float` (the metric) as output.