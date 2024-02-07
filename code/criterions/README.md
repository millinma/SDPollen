### Criterions

`CRITERION_REGISTRY` exports a registry of criterions. Criterions can be any class or function that takes in the model output and the target as input to their `__call__` function and returns a `torch.Tensor` (the loss) as output.

Criterions are specified in the `dataset` configuration file, so a criterion is always dataset specific.
Hydra configurations can be found in the `conf/dataset/criterion` directory.

Weighted criterions are **still WIP**.

#### CrossEntropyLoss

See [torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss).

#### MSELoss

See [torch.nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html).
