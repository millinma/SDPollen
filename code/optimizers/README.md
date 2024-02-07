### Optimizers

`OPTIMIZER_REGISTRY` exports a registry of optimizers. Optimizers can be any class that inherits from `torch.optim.Optimizer`.

All optimizers can be loaded from a pretrained checkpoint by specifying `pretrained` with the path to the checkpoint in the model configuration file.

Additionally, optimizers can implement a `custom_step` function, which is called instead of the default training step. This can be used to implement custom training steps for e.g. the `SAM` optimizer.
You dont have to worry about the device, since the model, data, targets, and optimizer are already moved to the correct device before calling the `custom_step` function.
The `custom_step` function should have the following signature:

```python
def custom_step(self, model, data, target, criterion)-> float:
    ...
    return loss.item()
```

#### Adam

See [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html).

#### SGD

See [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html).

#### SAM

See [Sharpness-Aware Minimization](https://github.com/davda54/sam).\
SAM relies on a second optimizer, which is used to compute the gradients for the sharpness-aware update.
The second optimizer is specified in the model configuration file and has to be in the registry.
