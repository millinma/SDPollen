import torch
import optimizers


class SAM(torch.optim.Optimizer):

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # ? only support optimizers in the registry for now
        self.base_optimizer = optimizers.OPTIMIZER_REGISTRY(
            name=base_optimizer,
            params=self.param_groups,
            **kwargs,
        )
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2)
                       if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    # @torch.no_grad()
    # def first_step_uphill(self, zero_grad=False):
    #     grad_norm = self._grad_norm()
    #     for group in self.param_groups:
    #         scale = group["rho"] / (grad_norm + 1e-12)

    #         for p in group["params"]:
    #             if p.grad is None: continue
    #             self.state[p]["old_p"] = p.data.clone()
    #             e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
    #             p.add_(e_w)  # climb to the local maximum "w + e(w)"

    #     if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # get back to "w" from "w + e(w)"
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # the closure should do a full forward-backward pass
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0)
                         * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                        ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    # ? Training Tips functions from the Github repo ref: https://github.com/davda54/sam
    # TODO: Check if Batch Norm Tip needs to be applied
    def custom_step(self, model, data, target, criterion):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        self.first_step(zero_grad=True)

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        self.second_step(zero_grad=True)
        _loss = loss.item()
        return _loss
