import torch
from efficientnet_pytorch import EfficientNet
from helpers import SuppressStdout


class ModifiedEfficientNet(torch.nn.Module):
    def __init__(self, output_dim, scaling_type="efficientnet-b0", transfer=False):
        super(ModifiedEfficientNet, self).__init__()
        self.output_dim = output_dim

        with SuppressStdout():
            if transfer:
                self.effnet = EfficientNet.from_pretrained(scaling_type)
            else:
                self.effnet = EfficientNet.from_name(scaling_type)

        self.effnet._fc = torch.nn.Linear(
            self.effnet._fc.in_features, output_dim)

    def forward(self, x):
        x = self.effnet(x)
        return x
