import torch.nn as nn
from .fhm import FHM
from .fem import FEM

class SAM(nn.Module):
    def __init__(self, low_channels, high_channels):
        super().__init__()
        self.fhm = FHM(low_channels)
        self.fem = FEM(low_channels, high_channels)

    def forward(self, low_feat, high_feat):
        new_low, mask_large = self.fhm(low_feat, high_feat)
        new_high = self.fem(high_feat, low_feat, mask_large)
        return new_low, new_high
