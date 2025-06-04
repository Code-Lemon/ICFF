import torch
import torch.nn as nn
import torch.nn.functional as F

class FHM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, low_feat, up_feat):
        up_feat = F.interpolate(up_feat, size=low_feat.shape[-2:], mode='nearest')
        avg = torch.mean(up_feat, dim=1, keepdim=True)
        max_ = torch.max(up_feat, dim=1, keepdim=True)[0]
        mask_large = self.sigmoid(avg * max_)
        mask_small = 1 - mask_large
        out = low_feat * mask_small
        return out, mask_large
