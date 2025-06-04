import torch
import torch.nn as nn
import torch.nn.functional as F

class CAM(nn.Module):
    def __init__(self, in_channels, num_classes=21):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.context_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k // 2)
            for k in [1, 3, 5]
        ])
        self.fuse = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.cls_map = nn.Conv2d(in_channels, num_classes + 1, kernel_size=1)
        self.score_sep = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ctx_feats = [conv(x) for conv in self.context_convs]
        ctx = torch.cat(ctx_feats, dim=1)
        ctx = self.fuse(ctx)
        fcls = self.cls_map(ctx)
        sep_score = self.score_sep(ctx)
        weight = self.sigmoid(sep_score)
        return x * weight
