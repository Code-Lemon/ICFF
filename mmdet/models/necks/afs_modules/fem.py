import torch
import torch.nn as nn
import torch.nn.functional as F

class FEM(nn.Module):
    def __init__(self, in_channels_low, in_channels_high):
        super().__init__()
        self.align_conv = nn.Conv2d(in_channels_low, in_channels_high, kernel_size=1)

    def forward(self, high_feat, low_feat, mask_large):
        plarge = low_feat * mask_large
        plarge_down = F.adaptive_avg_pool2d(plarge, output_size=high_feat.shape[-2:])
        plarge_down = self.align_conv(plarge_down)  # ✅ 通道对齐
        out = high_feat + plarge_down
        return out

