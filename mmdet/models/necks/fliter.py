# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor
from mmdet.registry import MODELS


@MODELS.register_module()
class Frequency_Filter(BaseModule):
    def __init__(self,
                 feat_num=5,
                 in_channels=256,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat_num = feat_num
        self.in_channels = in_channels
        self.filter = nn.ModuleList()
        for i in range(self.feat_num):
            filter = AFF(self.in_channels)
            self.filter.append(filter)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        outs = [
            self.filter[i](x[i]) for i in range(self.feat_num)
        ]
        return tuple(outs)


class AFF(nn.Module):
    def __init__(self, in_channels):
        super(AFF, self).__init__()
        self.real_filter = nn.Parameter(torch.randn(1, in_channels, 1, 1))
        self.imag_filter = nn.Parameter(torch.randn(1, in_channels, 1, 1))
        # Optional gating mechanism
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.ones(2))

    def forward(self, x):
        # 2D FFT
        x_freq = torch.fft.fft2(x, norm='ortho')
        real, imag = x_freq.real, x_freq.imag

        # Filter in frequency domain
        real_f = real * self.real_filter - imag * self.imag_filter
        imag_f = real * self.imag_filter + imag * self.real_filter
        x_filtered_freq = torch.complex(real_f, imag_f)

        # Inverse FFT back to spatial domain
        x_out = torch.fft.ifft2(x_filtered_freq, norm='ortho').real
        # Optional: spatial gate for residual
        gate_weight = self.gate(x)
        out = x + gate_weight * x_out  # residual + gated filtered output
        return out
