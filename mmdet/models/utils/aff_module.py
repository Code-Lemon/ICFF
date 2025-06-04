import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F


class AFF(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(AFF, self).__init__()
        self.mask_generator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Tanh()
        )
        self.alpha = nn.Parameter(torch.ones(2))

    def forward(self, x):
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_real = x_fft.real
        x_imag = x_fft.imag
        magnitude = torch.sqrt(x_real ** 2 + x_imag ** 2 + 1e-6)
        mask = self.mask_generator(magnitude)
        x_real_filtered = x_real * mask
        x_imag_filtered = x_imag * mask
        x_fft_filtered = torch.complex(x_real_filtered, x_imag_filtered)
        x_out = torch.fft.ifft2(x_fft_filtered, norm='ortho').real
        weights = F.sigmoid(self.alpha)
        x_out = weights[0] * x_out + weights[1] * x
        return x_out

if __name__ == "__main__":
    x = torch.randn(4, 64, 64, 64)  # batch=4, channels=64, size=64x64
    module = AFF(in_channels=64)
    y = module(x)
    print("Output shape:", y.shape)