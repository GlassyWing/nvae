import torch.nn as nn
import torch
import numpy as np

from nvae.utils import input_mapping


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim), Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self._seq(x)


# class FourierMapping(nn.Module):
#
#     def __init__(self, dims, seed):
#         super().__init__()
#         np.random.seed(seed)
#         B = np.random.randn(*dims) * 10
#         np.random.seed(None)
#         self.B = torch.tensor(B, dtype=torch.float32)
#
#     def forward(self, x):
#         x = input_mapping(x, self.B.to(x.device))
#         return x


class EncoderResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.seq = nn.Sequential(

            nn.Conv2d(dim, dim, kernel_size=5, padding=2),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim), Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self.seq(x)


class DecoderResidualBlock(nn.Module):

    def __init__(self, dim, n_group):
        super().__init__()

        self._seq = nn.Sequential(
            nn.Conv2d(dim, n_group * dim, kernel_size=1),
            nn.BatchNorm2d(n_group * dim), Swish(),
            nn.Conv2d(n_group * dim, n_group * dim, kernel_size=5, padding=2, groups=n_group),
            nn.BatchNorm2d(n_group * dim), Swish(),
            nn.Conv2d(n_group * dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self._seq(x)
