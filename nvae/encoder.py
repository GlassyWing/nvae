import torch
import torch.nn as nn

from nvae.common import EncoderResidualBlock, Swish


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self._seq = nn.Sequential(

            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=1),
            nn.BatchNorm2d(out_channel // 2), Swish(),
            nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel), Swish()
        )

    def forward(self, x):
        return self._seq(x)


class EncoderBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        modules = []
        for i in range(len(channels) - 1):
            modules.append(ConvBlock(channels[i], channels[i + 1]))

        self.modules_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x


class Encoder(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock([3, z_dim // 16, z_dim // 8]),  # (16, 16)
            EncoderBlock([z_dim // 8, z_dim // 4, z_dim // 2]),  # (4, 4)
            EncoderBlock([z_dim // 2, z_dim]),  # (2, 2)
        ])

        self.encoder_residual_blocks = nn.ModuleList([
            EncoderResidualBlock(z_dim // 8),
            EncoderResidualBlock(z_dim // 2),
            EncoderResidualBlock(z_dim),
        ])

        self.condition_x = nn.Sequential(
            Swish(),
            nn.Conv2d(z_dim, z_dim * 2, kernel_size=1)
        )

    def forward(self, x):
        xs = []
        last_x = x
        for e, r in zip(self.encoder_blocks, self.encoder_residual_blocks):
            x = r(e(x))
            last_x = x
            xs.append(x)

        mu, log_var = self.condition_x(last_x).chunk(2, dim=1)

        return mu, log_var, xs[:-1][::-1]
