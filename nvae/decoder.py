import torch.nn as nn
import torch

from nvae.common import EncoderResidualBlock, Swish, DecoderResidualBlock, ResidualBlock
from nvae.losses import kl_2
from nvae.utils import reparameterize


def create_grid(h, w, device):
    grid_y, grid_x = torch.meshgrid([torch.linspace(-1, 1, steps=h),
                                     torch.linspace(-1, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self._seq = nn.Sequential(

            nn.ConvTranspose2d(in_channel,
                               out_channel,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(out_channel), Swish(),
        )

    def forward(self, x):
        return self._seq(x)


class DecoderBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        modules = []
        for i in range(len(channels) - 1):
            modules.append(UpsampleBlock(channels[i], channels[i + 1]))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x


class Decoder(nn.Module):

    def __init__(self, z_dim):
        super().__init__()

        # Input channels = z_channels * 2 = x_channels + z_channels
        # Output channels = z_channels
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock([z_dim * 2, z_dim // 2]),  # 2x upsample
            DecoderBlock([z_dim, z_dim // 4, z_dim // 8]),  # 4x upsample
            DecoderBlock([z_dim // 4, z_dim // 16, z_dim // 32])  # 4x uplsampe
        ])
        self.decoder_residual_blocks = nn.ModuleList([
            DecoderResidualBlock(z_dim // 2, n_group=1),
            DecoderResidualBlock(z_dim // 8, n_group=2),
            DecoderResidualBlock(z_dim // 32, n_group=4)
        ])

        # p(z_l | z_(l-1))
        self.condition_z = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(z_dim // 2),
                nn.AdaptiveAvgPool2d(1),
                Swish(),
                nn.Conv2d(z_dim // 2, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim // 8),
                nn.AdaptiveAvgPool2d(1),
                Swish(),
                nn.Conv2d(z_dim // 8, z_dim // 4, kernel_size=1)
            )
        ])

        # p(z_l | x, z_(l-1))
        self.condition_xz = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(z_dim),
                nn.Conv2d(z_dim, z_dim // 2, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
                Swish(),
                nn.Conv2d(z_dim // 2, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim // 4),
                nn.Conv2d(z_dim // 4, z_dim // 8, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
                Swish(),
                nn.Conv2d(z_dim // 8, z_dim // 4, kernel_size=1)
            )
        ])

        self.map_from_z = nn.ModuleList([
            nn.Conv2d(z_dim + 2, z_dim, kernel_size=1),
            nn.Conv2d(z_dim // 2 + 2, z_dim // 2, kernel_size=1),
            nn.Conv2d(z_dim // 8 + 2, z_dim // 8, kernel_size=1)
        ])

        self.recon = nn.Conv2d(z_dim // 32, 3, kernel_size=1)
        self.pos_map = nn.Linear(2, 2)

    def forward(self, z, xs=None):
        """

        :param z: shape. = (B, z_dim)
        :return:
        """

        B, D = z.shape

        map_h, map_w = 2, 2
        decoder_out = torch.zeros(B, D, map_h, map_w, device=z.device, dtype=z.dtype)

        kl_losses = []

        for i in range(len(self.map_from_z)):
            # (B, m_h, m_w, z_dim)
            z_rep = z.unsqueeze(1).repeat(1, map_h * map_w, 1).reshape(z.shape[0], map_h, map_w, z.shape[1])

            # (B, m_h, m_w, 2)
            grid = create_grid(map_h, map_w, z.device).unsqueeze(0).repeat(z.shape[0], 1, 1, 1)
            grid = self.pos_map(grid)

            # (B, z_dim, m_h, m_w)
            z_sample = self.map_from_z[i](torch.cat([z_rep, grid], dim=-1).permute(0, 3, 1, 2).contiguous())

            z_sample = torch.cat([decoder_out, z_sample], dim=1)
            decoder_out = self.decoder_residual_blocks[i](self.decoder_blocks[i](z_sample))

            if i == len(self.map_from_z) - 1:
                break

            mu, log_var = self.condition_z[i](decoder_out).squeeze(-1).squeeze(-1).chunk(2, dim=-1)

            if xs is not None:
                delta_mu, delta_log_var = self.condition_xz[i](torch.cat([xs[i], decoder_out], dim=1)) \
                    .squeeze(-1).squeeze(-1).chunk(2, dim=-1)
                kl_losses.append(kl_2(delta_mu, delta_log_var, mu, log_var))
                mu = mu + delta_mu
                log_var = log_var + delta_log_var

            z = reparameterize(mu, torch.exp(0.5 * log_var))
            map_h *= 2 ** (len(self.decoder_blocks[i].channels) - 1)
            map_w *= 2 ** (len(self.decoder_blocks[i].channels) - 1)

        x_hat = self.recon(decoder_out)

        return x_hat, kl_losses
