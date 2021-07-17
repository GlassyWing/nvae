import torch.nn as nn
import torch

from nvae.common import Swish, DecoderResidualBlock, ResidualBlock
from nvae.losses import kl_2
from nvae.utils import reparameterize


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
            # nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
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
            DecoderResidualBlock(z_dim // 2, n_group=4),
            DecoderResidualBlock(z_dim // 8, n_group=2),
            DecoderResidualBlock(z_dim // 32, n_group=1)
        ])

        # p(z_l | z_(l-1))
        self.condition_z = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(z_dim // 2),
                Swish(),
                nn.Conv2d(z_dim // 2, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim // 8),
                Swish(),
                nn.Conv2d(z_dim // 8, z_dim // 4, kernel_size=1)
            )
        ])

        # p(z_l | x, z_(l-1))
        self.condition_xz = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(z_dim),
                nn.Conv2d(z_dim, z_dim // 2, kernel_size=1),
                Swish(),
                nn.Conv2d(z_dim // 2, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim // 4),
                nn.Conv2d(z_dim // 4, z_dim // 8, kernel_size=1),
                Swish(),
                nn.Conv2d(z_dim // 8, z_dim // 4, kernel_size=1)
            )
        ])

        self.recon = nn.Sequential(
            ResidualBlock(z_dim // 32),
            nn.Conv2d(z_dim // 32, 3, kernel_size=1),
        )

        self.zs = []

    def forward(self, z, xs=None, mode="random", freeze_level=-1):
        """

        :param z: shape. = (B, z_dim, map_h, map_w)
        :return:
        """

        B, D, map_h, map_w = z.shape

        # The init h (hidden state), can be replace with learned param, but it didn't work much
        decoder_out = torch.zeros(B, D, map_h, map_w, device=z.device, dtype=z.dtype)

        kl_losses = []
        if freeze_level != -1 and len(self.zs) == 0 :
            self.zs.append(z)

        for i in range(len(self.decoder_residual_blocks)):

            z_sample = torch.cat([decoder_out, z], dim=1)
            decoder_out = self.decoder_residual_blocks[i](self.decoder_blocks[i](z_sample))

            if i == len(self.decoder_residual_blocks) - 1:
                break

            mu, log_var = self.condition_z[i](decoder_out).chunk(2, dim=1)

            if xs is not None:
                delta_mu, delta_log_var = self.condition_xz[i](torch.cat([xs[i], decoder_out], dim=1)) \
                    .chunk(2, dim=1)
                kl_losses.append(kl_2(delta_mu, delta_log_var, mu, log_var))
                mu = mu + delta_mu
                log_var = log_var + delta_log_var

            if mode == "fix" and i < freeze_level:
                if len(self.zs) < freeze_level + 1:
                    z = reparameterize(mu, 0)
                    self.zs.append(z)
                else:
                    z = self.zs[i + 1]
            elif mode == "fix":
                z = reparameterize(mu, 0 if i == 0 else torch.exp(0.5 * log_var))
            else:
                z = reparameterize(mu, torch.exp(0.5 * log_var))

            map_h *= 2 ** (len(self.decoder_blocks[i].channels) - 1)
            map_w *= 2 ** (len(self.decoder_blocks[i].channels) - 1)

        x_hat = torch.sigmoid(self.recon(decoder_out))

        return x_hat, kl_losses
