import torch
import torch.nn as nn

from nvae.decoder import Decoder
from nvae.encoder import Encoder
from nvae.losses import recon, kl
from nvae.utils import reparameterize


class NVAE(nn.Module):

    def __init__(self, z_dim, img_dim, M_N=0.005):
        super().__init__()

        self.M_N = M_N

        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        """

        :param x: Tensor. shape = (B, C, H, W)
        :return:
        """

        mu, log_var, xs = self.encoder(x)

        # (B, D_Z)
        z = reparameterize(mu, torch.exp(0.5 * log_var))

        decoder_output, losses = self.decoder(z, xs)

        recon_loss = recon(decoder_output, x)
        kl_loss = kl(mu, log_var)

        vae_loss = recon_loss + self.M_N * (kl_loss + 1 / 2 * losses[0] + 1 / 8 * losses[1])

        return decoder_output, vae_loss


if __name__ == '__main__':
    vae = NVAE(512, (64, 64))
    img = torch.rand(2, 3, 64, 64)
    img_recon, vae_loss = vae(img)
    print(img_recon.shape)
    print(vae_loss)
