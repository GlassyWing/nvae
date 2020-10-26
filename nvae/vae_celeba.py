import torch
import torch.nn as nn
import torch.nn.functional as F
from nvae.decoder import Decoder
from nvae.encoder import Encoder
from nvae.losses import recon, kl
from nvae.utils import reparameterize

import robust_loss_pytorch
import numpy as np


class NVAE(nn.Module):

    def __init__(self, z_dim, img_dim):
        super().__init__()

        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

        self.adaptive_loss = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims=1, float_dtype=np.float32, device="cpu")

    def forward(self, x):
        """

        :param x: Tensor. shape = (B, C, H, W)
        :return:
        """

        mu, log_var, xs = self.encoder(x)

        # (B, D_Z)
        z = reparameterize(mu, torch.exp(0.5 * log_var))

        decoder_output, losses = self.decoder(z, xs)

        # Treat p(x|z) as discretized_mix_logistic distribution cost so much, this is an alternative way
        # witch combine multi distribution.
        recon_loss = torch.mean(self.adaptive_loss.lossfun(
            torch.mean(F.binary_cross_entropy(decoder_output, x, reduction='none'), dim=[1, 2, 3])[:, None]))

        kl_loss = kl(mu, log_var)

        return decoder_output, recon_loss, [kl_loss] + losses


if __name__ == '__main__':
    vae = NVAE(512, (64, 64))
    img = torch.rand(2, 3, 64, 64)
    img_recon, vae_loss = vae(img)
    print(img_recon.shape)
    print(vae_loss)
