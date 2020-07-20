import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def add_sn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return spectral_norm(m)
    else:
        return m

def reparameterize(mu, std):
    z = torch.randn_like(mu) * std + mu
    return z
