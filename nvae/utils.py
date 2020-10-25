import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm


def add_sn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return spectral_norm(m)
    else:
        return m


def reparameterize(mu, std):
    z = torch.randn_like(mu) * std + mu
    return z


def create_grid(h, w, device):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def random_uniform_like(tensor, min_val, max_val):
    return (max_val - min_val) * torch.rand_like(tensor) + min_val


def sample_from_discretized_mix_logistic(y, img_channels=3, log_scale_min=-7.):
    """

    :param y: Tensor, shape=(batch_size, 3 * num_mixtures * img_channels, height, width),
    :return: Tensor: sample in range of [-1, 1]
    """

    # unpack parameters, [batch_size, num_mixtures * img_channels, height, width] x 3
    logit_probs, means, log_scales = y.chunk(3, dim=1)

    temp = random_uniform_like(logit_probs, min_val=1e-5, max_val=1. - 1e-5)
    temp = logit_probs - torch.log(-torch.log(temp))

    ones = torch.eye(means.size(1) // img_channels, dtype=means.dtype, device=means.device)

    sample = []
    for logit_prob, mean, log_scale, tmp in zip(logit_probs.chunk(img_channels, dim=1),
                                                means.chunk(img_channels, dim=1),
                                                log_scales.chunk(img_channels, dim=1),
                                                temp.chunk(img_channels, dim=1)):
        # (batch_size, height, width)
        argmax = torch.max(tmp, dim=1)[1]
        B, H, W = argmax.shape

        one_hot = ones.index_select(0, argmax.flatten())
        one_hot = one_hot.view(B, H, W, mean.size(1)).permute(0, 3, 1, 2).contiguous()

        # (batch_size, 1, height, width)
        mean = torch.sum(mean * one_hot, dim=1)
        log_scale = torch.clamp_max(torch.sum(log_scale * one_hot, dim=1), log_scale_min)

        u = random_uniform_like(mean, min_val=1e-5, max_val=1. - 1e-5)
        x = mean + torch.exp(log_scale) * (torch.log(u) - torch.log(1 - u))
        sample.append(x)

    # (batch_size, img_channels, height, width)
    sample = torch.stack(sample, dim=1)

    return sample
