import torch
import numpy as np
from torch.nn import functional as F


def recon(output, target):
    """
    Treat q(x|z) as norm distribution
    :param output: Tensor. shape = (B, C, H, W)
    :param target: Tensor. shape = (B, C, H, W)
    :return:
    """

    loss = F.mse_loss(output, target)
    return loss


def kl(mu, log_var):
    """

    :param mu:
    :param log_var:
    :return:
    """
    loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=1)
    return torch.mean(loss, dim=0)


def kl_2(delta_mu, delta_log_var, mu, log_var):
    var = torch.exp(log_var)
    delta_var = torch.exp(delta_log_var)

    loss = -0.5 * torch.sum(1 + delta_log_var - delta_mu ** 2 / var - delta_var)
    return torch.mean(loss, dim=0)
