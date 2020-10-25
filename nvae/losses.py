from functools import reduce

import numpy as np
import torch
from torch.nn import functional as F


def recon(output, target):
    """
    recon loss
    :param output: Tensor. shape = (B, C, H, W)
    :param target: Tensor. shape = (B, C, H, W)
    :return:
    """

    # Treat q(x|z) as Norm distribution
    # loss = F.mse_loss(output, target)

    # Treat q(x|z) as Bernoulli distribution.
    loss = F.binary_cross_entropy(output, target)
    return loss


def kl(mu, log_var):
    """
    kl loss with standard norm distribute
    :param mu:
    :param log_var:
    :return:
    """
    loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=[1, 2, 3])
    return torch.mean(loss, dim=0)


def kl_2(delta_mu, delta_log_var, mu, log_var):
    var = torch.exp(log_var)
    delta_var = torch.exp(delta_log_var)

    loss = -0.5 * torch.sum(1 + delta_log_var - delta_mu ** 2 / var - delta_var, dim=[1, 2, 3])
    return torch.mean(loss, dim=0)


def log_sum_exp(x):
    """

    :param x: Tensor. shape = (batch_size, num_mixtures, height, width)
    :return:
    """

    m2 = torch.max(x, dim=1, keepdim=True)[0]
    m = m2.unsqueeze(1)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=1))


def discretized_mix_logistic_loss(y_hat: torch.Tensor, y: torch.Tensor, num_classes=256, log_scale_min=-7.0):
    """Discretized mix of logistic distributions loss.

    Note that it is assumed that input is scaled to [-1, 1]



    :param y_hat: Tensor. shape=(batch_size, 3 * num_mixtures * img_channels, height, width), predict output.
    :param y: Tensor. shape=(batch_size, img_channels, height, width), Target.
    :return: Tensor loss
    """

    # unpack parameters, [batch_size, num_mixtures * img_channels, height, width] x 3
    logit_probs, means, log_scales = y_hat.chunk(3, dim=1)
    log_scales = torch.clamp_max(log_scales, log_scale_min)

    num_mixtures = y_hat.size(1) // y.size(1) // 3

    B, C, H, W = y.shape
    y = y.unsqueeze(1).repeat(1, num_mixtures, 1, 1, 1).permute(0, 2, 1, 3, 4).reshape(B, -1, H, W)

    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    log_pdf_mid = min_in - log_scales - 2. * F.softplus(mid_in)

    log_probs = torch.where(y < -0.999, log_cdf_plus,
                            torch.where(y > 0.999, log_one_minus_cdf_min,
                                        torch.where(cdf_delta > 1e-5, torch.clamp_max(cdf_delta, 1e-12),
                                                    log_pdf_mid - np.log((num_classes - 1) / 2))))

    # (batch_size, num_mixtures * img_channels, height, width)
    log_probs = log_probs + F.softmax(log_probs, dim=1)

    log_probs = [log_sum_exp(log_prob) for log_prob in log_probs.chunk(y.size(1), dim=1)]
    log_probs = reduce(lambda a, b: a + b, log_probs)

    return -torch.sum(log_probs)
