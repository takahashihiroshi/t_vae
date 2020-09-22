import math

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def from_numpy(array, device, dtype=np.float32):
    return torch.from_numpy(array.astype(dtype)).to(device)


def to_numpy(tensor, device):
    if device.type == 'cuda':
        tensor = tensor.cpu()

    return tensor.data.numpy()


def make_data_loader(array, device, batch_size):
    return DataLoader(
        TensorDataset(from_numpy(array, device)),
        batch_size=batch_size, shuffle=True)


def reparameterize(mu, ln_var):
    std = torch.exp(0.5 * ln_var)
    eps = torch.randn_like(std)
    z = mu + std * eps
    return z


def gaussian_nll(x, mu, ln_var, dim=1):
    prec = torch.exp(-1 * ln_var)
    x_diff = x - mu
    x_power = (x_diff * x_diff) * prec * -0.5
    return torch.sum((ln_var + math.log(2 * math.pi)) * 0.5 - x_power, dim=dim)


def standard_gaussian_nll(x, dim=1):
    return torch.sum(0.5 * math.log(2 * math.pi) + 0.5 * x * x, dim=dim)


def students_t_nll(x, ln_df, loc, ln_scale, dim=1):
    nll = torch.lgamma((torch.exp(ln_df) + 1) / 2) - torch.lgamma(torch.exp(ln_df) / 2)
    nll += (ln_scale - math.log(math.pi) - ln_df) / 2
    nll -= (torch.exp(ln_df) + 1) * torch.log(1 + (torch.exp(ln_scale) * ((x - loc) ** 2) / torch.exp(ln_df))) / 2
    return torch.sum(-1 * nll, dim=dim)


def gaussian_kl_divergence(mu, ln_var, dim=1):
    return torch.sum(-0.5 * (1 + ln_var - mu.pow(2) - ln_var.exp()), dim=dim)
