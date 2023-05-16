import yaml
import torch
import numpy as np

from torch.utils.data import DataLoader


def compute_mu_sig_images(dset, batch_size = 256):
    """
    Compute mean and standard variance of a dataset (use batching with large datasets)
    """
    # Load samples in batches
    n_dset = len(dset)
    loader = DataLoader(dset, batch_size)
    n_channels = next(iter(loader))[0].shape[1]

    # Calculate mean
    mu = torch.zeros(n_channels)
    for x, _ in loader:
        for c in np.arange(n_channels):
            x_c = x[:, c, :, :]
            weight = x.shape[0] / n_dset
            mu[c] += weight * torch.mean(x_c).item()

    # Calculate std
    D_sq = torch.zeros(n_channels)
    for x, _ in loader:
        for c in np.arange(n_channels):
            x_c = x[:, c, :, :]
            D_sq += torch.sum((x_c - mu[c]) ** 2)
    sig = (D_sq / (n_dset * x.shape[-1] * x.shape[-2])) ** 0.5

    mu, sig = tuple(mu.tolist()), tuple(sig.tolist())
    print(f"mu: {mu}, std: {sig}")
    return mu, sig

