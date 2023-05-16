import torch
import yaml
import torchvision.transforms as T
import logging
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics as tm

from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from torchmetrics.functional import accuracy
from torch.utils.data import DataLoader

from dataset import MighteeZoo
from paths import Path_Handler
from utilities import compute_mu_sig_images

from collections import Counter


def aggregate(preds: list) -> tuple:
    counter = Counter(preds)
    agg_pred, count = counter.most_common(1)[0]
    vote_frac = count / len(preds)

    return agg_pred, vote_frac


if __name__ == "__main__":
    # Get paths
    paths = Path_Handler()._dict()

    # Calculate norm from mightee data
    transform = T.Compose(
        [
            T.ToTensor(),
            # Rescale to adjust for resolution difference between MIGHTEE & RGZ
            T.Resize(70, antialias=None),
        ]
    )

    data = MighteeZoo(paths["mightee"], transform=transform, set="all")
    mu, sig = compute_mu_sig_images(data)

    print(f"MighteeZoo dataset | mean: {mu[0]}, std: {sig[0]}")
