import torch
import torchvision.transforms as T
import numpy as np

from torch.utils.data import DataLoader

from dataset import MighteeZoo
from paths import Path_Handler

transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize(70),  # Rescale to adjust for resolution difference between MIGHTEE & RGZ
        T.Normalize((1.59965605788234e-05,), (0.0038063037602458706,)),
    ]
)

paths = Path_Handler()._dict()

data = MighteeZoo(path=paths["mightee"], transform=transform, set="uncertain")

print(f"Length of data set: {len(data)}")
print("Printing first batch:")
print(next(iter(DataLoader(data, batch_size=1))))
