import torch
import torchvision.transforms as T
import numpy as np

from torch.utils.data import DataLoader
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from dataset import MighteeZoo
from paths import Path_Handler

transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize(70),  # Rescale to adjust for resolution difference between MIGHTEE & RGZ
        T.Normalize((1.59965605788234e-05,), (0.0038063037602458706,)),
    ]
)

set = 'certain'

paths = Path_Handler()._dict()

data = MighteeZoo(path=paths["mightee"], transform=transform, set="certain")

print(f"Length of data set: {len(data)}")
print("Printing first batch:")
print(next(iter(DataLoader(data, batch_size=1))))

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img

for b, (im_batch, y_batch) in enumerate(DataLoader(data, batch_size=16)):
        fig = plt.figure(figsize=(13.0, 13.0))
        fig.subplots_adjust(0, 0, 1, 1)
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0)
        
        for i, (ax, im, y) in enumerate(zip(grid, list(im_batch), list(y_batch))):

            # im = im_batch[i].squeeze().numpy() 
            im = im.squeeze().numpy()
            
            ax.axis("off")
            text = f"FR{y+1}"

            ax.text(1, 66, text, fontsize=23, color="yellow")

            # contours
            threshold = 1
            ax.contour(np.where(im > threshold, 1, 0), cmap="cool", alpha=0.1)

            ax.imshow(im, cmap="hot")

        plt.axis("off")
        pil_img = fig2img(fig)
        plt.savefig(f"samples/{set}_{b}.png")
        plt.close(fig)
