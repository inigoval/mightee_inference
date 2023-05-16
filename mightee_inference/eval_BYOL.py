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


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.linear(x)
        return x


class FineTune(pl.LightningModule):
    """
    Parent class for self-supervised LightningModules to perform linear evaluation with multiple
    data-sets.
    """

    def __init__(
        self,
        encoder: nn.Module,
        dim: int,
        n_classes,
        n_epochs=100,
        n_layers=0,
        batch_size=1024,
        lr_decay=0.75,
        seed=69,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "head"])

        self.n_layers = n_layers
        self.batch_size = batch_size
        self.encoder = encoder
        self.lr_decay = lr_decay
        self.n_epochs = n_epochs
        self.seed = seed
        self.n_classes = n_classes
        self.layers = []

        self.head = LogisticRegression(input_dim=dim, output_dim=n_classes)
        self.head_type = "linear"

        # Set finetuning layers for easy access
        if self.n_layers:
            layers = self.encoder.finetuning_layers
            assert self.n_layers <= len(
                layers
            ), f"Network only has {len(layers)} layers, {self.n_layers} specified for finetuning"

            self.layers = layers[::-1][:n_layers]

        self.train_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)

        self.val_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)

        self.test_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        x = self.head(x)
        return x

    def on_fit_start(self):
        # Log size of data-sets #

        self.train_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)
        self.val_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)

        self.test_acc = nn.ModuleList(
            [
                tm.Accuracy(
                    task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
                ).to(self.device)
            ]
            * len(self.trainer.datamodule.data["test"])
        )

        logging_params = {f"n_{key}": len(value) for key, value in self.trainer.datamodule.data.items()}
        self.logger.log_hyperparams(logging_params)

        # Make sure network that isn't being finetuned is frozen
        # probably unnecessary but best to be sure
        set_grads(self.encoder, False)
        if self.n_layers:
            for layer in self.layers:
                set_grads(layer, True)

    def training_step(self, batch, batch_idx):
        # Load data and targets
        x, y = batch
        logits = self.forward(x)
        y_pred = logits.softmax(dim=-1)
        loss = F.cross_entropy(y_pred, y, label_smoothing=0.1 if self.n_layers else 0)
        self.log("finetuning/train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        preds = self.forward(x)
        self.val_acc(preds, y)
        self.log("finetuning/val_acc", self.val_acc, on_step=False, on_epoch=True)

    # def test_step(self, batch, batch_idx, dataloader_idx=0):
    #     x, y = batch
    #     preds = self.forward(x)
    #     self.test_acc(preds, y)
    #     self.log("finetuning/test_acc", self.test_acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        name = list(self.trainer.datamodule.data["test"].keys())[dataloader_idx]

        preds = self.forward(x)
        self.test_acc[dataloader_idx](preds, y)
        self.log(
            f"finetuning/test/{name}_acc", self.test_acc[dataloader_idx], on_step=False, on_epoch=True
        )

    def configure_optimizers(self):
        if not self.n_layers and self.head_type == "linear":
            # Scale base lr=0.1
            lr = 0.1 * self.batch_size / 256
            params = self.head.parameters()
            return torch.optim.SGD(params, momentum=0.9, lr=lr)
        else:
            lr = 0.001 * self.batch_size / 256
            params = [{"params": self.head.parameters(), "lr": lr}]
            # layers.reverse()

            # Append parameters of layers for finetuning along with decayed learning rate
            for i, layer in enumerate(self.layers):
                params.append({"params": layer.parameters(), "lr": lr * (self.lr_decay**i)})

            # Initialize AdamW optimizer with cosine decay learning rate
            opt = torch.optim.AdamW(params, weight_decay=0.05, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.n_epochs)
            return [opt], [scheduler]


def set_bn_train(module):
    """
    Recursively sets all BatchNorm2d layers in module to eval mode.
    """
    for child in module.children():
        if isinstance(child, nn.BatchNorm2d):
            child.train()
            child.reset_running_stats()
        else:
            set_bn_train(child)


def aggregate(preds: list) -> tuple:
    counter = Counter(preds)
    agg_pred, count = counter.most_common(1)[0]
    vote_frac = count / len(preds)

    return agg_pred, vote_frac


if __name__ == "__main__":
    # Get paths
    paths = Path_Handler()._dict()

    # Load config file
    # global_path = paths["main"] / "config.yml"
    with open("config.yml", "r") as ymlconfig:
        config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    if config["data"]["norm"] == "mightee":
        # Calculate norm from mightee data
        transform = T.Compose(
            [
                T.ToTensor(),
                # Rescale to adjust for resolution difference between MIGHTEE & RGZ
                T.Resize(70, antialias=None),
            ]
        )

        data = MighteeZoo(paths["mightee"], transform=transform, set=config["data"]["set"])
        mu, sig = compute_mu_sig_images(data)

    elif config["data"]["norm"] == "rgz":
        # Precalculated RGZ values
        mu, sig = ((0.008008896,), (0.05303395,))
    else:
        raise ValueError(f"Normalization {config['data']['norm']} not implemented")

    transform = T.Compose(
        [
            T.ToTensor(),
            # Rescale to adjust for resolution difference between MIGHTEE & RGZ
            T.Resize(70, antialias=None),
            T.Normalize(mu, sig),
        ]
    )

    data = MighteeZoo(paths["mightee"], transform=transform, set=config["data"]["set"])

    model_dir = paths["data"] / "weights" / "finetune"
    # Make predictions on RGZ data

    y = torch.tensor(data.get_labels()).flatten()
    preds = [[] for _ in model_dir.iterdir()]

    for i, model_path in enumerate(model_dir.iterdir()):
        print(f"Loading model {i}...")
        model = FineTune.load_from_checkpoint(model_path)
        encoder = model.encoder
        head = model.head

        network = torch.nn.Sequential(model.encoder, Rearrange("b c h w -> b (c h w)"), model.head)
        # encoder.eval()

        # head.eval()
        encoder.eval()
        head.eval()

        if config["bn"]:
            set_bn_train(encoder)
            set_bn_train(head)
        # set_bn_track_running_stats(encoder)

        print("Performing inference on mightee data...")
        for x, _ in DataLoader(data, batch_size=64, shuffle=False, pin_memory=False):
            x = x.float()

            # logits = head(encoder(x))
            logits = network(x)
            y_preds = logits.softmax(dim=-1)
            y_softmax, y_pred = torch.max(y_preds, dim=1)

            preds[i].append(y_pred)

        print(
            f"Model {i} accuracy: {accuracy(torch.cat(preds[i]), y, task='multiclass', num_classes=2).item()}"
        )

        preds[i] = torch.cat(preds[i], dim=0)

    preds = torch.stack(preds, dim=0)
    preds = rearrange(preds, "m n -> n m")
    preds = list(preds)
    preds = torch.tensor([torch.mode(pred).values for pred in preds]).flatten()

    print(f"Model accuracy: ", accuracy(preds, y, task="multiclass", num_classes=2).item())

    #
    # print("Aggregating predictions...")

    # df_vals = []
    # for name, value in tqdm(y.items()):
    #     preds, softmax = value["pred"], value["softmax"]
    #
    #     # print(preds, softmax)
    #
    #     agg_pred, vote_frac = aggregate(preds)

    #     df_vals.append([name, agg_pred, vote_frac])

    # df = pd.DataFrame(df_vals, columns=["name", "pred", "vote_frac"])

    # for i, (X, names) in tqdm(
    #     enumerate(DataLoader(mightee_data, batch_size=16, shuffle=False)), total=len(mightee_data)
    # ):
    #     fig = plt.figure(figsize=(13.0, 13.0))
    #     grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.1)

    #     for name, ax, im in zip(names, grid, list(X)):
    #         im = torch.squeeze(im)
    #         ax.axis("off")
    #         ax.imshow(im, cmap="hot")

    #         # Add text
    #         pred = df.loc[df["name"] == name, "pred"].values.item()
    #         vote_frac = df.loc[df["name"] == name, "vote_frac"].values.item()

    #         text = f"FR{pred}, C={vote_frac}"
    #         ax.text(1, 66, text, fontsize=23, color="yellow")

    #     plt.axis("off")
    #     plt.savefig(
    #         paths["main"] / "analysis" / "imgs" / "mightee" / f"grid_{i:03d}.png", bbox_inches="tight"
    #     )
    #     plt.close(fig)
    #     break

    #
