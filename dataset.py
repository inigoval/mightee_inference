import numpy as np
import pandas as pd

from cata2data import CataData
from astropy.stats import sigma_clipped_stats
from typing import Optional
from pathlib import Path
from astropy.wcs import WCS


def image_preprocessing(image: np.ndarray, field: str) -> np.ndarray:
    """Example preprocessing function for basic images.
    Args:
        image (np.ndarray): image
        field (str): Not Implemented here, but will be passed.
    Returns:
        np.ndarray: Squeezed image. I.e. removed empty axis.
    """
    return np.squeeze(image)


def wcs_preprocessing(wcs, field: str):
    """Example preprocessing function for wcs (world coordinate system).
    Args:
        wcs: Input wcs.
        field (str): field name matching the respective wcs.
    Returns:
        Altered wcs.
    """
    if field in ["COSMOS"]:
        return (wcs.dropaxis(3).dropaxis(2),)
    elif field in ["XMMLSS"]:
        raise UserWarning(
            f"This may cause issues in the future. It is unclear where header would have been defined."
        )
        wcs = WCS(header, naxis=2)  # This surely causes a bug right?
    else:
        return wcs


class catalogue_preprocessing:
    def __init__(self, set: str = "certain", random_state: Optional[int] = None):
        """
        Args:
            set (str, optional): Set to use. Defaults to "certain".
            random_state (Optional[int], optional): Random state. Defaults to None.
        """
        self.set = set
        self.df_zoo = pd.read_parquet("zooniverse_mightee_classifications.parquet")
        # Filter down to sources with predictions, cuts on other things have been made already for
        # the zooniverse data set.
        self.df_zoo = pd.read_parquet("zooniverse_mightee_classifications.parquet")
        self.df_zoo = self.df_zoo[["filename", "majority_classification", "vote_fraction"]]
        self.df_zoo = self.df_zoo.rename({"filename": "NAME"}, axis="columns", errors="raise")

        # Cut out certain/uncertain samples
        if set == "certain":
            self.df_zoo = self.df_zoo.query("vote_fraction > 0.65")
        elif set == "uncertain":
            self.df_zoo = self.df_zoo.query("vote_fraction <= 0.65 and vote_fraction > 0.5")
        elif set == "all":
            self.df_zoo = self.df_zoo.query("vote_fraction >= 0.5")
        else:
            raise ValueError(f"Invalid set: {set}")

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(self.df_zoo, on="NAME", how="inner")

        # Map FRI to label 0 and FRII to label 1
        df["y"] = df["majority_classification"].map({"FRI": 0, "FRII": 1})

        return df.reset_index(drop=True)


class MighteeZoo:
    def __init__(self, path: Path, transform, set: str = "certain"):
        """
        Args:
            path (Path): Path to the data set.
            transform (torchvision.transforms): Transform to apply to the images.
            set (str, optional): Which set to use. ["certain", "uncertain", "all"].

        """
        self.transform = transform

        catalogue_paths = [
            path / "COSMOS_source_catalogue.fits",
            path / "XMMLSS_source_catalogue.fits",
        ]
        image_paths = [
            path / "COSMOS_image.fits",
            path / "XMMLSS_image.fits",
        ]

        field_names = ["COSMOS", "XMMLSS"]

        # Create Data Set #
        self.catadata = CataData(
            catalogue_paths=catalogue_paths,
            image_paths=image_paths,
            field_names=field_names,
            cutout_width=114,
            catalogue_preprocessing=catalogue_preprocessing(set=set),
            image_preprocessing=image_preprocessing,
        )

    def __getitem__(self, index: int) -> tuple:
        # rms = self.catadata.df.loc[index, "ISL_RMS"]

        img = self.catadata[index]

        _, _, rms = sigma_clipped_stats(img)

        # Clip values below 3 sigma
        img[np.where(img <= 3 * rms)] = 0.0

        # Remove NaNs
        img = np.nan_to_num(img, nan=0.0)

        # Apply transform
        img = self.transform(np.squeeze(img))

        y = self.catadata.df.loc[index, "y"]

        return (img, y)

    def __len__(self):
        return self.catadata.__len__()
