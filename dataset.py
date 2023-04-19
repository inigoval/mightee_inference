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


def catalogue_preprocessing(df: pd.DataFrame, random_state: Optional[int] = None) -> pd.DataFrame:
    """Example Function to make preselections on the catalog to specific
    sources meeting given criteria.
    Args:
        df (pd.DataFrame): Data frame containing catalogue information.
        random_state (Optional[int], optional): Random state seed. Defaults to None.
    Returns:
        pd.DataFrame: Subset catalogue.
    """
    # Filter down to sources with predictions, cuts on other things have been made already for
    # the zooniverse data set.
    df_zoo = pd.read_parquet("zooniverse_mightee_classifications.parquet")
    df_zoo = df_zoo[["filename", "majority_classification", "vote_fraction"]]
    df_zoo = df_zoo.rename({"filename": "NAME"}, axis="columns", errors="raise")
    df = df.merge(df_zoo, on="NAME", how="inner")

    # Map FRI to label 0 and FRII to label 1
    df["y"] = df["majority_classification"].map({"FRI": 0, "FRII": 1})

    return df.reset_index(drop=True)


class MighteeZoo:
    def __init__(self, path: Path, transform):
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
            catalogue_preprocessing=catalogue_preprocessing,
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
