"""Stratified Group K-Fold splitter that respects image boundaries."""
import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator
from sklearn.model_selection import StratifiedKFold, GroupKFold


class StratifiedGroupKFoldSplitter:
    """Custom cross-validation splitter that groups by image_id and
    attempts to stratify by biomass quantile × State."""

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, df: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, val_indices) for each fold.

        The indices refer to positions in the dataframe (all 1785 rows).
        Grouping by image_path ensures all 5 rows of an image stay together.
        """
        # Get unique images
        unique_images = df[["image_path"]].drop_duplicates()
        unique_images = unique_images.reset_index(drop=True)

        # Create stratification labels: biomass_quartile × State
        image_stats = df.groupby("image_path").agg(
            dry_total_mean=("target", lambda x: x[df.loc[x.index, "target_name"] == "Dry_Total_g"].values[0]),
            state=("State", "first"),
        ).reset_index()

        # Ensure alignment with unique_images
        image_stats = image_stats.set_index("image_path").loc[unique_images["image_path"]].reset_index()

        # Create strata
        total_bins = pd.qcut(image_stats["dry_total_mean"], q=min(4, self.n_splits),
                             labels=False, duplicates="drop")
        strata = total_bins.astype(str) + "_" + image_stats["state"]
        strata_cat = pd.Categorical(strata).codes

        # Use StratifiedKFold on unique images
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_img_idx, val_img_idx in skf.split(unique_images, strata_cat):
            train_images = set(unique_images.iloc[train_img_idx]["image_path"])
            val_images = set(unique_images.iloc[val_img_idx]["image_path"])

            train_idx = df[df["image_path"].isin(train_images)].index.values
            val_idx = df[df["image_path"].isin(val_images)].index.values

            yield train_idx, val_idx

    def get_image_splits(self, df: pd.DataFrame) -> Iterator[Tuple[List[str], List[str]]]:
        """Yield (train_image_paths, val_image_paths) for each fold."""
        unique_images = df[["image_path"]].drop_duplicates()["image_path"].tolist()

        # Create strata (same logic)
        image_stats = df.groupby("image_path").agg(
            dry_total_mean=("target", lambda x: x[df.loc[x.index, "target_name"] == "Dry_Total_g"].values[0]),
            state=("State", "first"),
        ).reset_index()

        image_stats = image_stats.set_index("image_path").loc[unique_images].reset_index()

        total_bins = pd.qcut(image_stats["dry_total_mean"], q=min(4, self.n_splits),
                             labels=False, duplicates="drop")
        strata = total_bins.astype(str) + "_" + image_stats["state"]
        strata_cat = pd.Categorical(strata).codes

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in skf.split(unique_images, strata_cat):
            train_imgs = [unique_images[i] for i in train_idx]
            val_imgs = [unique_images[i] for i in val_idx]
            yield train_imgs, val_imgs
