"""PyTorch Dataset for pasture biomass prediction."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import torchvision.transforms as T

from .tabular_encoder import TabularPreprocessor


TARGET_NAMES = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]
BASE_TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"]


class PastureDataset(Dataset):
    """Dataset that loads pasture images and returns all 5 biomass targets.

    Each sample is one image with its 5 target values and tabular features.
    Since each image appears 5 times in the CSV (once per target), this dataset
    works at the image level and returns a dict with all 5 targets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        preprocessor: TabularPreprocessor,
        image_size: Tuple[int, int] = (512, 256),
        transforms: Optional[T.Compose] = None,
        use_grid_features: bool = False,
        grid_rows: int = 4,
        grid_cols: int = 2,
        use_color_histogram: bool = False,
    ):
        """
        Args:
            df: Dataframe with all 1785 rows (or a subset).
            image_dir: Directory containing the .jpg images.
            preprocessor: Fitted TabularPreprocessor.
            image_size: (width, height) to resize images to.
            transforms: torchvision transforms (augmentation or normalization).
            use_grid_features: Extract features from image grid tiles.
            grid_rows, grid_cols: Grid dimensions for tile extraction.
            use_color_histogram: Compute RGB color histograms as features.
        """
        super().__init__()
        self.df = df
        self.image_dir = Path(image_dir)
        self.preprocessor = preprocessor
        self.image_size = image_size
        self.transforms = transforms
        self.use_grid_features = use_grid_features
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.use_color_histogram = use_color_histogram

        # Build image-level index: one entry per unique image
        self.image_paths = sorted(self.df["image_path"].unique())
        self._build_image_targets()

    def _build_image_targets(self):
        """Pre-compute target arrays for each image."""
        self.targets = {}
        self.tabular_rows = {}  # Store one representative row per image

        for img_path in self.image_paths:
            img_df = self.df[self.df["image_path"] == img_path]
            target_dict = {}
            for _, row in img_df.iterrows():
                target_dict[row["target_name"]] = row["target"]
            self.targets[img_path] = np.array(
                [target_dict.get(t, np.nan) for t in TARGET_NAMES], dtype=np.float32
            )
            self.tabular_rows[img_path] = img_df.iloc[0]

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, img_path: str) -> Image.Image:
        """Load a single RGB image."""
        full_path = self.image_dir / Path(img_path).name
        img = Image.open(full_path).convert("RGB")
        return img

    def _extract_grid_tiles(self, img: Image.Image) -> torch.Tensor:
        """Split image into grid_rows × grid_cols tiles and return as a batch tensor."""
        w, h = img.size
        tile_w, tile_h = w // self.grid_cols, h // self.grid_rows
        tiles = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                left, top = c * tile_w, r * tile_h
                tile = img.crop((left, top, left + tile_w, top + tile_h))
                if self.transforms:
                    tile = self.transforms(tile)
                tiles.append(tile)
        return torch.stack(tiles, dim=0)  # (grid_rows * grid_cols, 3, H, W)

    def _compute_color_histogram(self, img: Image.Image) -> torch.Tensor:
        """Compute 64-bin RGB histogram as a normalized feature vector."""
        arr = np.array(img.resize((256, 128)))  # Downscale for speed
        hist_features = []
        for c in range(3):  # R, G, B channels
            hist, _ = np.histogram(arr[:, :, c], bins=64, range=(0, 255), density=True)
            hist_features.extend(hist)
        return torch.tensor(hist_features, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.image_paths[idx]
        img = self._load_image(img_path)
        original_img = img.copy()

        # Apply transforms
        if self.transforms:
            img_tensor = self.transforms(img)
        else:
            img_tensor = T.ToTensor()(img)

        # Tabular features
        row = self.tabular_rows[img_path]
        tab_feats = self.preprocessor.transform(row)
        tab_tensor = torch.tensor(tab_feats, dtype=torch.float32)

        # Targets
        targets = torch.tensor(self.targets[img_path], dtype=torch.float32)

        result = {
            "image": img_tensor,
            "tabular": tab_tensor,
            "targets": targets,  # [Clover, Dead, Green, Total, GDM]
            "image_path": img_path,
        }

        # Grid features
        if self.use_grid_features:
            from .augmentations import get_val_transforms
            grid_transform = get_val_transforms((self.image_size[0] // self.grid_cols,
                                                  self.image_size[1] // self.grid_rows))
            result["grid_tiles"] = self._extract_grid_tiles(original_img)

        # Color histogram
        if self.use_color_histogram:
            result["color_hist"] = self._compute_color_histogram(original_img)

        return result


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for PastureDataset."""
    images = torch.stack([item["image"] for item in batch])
    tabular = torch.stack([item["tabular"] for item in batch])
    targets = torch.stack([item["targets"] for item in batch])
    paths = [item["image_path"] for item in batch]

    result = {
        "image": images,
        "tabular": tabular,
        "targets": targets,
        "image_path": paths,
    }

    # Optional grid tiles
    if "grid_tiles" in batch[0]:
        result["grid_tiles"] = torch.stack([item["grid_tiles"] for item in batch])

    # Optional color histograms
    if "color_hist" in batch[0]:
        result["color_hist"] = torch.stack([item["color_hist"] for item in batch])

    return result


def create_dataloaders(
    df: pd.DataFrame,
    image_dir: Path,
    preprocessor: TabularPreprocessor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 256),
    use_grid_features: bool = False,
    use_color_histogram: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    from .augmentations import get_train_transforms, get_val_transforms

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    train_dataset = PastureDataset(
        df=train_df,
        image_dir=image_dir,
        preprocessor=preprocessor,
        image_size=image_size,
        transforms=train_transform,
        use_grid_features=use_grid_features,
        use_color_histogram=use_color_histogram,
    )

    val_dataset = PastureDataset(
        df=val_df,
        image_dir=image_dir,
        preprocessor=preprocessor,
        image_size=image_size,
        transforms=val_transform,
        use_grid_features=use_grid_features,
        use_color_histogram=use_color_histogram,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    return train_loader, val_loader
