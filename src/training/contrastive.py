"""SimCLR-style contrastive pre-training for pasture images.

Trains the image encoder to produce rich representations without labels,
which is critical with only 357 labeled images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from tqdm import tqdm

from src.models.image_encoder import ImageEncoder
from src.data.augmentations import get_contrastive_transforms


class ContrastivePretrainDataset(Dataset):
    """Dataset for contrastive pre-training: returns two augmented views per image."""

    def __init__(self, image_paths: list, image_dir: Path,
                 transform=None, image_size=(512, 256)):
        self.image_paths = image_paths
        self.image_dir = Path(image_dir)
        self.transform = transform or get_contrastive_transforms(image_size)
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.image_dir / Path(self.image_paths[idx]).name
        img = Image.open(img_path).convert("RGB")

        # Two augmented views for SimCLR
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2


class ProjectionHead(nn.Module):
    """Projection head for SimCLR: maps encoder features to embedding space."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class SimCLR(nn.Module):
    """SimCLR wrapper combining encoder + projection head."""

    def __init__(self, encoder: ImageEncoder, projection_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.projection = ProjectionHead(
            input_dim=encoder.feature_dim,
            hidden_dim=256,
            output_dim=projection_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder.backbone(x)
        return self.projection(features)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy loss.

    Args:
        z1, z2: (B, D) normalized embeddings from two views
        temperature: softmax temperature
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    # Compute similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Positive pairs: (i, i+B) and (i+B, i) for all i in [0, B)
    pos_mask = torch.zeros(2 * B, 2 * B, device=z.device)
    for i in range(B):
        pos_mask[i, i + B] = 1.0
        pos_mask[i + B, i] = 1.0

    # Negative mask (exclude self-similarity)
    neg_mask = 1.0 - torch.eye(2 * B, device=z.device) - pos_mask

    # For each anchor, compute the loss
    # log(exp(sim_pos) / (exp(sim_pos) + sum(exp(sim_neg))))
    pos_sim = (sim * pos_mask).sum(dim=-1)  # (2B,)
    neg_exp_sum = (torch.exp(sim) * neg_mask).sum(dim=-1)  # (2B,)

    loss = -pos_sim + torch.log(torch.exp(pos_sim) + neg_exp_sum + 1e-8)
    return loss.mean()


def contrastive_pretraining(
    image_paths: list,
    image_dir: Path,
    backbone: str = "efficientnet_b0",
    output_dim: int = 256,
    projection_dim: int = 128,
    temperature: float = 0.1,
    batch_size: int = 16,
    epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    image_size: Tuple[int, int] = (512, 256),
    device: str = "cuda",
    save_path: Optional[Path] = None,
) -> ImageEncoder:
    """Run contrastive pre-training on pasture images.

    Returns:
        Pre-trained ImageEncoder (without projection head).
    """
    print(f"\n{'='*50}")
    print("Contrastive Pre-training (SimCLR)")
    print(f"{'='*50}")
    print(f"Images: {len(image_paths)}, Epochs: {epochs}, Batch: {batch_size}")

    # Dataset
    transform = get_contrastive_transforms(image_size)
    dataset = ContrastivePretrainDataset(image_paths, image_dir, transform, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    # Model
    encoder = ImageEncoder(backbone_name=backbone, pretrained=True, output_dim=output_dim)
    simclr = SimCLR(encoder, projection_dim=projection_dim).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(simclr.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    best_loss = float("inf")
    for epoch in range(epochs):
        simclr.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Contrastive Epoch {epoch+1}/{epochs}")

        for view1, view2 in pbar:
            view1, view2 = view1.to(device), view2.to(device)
            B = view1.shape[0]

            # Forward
            z1 = simclr(view1)
            z2 = simclr(view2)

            # Loss
            loss = nt_xent_loss(z1, z2, temperature)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader)
        scheduler.step()
        print(f"  Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}")

        if save_path and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), save_path)
            print(f"  Saved best encoder to {save_path}")

    # Return the encoder (without projection head)
    if save_path and save_path.exists():
        encoder.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
    return encoder
