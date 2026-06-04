"""CNN image encoders for pasture images."""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ImageEncoder(nn.Module):
    """Image encoder using a pretrained CNN backbone.

    Supports EfficientNet-B0 (default) and ResNet18.
    """

    BACKBONES = {
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights),
        "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights),
        "resnet18": (models.resnet18, models.ResNet18_Weights),
        "resnet34": (models.resnet34, models.ResNet34_Weights),
        "resnet50": (models.resnet50, models.ResNet50_Weights),
    }

    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = True,
                 output_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.backbone_name = backbone_name

        model_fn, weights_cls = self.BACKBONES.get(
            backbone_name, self.BACKBONES["efficientnet_b0"]
        )
        weights = weights_cls.DEFAULT if pretrained else None
        self.backbone = model_fn(weights=weights)

        # Determine feature dimension
        if "efficientnet" in backbone_name:
            self.feature_dim = self.backbone.classifier[1].in_features
            # Replace classifier with identity
            self.backbone.classifier = nn.Identity()
        else:  # ResNet
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) -> (B, output_dim)"""
        features = self.backbone(x)
        return self.projection(features)


class GridImageEncoder(nn.Module):
    """Encodes grid tiles from an image using a shared smaller CNN backbone.

    Input: (B, num_tiles, 3, tile_H, tile_W)
    Output: (B, output_dim) via attention pooling over tiles.
    """

    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = True,
                 output_dim: int = 128, num_tiles: int = 8, dropout: float = 0.2):
        super().__init__()
        self.num_tiles = num_tiles
        self.tile_encoder = ImageEncoder(backbone_name, pretrained, output_dim, dropout)
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=4, batch_first=True, dropout=dropout
        )
        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        """tiles: (B, T, 3, H_tile, W_tile)"""
        B, T = tiles.shape[0], tiles.shape[1]
        # Encode each tile independently
        tiles_flat = tiles.view(B * T, *tiles.shape[2:])
        tile_feats = self.tile_encoder(tiles_flat)  # (B*T, D)
        tile_feats = tile_feats.view(B, T, -1)  # (B, T, D)

        # Attention pooling: learn which tiles are most informative
        pooled, _ = self.attention_pool(tile_feats, tile_feats, tile_feats)
        # Mean pool across tiles
        pooled = pooled.mean(dim=1)  # (B, D)
        return self.output_proj(pooled)
