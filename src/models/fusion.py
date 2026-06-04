"""Multi-modal fusion modules."""
import torch
import torch.nn as nn
from typing import List


class FusionModule(nn.Module):
    """Simple concatenation-based fusion followed by MLP."""

    def __init__(self, image_dim: int, tabular_dim: int,
                 hidden_dims: List[int] = (512, 256, 128), dropout: float = 0.3):
        super().__init__()
        self.input_dim = image_dim + tabular_dim
        layers = []
        prev_dim = self.input_dim
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hdim
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else self.input_dim

    def forward(self, image_features: torch.Tensor, tabular_features: torch.Tensor) -> torch.Tensor:
        """Fuse image and tabular features."""
        combined = torch.cat([image_features, tabular_features], dim=-1)
        return self.net(combined)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion: image features attend to tabular features and vice versa."""

    def __init__(self, image_dim: int, tabular_dim: int, hidden_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.tabular_proj = nn.Linear(tabular_dim, hidden_dim)

        # Image attends to tabular
        self.img_to_tab = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        # Tabular attends to image
        self.tab_to_img = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(self, image_features: torch.Tensor, tabular_features: torch.Tensor) -> torch.Tensor:
        """Cross-attention fusion.

        Args:
            image_features: (B, image_dim)
            tabular_features: (B, tabular_dim)
        Returns:
            fused: (B, hidden_dim)
        """
        img = self.image_proj(image_features).unsqueeze(1)  # (B, 1, H)
        tab = self.tabular_proj(tabular_features).unsqueeze(1)  # (B, 1, H)

        # Image attends to tabular
        img_attended, _ = self.img_to_tab(img, tab, tab)  # (B, 1, H)
        # Tabular attends to image
        tab_attended, _ = self.tab_to_img(tab, img, img)  # (B, 1, H)

        combined = torch.cat([img_attended.squeeze(1), tab_attended.squeeze(1)], dim=-1)
        return self.output_proj(combined)
