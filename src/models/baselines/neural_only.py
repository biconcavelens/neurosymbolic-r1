"""Neural network baselines: pure CNN+MLP fusion without logical constraints."""
import torch
import torch.nn as nn
from typing import Dict, Optional

from src.models.image_encoder import ImageEncoder, GridImageEncoder
from src.models.tabular_encoder import TabularEncoder
from src.models.fusion import FusionModule


TARGET_NAMES = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]
BASE_TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"]


class NeuralOnlyModel(nn.Module):
    """Pure neural network baseline: CNN + tabular MLP → fusion → 5 independent heads.

    No logical constraints. Trained with standard MSE loss.
    """

    def __init__(self, tabular_input_dim: int,
                 backbone: str = "efficientnet_b0",
                 image_feature_dim: int = 256,
                 tabular_hidden: tuple = (64, 32, 16),
                 fusion_hidden: tuple = (512, 256, 128),
                 dropout: float = 0.3,
                 use_grid_features: bool = False,
                 use_color_histogram: bool = False,
                 color_hist_dim: int = 192):
        super().__init__()
        self.use_grid_features = use_grid_features
        self.use_color_histogram = use_color_histogram

        # Image encoder (global view)
        self.image_encoder = ImageEncoder(
            backbone_name=backbone, pretrained=True,
            output_dim=image_feature_dim, dropout=dropout,
        )

        # Optional grid encoder
        if use_grid_features:
            self.grid_encoder = GridImageEncoder(
                backbone_name=backbone, pretrained=True,
                output_dim=128, num_tiles=8, dropout=dropout,
            )

        # Tabular encoder
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim, hidden_dims=tabular_hidden, dropout=dropout,
        )

        # Compute fusion input dimension
        fusion_input_img_dim = image_feature_dim
        if use_grid_features:
            fusion_input_img_dim += 128
        if use_color_histogram:
            fusion_input_img_dim += color_hist_dim

        # Fusion
        self.fusion = FusionModule(
            image_dim=fusion_input_img_dim,
            tabular_dim=tabular_hidden[-1],
            hidden_dims=fusion_hidden,
            dropout=dropout,
        )

        # 5 independent prediction heads
        fusion_out_dim = fusion_hidden[-1] if fusion_hidden else fusion_input_img_dim + tabular_hidden[-1]
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(fusion_out_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
            ) for name in TARGET_NAMES
        })

    def forward(self, batch: Dict) -> Dict:
        """Forward pass returning all 5 predictions."""
        # Image features
        img_features = self.image_encoder(batch["image"])

        # Grid features
        extra_features = [img_features]
        if self.use_grid_features and "grid_tiles" in batch:
            grid_features = self.grid_encoder(batch["grid_tiles"])
            extra_features.append(grid_features)
        if self.use_color_histogram and "color_hist" in batch:
            extra_features.append(batch["color_hist"])

        combined_img = torch.cat(extra_features, dim=-1)

        # Tabular features
        tab_features = self.tabular_encoder(batch["tabular"])

        # Fusion
        fused = self.fusion(combined_img, tab_features)

        # Predict each target
        preds = {}
        for name in TARGET_NAMES:
            preds[name] = self.heads[name](fused).squeeze(-1)

        # Stack in canonical order
        predictions = torch.stack([preds[t] for t in TARGET_NAMES], dim=-1)
        return {"predictions": predictions, "fused_features": fused}

    def freeze_backbone(self):
        """Freeze the CNN backbone for initial training."""
        for param in self.image_encoder.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze the CNN backbone."""
        for param in self.image_encoder.backbone.parameters():
            param.requires_grad = True


class NeuralConstraintModel(NeuralOnlyModel):
    """Neural network + soft constraint penalty loss.

    Same architecture as NeuralOnlyModel but trained with an additional MSE
    penalty on constraint violations. This is the critical ablation:
    it tests whether simply adding a penalty term matches LTN's fuzzy logic.
    """

    def __init__(self, *args, constraint_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint_weight = constraint_weight

    def compute_constraint_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute MSE penalty for constraint violations.

        predictions: (B, 5) with order [Clover, Dead, Green, Total, GDM]
        """
        clover, dead, green, total, gdm = predictions.unbind(dim=-1)

        # Constraint 1: Total should equal Green + Dead + Clover
        c1 = (total - (green + dead + clover)) ** 2

        # Constraint 2: GDM should equal Green + Clover
        c2 = (gdm - (green + clover)) ** 2

        return (c1 + c2).mean()
