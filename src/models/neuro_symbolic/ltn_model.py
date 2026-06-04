"""Logical Tensor Network model for neuro-symbolic biomass prediction.

Key architecture: Predict 3 BASE targets (Green, Dead, Clover), derive 2 composites
(Total = Green+Dead+Clover, GDM = Green+Clover). This structurally guarantees
the physical constraints while the fuzzy logic loss provides additional regularization.

The model also includes:
- Evidential regression heads for uncertainty quantification
- Learned predicate weights via attention gating
- Learnable predicate membership functions for rule discovery
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from src.models.image_encoder import ImageEncoder, GridImageEncoder
from src.models.tabular_encoder import TabularEncoder
from src.models.fusion import FusionModule, CrossAttentionFusion
from .predicates import (
    DomainPredicate, PredicateWeightNetwork,
    LearnedPredicate,
)
from .constraints import (
    compute_constraint_satisfaction, compute_ltn_loss,
    aggregate_satisfactions,
)

# Target order in predictions tensor
BASE_TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"]
ALL_TARGETS = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]


class LTNModel(nn.Module):
    """Neuro-symbolic model with Logical Tensor Network constraints.

    Predicts 3 base components, derives 2 composite targets, and applies
    fuzzy logic constraints during training.
    """

    def __init__(
        self,
        tabular_input_dim: int,
        backbone: str = "efficientnet_b0",
        image_feature_dim: int = 256,
        tabular_hidden: tuple = (64, 32, 16),
        fusion_hidden: tuple = (512, 256, 128),
        dropout: float = 0.3,
        use_grid_features: bool = False,
        use_color_histogram: bool = False,
        color_hist_dim: int = 192,
        use_cross_attention: bool = False,
        # LTN options
        use_evidential: bool = False,
        use_learned_weights: bool = True,
        use_learned_predicates: bool = True,
        constraint_eps: float = 0.5,
        # Clover two-stage
        use_two_stage_clover: bool = False,
    ):
        super().__init__()
        self.use_grid_features = use_grid_features
        self.use_color_histogram = use_color_histogram
        self.use_evidential = use_evidential
        self.use_learned_weights = use_learned_weights
        self.use_learned_predicates = use_learned_predicates
        self.use_two_stage_clover = use_two_stage_clover

        # ── Encoders ──
        self.image_encoder = ImageEncoder(
            backbone_name=backbone, pretrained=True,
            output_dim=image_feature_dim, dropout=dropout,
        )

        if use_grid_features:
            self.grid_encoder = GridImageEncoder(
                backbone_name=backbone, pretrained=True,
                output_dim=128, num_tiles=8, dropout=dropout,
            )

        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim, hidden_dims=tabular_hidden, dropout=dropout,
        )

        # ── Fusion ──
        fusion_input_img_dim = image_feature_dim
        if use_grid_features:
            fusion_input_img_dim += 128
        if use_color_histogram:
            fusion_input_img_dim += color_hist_dim

        if use_cross_attention:
            self.fusion = CrossAttentionFusion(
                image_dim=fusion_input_img_dim,
                tabular_dim=tabular_hidden[-1],
                hidden_dim=fusion_hidden[-1],
                dropout=dropout,
            )
            fusion_out_dim = self.fusion.output_dim
        else:
            self.fusion = FusionModule(
                image_dim=fusion_input_img_dim,
                tabular_dim=tabular_hidden[-1],
                hidden_dims=fusion_hidden,
                dropout=dropout,
            )
            fusion_out_dim = fusion_hidden[-1] if fusion_hidden else fusion_input_img_dim + tabular_hidden[-1]

        # ── Prediction Heads ──
        # 3 base targets: Green, Dead, Clover
        if use_evidential:
            # Evidential: predict 4 NIG parameters per target
            self.base_heads = nn.ModuleDict({
                name: EvidentialHead(fusion_out_dim, dropout)
                for name in BASE_TARGETS
            })
        elif use_two_stage_clover:
            # Two-stage clover + standard heads for green and dead
            self.head_green = self._make_head(fusion_out_dim, dropout)
            self.head_dead = self._make_head(fusion_out_dim, dropout)
            # Two-stage clover head
            self.clover_zero_classifier = nn.Sequential(
                nn.Linear(fusion_out_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )
            self.clover_regressor = self._make_head(fusion_out_dim, dropout)
        else:
            self.base_heads = nn.ModuleDict({
                name: self._make_head(fusion_out_dim, dropout)
                for name in BASE_TARGETS
            })

        # ── LTN Components ──
        self.domain_pred = DomainPredicate("domain_knowledge")
        self.constraint_eps = constraint_eps

        if use_learned_weights:
            self.predicate_weights = PredicateWeightNetwork(fusion_out_dim, num_predicate_groups=7)

        if use_learned_predicates:
            self.learned_predicate = LearnedPredicate(
                "discovered_rule", fusion_out_dim, hidden_dim=32
            )

    @staticmethod
    def _make_head(in_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def _predict_bases(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict the 3 base components from fused features."""
        if self.use_evidential:
            preds = {}
            for name in BASE_TARGETS:
                gamma, nu, alpha, beta = self.base_heads[name](fused)
                # Store mean prediction
                preds[name] = gamma.squeeze(-1)
                # Store evidential params for loss
                preds[f"{name}_nu"] = nu.squeeze(-1)
                preds[f"{name}_alpha"] = alpha.squeeze(-1)
                preds[f"{name}_beta"] = beta.squeeze(-1)
            return preds
        elif self.use_two_stage_clover:
            green = self.head_green(fused).squeeze(-1)
            dead = self.head_dead(fused).squeeze(-1)
            # Two-stage clover
            zero_logit = self.clover_zero_classifier(fused).squeeze(-1)
            clover_reg = self.clover_regressor(fused).squeeze(-1)
            return {
                "Dry_Green_g": green,
                "Dry_Dead_g": dead,
                "clover_zero_logit": zero_logit,
                "clover_reg": clover_reg,
            }
        else:
            return {name: self.base_heads[name](fused).squeeze(-1)
                    for name in BASE_TARGETS}

    def _derive_composites(self, base_preds: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Derive Dry_Total and GDM from base predictions.

        This STRUCTURALLY guarantees the constraints, not just penalizes violations.
        """
        green = base_preds["Dry_Green_g"]
        dead = base_preds["Dry_Dead_g"]
        clover = base_preds.get("Dry_Clover_g", torch.zeros_like(green))

        total = green + dead + clover
        gdm = green + clover

        # Return in canonical order: [Clover, Dead, Green, Total, GDM]
        return torch.stack([clover, dead, green, total, gdm], dim=-1)

    def forward(self, batch: Dict) -> Dict:
        """Forward pass with structural constraint enforcement.

        Returns:
            Dict with:
                - predictions: (B, 5) tensor [Clover, Dead, Green, Total, GDM]
                - fused_features: (B, fusion_dim)
                - base_preds: dict of base component predictions
                - constraint_sats: dict of constraint satisfaction values
        """
        # ── Image features ──
        img_features = self.image_encoder(batch["image"])
        extra_features = [img_features]

        if self.use_grid_features and "grid_tiles" in batch:
            grid_features = self.grid_encoder(batch["grid_tiles"])
            extra_features.append(grid_features)
        if self.use_color_histogram and "color_hist" in batch:
            extra_features.append(batch["color_hist"])

        combined_img = torch.cat(extra_features, dim=-1)

        # ── Tabular features ──
        tab_features = self.tabular_encoder(batch["tabular"])

        # ── Fusion ──
        fused = self.fusion(combined_img, tab_features)

        # ── Base predictions ──
        base_preds = self._predict_bases(fused)

        # ── Derive composites (structural constraint enforcement) ──
        predictions = self._derive_composites(base_preds)

        # ── Constraint satisfaction (for logging and loss) ──
        # Extract NDVI and height from tabular features
        ndvi = batch["tabular"][:, 0]  # First feature is NDVI
        height = batch["tabular"][:, 1]  # Second is log-height

        # Species one-hot encoding (for species-clover constraint)
        # Tabular features: [ndvi(1), height(1), state(4), species(15), date(2)]
        num_state = 4
        num_species = 15
        species_start = 2 + num_state
        species_end = species_start + num_species
        species_indices = batch["tabular"][:, species_start:species_end]

        # Non-clover species mask
        non_clover_names = ["Fescue", "Lucerne", "Mixed", "Phalaris", "Ryegrass"]
        non_clover_mask = torch.zeros(num_species, device=predictions.device)
        # We rely on the preprocessor's species ordering — indices 2, 5, 7, 10, 13 are approximate
        # For now use a simple approach
        non_clover_mask = batch.get("non_clover_mask", non_clover_mask)

        constraint_sats = compute_constraint_satisfaction(
            predictions=predictions,
            ndvi=ndvi,
            height=height,
            species_indices=species_indices,
            non_clover_mask=non_clover_mask,
            domain_pred=self.domain_pred,
            eps=self.constraint_eps,
        )

        # ── Learned predicates ──
        if self.use_learned_predicates:
            learned_sat = self.learned_predicate(fused)
            # Discovered rule: predictions should explain the fused features
            constraint_sats["learned_rule"] = learned_sat

        # ── Learned predicate weights ──
        if self.use_learned_weights:
            pw = self.predicate_weights(fused)  # (B, 7)
            constraint_sats["predicate_weights"] = pw

        return {
            "predictions": predictions,
            "fused_features": fused,
            "base_preds": base_preds,
            "constraint_sats": constraint_sats,
        }

    def freeze_backbone(self):
        for param in self.image_encoder.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.image_encoder.backbone.parameters():
            param.requires_grad = True

    def get_phase_weights(self, epoch: int, total_epochs: int) -> Tuple[float, float, float, float]:
        """Get constraint loss weights based on training phase."""
        if epoch < total_epochs * 0.15:
            # Phase 1: focus on regression
            return (1.0, 0.5, 0.0, 0.0)
        elif epoch < total_epochs * 0.40:
            # Phase 2: introduce constraints
            return (1.0, 1.5, 0.5, 0.2)
        else:
            # Phase 3: full neuro-symbolic
            return (1.0, 2.0, 1.0, 0.5)


class AlphaShift(nn.Module):
    """Adds 1.0 to ensure alpha > 1 (required by NIG distribution)."""
    def forward(self, x): return x + 1.0


class EvidentialHead(nn.Module):
    """Deep Evidential Regression head (Amini et al., NeurIPS 2020).

    Predicts parameters of a Normal-Inverse-Gamma distribution:
    - gamma: mean prediction (the actual regression output)
    - nu: virtual observations (> 0)
    - alpha: shape parameter (> 1)
    - beta: scale parameter (> 0)

    Uncertainty: epistemic = beta / (nu * (alpha - 1))
                 aleatoric = beta / (alpha - 1)
    """

    def __init__(self, in_dim: int, dropout: float = 0.2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gamma_head = nn.Linear(64, 1)
        self.nu_head = nn.Sequential(nn.Linear(64, 1), nn.Softplus())
        self.alpha_head = nn.Sequential(nn.Linear(64, 1), nn.Softplus(), AlphaShift())
        self.beta_head = nn.Sequential(nn.Linear(64, 1), nn.Softplus())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        shared = self.shared(x)
        gamma = self.gamma_head(shared)
        nu = self.nu_head(shared)
        alpha = self.alpha_head(shared)
        beta = self.beta_head(shared)
        return gamma, nu.clamp(min=1e-6), alpha.clamp(min=1.0 + 1e-6), beta.clamp(min=1e-6)

    @staticmethod
    def evidential_loss(gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor,
                        beta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of NIG distribution."""
        # Student-t log likelihood approximation
        from torch import lgamma as lg
        two_blambda = 2 * beta * (1 + nu)
        nll = (0.5 * torch.log(torch.pi / nu) -
               alpha * torch.log(two_blambda) +
               (alpha + 0.5) * torch.log(nu * (target - gamma) ** 2 + two_blambda) +
               lg(alpha) - lg(alpha + 0.5))
        # Add regularization
        reg = torch.abs(target - gamma) * (2 * nu + alpha)
        return (nll + 1e-2 * reg).mean()
