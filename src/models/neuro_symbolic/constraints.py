"""Logical constraint formulations for the LTN model.

Each function returns truth values in [0, 1] representing the degree
to which a logical formula is satisfied for each sample in the batch.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from .predicates import (
    ConstraintPredicate, GDMConstraintPredicate, DomainPredicate,
    HierarchicalPredicate, MonotonicityPredicate, SpeciesCloverPredicate,
    pmean_aggregation, product_tnorm,
)


# ── Constraint Builders ─────────────────────────────────────────────────

def aggregation_constraint(total: torch.Tensor, green: torch.Tensor,
                           dead: torch.Tensor, clover: torch.Tensor,
                           eps: float = 0.5) -> torch.Tensor:
    """Truth of: Dry_Total ≈ Dry_Green + Dry_Dead + Dry_Clover."""
    deviation = torch.abs(total - (green + dead + clover))
    return torch.exp(-deviation ** 2 / (2 * eps ** 2 + 1e-8))


def green_matter_constraint(gdm: torch.Tensor, green: torch.Tensor,
                            clover: torch.Tensor, eps: float = 0.5) -> torch.Tensor:
    """Truth of: GDM ≈ Dry_Green + Dry_Clover."""
    deviation = torch.abs(gdm - (green + clover))
    return torch.exp(-deviation ** 2 / (2 * eps ** 2 + 1e-8))


def ndvi_green_implication(ndvi: torch.Tensor, green_biomass: torch.Tensor,
                           ndvi_high: nn.Module, biomass_high: nn.Module) -> torch.Tensor:
    """Truth of: NDVI_is_high -> GreenBiomass_is_high."""
    ndvi_h = ndvi_high(ndvi)
    biomass_h = biomass_high(green_biomass)
    # Reichenbach implication
    return 1.0 - ndvi_h + ndvi_h * biomass_h


def monotonicity_constraint(height: torch.Tensor, total_biomass: torch.Tensor) -> torch.Tensor:
    """Truth of monotonic relationship between height and total biomass."""
    # Batch-level check: Spearman-like rank consistency
    B = height.shape[0]
    if B < 3:
        return torch.ones(1, device=height.device)

    # Compute pairwise signs
    h_diff = height.unsqueeze(0) - height.unsqueeze(1)  # (B, B)
    t_diff = total_biomass.unsqueeze(0) - total_biomass.unsqueeze(1)

    # Ignore diagonal and near-zero height differences
    mask = (h_diff.abs() > 2.0).float()
    # Check sign agreement
    sign_h = torch.sign(h_diff)
    sign_t = torch.sign(t_diff)
    agreement = (sign_h == sign_t).float()
    # For zero sign_t, count as agreement
    agreement = torch.where(sign_t == 0, torch.ones_like(agreement), agreement)

    total_mask = mask.sum()
    if total_mask < 1:
        return torch.ones(1, device=height.device)

    return (agreement * mask).sum() / total_mask


def species_clover_constraint(species_indices: torch.Tensor, clover_pred: torch.Tensor,
                              non_clover_mask: torch.Tensor, eps: float = 0.5) -> torch.Tensor:
    """Truth of: non-clover species -> clover biomass near zero.

    Args:
        species_indices: (B, num_species) one-hot
        clover_pred: (B,) predicted clover
        non_clover_mask: (num_species,) binary mask for non-clover species
    """
    is_non_clover = (species_indices * non_clover_mask.unsqueeze(0)).sum(dim=-1)  # (B,)
    clover_near_zero = torch.exp(-clover_pred ** 2 / (2 * eps ** 2 + 1e-8))
    # If not non-clover, vacuously true
    return is_non_clover * clover_near_zero + (1.0 - is_non_clover) * 1.0


def hierarchy_constraint(gdm: torch.Tensor, total: torch.Tensor, eps: float = 0.5) -> torch.Tensor:
    """Truth of: GDM <= Total."""
    violation = torch.clamp(gdm - total, min=0.0)
    return torch.exp(-violation ** 2 / (2 * eps ** 2 + 1e-8))


# ── Combined Constraint Satisfaction ────────────────────────────────────

def compute_constraint_satisfaction(
    predictions: torch.Tensor,
    ndvi: torch.Tensor,
    height: torch.Tensor,
    species_indices: torch.Tensor,
    non_clover_mask: torch.Tensor,
    domain_pred: DomainPredicate,
    eps: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Compute satisfaction (truth values) for all logical constraints.

    Args:
        predictions: (B, 5) with order [Clover, Dead, Green, Total, GDM]
        ndvi: (B,) normalized NDVI values
        height: (B,) normalized log-height values
        species_indices: (B, num_species) one-hot
        non_clover_mask: (num_species,) mask
        domain_pred: DomainPredicate module

    Returns:
        Dict mapping constraint name to truth value tensor (B,).
        Also returns aggregated scalar satisfactions.
    """
    clover, dead, green, total, gdm = predictions.unbind(dim=-1)

    satisfactions = {}

    # 1. Mass conservation: Total ≈ Green + Dead + Clover
    satisfactions["mass_conservation"] = aggregation_constraint(total, green, dead, clover, eps)

    # 2. GDM identity: GDM ≈ Green + Clover
    satisfactions["gdm_identity"] = green_matter_constraint(gdm, green, clover, eps)

    # 3. NDVI → Green implication
    sat_ndvi = ndvi_green_implication(ndvi, green, domain_pred.ndvi_high, domain_pred.biomass_high)
    satisfactions["ndvi_implies_green"] = sat_ndvi

    # 4. NDVI → Low Dead implication
    ndvi_h = domain_pred.ndvi_high(ndvi)
    dead_low = 1.0 - domain_pred.biomass_high(dead)
    satisfactions["ndvi_implies_low_dead"] = 1.0 - ndvi_h + ndvi_h * dead_low

    # 5. Height monotonicity (batch-level, returns scalar — broadcast to batch)
    mono_sat = monotonicity_constraint(height, total)
    if mono_sat.ndim == 0 or mono_sat.shape[0] == 1:
        satisfactions["height_monotonic"] = mono_sat.expand(predictions.shape[0])
    else:
        satisfactions["height_monotonic"] = mono_sat

    # 6. Species → Clover constraint
    satisfactions["species_clover"] = species_clover_constraint(
        species_indices, clover, non_clover_mask, eps
    )

    # 7. Hierarchy: GDM subset of Total
    satisfactions["gdm_subset_total"] = hierarchy_constraint(gdm, total, eps)

    return satisfactions


def aggregate_satisfactions(satisfactions: Dict[str, torch.Tensor],
                            p: int = 2) -> torch.Tensor:
    """Aggregate all satisfactions into a single scalar using p-mean.

    Lower value = worse satisfaction. We return 1 - agg so it can be used as loss.
    """
    all_sats = list(satisfactions.values())
    return pmean_aggregation(all_sats, p=p)


def compute_ltn_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    satisfactions: Dict[str, torch.Tensor],
    weights: Tuple[float, float, float, float] = (1.0, 2.0, 1.0, 0.5),
    huber_delta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the full LTN loss.

    Args:
        predictions: (B, 5) [Clover, Dead, Green, Total, GDM]
        targets: (B, 5) ground truth
        satisfactions: Dict of constraint satisfaction tensors
        weights: (w_reg, w_constraint, w_domain, w_hierarchy)

    Returns:
        total_loss: scalar
        loss_components: dict of individual loss terms
    """
    w_reg, w_constraint, w_domain, w_hierarchy = weights

    # Regression loss (Huber)
    reg_loss = torch.nn.functional.smooth_l1_loss(predictions, targets, beta=huber_delta)

    # Constraint loss (1 - satisfaction)
    constraint_loss = 0.5 * (
        (1.0 - satisfactions["mass_conservation"]).mean() +
        (1.0 - satisfactions["gdm_identity"]).mean()
    )

    # Domain loss
    domain_loss = (1.0 - satisfactions["ndvi_implies_green"]).mean() * 0.5 + \
                  (1.0 - satisfactions["ndvi_implies_low_dead"]).mean() * 0.5

    # Hierarchy loss
    hierarchy_loss = (1.0 - satisfactions["gdm_subset_total"]).mean()

    total = (w_reg * reg_loss +
             w_constraint * constraint_loss +
             w_domain * domain_loss +
             w_hierarchy * hierarchy_loss)

    components = {
        "reg_loss": reg_loss,
        "constraint_loss": constraint_loss,
        "domain_loss": domain_loss,
        "hierarchy_loss": hierarchy_loss,
        "total_loss": total,
    }

    return total, components
