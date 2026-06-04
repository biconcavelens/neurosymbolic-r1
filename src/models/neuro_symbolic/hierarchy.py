"""Hierarchical ontology predicates for biomass composition.

Models the compositional hierarchy:
    TotalBiomass
    ├── GreenDryMatter (GDM)
    │   ├── Dry_Green_g (grass green)
    │   └── Dry_Clover_g (clover green)
    └── Dry_Dead_g (senescent material)
"""
import torch
import torch.nn as nn
from .predicates import FuzzyPredicate


class HierarchicalOntologyPredicates(nn.Module):
    """Collection of predicates enforcing the biomass ontology hierarchy."""

    def __init__(self, eps: float = 0.5):
        super().__init__()
        self.eps = eps

    def parent_equals_sum_of_children(
        self, parent: torch.Tensor, children: list[torch.Tensor]
    ) -> torch.Tensor:
        """Truth value that parent biomass equals sum of child biomasses.
        Uses Gaussian kernel for fuzzy equality.
        """
        child_sum = sum(children)
        deviation = torch.abs(parent - child_sum)
        return torch.exp(-deviation ** 2 / (2 * self.eps ** 2 + 1e-8))

    def child_less_than_parent(
        self, child: torch.Tensor, parent: torch.Tensor, margin: float = 0.0
    ) -> torch.Tensor:
        """Truth value that child biomass is less than or equal to parent.
        margin allows a small tolerance (e.g., rounding errors).
        """
        violation = torch.clamp(child - parent + margin, min=0.0)
        return torch.exp(-violation ** 2 / (2 * self.eps ** 2 + 1e-8))

    def forward(self, green: torch.Tensor, dead: torch.Tensor,
                clover: torch.Tensor, total: torch.Tensor,
                gdm: torch.Tensor) -> dict:
        """Compute all hierarchical constraint satisfactions.

        Args:
            green: (B,) Dry_Green_g
            dead: (B,) Dry_Dead_g
            clover: (B,) Dry_Clover_g
            total: (B,) Dry_Total_g
            gdm: (B,) GDM_g

        Returns:
            dict of satisfaction tensors
        """
        results = {}

        # Total = GDM + Dead (but GDM = Green + Clover)
        # So Total = Green + Clover + Dead (already in constraints.py)
        # This is redundant but verifies the hierarchy

        # GDM = Green + Clover (leaf-level constraint)
        results["gdm_children"] = self.parent_equals_sum_of_children(
            gdm, [green, clover]
        )

        # Green <= GDM (child <= parent)
        results["green_subset_gdm"] = self.child_less_than_parent(green, gdm)

        # Clover <= GDM
        results["clover_subset_gdm"] = self.child_less_than_parent(clover, gdm)

        # Dead <= Total
        results["dead_subset_total"] = self.child_less_than_parent(dead, total)

        # GDM <= Total
        results["gdm_subset_total"] = self.child_less_than_parent(gdm, total)

        return results
