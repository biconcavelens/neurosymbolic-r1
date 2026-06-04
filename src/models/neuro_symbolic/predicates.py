"""Fuzzy logic predicates for the Logical Tensor Network.

Each predicate outputs a truth value in [0, 1]. The model is trained
to maximize the satisfaction (truth value) of logical formulas over these predicates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ── T-norms ──────────────────────────────────────────────────────────────

def lukasiewicz_tnorm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Łukasiewicz t-norm: max(0, a + b - 1)."""
    return torch.clamp(a + b - 1.0, min=0.0, max=1.0)


def product_tnorm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Product t-norm: a * b."""
    return a * b


def godel_tnorm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Gödel t-norm: min(a, b)."""
    return torch.min(a, b)


# ── Fuzzy Implications ───────────────────────────────────────────────────

def reichenbach_implication(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reichenbach implication: 1 - a + a * b."""
    return 1.0 - a + a * b


def lukasiewicz_implication(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Łukasiewicz implication: min(1, 1 - a + b)."""
    return torch.clamp(1.0 - a + b, min=0.0, max=1.0)


# ── Fuzzy Aggregation ────────────────────────────────────────────────────

def pmean_aggregation(values: list, p: int = 2) -> torch.Tensor:
    """p-mean error aggregation with numerical stability.

    Higher p gives more weight to the worst-satisfied constraint.
    """
    stacked = torch.stack(values, dim=0)  # (N, B)
    # Use stable computation: mean(|1 - sat|^p)^(1/p)
    deviations = (1.0 - stacked).abs() ** p
    mean_dev = deviations.mean(dim=0)
    return 1.0 - mean_dev ** (1.0 / p)


# ── Base Predicates ──────────────────────────────────────────────────────

class FuzzyPredicate(nn.Module):
    """Base class for fuzzy logic predicates."""
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Return truth values in [0, 1]. Shape: (B,)"""
        raise NotImplementedError


class GaussianEqualityPredicate(FuzzyPredicate):
    """Fuzzy equality using Gaussian kernel: exp(-|x - y|^2 / (2*eps^2))."""

    def __init__(self, name: str = "approx_equal", eps: float = 0.5):
        super().__init__(name)
        self.eps = nn.Parameter(torch.tensor(eps), requires_grad=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x, y: (B,) tensors."""
        return torch.exp(-(x - y) ** 2 / (2 * self.eps ** 2 + 1e-8))


class SigmoidMembershipPredicate(FuzzyPredicate):
    """Learned fuzzy membership: sigmoid(alpha * (x - beta)).

    alpha: steepness (positive = increasing membership)
    beta: threshold (where membership = 0.5)
    """

    def __init__(self, name: str = "fuzzy_member", alpha_init: float = 1.0, beta_init: float = 0.0):
        super().__init__(name)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,) tensor."""
        return torch.sigmoid(self.alpha * (x - self.beta))


# ── Domain-Specific Predicates ───────────────────────────────────────────

class ConstraintPredicate(FuzzyPredicate):
    """Predicate for physical constraint: Total ≈ Green + Dead + Clover."""

    def __init__(self, name: str = "mass_conservation", eps: float = 0.5):
        super().__init__(name)
        self.equality = GaussianEqualityPredicate("eq", eps)

    def forward(self, total: torch.Tensor, green: torch.Tensor,
                dead: torch.Tensor, clover: torch.Tensor) -> torch.Tensor:
        """Truth value of Total == Green + Dead + Clover."""
        predicted_sum = green + dead + clover
        return self.equality(total, predicted_sum)


class GDMConstraintPredicate(FuzzyPredicate):
    """Predicate for: GDM == Green + Clover."""

    def __init__(self, name: str = "gdm_identity", eps: float = 0.5):
        super().__init__(name)
        self.equality = GaussianEqualityPredicate("eq", eps)

    def forward(self, gdm: torch.Tensor, green: torch.Tensor, clover: torch.Tensor) -> torch.Tensor:
        return self.equality(gdm, green + clover)


class DomainPredicate(FuzzyPredicate):
    """Domain knowledge predicates with learnable membership functions."""

    def __init__(self, name: str = "domain_knowledge"):
        super().__init__(name)
        # Learned fuzzy thresholds
        self.ndvi_high = SigmoidMembershipPredicate("ndvi_high", alpha_init=5.0, beta_init=0.6)
        self.biomass_high = SigmoidMembershipPredicate("biomass_high", alpha_init=0.1, beta_init=30.0)
        self.biomass_low = SigmoidMembershipPredicate("biomass_low", alpha_init=-0.1, beta_init=20.0)

    def ndvi_implies_green(self, ndvi: torch.Tensor, green_biomass: torch.Tensor) -> torch.Tensor:
        """High NDVI implies high green biomass."""
        ndvi_high = self.ndvi_high(ndvi)
        biomass_high = self.biomass_high(green_biomass)
        return reichenbach_implication(ndvi_high, biomass_high)

    def ndvi_implies_low_dead(self, ndvi: torch.Tensor, dead_biomass: torch.Tensor) -> torch.Tensor:
        """High NDVI implies low dead biomass."""
        ndvi_high = self.ndvi_high(ndvi)
        dead_low = 1.0 - self.biomass_high(dead_biomass)
        return reichenbach_implication(ndvi_high, dead_low)


class HierarchicalPredicate(FuzzyPredicate):
    """Hierarchical ontology: GDM is a subset of Total."""

    def __init__(self, name: str = "gdm_subset_of_total", eps: float = 0.5):
        super().__init__(name)
        self.eps_val = eps

    def forward(self, gdm: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
        """Truth value of GDM <= Total.
        Returns 0 if GDM > Total, 1 if GDM <= Total.
        Fuzzy version: exp(-max(0, GDM - Total)^2 / eps^2)
        """
        violation = torch.clamp(gdm - total, min=0.0)
        return torch.exp(-violation ** 2 / (2 * self.eps_val ** 2 + 1e-8))


class MonotonicityPredicate(FuzzyPredicate):
    """Height monotonically relates to total biomass within a batch."""

    def __init__(self, name: str = "height_monotonic", eps: float = 0.5):
        super().__init__(name)
        self.eps_val = eps

    def forward(self, height: torch.Tensor, total_biomass: torch.Tensor) -> torch.Tensor:
        """Check monotonicity for all pairs where height differs significantly."""
        B = height.shape[0]
        if B < 2:
            return torch.ones(1, device=height.device)

        # Pairwise differences
        h_i = height.unsqueeze(0)  # (1, B)
        h_j = height.unsqueeze(1)  # (B, 1)
        t_i = total_biomass.unsqueeze(0)
        t_j = total_biomass.unsqueeze(1)

        # Only consider pairs where height differs by at least 1cm
        h_diff = h_i - h_j
        significant = (h_diff.abs() > 2.0).float()  # at least 2cm difference

        # Height ordering should match biomass ordering
        same_ordering = ((h_diff > 0) == (t_i > t_j)).float()
        # Also handle near-equal cases
        near_equal = ((h_diff.abs() <= 2.0) | ((t_i - t_j).abs() < 1.0)).float()

        pair_sat = torch.where(near_equal > 0.5,
                              torch.ones_like(same_ordering),
                              same_ordering)

        # Mean over all valid pairs
        valid_pairs = significant + near_equal
        valid_pairs = valid_pairs.clamp(max=1.0)

        total_pairs = valid_pairs.sum()
        if total_pairs < 1:
            return torch.ones(1, device=height.device)

        return (pair_sat * valid_pairs).sum() / total_pairs


class SpeciesCloverPredicate(FuzzyPredicate):
    """Species-specific clover presence predicate."""

    # Species that typically have zero clover
    NON_CLOVER_SPECIES = {"Fescue", "Lucerne", "Mixed", "Phalaris", "Ryegrass"}

    def __init__(self, name: str = "species_clover", eps: float = 0.5):
        super().__init__(name)
        self.eps_val = eps

    def forward(self, species_indices: torch.Tensor, clover_pred: torch.Tensor,
                species_to_idx: dict) -> torch.Tensor:
        """For non-clover species, clover prediction should be near zero.

        Args:
            species_indices: (B, num_species) one-hot encoded species
            clover_pred: (B,) predicted clover biomass
            species_to_idx: dict mapping species name to index
        """
        # Build mask for non-clover species
        non_clover_indices = [
            species_to_idx[s] for s in self.NON_CLOVER_SPECIES
            if s in species_to_idx
        ]
        if not non_clover_indices:
            return torch.ones(1, device=clover_pred.device)

        is_non_clover = species_indices[:, non_clover_indices].sum(dim=-1)  # (B,)

        # Fuzzy truth: for non-clover species, clover should be near zero
        clover_near_zero = torch.exp(-clover_pred ** 2 / (2 * self.eps_val ** 2 + 1e-8))

        # Only applies to non-clover species (for clover species, constraint is vacuously true)
        result = is_non_clover * clover_near_zero + (1.0 - is_non_clover) * 1.0
        return result


# ── Learned Predicate Weights ────────────────────────────────────────────

class PredicateWeightNetwork(nn.Module):
    """Learns per-sample weights for each predicate based on fused features.

    This allows the model to adaptively emphasize different constraints:
    - Non-clover species get lower weight on clover constraints
    - WA state gets lower weight on dead material predicates
    - High-NDVI samples get higher weight on green biomass predicates
    """

    def __init__(self, feature_dim: int, num_predicate_groups: int = 5):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_predicate_groups),
            nn.Softmax(dim=-1),
        )
        # Base weights (learned scalar bias for each group)
        self.base_weights = nn.Parameter(torch.ones(num_predicate_groups))

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """Returns per-sample predicate group weights: (B, num_groups)."""
        attn_weights = self.attention(fused_features)
        return attn_weights * self.base_weights.unsqueeze(0)


# ── Learnable Predicate (Rule Discovery) ─────────────────────────────────

class LearnedPredicate(FuzzyPredicate):
    """A predicate with fully learnable membership function for rule discovery.

    Can learn rules like: "When feature combination X exceeds threshold Y,
    then biomass should be in range Z."
    """

    def __init__(self, name: str, input_dim: int, hidden_dim: int = 32):
        super().__init__(name)
        self.feature_selector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Learned membership parameters
        self.threshold = nn.Parameter(torch.zeros(1))
        self.steepness = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) feature vector.
        Returns: (B,) truth values.
        """
        # Learned feature combination
        selected = self.feature_selector(x).squeeze(-1)  # (B,)
        # Fuzzy membership
        return torch.sigmoid(self.steepness * (selected - self.threshold))
