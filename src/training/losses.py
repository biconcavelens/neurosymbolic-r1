"""Loss functions for training."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberLoss(nn.Module):
    """Huber (smooth L1) loss, more robust to outliers than MSE."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target, beta=self.delta)


class QuantileLoss(nn.Module):
    """Quantile loss for uncertainty-aware prediction."""

    def __init__(self, quantile: float = 0.5):
        super().__init__()
        self.quantile = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = target - pred
        loss = torch.where(diff >= 0, self.quantile * diff, (self.quantile - 1) * diff)
        return loss.mean()


class CompositeLoss(nn.Module):
    """Combined loss with optional constraint term."""

    def __init__(self, task_loss: nn.Module = None, constraint_weight: float = 0.0):
        super().__init__()
        self.task_loss = task_loss or HuberLoss()
        self.constraint_weight = constraint_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            predictions: (B, 5) tensor [Clover, Dead, Green, Total, GDM]
            targets: (B, 5) tensor in same order
        """
        task = self.task_loss(predictions, targets)

        if self.constraint_weight > 0:
            clover, dead, green, total, gdm = predictions.unbind(dim=-1)
            c1 = ((total - (green + dead + clover)) ** 2).mean()
            c2 = ((gdm - (green + clover)) ** 2).mean()
            constraint = c1 + c2
            return task + self.constraint_weight * constraint

        return task
