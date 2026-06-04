"""Training callbacks: early stopping, LR scheduling, checkpointing."""
import numpy as np
import torch
from pathlib import Path
from typing import Optional


class EarlyStopping:
    """Early stopping that monitors a metric and saves best model."""

    def __init__(self, patience: int = 20, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """Returns True if this is a new best score."""
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


class LRSchedulerCallback:
    """Wrapper for learning rate schedulers."""

    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler,
                 monitor: str = "val_loss", mode: str = "min"):
        self.scheduler = scheduler
        self.monitor = monitor
        self.mode = mode
        self.is_reduce_on_plateau = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        )

    def step(self, metric: Optional[float] = None):
        if self.is_reduce_on_plateau:
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, metrics: dict, filepath: Path):
    """Save model checkpoint."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, filepath)


def load_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
                    filepath: Path, device: str = "cuda") -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
