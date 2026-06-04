"""Generic training loop for neural models."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional, Callable
import numpy as np
from tqdm import tqdm
import time

from .callbacks import EarlyStopping, LRSchedulerCallback, save_checkpoint


class Trainer:
    """Generic trainer for multi-target regression models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[LRSchedulerCallback] = None,
        device: str = "cuda",
        use_amp: bool = True,
        grad_clip: float = 1.0,
        log_interval: int = 10,
        output_dir: Path = Path("outputs"),
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and device == "cuda"
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.output_dir = Path(output_dir)

        self.scaler = GradScaler(enabled=self.use_amp)
        self.early_stopping = EarlyStopping(patience=20)
        self.history = {"train_loss": [], "val_loss": [], "val_rmse": [], "lr": []}

    def train_epoch(self) -> float:
        """Train one epoch. Returns average training loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(batch)
                    loss = self.loss_fn(outputs["predictions"], batch["targets"])
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                loss = self.loss_fn(outputs["predictions"], batch["targets"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model. Returns dict of metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            if self.use_amp:
                with autocast():
                    outputs = self.model(batch)
                    loss = self.loss_fn(outputs["predictions"], batch["targets"])
            else:
                outputs = self.model(batch)
                loss = self.loss_fn(outputs["predictions"], batch["targets"])

            total_loss += loss.item()
            all_preds.append(outputs["predictions"].cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())
            num_batches += 1

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Per-target RMSE
        rmse_per_target = np.sqrt(np.mean((all_preds - all_targets) ** 2, axis=0))
        overall_rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

        return {
            "val_loss": total_loss / max(num_batches, 1),
            "val_rmse": float(overall_rmse),
            "val_rmse_per_target": rmse_per_target.tolist(),
            "predictions": all_preds,
            "targets": all_targets,
        }

    def fit(self, num_epochs: int, start_epoch: int = 0) -> Dict:
        """Full training loop."""
        best_val_rmse = float("inf")
        best_model_path = None

        for epoch in range(start_epoch, start_epoch + num_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)

            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics["val_loss"]
            val_rmse = val_metrics["val_rmse"]
            self.history["val_loss"].append(val_loss)
            self.history["val_rmse"].append(val_rmse)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)

            # Scheduler step
            if self.scheduler is not None:
                try:
                    self.scheduler.step(val_rmse)
                except TypeError:
                    self.scheduler.step()

            # Early stopping
            is_best = self.early_stopping(val_rmse, epoch)
            if is_best:
                best_val_rmse = val_rmse
                best_model_path = self.output_dir / "models" / "best_model.pt"
                save_checkpoint(self.model, self.optimizer, epoch, val_metrics, best_model_path)

            # Logging
            if epoch % self.log_interval == 0 or is_best:
                epoch_time = time.time() - epoch_start
                rmse_str = ", ".join(
                    f"{v:.3f}" for v in val_metrics["val_rmse_per_target"]
                )
                print(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val RMSE: {val_rmse:.4f} [{rmse_str}] | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                    + (" *" if is_best else "")
                )

            if self.early_stopping.should_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Load best model
        if best_model_path and best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        return {
            "best_val_rmse": best_val_rmse,
            "best_epoch": self.early_stopping.best_epoch,
            "history": self.history,
            "best_model_path": best_model_path,
        }
