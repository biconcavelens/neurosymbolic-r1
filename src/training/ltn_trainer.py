"""Specialized trainer for LTN models with 3-phase constraint schedule."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
import time

from .callbacks import EarlyStopping, save_checkpoint


class LTNTrainer:
    """Trainer for the LTN neuro-symbolic model with phased constraint introduction.

    Phase 1 (15% of epochs): Focus on regression, light constraints
    Phase 2 (15-40%): Introduce domain predicates
    Phase 3 (40-100%): Full neuro-symbolic training
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        device: str = "cuda",
        use_amp: bool = True,
        grad_clip: float = 1.0,
        log_interval: int = 10,
        output_dir: Path = Path("outputs"),
        total_epochs: int = 100,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and device == "cuda"
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.output_dir = Path(output_dir)
        self.total_epochs = total_epochs

        self.scaler = GradScaler(enabled=self.use_amp)
        self.early_stopping = EarlyStopping(patience=25)
        self.history = {
            "train_loss": [], "train_reg_loss": [], "train_constraint_loss": [],
            "val_loss": [], "val_rmse": [],
            "constraint_satisfaction": [],
            "lr": [],
        }

    def _get_phase_weights(self, epoch: int) -> Tuple[float, float, float, float]:
        """Get (w_reg, w_constraint, w_domain, w_hierarchy) for current epoch."""
        fraction = epoch / max(self.total_epochs, 1)
        if fraction < 0.15:
            return (1.0, 0.3, 0.0, 0.0)
        elif fraction < 0.40:
            return (1.0, 1.0, 0.3, 0.1)
        else:
            return (1.0, 2.0, 1.0, 0.5)

    def compute_loss(self, outputs: Dict, batch: Dict, epoch: int) -> Tuple[torch.Tensor, Dict]:
        """Compute the full LTN loss with phase-aware weighting."""
        from src.models.neuro_symbolic.constraints import compute_ltn_loss

        predictions = outputs["predictions"]
        targets = batch["targets"]
        constraint_sats = outputs.get("constraint_sats", {})

        weights = self._get_phase_weights(epoch)

        # If using evidential heads, use evidential loss for regression
        if hasattr(self.model, "use_evidential") and self.model.use_evidential:
            from src.models.neuro_symbolic.ltn_model import EvidentialHead, BASE_TARGETS
            base_preds = outputs["base_preds"]
            evid_loss = 0.0
            for name in BASE_TARGETS:
                gamma = base_preds[name]
                nu = base_preds[f"{name}_nu"]
                alpha = base_preds[f"{name}_alpha"]
                beta = base_preds[f"{name}_beta"]
                # Get corresponding target
                tgt_idx = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"].index(name)
                if name == "Dry_Clover_g":
                    tgt_val = targets[:, 0]
                elif name == "Dry_Dead_g":
                    tgt_val = targets[:, 1]
                else:
                    tgt_val = targets[:, 2]
                evid_loss += EvidentialHead.evidential_loss(gamma, nu, alpha, beta, tgt_val)
            reg_loss = evid_loss / 3.0
        else:
            reg_loss = torch.nn.functional.smooth_l1_loss(predictions, targets, beta=1.0)

        # Constraint loss
        if constraint_sats:
            constraint_loss = 0.5 * (
                (1.0 - constraint_sats.get("mass_conservation", torch.ones(1, device=self.device))).mean() +
                (1.0 - constraint_sats.get("gdm_identity", torch.ones(1, device=self.device))).mean()
            )
            domain_loss = 0.0
            for key in ["ndvi_implies_green", "ndvi_implies_low_dead", "species_clover"]:
                if key in constraint_sats:
                    domain_loss += (1.0 - constraint_sats[key]).mean()
            domain_loss = domain_loss / max(1, sum(1 for k in ["ndvi_implies_green", "ndvi_implies_low_dead", "species_clover"] if k in constraint_sats))

            hierarchy_loss = (1.0 - constraint_sats.get("gdm_subset_total", torch.ones(1, device=self.device))).mean()
        else:
            constraint_loss = torch.tensor(0.0, device=self.device)
            domain_loss = torch.tensor(0.0, device=self.device)
            hierarchy_loss = torch.tensor(0.0, device=self.device)

        w_reg, w_constraint, w_domain, w_hierarchy = weights

        total_loss = (w_reg * reg_loss +
                      w_constraint * constraint_loss +
                      w_domain * domain_loss +
                      w_hierarchy * hierarchy_loss)

        return total_loss, {
            "reg_loss": reg_loss.detach(),
            "constraint_loss": constraint_loss.detach(),
            "domain_loss": domain_loss.detach(),
            "hierarchy_loss": hierarchy_loss.detach(),
            "total_loss": total_loss.detach(),
        }

    def train_epoch(self, epoch: int) -> Dict:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        total_reg = 0.0
        total_constraint = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(batch)
                    loss, components = self.compute_loss(outputs, batch, epoch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                loss, components = self.compute_loss(outputs, batch, epoch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            total_reg += components["reg_loss"].item()
            total_constraint += components["constraint_loss"].item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        n = max(num_batches, 1)
        return {
            "train_loss": total_loss / n,
            "train_reg_loss": total_reg / n,
            "train_constraint_loss": total_constraint / n,
        }

    @torch.no_grad()
    def validate(self) -> Dict:
        """Validate and return comprehensive metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        all_constraint_sats = {}
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            if self.use_amp:
                with autocast():
                    outputs = self.model(batch)
                    loss, _ = self.compute_loss(outputs, batch, self.total_epochs)  # Use phase 3 weights for val
            else:
                outputs = self.model(batch)
                loss, _ = self.compute_loss(outputs, batch, self.total_epochs)

            total_loss += loss.item()
            all_preds.append(outputs["predictions"].cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())

            # Track constraint satisfactions
            if "constraint_sats" in outputs:
                for k, v in outputs["constraint_sats"].items():
                    if isinstance(v, torch.Tensor) and v.ndim <= 1:
                        if k not in all_constraint_sats:
                            all_constraint_sats[k] = []
                        all_constraint_sats[k].append(v.cpu().numpy())
            num_batches += 1

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        rmse_per_target = np.sqrt(np.mean((all_preds - all_targets) ** 2, axis=0))
        overall_rmse = float(np.sqrt(np.mean((all_preds - all_targets) ** 2)))

        # Average constraint satisfactions
        avg_sats = {}
        for k, v_list in all_constraint_sats.items():
            try:
                avg_sats[k] = float(np.mean(np.concatenate(v_list)))
            except (ValueError, TypeError):
                avg_sats[k] = float(np.mean([np.mean(v) for v in v_list]))

        return {
            "val_loss": total_loss / max(num_batches, 1),
            "val_rmse": overall_rmse,
            "val_rmse_per_target": rmse_per_target.tolist(),
            "constraint_sats": avg_sats,
            "predictions": all_preds,
            "targets": all_targets,
        }

    def fit(self, num_epochs: int, start_epoch: int = 0,
            unfreeze_backbone_at: int = 10) -> Dict:
        """Full training loop with phased constraint introduction."""
        best_val_rmse = float("inf")
        best_model_path = None

        # Freeze backbone initially
        if hasattr(self.model, 'freeze_backbone'):
            self.model.freeze_backbone()
            print("Backbone frozen for initial training")

        for epoch in range(start_epoch, start_epoch + num_epochs):
            epoch_start = time.time()

            # Unfreeze backbone after warmup
            if epoch == unfreeze_backbone_at and hasattr(self.model, 'unfreeze_backbone'):
                self.model.unfreeze_backbone()
                print(f"Backbone unfrozen at epoch {epoch}")

            # Train
            train_metrics = self.train_epoch(epoch)
            for k, v in train_metrics.items():
                if k not in self.history:
                    self.history[k] = []
                self.history[k].append(v)

            # Validate
            val_metrics = self.validate()
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_rmse"].append(val_metrics["val_rmse"])

            # Track average constraint satisfaction
            if val_metrics["constraint_sats"]:
                avg_sat = np.mean(list(val_metrics["constraint_sats"].values()))
                self.history["constraint_satisfaction"].append(avg_sat)
            else:
                self.history["constraint_satisfaction"].append(0.0)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)

            # Scheduler step
            if self.scheduler is not None:
                try:
                    self.scheduler.step(val_metrics["val_rmse"])
                except TypeError:
                    self.scheduler.step()

            # Early stopping
            is_best = self.early_stopping(val_metrics["val_rmse"], epoch)
            if is_best:
                best_val_rmse = val_metrics["val_rmse"]
                best_model_path = self.output_dir / "models" / "best_ltn_model.pt"
                save_checkpoint(self.model, self.optimizer, epoch, val_metrics, best_model_path)

            # Logging
            if epoch % self.log_interval == 0 or is_best:
                epoch_time = time.time() - epoch_start
                phase_weights = self._get_phase_weights(epoch)
                rmse_str = ", ".join(f"{v:.3f}" for v in val_metrics["val_rmse_per_target"])
                sat_str = ""
                if val_metrics["constraint_sats"]:
                    sat_str = " | Sat: " + ", ".join(
                        f"{k.split('_')[0]}:{v:.3f}"
                        for k, v in sorted(val_metrics["constraint_sats"].items())
                        if isinstance(v, (int, float))
                    )
                print(
                    f"Epoch {epoch:3d} | "
                    f"Loss: {train_metrics['train_loss']:.4f} | "
                    f"Val RMSE: {val_metrics['val_rmse']:.4f} [{rmse_str}] | "
                    f"w={phase_weights}{sat_str} | "
                    f"LR: {current_lr:.2e} | "
                    f"{epoch_time:.1f}s"
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
