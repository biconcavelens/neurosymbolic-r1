"""Cross-validation orchestrator for model evaluation."""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Type
from copy import deepcopy
import json

from src.data.splitter import StratifiedGroupKFoldSplitter
from src.data.dataset import create_dataloaders, PastureDataset
from src.data.tabular_encoder import TabularPreprocessor
from src.evaluation.metrics import compute_metrics, compute_constraint_metrics, TARGET_NAMES
from src.config import default_config, ExperimentConfig, OUTPUT_DIR


class CrossValidator:
    """Orchestrates k-fold cross-validation for model evaluation."""

    def __init__(self, config: ExperimentConfig = None):
        self.config = config or default_config
        self.splitter = StratifiedGroupKFoldSplitter(
            n_splits=self.config.training.num_folds,
            random_state=self.config.training.seed,
        )
        self.results = {}

    def run_baseline_cv(
        self, df: pd.DataFrame, model_class, preprocessor: TabularPreprocessor,
        model_name: str, **model_kwargs
    ) -> Dict:
        """Run cross-validation for a tree-based or linear baseline.

        These models use tabular features only (no images).
        """
        print(f"\n{'='*60}")
        print(f"Running {model_name} — 5-fold Cross-Validation")
        print(f"{'='*60}")

        fold_metrics = []
        all_fold_preds = []
        all_fold_targets = []

        for fold_idx, (train_imgs, val_imgs) in enumerate(self.splitter.get_image_splits(df)):
            print(f"\n--- Fold {fold_idx + 1}/{self.config.training.num_folds} ---")

            train_df = df[df["image_path"].isin(train_imgs)]
            val_df = df[df["image_path"].isin(val_imgs)]

            # Train
            model = model_class(**model_kwargs)
            model.fit(train_df, preprocessor)

            # Predict
            y_pred = model.predict(val_df, preprocessor)

            # Get ground truth
            from src.models.baselines.tree_models import _prepare_tabular_data
            _, y_true = _prepare_tabular_data(val_df, preprocessor)

            # Metrics
            metrics = compute_metrics(y_pred, y_true)
            constraint_metrics = compute_constraint_metrics(y_pred)
            metrics.update(constraint_metrics)
            metrics["fold"] = fold_idx
            fold_metrics.append(metrics)

            all_fold_preds.append(y_pred)
            all_fold_targets.append(y_true)

            print(f"  RMSE: {metrics['overall_rmse']:.4f} | "
                  f"Constraint RMSE: {metrics['combined_constraint_rmse']:.4f}")

        self.results[model_name] = {
            "fold_metrics": fold_metrics,
            "predictions": all_fold_preds,
            "targets": all_fold_targets,
        }

        self._print_summary(model_name, fold_metrics)
        return self.results[model_name]

    def run_neural_cv(
        self, df: pd.DataFrame, model_class, preprocessor: TabularPreprocessor,
        model_name: str, image_dir: Path, use_ltn_trainer: bool = False,
        **model_kwargs
    ) -> Dict:
        """Run cross-validation for a neural network model."""
        print(f"\n{'='*60}")
        print(f"Running {model_name} — 5-fold Cross-Validation")
        print(f"{'='*60}")

        fold_metrics = []
        all_fold_preds = []
        all_fold_targets = []

        for fold_idx, (train_imgs, val_imgs) in enumerate(self.splitter.get_image_splits(df)):
            print(f"\n--- Fold {fold_idx + 1}/{self.config.training.num_folds} ---")

            train_idx = df[df["image_path"].isin(train_imgs)].index.values
            val_idx = df[df["image_path"].isin(val_imgs)].index.values

            # Create dataloaders
            train_loader, val_loader = create_dataloaders(
                df=df, image_dir=image_dir, preprocessor=preprocessor,
                train_idx=train_idx, val_idx=val_idx,
                batch_size=self.config.training.batch_size,
                num_workers=self.config.training.num_workers,
                image_size=self.config.data.image_size,
            )

            # Build model
            tabular_input_dim = preprocessor.output_dim
            model = model_class(tabular_input_dim=tabular_input_dim, **model_kwargs)
            model = model.to(self.config.device)

            # Optimizer — handle all model parameters (some models may not have all submodules)
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if "backbone" in name and "image_encoder" in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)

            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": self.config.training.lr_backbone} if backbone_params else None,
                {"params": head_params, "lr": self.config.training.lr_heads},
            ], weight_decay=self.config.training.weight_decay)
            # Filter out None groups
            optimizer.param_groups = [g for g in optimizer.param_groups if "params" in g and g["params"]]

            # LR scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.training.num_epochs,
                eta_min=self.config.training.min_lr,
            )
            from src.training.callbacks import LRSchedulerCallback
            lr_callback = LRSchedulerCallback(scheduler)

            # Train
            if use_ltn_trainer:
                from src.training.ltn_trainer import LTNTrainer
                trainer = LTNTrainer(
                    model=model, train_loader=train_loader, val_loader=val_loader,
                    optimizer=optimizer, device=self.config.device,
                    use_amp=self.config.training.use_amp,
                    output_dir=OUTPUT_DIR,
                    total_epochs=self.config.training.num_epochs,
                )
                trainer.scheduler = lr_callback
                result = trainer.fit(self.config.training.num_epochs)
            else:
                from src.training.trainer import Trainer
                loss_fn = torch.nn.SmoothL1Loss(beta=1.0)
                trainer = Trainer(
                    model=model, train_loader=train_loader, val_loader=val_loader,
                    loss_fn=loss_fn, optimizer=optimizer, device=self.config.device,
                    use_amp=self.config.training.use_amp,
                    output_dir=OUTPUT_DIR,
                )
                trainer.scheduler = lr_callback
                result = trainer.fit(self.config.training.num_epochs)

            # Evaluate on validation fold
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    outputs = model(batch)
                    all_preds.append(outputs["predictions"].cpu().numpy())
                    all_targets.append(batch["targets"].cpu().numpy())

            y_pred = np.concatenate(all_preds, axis=0)
            y_true = np.concatenate(all_targets, axis=0)

            # Metrics
            metrics = compute_metrics(y_pred, y_true)
            constraint_metrics = compute_constraint_metrics(y_pred)
            metrics.update(constraint_metrics)
            metrics["fold"] = fold_idx
            metrics["best_val_rmse"] = result.get("best_val_rmse", float("inf"))
            fold_metrics.append(metrics)

            all_fold_preds.append(y_pred)
            all_fold_targets.append(y_true)

            print(f"  RMSE: {metrics['overall_rmse']:.4f} | "
                  f"Constraint RMSE: {metrics['combined_constraint_rmse']:.4f}")

            # Clean up to free memory
            del model, optimizer, trainer
            torch.cuda.empty_cache()

        self.results[model_name] = {
            "fold_metrics": fold_metrics,
            "predictions": all_fold_preds,
            "targets": all_fold_targets,
        }

        self._print_summary(model_name, fold_metrics)
        return self.results[model_name]

    def _print_summary(self, model_name: str, fold_metrics: List[Dict]):
        """Print cross-validation summary."""
        print(f"\n--- {model_name} Summary ---")
        rmse_vals = [m["overall_rmse"] for m in fold_metrics]
        c_rmse_vals = [m["combined_constraint_rmse"] for m in fold_metrics]
        print(f"  RMSE: {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}")
        print(f"  Constraint RMSE: {np.mean(c_rmse_vals):.4f} ± {np.std(c_rmse_vals):.4f}")

        # Per-target
        for t in TARGET_NAMES:
            t_rmse = [m[f"{t}_rmse"] for m in fold_metrics]
            t_r2 = [m[f"{t}_r2"] for m in fold_metrics]
            print(f"  {t}: RMSE={np.mean(t_rmse):.3f}±{np.std(t_rmse):.3f}, R²={np.mean(t_r2):.3f}±{np.std(t_r2):.3f}")

    def save_results(self, filepath: Path):
        """Save fold results to JSON (excluding large arrays)."""
        serializable = {}
        for model_name, data in self.results.items():
            serializable[model_name] = {
                "fold_metrics": data["fold_metrics"],
            }
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"Results saved to {filepath}")
