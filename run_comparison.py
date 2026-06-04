"""Focused comparison run: 1 fold, 20 epochs per neural model, all baselines.

Shows the key results quickly (~30 min):
- Tabular baselines (already fast): XGBoost, LightGBM, Ridge
- Neural-Only vs Neural+Constraint vs LTN on 1 fold
- Generates comparison figures
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

import pandas as pd, numpy as np, torch
from pathlib import Path
from src.config import CSV_PATH, IMAGE_DIR, OUTPUT_DIR, FIGURE_DIR
from src.data.tabular_encoder import TabularPreprocessor
from src.data.splitter import StratifiedGroupKFoldSplitter
from src.data.dataset import create_dataloaders
from src.evaluation.metrics import compute_metrics, compute_constraint_metrics, TARGET_NAMES
from src.evaluation.statistical_tests import compare_models

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Load data ──
df = pd.read_csv(CSV_PATH)
preprocessor = TabularPreprocessor()
preprocessor.fit(df)

# ── Get one fold ──
splitter = StratifiedGroupKFoldSplitter(n_splits=5, random_state=42)
train_imgs, val_imgs = next(splitter.get_image_splits(df))
train_idx = df[df["image_path"].isin(train_imgs)].index.values
val_idx = df[df["image_path"].isin(val_imgs)].index.values
print(f"Train images: {len(train_imgs)}, Val images: {len(val_imgs)}")

train_loader, val_loader = create_dataloaders(
    df=df, image_dir=IMAGE_DIR, preprocessor=preprocessor,
    train_idx=train_idx, val_idx=val_idx,
    batch_size=16, num_workers=0, image_size=(512, 256),
)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

results = {}
EPOCHS = 20
tabular_dim = preprocessor.output_dim

# ── 1. Tabular Baselines ──
print("\n" + "="*60)
print("TABULAR BASELINES")
print("="*60)

train_df = df[df["image_path"].isin(train_imgs)]
val_df = df[df["image_path"].isin(val_imgs)]

from src.models.baselines.tree_models import XGBoostBaseline, LightGBMBaseline, _prepare_tabular_data
from src.models.baselines.linear_models import RidgeBaseline

for name, model_cls, kwargs in [
    ("XGBoost", XGBoostBaseline, {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05}),
    ("LightGBM", LightGBMBaseline, {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05}),
    ("Ridge", RidgeBaseline, {"alpha": 1.0, "use_poly": True, "poly_degree": 2}),
]:
    print(f"\n--- {name} ---")
    model = model_cls(**kwargs)
    model.fit(train_df, preprocessor)
    y_pred = model.predict(val_df, preprocessor)
    _, y_true = _prepare_tabular_data(val_df, preprocessor)
    metrics = compute_metrics(y_pred, y_true)
    metrics.update(compute_constraint_metrics(y_pred))
    results[name] = {"predictions": y_pred, "targets": y_true, "metrics": metrics}
    print(f"  RMSE: {metrics['overall_rmse']:.3f}, Constraint RMSE: {metrics['combined_constraint_rmse']:.4f}")
    for t in TARGET_NAMES:
        print(f"    {t}: RMSE={metrics[f'{t}_rmse']:.3f}, R²={metrics[f'{t}_r2']:.3f}")

# ── 2. Neural-Only ──
print("\n" + "="*60)
print("NEURAL-ONLY (CNN + Tabular, No Constraints)")
print("="*60)

from src.models.baselines.neural_only import NeuralOnlyModel
from src.training.trainer import Trainer

model_nn = NeuralOnlyModel(
    tabular_input_dim=tabular_dim,
    image_feature_dim=256,
    tabular_hidden=(64, 32, 16),
    fusion_hidden=(512, 256, 128),
    dropout=0.3,
).to(device)

optimizer = torch.optim.AdamW(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
from src.training.callbacks import LRSchedulerCallback

trainer_nn = Trainer(
    model_nn, train_loader, val_loader,
    torch.nn.SmoothL1Loss(beta=1.0), optimizer, device=device,
    use_amp=True, output_dir=OUTPUT_DIR,
)
trainer_nn.scheduler = LRSchedulerCallback(scheduler)
result_nn = trainer_nn.fit(EPOCHS)

# Get predictions
model_nn.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model_nn(batch)
        all_preds.append(out["predictions"].cpu().numpy())
        all_targets.append(batch["targets"].cpu().numpy())
y_pred_nn = np.concatenate(all_preds)
y_true_nn = np.concatenate(all_targets)

metrics_nn = compute_metrics(y_pred_nn, y_true_nn)
metrics_nn.update(compute_constraint_metrics(y_pred_nn))
results["NeuralOnly"] = {"predictions": y_pred_nn, "targets": y_true_nn, "metrics": metrics_nn}
print(f"\nNeuralOnly Best Val RMSE: {result_nn['best_val_rmse']:.3f}")
print(f"Constraint RMSE: {metrics_nn['combined_constraint_rmse']:.4f}")
del model_nn, optimizer, trainer_nn; torch.cuda.empty_cache()

# ── 3. Neural + Constraint Penalty ──
print("\n" + "="*60)
print("NEURAL + SOFT CONSTRAINT PENALTY (Ablation)")
print("="*60)

from src.models.baselines.neural_only import NeuralConstraintModel

model_nc = NeuralConstraintModel(
    tabular_input_dim=tabular_dim,
    image_feature_dim=256,
    tabular_hidden=(64, 32, 16),
    fusion_hidden=(512, 256, 128),
    dropout=0.3,
    constraint_weight=2.0,
).to(device)

optimizer = torch.optim.AdamW(model_nc.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
trainer_nc = Trainer(
    model_nc, train_loader, val_loader,
    torch.nn.SmoothL1Loss(beta=1.0), optimizer, device=device,
    use_amp=True, output_dir=OUTPUT_DIR,
)
trainer_nc.scheduler = LRSchedulerCallback(scheduler)
result_nc = trainer_nc.fit(EPOCHS)

model_nc.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model_nc(batch)
        all_preds.append(out["predictions"].cpu().numpy())
        all_targets.append(batch["targets"].cpu().numpy())
y_pred_nc = np.concatenate(all_preds)
y_true_nc = np.concatenate(all_targets)

metrics_nc = compute_metrics(y_pred_nc, y_true_nc)
metrics_nc.update(compute_constraint_metrics(y_pred_nc))
results["Neural+Constraint"] = {"predictions": y_pred_nc, "targets": y_true_nc, "metrics": metrics_nc}
print(f"\nNeural+Constraint Best Val RMSE: {result_nc['best_val_rmse']:.3f}")
print(f"Constraint RMSE: {metrics_nc['combined_constraint_rmse']:.4f}")
del model_nc, optimizer, trainer_nc; torch.cuda.empty_cache()

# ── 4. LTN Neuro-Symbolic ──
print("\n" + "="*60)
print("LTN NEURO-SYMBOLIC (Fuzzy Logic Constraints)")
print("="*60)

from src.models.neuro_symbolic.ltn_model import LTNModel
from src.training.ltn_trainer import LTNTrainer

model_ltn = LTNModel(
    tabular_input_dim=tabular_dim,
    image_feature_dim=256,
    tabular_hidden=(64, 32, 16),
    fusion_hidden=(512, 256, 128),
    dropout=0.3,
    use_learned_weights=True,
    use_learned_predicates=True,
    constraint_eps=0.5,
).to(device)

optimizer = torch.optim.AdamW(model_ltn.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
trainer_ltn = LTNTrainer(
    model_ltn, train_loader, val_loader,
    optimizer, device=device,
    use_amp=True, output_dir=OUTPUT_DIR, total_epochs=EPOCHS,
)
trainer_ltn.scheduler = LRSchedulerCallback(scheduler)
result_ltn = trainer_ltn.fit(EPOCHS)

model_ltn.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model_ltn(batch)
        all_preds.append(out["predictions"].cpu().numpy())
        all_targets.append(batch["targets"].cpu().numpy())
y_pred_ltn = np.concatenate(all_preds)
y_true_ltn = np.concatenate(all_targets)

metrics_ltn = compute_metrics(y_pred_ltn, y_true_ltn)
metrics_ltn.update(compute_constraint_metrics(y_pred_ltn))
results["LTN_NeuroSymbolic"] = {"predictions": y_pred_ltn, "targets": y_true_ltn, "metrics": metrics_ltn}
print(f"\nLTN Best Val RMSE: {result_ltn['best_val_rmse']:.3f}")
print(f"Constraint RMSE: {metrics_ltn['combined_constraint_rmse']:.4f}")
if "constraint_satisfaction" in result_ltn.get("history", {}):
    print(f"Final constraint satisfaction: {result_ltn['history']['constraint_satisfaction'][-1]:.4f}")

# ── Summary ──
print("\n" + "="*70)
print("FINAL COMPARISON")
print("="*70)
header = f"{'Model':<22} {'RMSE':>8} {'Constr.RMSE':>12} {'R²':>8} {'C1_Total':>10} {'C2_GDM':>10}"
print(header)
print("-" * len(header))
for name in ["XGBoost", "LightGBM", "Ridge", "NeuralOnly", "Neural+Constraint", "LTN_NeuroSymbolic"]:
    m = results[name]["metrics"]
    print(f"{name:<22} {m['overall_rmse']:>8.3f} {m['combined_constraint_rmse']:>12.4f} "
          f"{m['overall_r2']:>8.3f} {m['c1_total_equals_sum_rmse']:>10.4f} {m['c2_gdm_equals_sum_rmse']:>10.4f}")

# ── Generate Figures ──
print("\n" + "="*60)
print("GENERATING FIGURES")
print("="*60)

from src.visualization.predictions import (
    plot_predictions_vs_actual, plot_constraint_satisfaction, plot_model_comparison,
)
from src.visualization.embeddings import create_summary_table

# Collect predictions for plotting
model_names = ["XGBoost", "LightGBM", "Ridge", "NeuralOnly", "Neural+Constraint", "LTN_NeuroSymbolic"]
model_preds = [results[n]["predictions"] for n in model_names]
targets_viz = results["XGBoost"]["targets"]

plot_predictions_vs_actual(
    model_preds, targets_viz, model_names,
    save_path=FIGURE_DIR / "fig1_predictions_vs_actual.png",
)

# Constraint satisfaction
constraint_metrics = {}
for n in model_names:
    m = results[n]["metrics"]
    constraint_metrics[n] = {
        "c1_total_equals_sum_rmse": [m["c1_total_equals_sum_rmse"]],
        "c2_gdm_equals_sum_rmse": [m["c2_gdm_equals_sum_rmse"]],
        "c1_violation_rate": [m["c1_violation_rate"]],
        "c2_violation_rate": [m["c2_violation_rate"]],
        "combined_constraint_rmse": [m["combined_constraint_rmse"]],
    }

plot_constraint_satisfaction(
    constraint_metrics, model_names,
    save_path=FIGURE_DIR / "fig3_constraint_satisfaction.png",
)

# Model comparison
all_metrics_for_viz = {}
for n in model_names:
    all_metrics_for_viz[n] = {"fold_metrics": [results[n]["metrics"]]}

plot_model_comparison(
    all_metrics_for_viz, model_names,
    save_path=FIGURE_DIR / "fig5_model_comparison.png",
)

# Summary table
print("\n" + create_summary_table(all_metrics_for_viz, model_names))

# Save results
with open(OUTPUT_DIR / "comparison_results.json", "w") as f:
    json.dump({n: {"metrics": results[n]["metrics"]} for n in model_names}, f, indent=2, default=str)

print(f"\nFigures saved to {FIGURE_DIR}")
print(f"Results saved to {OUTPUT_DIR / 'comparison_results.json'}")
print("\nDone!")
