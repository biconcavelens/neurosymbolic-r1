"""Focused LTN training showing convergence to competitive RMSE. ~20 min."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'.')
import pandas as pd, numpy as np, torch
from src.config import CSV_PATH, IMAGE_DIR, OUTPUT_DIR, FIGURE_DIR
from src.data.tabular_encoder import TabularPreprocessor
from src.data.splitter import StratifiedGroupKFoldSplitter
from src.data.dataset import create_dataloaders
from src.evaluation.metrics import compute_metrics, compute_constraint_metrics, TARGET_NAMES
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv(CSV_PATH)
preprocessor = TabularPreprocessor(); preprocessor.fit(df)
tabular_dim = preprocessor.output_dim

splitter = StratifiedGroupKFoldSplitter(n_splits=5, random_state=42)
train_imgs, val_imgs = next(splitter.get_image_splits(df))
train_idx = df[df["image_path"].isin(train_imgs)].index.values
val_idx = df[df["image_path"].isin(val_imgs)].index.values

train_loader, val_loader = create_dataloaders(
    df=df, image_dir=IMAGE_DIR, preprocessor=preprocessor,
    train_idx=train_idx, val_idx=val_idx,
    batch_size=16, num_workers=0, image_size=(512, 256),
)

# Tabular baselines
train_df = df[df["image_path"].isin(train_imgs)]
val_df = df[df["image_path"].isin(val_imgs)]
from src.models.baselines.tree_models import XGBoostBaseline, _prepare_tabular_data
xgb = XGBoostBaseline(n_estimators=200, max_depth=5, learning_rate=0.05)
xgb.fit(train_df, preprocessor)
xgb_pred = xgb.predict(val_df, preprocessor)
_, xgb_true = _prepare_tabular_data(val_df, preprocessor)
xgb_rmse = float(np.sqrt(np.mean((xgb_pred - xgb_true)**2)))
print(f"XGBoost RMSE: {xgb_rmse:.3f}")

# LTN model — 50 epochs, unfreeze backbone at epoch 5
from src.models.neuro_symbolic.ltn_model import LTNModel
from src.training.ltn_trainer import LTNTrainer
from src.training.callbacks import LRSchedulerCallback

model = LTNModel(tabular_input_dim=tabular_dim, image_feature_dim=256,
                 tabular_hidden=(64,32,16), fusion_hidden=(512,256,128),
                 dropout=0.3, use_learned_weights=True,
                 use_learned_predicates=True, constraint_eps=0.5).to(device)

optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=50, eta_min=1e-6)
trainer = LTNTrainer(model, train_loader, val_loader, optim, device=device,
                     use_amp=True, output_dir=OUTPUT_DIR, total_epochs=50)
trainer.scheduler = LRSchedulerCallback(sched)

print("\nTraining LTN for 50 epochs (unfreeze backbone at epoch 5)...")
result = trainer.fit(50, unfreeze_backbone_at=5)

# Evaluate
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch)
        all_preds.append(out["predictions"].cpu().numpy())
        all_targets.append(batch["targets"].cpu().numpy())
yp = np.concatenate(all_preds); yt = np.concatenate(all_targets)
m = compute_metrics(yp, yt); m.update(compute_constraint_metrics(yp))
ltn_rmse = m["overall_rmse"]

print(f"\n{'='*60}")
print(f"FINAL: XGBoost RMSE={xgb_rmse:.3f} | LTN RMSE={ltn_rmse:.3f} | LTN ConstrRMSE={m['combined_constraint_rmse']:.4f}")
print(f"LTN closes gap: LTN/XGBoost = {ltn_rmse/xgb_rmse:.2f}x")
print(f"LTN constraint violation rate: {m['c1_violation_rate']*100:.1f}%")
for t in TARGET_NAMES:
    print(f"  {t}: LTN R²={m[f'{t}_r2']:.3f} | XGBoost R²={compute_metrics(xgb_pred, xgb_true)[f'{t}_r2']:.3f}")

# Convergence plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(result["history"]["val_rmse"], "b-", label="LTN Val RMSE", linewidth=2)
ax1.axhline(y=xgb_rmse, color="orange", linestyle="--", label=f"XGBoost ({xgb_rmse:.1f})", linewidth=2)
ax1.axvline(x=5, color="gray", linestyle=":", alpha=0.5, label="Backbone unfrozen")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("RMSE (g)"); ax1.set_title("LTN Convergence"); ax1.legend()
if "constraint_satisfaction" in result["history"]:
    ax2.plot(result["history"]["constraint_satisfaction"], "g-", linewidth=2)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Constraint Satisfaction")
ax2.set_title("Logical Constraint Satisfaction (0→1)"); ax2.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "fig_convergence.png", dpi=150)
print(f"\nConvergence plot: {FIGURE_DIR / 'fig_convergence.png'}")

# Save
import json
with open(OUTPUT_DIR / "final_ltn_results.json", "w") as f:
    json.dump({"xgb_rmse": xgb_rmse, "ltn_rmse": ltn_rmse, "ltn_constraint_rmse": m["combined_constraint_rmse"],
               "per_target": {t: {"ltn_r2": m[f"{t}_r2"], "ltn_rmse": m[f"{t}_rmse"]} for t in TARGET_NAMES}}, f, indent=2)
print("Done!")
