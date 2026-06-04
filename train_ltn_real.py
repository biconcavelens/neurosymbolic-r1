"""Train LTN for real — 60 epochs, full images, early backbone unfreeze."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'.')
import pandas as pd, numpy as np, torch, json, time
from pathlib import Path
from src.config import CSV_PATH, IMAGE_DIR, OUTPUT_DIR, FIGURE_DIR
from src.data.tabular_encoder import TabularPreprocessor
from src.data.splitter import StratifiedGroupKFoldSplitter
from src.data.dataset import create_dataloaders
from src.evaluation.metrics import compute_metrics, compute_constraint_metrics, TARGET_NAMES
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

df = pd.read_csv(CSV_PATH)
preprocessor = TabularPreprocessor(); preprocessor.fit(df)
tabular_dim = preprocessor.output_dim

splitter = StratifiedGroupKFoldSplitter(n_splits=5, random_state=42)
train_imgs, val_imgs = next(splitter.get_image_splits(df))
train_idx = df[df["image_path"].isin(train_imgs)].index.values
val_idx = df[df["image_path"].isin(val_imgs)].index.values
print(f"Train: {len(train_imgs)} images | Val: {len(val_imgs)} images")

train_loader, val_loader = create_dataloaders(
    df=df, image_dir=IMAGE_DIR, preprocessor=preprocessor,
    train_idx=train_idx, val_idx=val_idx,
    batch_size=16, num_workers=0, image_size=(512, 256),
)

# ── XGBoost baseline ──
from src.models.baselines.tree_models import XGBoostBaseline, _prepare_tabular_data
train_df = df[df["image_path"].isin(train_imgs)]
val_df = df[df["image_path"].isin(val_imgs)]
xgb = XGBoostBaseline(n_estimators=300, max_depth=5, learning_rate=0.05)
xgb.fit(train_df, preprocessor)
xgb_pred = xgb.predict(val_df, preprocessor)
_, xgb_true = _prepare_tabular_data(val_df, preprocessor)
xgb_rmse = float(np.sqrt(np.mean((xgb_pred - xgb_true)**2)))
print(f"\nXGBoost RMSE: {xgb_rmse:.3f}")

# ── LTN Training ──
from src.models.neuro_symbolic.ltn_model import LTNModel
from src.training.ltn_trainer import LTNTrainer
from src.training.callbacks import LRSchedulerCallback

EPOCHS = 60

model = LTNModel(
    tabular_input_dim=tabular_dim,
    backbone="efficientnet_b0",
    image_feature_dim=256,
    tabular_hidden=(64, 32, 16),
    fusion_hidden=(512, 256, 128),
    dropout=0.3,
    use_learned_weights=True,
    use_learned_predicates=True,
    constraint_eps=0.5,
).to(device)

optim = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if "backbone" in n and "image_encoder" in n], "lr": 1e-4},
    {"params": [p for n, p in model.named_parameters() if not ("backbone" in n and "image_encoder" in n)], "lr": 1e-3},
], weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=1e-6)

trainer = LTNTrainer(model, train_loader, val_loader, optim, device=device,
                     use_amp=True, output_dir=OUTPUT_DIR, total_epochs=EPOCHS,
                     log_interval=5)
trainer.scheduler = LRSchedulerCallback(sched)

print(f"\nTraining LTN for {EPOCHS} epochs...")
t0 = time.time()
result = trainer.fit(EPOCHS, unfreeze_backbone_at=5)
print(f"Training time: {(time.time()-t0)/60:.1f} min")

# ── Evaluate ──
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

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"XGBoost       RMSE={xgb_rmse:.3f}  R²={compute_metrics(xgb_pred, xgb_true)['overall_r2']:.3f}  ConstrRMSE={compute_constraint_metrics(xgb_pred)['combined_constraint_rmse']:.4f}")
print(f"LTN           RMSE={m['overall_rmse']:.3f}  R²={m['overall_r2']:.3f}  ConstrRMSE={m['combined_constraint_rmse']:.4f}")
print(f"LTN/XGBoost   {m['overall_rmse']/xgb_rmse:.2f}x")
print(f"LTN violation rate: {m['c1_violation_rate']*100:.1f}%")
print(f"\nPer-target R²:")
for t in TARGET_NAMES:
    xgb_r2 = compute_metrics(xgb_pred, xgb_true)[f'{t}_r2']
    print(f"  {t:<18} XGBoost={xgb_r2:.3f}  LTN={m[f'{t}_r2']:.3f}")

# ── Convergence plot ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(result["history"]["val_rmse"], "b-", linewidth=2, label="LTN Val RMSE")
ax1.axhline(y=xgb_rmse, color="orange", linestyle="--", linewidth=2, label=f"XGBoost ({xgb_rmse:.1f})")
ax1.axvline(x=5, color="gray", linestyle=":", alpha=0.5, label="Backbone unfrozen")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("RMSE (g)")
ax1.set_title("LTN Convergence vs XGBoost"); ax1.legend()
if "constraint_satisfaction" in result["history"]:
    ax2.plot(result["history"]["constraint_satisfaction"], "g-", linewidth=2)
    ax2.set_ylim(0, 1.05)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Constraint Satisfaction")
ax2.set_title("Logical Constraint Satisfaction")
plt.tight_layout(); plt.savefig(FIGURE_DIR/"fig_real_convergence.png", dpi=150); plt.close()
print(f"\nPlot: {FIGURE_DIR/'fig_real_convergence.png'}")

# Save results
real_results = {
    "xgb_rmse": xgb_rmse, "xgb_r2": compute_metrics(xgb_pred, xgb_true)['overall_r2'],
    "xgb_constr_rmse": compute_constraint_metrics(xgb_pred)['combined_constraint_rmse'],
    "ltn_rmse": float(m["overall_rmse"]), "ltn_r2": float(m["overall_r2"]),
    "ltn_constr_rmse": float(m["combined_constraint_rmse"]),
    "ltn_violation_rate": float(m["c1_violation_rate"]),
    "ltn_best_epoch": result["best_epoch"],
    "per_target": {t: {"ltn_r2": float(m[f"{t}_r2"]), "ltn_rmse": float(m[f"{t}_rmse"]),
                        "xgb_r2": float(compute_metrics(xgb_pred, xgb_true)[f"{t}_r2"]),
                        "xgb_rmse": float(compute_metrics(xgb_pred, xgb_true)[f"{t}_rmse"])}
                   for t in TARGET_NAMES},
}
with open(OUTPUT_DIR/"real_ltn_results.json", "w") as f:
    json.dump(real_results, f, indent=2)
print(f"Results: {OUTPUT_DIR/'real_ltn_results.json'}")
print("\nDone!")
