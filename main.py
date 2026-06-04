#!/usr/bin/env python
"""Main entry point for the Neuro-Symbolic Biomass Prediction project.

Usage:
    python main.py --mode baselines    # Train all baselines with 5-fold CV
    python main.py --mode ltn          # Train LTN models with 5-fold CV
    python main.py --mode all          # Run everything (baselines + LTN + evaluation)
    python main.py --mode evaluate     # Generate evaluation figures from saved results
    python main.py --mode quick        # Quick test run (1 fold, 5 epochs, reduced size)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="Neuro-Symbolic Biomass Prediction")
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["baselines", "ltn", "all", "evaluate", "quick"],
                        help="Run mode")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--folds", type=int, default=None,
                        help="Override number of CV folds")
    args = parser.parse_args()

    if args.mode == "baselines":
        print("Running baseline evaluation...")
        import run_baselines
        run_baselines.main()

    elif args.mode == "ltn":
        print("Running LTN neuro-symbolic evaluation...")
        import run_ltn
        run_ltn.main()

    elif args.mode == "all":
        print("=" * 60)
        print("FULL NEURO-SYMBOLIC BIOMASS PREDICTION PIPELINE")
        print("=" * 60)
        print("\nStep 1/3: Training Baselines...")
        import run_baselines
        run_baselines.main()

        print("\nStep 2/3: Training LTN Models...")
        import run_ltn
        run_ltn.main()

        print("\nStep 3/3: Generating Evaluation Report...")
        import run_evaluation
        run_evaluation.main()
        print("\nPipeline complete! Check outputs/ for results.")

    elif args.mode == "evaluate":
        print("Generating evaluation report from saved results...")
        import run_evaluation
        run_evaluation.main()

    elif args.mode == "quick":
        print("Quick test run — 1 fold, 10 epochs, reduced resolution...")
        _quick_test(args)


def _quick_test(args):
    """Small test run to verify everything works."""
    import pandas as pd
    import torch
    import numpy as np
    from src.config import CSV_PATH, IMAGE_DIR, OUTPUT_DIR, default_config
    from src.data.tabular_encoder import TabularPreprocessor
    from src.data.splitter import StratifiedGroupKFoldSplitter
    from src.data.dataset import create_dataloaders

    df = pd.read_csv(CSV_PATH)
    preprocessor = TabularPreprocessor()
    preprocessor.fit(df)

    splitter = StratifiedGroupKFoldSplitter(n_splits=5, random_state=42)
    train_imgs, val_imgs = next(splitter.get_image_splits(df))
    train_idx = df[df["image_path"].isin(train_imgs)].index.values
    val_idx = df[df["image_path"].isin(val_imgs)].index.values

    epochs = args.epochs or 10
    batch_size = args.batch_size or 8

    train_loader, val_loader = create_dataloaders(
        df=df, image_dir=IMAGE_DIR, preprocessor=preprocessor,
        train_idx=train_idx, val_idx=val_idx,
        batch_size=batch_size, num_workers=0, image_size=(256, 128),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Neural baseline
    from src.models.baselines.neural_only import NeuralOnlyModel
    from src.training.trainer import Trainer

    print("\n--- Neural-Only Baseline ---")
    model = NeuralOnlyModel(
        tabular_input_dim=preprocessor.output_dim,
        image_feature_dim=128, tabular_hidden=(32, 16),
        fusion_hidden=(128, 64), dropout=0.3,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, train_loader, val_loader,
                      torch.nn.SmoothL1Loss(), optimizer, device=device,
                      use_amp=(device == "cuda"), output_dir=OUTPUT_DIR)
    result = trainer.fit(num_epochs=epochs)
    print(f"Neural-Only Best Val RMSE: {result['best_val_rmse']:.4f}")

    # LTN model
    from src.models.neuro_symbolic.ltn_model import LTNModel
    from src.training.ltn_trainer import LTNTrainer

    print("\n--- LTN Neuro-Symbolic ---")
    model2 = LTNModel(
        tabular_input_dim=preprocessor.output_dim,
        image_feature_dim=128, tabular_hidden=(32, 16),
        fusion_hidden=(128, 64), dropout=0.3,
        use_learned_weights=True, use_learned_predicates=True,
    ).to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    trainer2 = LTNTrainer(model2, train_loader, val_loader,
                          optimizer2, device=device,
                          use_amp=(device == "cuda"), output_dir=OUTPUT_DIR,
                          total_epochs=epochs)
    result2 = trainer2.fit(num_epochs=epochs)
    print(f"LTN Best Val RMSE: {result2['best_val_rmse']:.4f}")
    if "constraint_satisfaction" in result2.get("history", {}):
        print(f"Final constraint satisfaction: {result2['history']['constraint_satisfaction'][-1]:.4f}")

    # XGBoost
    from src.models.baselines.tree_models import XGBoostBaseline, _prepare_tabular_data
    train_df = df[df["image_path"].isin(train_imgs)]
    val_df = df[df["image_path"].isin(val_imgs)]
    xgb = XGBoostBaseline(n_estimators=100)
    xgb.fit(train_df, preprocessor)
    y_pred = xgb.predict(val_df, preprocessor)
    _, y_true = _prepare_tabular_data(val_df, preprocessor)
    xgb_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print(f"\nXGBoost RMSE: {xgb_rmse:.4f}")

    print(f"\nQuick test complete!")
    print(f"Neural-Only: {result['best_val_rmse']:.2f} | LTN: {result2['best_val_rmse']:.2f} | XGBoost: {xgb_rmse:.2f}")


if __name__ == "__main__":
    main()
