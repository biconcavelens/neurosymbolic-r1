#!/usr/bin/env python
"""Train and evaluate all baseline models using 5-fold cross-validation.

Baselines:
1. XGBoost (tabular only)
2. LightGBM (tabular only)
3. Ridge Regression with polynomial features
4. Neural-Only (CNN + tabular fusion, no constraints)
5. Neural + Soft Constraint Penalty (critical ablation)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import default_config, DATA_DIR, IMAGE_DIR, CSV_PATH, OUTPUT_DIR
from src.data.tabular_encoder import TabularPreprocessor
from src.data.splitter import StratifiedGroupKFoldSplitter
from src.evaluation.cross_validator import CrossValidator
from src.evaluation.metrics import compute_metrics, compute_constraint_metrics, TARGET_NAMES
from src.models.baselines.tree_models import XGBoostBaseline, LightGBMBaseline, _prepare_tabular_data
from src.models.baselines.linear_models import RidgeBaseline, ElasticNetBaseline
from src.models.baselines.neural_only import NeuralOnlyModel, NeuralConstraintModel


def main():
    print("=" * 60)
    print("Pasture Biomass Prediction — Baseline Evaluation")
    print("=" * 60)

    # Load data
    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded {len(df)} rows from {CSV_PATH}")
    print(f"Unique images: {df['image_path'].nunique()}")

    # Setup preprocessor
    preprocessor = TabularPreprocessor(
        state_embedding_dim=default_config.data.state_embedding_dim,
        species_embedding_dim=default_config.data.species_embedding_dim,
        date_cyclical=default_config.data.date_cyclical,
    )
    preprocessor.fit(df)
    tabular_input_dim = preprocessor.output_dim
    print(f"Tabular feature dimension: {tabular_input_dim}")

    # Cross-validator
    cv = CrossValidator(default_config)

    # ── Baseline 1: XGBoost ──
    print("\n" + "=" * 60)
    print("BASELINE 1: XGBoost (Tabular Only)")
    print("=" * 60)
    cv.run_baseline_cv(
        df=df, model_class=XGBoostBaseline,
        preprocessor=preprocessor, model_name="XGBoost",
        n_estimators=300, max_depth=5, learning_rate=0.05,
    )

    # ── Baseline 2: LightGBM ──
    print("\n" + "=" * 60)
    print("BASELINE 2: LightGBM (Tabular Only)")
    print("=" * 60)
    cv.run_baseline_cv(
        df=df, model_class=LightGBMBaseline,
        preprocessor=preprocessor, model_name="LightGBM",
        n_estimators=300, num_leaves=31, learning_rate=0.05,
    )

    # ── Baseline 3: Ridge Regression ──
    print("\n" + "=" * 60)
    print("BASELINE 3: Ridge Regression (Tabular Only, Polynomial Features)")
    print("=" * 60)
    cv.run_baseline_cv(
        df=df, model_class=RidgeBaseline,
        preprocessor=preprocessor, model_name="Ridge",
        alpha=1.0, use_poly=True, poly_degree=2,
    )

    # ── Baseline 4: Neural-Only (CNN + Tabular) ──
    print("\n" + "=" * 60)
    print("BASELINE 4: Neural-Only (CNN + Tabular Fusion)")
    print("=" * 60)
    cv.run_neural_cv(
        df=df, model_class=NeuralOnlyModel,
        preprocessor=preprocessor, model_name="NeuralOnly",
        image_dir=IMAGE_DIR,
        backbone=default_config.model.backbone,
        image_feature_dim=default_config.model.image_feature_dim,
        tabular_hidden=tuple(default_config.model.tabular_hidden_dims),
        fusion_hidden=tuple(default_config.model.fusion_hidden_dims),
        dropout=0.3,
    )

    # ── Baseline 5: Neural + Soft Constraint ──
    print("\n" + "=" * 60)
    print("BASELINE 5: Neural + Soft Constraint Penalty (Ablation)")
    print("=" * 60)
    cv.run_neural_cv(
        df=df, model_class=NeuralConstraintModel,
        preprocessor=preprocessor, model_name="Neural+Constraint",
        image_dir=IMAGE_DIR,
        backbone=default_config.model.backbone,
        image_feature_dim=default_config.model.image_feature_dim,
        tabular_hidden=tuple(default_config.model.tabular_hidden_dims),
        fusion_hidden=tuple(default_config.model.fusion_hidden_dims),
        dropout=0.3,
        constraint_weight=2.0,
    )

    # ── Save Results ──
    cv.save_results(OUTPUT_DIR / "baseline_results.json")

    # Print summary table
    from src.visualization.embeddings import create_summary_table
    print("\n" + create_summary_table(cv.results, list(cv.results.keys())))

    print("\nBaseline evaluation complete!")


if __name__ == "__main__":
    main()
