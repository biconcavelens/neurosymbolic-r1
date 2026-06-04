#!/usr/bin/env python
"""Train and evaluate the Neuro-Symbolic LTN model with 5-fold cross-validation.

The LTN model:
1. Uses the same CNN+tabular backbone as the neural baseline
2. Predicts 3 base targets (Green, Dead, Clover), derives 2 composites (Total, GDM)
3. Applies fuzzy logic constraints during training
4. Features: learned predicate weights, evidential regression, rule discovery
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
from src.models.neuro_symbolic.ltn_model import LTNModel


def main():
    print("=" * 60)
    print("Neuro-Symbolic LTN Model — 5-fold Cross-Validation")
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

    # ── LTN Model Cross-Validation ──
    cv = CrossValidator(default_config)

    print("\n" + "=" * 60)
    print("NEURO-SYMBOLIC: LTN Model (Full)")
    print("=" * 60)

    cv.run_neural_cv(
        df=df,
        model_class=LTNModel,
        preprocessor=preprocessor,
        model_name="LTN_NeuroSymbolic",
        image_dir=IMAGE_DIR,
        use_ltn_trainer=True,
        backbone=default_config.model.backbone,
        image_feature_dim=default_config.model.image_feature_dim,
        tabular_hidden=tuple(default_config.model.tabular_hidden_dims),
        fusion_hidden=tuple(default_config.model.fusion_hidden_dims),
        dropout=0.3,
        use_grid_features=False,
        use_color_histogram=False,
        use_cross_attention=False,
        use_evidential=False,
        use_learned_weights=True,
        use_learned_predicates=True,
        constraint_eps=0.5,
    )

    # ── LTN + Evidential ──
    print("\n" + "=" * 60)
    print("NEURO-SYMBOLIC: LTN + Evidential Regression")
    print("=" * 60)

    cv.run_neural_cv(
        df=df,
        model_class=LTNModel,
        preprocessor=preprocessor,
        model_name="LTN_Evidential",
        image_dir=IMAGE_DIR,
        use_ltn_trainer=True,
        backbone=default_config.model.backbone,
        image_feature_dim=default_config.model.image_feature_dim,
        tabular_hidden=tuple(default_config.model.tabular_hidden_dims),
        fusion_hidden=tuple(default_config.model.fusion_hidden_dims),
        dropout=0.3,
        use_evidential=True,
        use_learned_weights=True,
        use_learned_predicates=True,
        constraint_eps=0.5,
    )

    # ── LTN + Cross-Attention Fusion ──
    print("\n" + "=" * 60)
    print("NEURO-SYMBOLIC: LTN + Cross-Attention Fusion")
    print("=" * 60)

    cv.run_neural_cv(
        df=df,
        model_class=LTNModel,
        preprocessor=preprocessor,
        model_name="LTN_CrossAttn",
        image_dir=IMAGE_DIR,
        use_ltn_trainer=True,
        backbone=default_config.model.backbone,
        image_feature_dim=default_config.model.image_feature_dim,
        tabular_hidden=tuple(default_config.model.tabular_hidden_dims),
        fusion_hidden=tuple(default_config.model.fusion_hidden_dims),
        dropout=0.3,
        use_cross_attention=True,
        use_evidential=False,
        use_learned_weights=True,
        use_learned_predicates=True,
        constraint_eps=0.5,
    )

    # ── Save Results ──
    cv.save_results(OUTPUT_DIR / "ltn_results.json")

    # Print summary
    from src.visualization.embeddings import create_summary_table
    print("\n" + create_summary_table(cv.results, list(cv.results.keys())))

    print("\nLTN evaluation complete!")


if __name__ == "__main__":
    main()
