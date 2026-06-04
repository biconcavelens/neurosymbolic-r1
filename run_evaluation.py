#!/usr/bin/env python
"""Generate all evaluation figures and the final comparison report.

Loads saved results from baseline and LTN runs, then generates:
Figure 1: Predictions vs Actual scatter
Figure 2: Residual analysis
Figure 3: Constraint satisfaction comparison
Figure 4: Learning curves
Figure 5: Cross-validation box plots
Figure 6: Embedding visualization
Figure 7: Model comparison summary
Figure 8: Summary table
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
import pickle
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import OUTPUT_DIR, FIGURE_DIR, CSV_PATH, IMAGE_DIR, default_config
from src.data.tabular_encoder import TabularPreprocessor
from src.data.splitter import StratifiedGroupKFoldSplitter
from src.evaluation.metrics import TARGET_NAMES
from src.evaluation.statistical_tests import compare_models
from src.visualization.predictions import (
    plot_predictions_vs_actual, plot_residuals,
    plot_constraint_satisfaction, plot_model_comparison,
)
from src.visualization.embeddings import (
    plot_learning_curves, plot_cross_validation_boxplots,
    create_summary_table, plot_embeddings,
)


def load_results(results_path: Path) -> dict:
    """Load saved JSON results."""
    with open(results_path) as f:
        return json.load(f)


def merge_results(*result_dicts) -> dict:
    """Merge multiple result dictionaries."""
    merged = {}
    for d in result_dicts:
        merged.update(d)
    return merged


def main():
    print("=" * 60)
    print("Neuro-Symbolic Biomass Prediction — Final Evaluation")
    print("=" * 60)

    # Try to load saved results; if not available, print instructions
    baseline_path = OUTPUT_DIR / "baseline_results.json"
    ltn_path = OUTPUT_DIR / "ltn_results.json"

    all_results = {}
    if baseline_path.exists():
        print(f"Loading baseline results from {baseline_path}")
        all_results.update(load_results(baseline_path))
    else:
        print(f"Baseline results not found at {baseline_path}")
        print("Run: python run_baselines.py first")

    if ltn_path.exists():
        print(f"Loading LTN results from {ltn_path}")
        all_results.update(load_results(ltn_path))
    else:
        print(f"LTN results not found at {ltn_path}")
        print("Run: python run_ltn.py first")

    if not all_results:
        print("\nNo results to evaluate. Please run baselines and/or LTN training first.")
        return

    model_names = sorted(all_results.keys())
    print(f"\nModels to compare: {model_names}")

    # ── Collect predictions for plotting ──
    # Use a single fold's predictions for the scatter/residual plots
    model_predictions = []
    for m_name in model_names:
        preds_list = all_results[m_name].get("predictions", [])
        targets_list = all_results[m_name].get("targets", [])
        if preds_list and targets_list:
            # Concatenate all folds
            all_preds = np.concatenate(preds_list, axis=0)
            model_predictions.append(all_preds)
        else:
            model_predictions.append(None)

    targets_for_viz = np.concatenate(
        [np.array(t) for t in all_results[model_names[0]].get("targets", [])], axis=0
    ) if all_results[model_names[0]].get("targets") else None

    # ── Figure 1: Predictions vs Actual ──
    if all(p is not None for p in model_predictions) and targets_for_viz is not None:
        print("\nGenerating Figure 1: Predictions vs Actual...")
        plot_predictions_vs_actual(
            predictions=model_predictions,
            targets=targets_for_viz,
            model_names=model_names,
            save_path=FIGURE_DIR / "fig1_predictions_vs_actual.png",
        )

        # ── Figure 2: Residual Analysis (best model) ──
        print("Generating Figure 2: Residual Analysis...")
        # Use the model with lowest constraint RMSE
        best_model_idx = 0
        best_c_rmse = float("inf")
        for i, m_name in enumerate(model_names):
            fm = all_results[m_name].get("fold_metrics", [])
            if fm:
                c_rmse = np.mean([f.get("combined_constraint_rmse", 999) for f in fm])
                if c_rmse < best_c_rmse:
                    best_c_rmse = c_rmse
                    best_model_idx = i

        if model_predictions[best_model_idx] is not None:
            plot_residuals(
                predictions=model_predictions[best_model_idx],
                targets=targets_for_viz,
                save_path=FIGURE_DIR / "fig2_residuals.png",
            )

    # ── Figure 3: Constraint Satisfaction ──
    print("Generating Figure 3: Constraint Satisfaction...")
    constraint_metrics = {}
    for m_name in model_names:
        fm = all_results[m_name].get("fold_metrics", [])
        if fm:
            cm = {}
            for key in ["c1_total_equals_sum_rmse", "c2_gdm_equals_sum_rmse",
                        "c1_violation_rate", "c2_violation_rate", "combined_constraint_rmse"]:
                cm[key] = [f.get(key, 0) for f in fm]
            constraint_metrics[m_name] = cm

    plot_constraint_satisfaction(
        constraint_metrics=constraint_metrics,
        model_names=model_names,
        save_path=FIGURE_DIR / "fig3_constraint_satisfaction.png",
    )

    # ── Figure 4: Cross-Validation Box Plots ──
    print("Generating Figure 4: Cross-Validation Stability...")
    plot_cross_validation_boxplots(
        all_metrics=all_results,
        model_names=model_names,
        save_path=FIGURE_DIR / "fig4_cv_boxplots.png",
    )

    # ── Figure 5: Model Comparison Summary ──
    print("Generating Figure 5: Model Comparison Summary...")
    plot_model_comparison(
        all_metrics=all_results,
        model_names=model_names,
        save_path=FIGURE_DIR / "fig5_model_comparison.png",
    )

    # ── Statistical Tests ──
    print("\nRunning statistical significance tests...")
    comparisons = compare_models(all_results, metric_key="overall_rmse")
    for pair, result in comparisons.items():
        print(f"\n{pair}:")
        if "dm" in result:
            print(f"  Diebold-Mariano: p={result['dm']['p_value']:.4f} — {result['dm']['conclusion']}")
        if "wilcoxon" in result:
            print(f"  Wilcoxon: p={result['wilcoxon']['p_value']:.4f} — {result['wilcoxon']['conclusion']}")

    # ── Summary Table ──
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(create_summary_table(all_results, model_names))

    # Save report
    report_path = OUTPUT_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("Neuro-Symbolic Biomass Prediction — Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(create_summary_table(all_results, model_names))
        f.write("\n\nStatistical Significance:\n")
        for pair, result in comparisons.items():
            f.write(f"\n{pair}:\n")
            if "dm" in result:
                f.write(f"  Diebold-Mariano: p={result['dm']['p_value']:.4f}\n")
            if "wilcoxon" in result:
                f.write(f"  Wilcoxon: p={result['wilcoxon']['p_value']:.4f}\n")

    print(f"\nReport saved to {report_path}")
    print(f"Figures saved to {FIGURE_DIR}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
