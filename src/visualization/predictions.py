"""Visualization module: prediction plots, residuals, constraints, feature importance."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("husl")

TARGET_NAMES = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]
TARGET_UNITS = ["g", "g", "g", "g", "g"]


def set_style():
    """Set consistent matplotlib style for paper-ready figures."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.figsize": (8, 6),
    })


def plot_predictions_vs_actual(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_names: List[str],
    save_path: Optional[Path] = None,
):
    """Figure 1: Actual vs Predicted scatter plots for all targets and models.

    Args:
        predictions: List of (N, 5) arrays, one per model
        targets: (N, 5) ground truth
        model_names: Names for each model
    """
    set_style()
    n_targets = len(TARGET_NAMES)
    n_models = len(model_names)

    fig, axes = plt.subplots(n_targets, n_models, figsize=(4 * n_models, 4 * n_targets))
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    for t_idx, t_name in enumerate(TARGET_NAMES):
        for m_idx, (preds, m_name) in enumerate(zip(predictions, model_names)):
            ax = axes[t_idx, m_idx]
            ax.scatter(targets[:, t_idx], preds[:, t_idx], alpha=0.5, s=10, edgecolors="none")

            # Perfect prediction line
            lims = [targets[:, t_idx].min(), targets[:, t_idx].max()]
            ax.plot(lims, lims, "r--", linewidth=1, alpha=0.7)

            # R² annotation
            ss_res = np.sum((targets[:, t_idx] - preds[:, t_idx]) ** 2)
            ss_tot = np.sum((targets[:, t_idx] - targets[:, t_idx].mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            rmse = np.sqrt(np.mean((targets[:, t_idx] - preds[:, t_idx]) ** 2))

            ax.set_xlabel(f"Actual ({TARGET_UNITS[t_idx]})")
            ax.set_ylabel(f"Predicted ({TARGET_UNITS[t_idx]})")
            ax.set_title(f"{m_name} — {t_name}\nR²={r2:.3f}, RMSE={rmse:.2f}")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_residuals(
    predictions: np.ndarray,
    targets: np.ndarray,
    species_labels: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
):
    """Figure 2: Residual analysis — residual vs fitted, Q-Q, per-target distribution."""
    set_style()
    residuals = predictions - targets  # (N, 5)
    fitted = predictions

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for t_idx, t_name in enumerate(TARGET_NAMES[:5]):
        ax = axes[t_idx // 3, t_idx % 3]
        ax.scatter(fitted[:, t_idx], residuals[:, t_idx], alpha=0.4, s=8,
                   c=species_labels if species_labels is not None else None, cmap="tab20")
        ax.axhline(y=0, color="r", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Fitted values (g)")
        ax.set_ylabel("Residuals (g)")
        ax.set_title(f"{t_name}")

    # Hide extra subplot
    if len(TARGET_NAMES) < 6:
        axes[1, 2].set_visible(False)

    plt.suptitle("Residual Analysis — Residuals vs Fitted Values", fontsize=14)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_constraint_satisfaction(
    constraint_metrics: Dict[str, List[float]],
    model_names: List[str],
    save_path: Optional[Path] = None,
):
    """Figure 3: Constraint satisfaction comparison across models."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Bar chart: constraint RMSE
    ax = axes[0]
    x = np.arange(len(model_names))
    width = 0.35
    c1_vals = [constraint_metrics[m].get("c1_total_equals_sum_rmse", [0]) for m in model_names]
    c2_vals = [constraint_metrics[m].get("c2_gdm_equals_sum_rmse", [0]) for m in model_names]
    c1_means = [np.mean(v) for v in c1_vals]
    c2_means = [np.mean(v) for v in c2_vals]
    c1_stds = [np.std(v) for v in c1_vals]
    c2_stds = [np.std(v) for v in c2_vals]

    ax.bar(x - width / 2, c1_means, width, yerr=c1_stds, label="Total = Green+Dead+Clover", capsize=3)
    ax.bar(x + width / 2, c2_means, width, yerr=c2_stds, label="GDM = Green+Clover", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylabel("Constraint RMSE (g)")
    ax.set_title("Physical Constraint Violations")
    ax.legend()

    # Violation rate
    ax = axes[1]
    violation_rates = []
    for m in model_names:
        c1_rate = constraint_metrics[m].get("c1_violation_rate", [0])
        violation_rates.append(np.mean(c1_rate) * 100)
    bars = ax.bar(model_names, violation_rates)
    ax.set_ylabel("Violation Rate (%)")
    ax.set_title("Samples Exceeding 1g Constraint Tolerance")
    for bar, rate in zip(bars, violation_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{rate:.1f}%", ha="center", fontsize=9)

    # Combined constraint RMSE
    ax = axes[2]
    combined = []
    for m in model_names:
        c = constraint_metrics[m].get("combined_constraint_rmse", [0])
        combined.append(np.mean(c))
    bars = ax.bar(model_names, combined, color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"][:len(model_names)])
    ax.set_ylabel("Combined Constraint RMSE (g)")
    ax.set_title("Overall Physical Consistency")
    for bar, val in zip(bars, combined):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9)

    plt.suptitle("Logical Constraint Satisfaction Analysis", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_model_comparison(
    all_metrics: Dict[str, Dict],
    model_names: List[str],
    save_path: Optional[Path] = None,
):
    """Figure 8: Comprehensive model comparison — table + radar plot."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: Bar chart of per-target RMSE
    ax = axes[0]
    x = np.arange(len(TARGET_NAMES))
    width = 0.8 / len(model_names)

    for i, m_name in enumerate(model_names):
        fold_metrics = all_metrics[m_name].get("fold_metrics", [])
        if not fold_metrics:
            continue
        rmse_means = [np.mean([fm[f"{t}_rmse"] for fm in fold_metrics]) for t in TARGET_NAMES]
        rmse_stds = [np.std([fm[f"{t}_rmse"] for fm in fold_metrics]) for t in TARGET_NAMES]
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, rmse_means, width, yerr=rmse_stds, label=m_name, capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_NAMES, rotation=20, ha="right")
    ax.set_ylabel("RMSE (g)")
    ax.set_title("Per-Target RMSE Comparison")
    ax.legend(loc="upper left")

    # Right: Radar/spider plot
    ax = axes[1]
    try:
        _plot_radar(ax, all_metrics, model_names)
    except Exception as e:
        print(f"Radar plot skipped: {e}")
        ax.text(0.5, 0.5, "Radar plot\nunavailable", ha="center", va="center", transform=ax.transAxes)

    plt.suptitle("Model Comparison Summary", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


def _plot_radar(ax, all_metrics: Dict, model_names: List[str]):
    """Helper: radar chart for model comparison."""
    # Normalize each metric to [0, 1] where 1 = best
    metrics_to_plot = TARGET_NAMES + ["Constraint"]
    n_metrics = len(metrics_to_plot)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    # Collect scores
    model_scores = {}
    for m_name in model_names:
        fold_metrics = all_metrics[m_name].get("fold_metrics", [])
        if not fold_metrics:
            continue
        # For each metric, compute mean RMSE across folds
        scores = [np.mean([fm[f"{t}_rmse"] for fm in fold_metrics]) for t in TARGET_NAMES]
        scores.append(np.mean([fm["combined_constraint_rmse"] for fm in fold_metrics]))
        model_scores[m_name] = scores

    # Normalize (invert: lower RMSE = higher score)
    all_values = np.array(list(model_scores.values()))
    mins = all_values.min(axis=0)
    maxs = all_values.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    for m_name, scores in model_scores.items():
        normalized = 1.0 - (np.array(scores) - mins) / ranges  # 1 = best
        values = normalized.tolist() + [normalized[0]]
        ax.fill(angles, values, alpha=0.15)
        ax.plot(angles, values, linewidth=2, label=m_name)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1.05)
    ax.set_title("Normalized Performance (higher = better)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
