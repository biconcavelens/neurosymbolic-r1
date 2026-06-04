"""Visualization module: embeddings, feature importance, dashboard."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def plot_embeddings(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[Dict[int, str]] = None,
    method: str = "tsne",
    title: str = "Learned Feature Embeddings",
    save_path: Optional[Path] = None,
):
    """Figure 6 supplement: t-SNE/UMAP of learned representations.

    Args:
        embeddings: (N, D) feature vectors
        labels: (N,) integer labels for coloring
        label_names: Dict mapping label int to readable name
        method: "tsne" or "pca"
    """
    if method == "tsne" and embeddings.shape[0] > 2:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0] // 3))
    else:
        reducer = PCA(n_components=2, random_state=42)

    reduced = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    if labels is not None:
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab20",
                             alpha=0.7, s=30, edgecolors="none")
        if label_names:
            handles, _ = scatter.legend_elements()
            legend_labels = [label_names.get(int(l), f"Class {l}")
                             for l in sorted(set(labels))]
            ax.legend(handles, legend_labels, title="", bbox_to_anchor=(1.02, 1),
                      loc="upper left", fontsize=7)
    else:
        ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=30)

    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_learning_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
):
    """Plot training and validation learning curves."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss curves
    ax = axes[0]
    ax.plot(history.get("train_loss", []), label="Train Loss", linewidth=1)
    ax.plot(history.get("val_loss", []), label="Val Loss", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.legend()

    # RMSE
    ax = axes[1]
    ax.plot(history.get("val_rmse", []), label="Val RMSE", linewidth=1, color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (g)")
    ax.set_title("Validation RMSE")
    ax.legend()

    # Constraint satisfaction
    ax = axes[2]
    if "constraint_satisfaction" in history:
        ax.plot(history["constraint_satisfaction"], label="Constraint Satisfaction",
                linewidth=1, color="purple")
        ax.set_ylim(0, 1.05)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Satisfaction")
    ax.set_title("Logical Constraint Satisfaction")
    ax.legend()

    plt.suptitle("Training Dynamics", fontsize=14)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_cross_validation_boxplots(
    all_metrics: Dict[str, Dict],
    model_names: List[str],
    save_path: Optional[Path] = None,
):
    """Figure 7: Box plots of per-fold RMSE across models."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for t_idx, t_name in enumerate(["Overall"] + ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]):
        ax = axes[t_idx]
        data = []
        labels = []
        for m_name in model_names:
            fold_metrics = all_metrics[m_name].get("fold_metrics", [])
            if t_name == "Overall":
                vals = [fm["overall_rmse"] for fm in fold_metrics]
            else:
                vals = [fm[f"{t_name}_rmse"] for fm in fold_metrics]
            if vals:
                data.append(vals)
                labels.append(m_name)

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], plt.cm.Set2(np.linspace(0, 1, len(data)))):
            patch.set_facecolor(color)

        ax.set_ylabel("RMSE (g)")
        ax.set_title(f"{t_name}")
        ax.tick_params(axis="x", rotation=15)

    if len(TARGET_NAMES) < 5:
        axes[-1].set_visible(False)

    plt.suptitle("Cross-Validation Stability", fontsize=14)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


def create_summary_table(
    all_metrics: Dict[str, Dict],
    model_names: List[str],
) -> str:
    """Generate a formatted summary table string."""
    header = f"{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Constr.RMSE':>12} {'Clover R²':>10} {'Total R²':>10}"
    separator = "-" * len(header)

    lines = [separator, header, separator]
    for m_name in model_names:
        fold_metrics = all_metrics[m_name].get("fold_metrics", [])
        if not fold_metrics:
            continue
        rmse = np.mean([fm["overall_rmse"] for fm in fold_metrics])
        mae = np.mean([fm["overall_mae"] for fm in fold_metrics])
        r2 = np.mean([fm["overall_r2"] for fm in fold_metrics])
        c_rmse = np.mean([fm.get("combined_constraint_rmse", 0) for fm in fold_metrics])
        clover_r2 = np.mean([fm["Dry_Clover_g_r2"] for fm in fold_metrics])
        total_r2 = np.mean([fm["Dry_Total_g_r2"] for fm in fold_metrics])

        rmse_std = np.std([fm["overall_rmse"] for fm in fold_metrics])
        lines.append(
            f"{m_name:<25} {rmse:>6.2f}±{rmse_std:<4.2f} {mae:>6.2f} "
            f"{r2:>6.3f} {c_rmse:>10.4f} {clover_r2:>8.3f} {total_r2:>8.3f}"
        )
    lines.append(separator)
    return "\n".join(lines)
