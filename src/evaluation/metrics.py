"""Evaluation metrics for multi-target regression."""
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

TARGET_NAMES = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]


def target_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Per-target RMSE."""
    return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))


def target_mae(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Per-target MAE."""
    return np.mean(np.abs(y_pred - y_true), axis=0)


def target_r2(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Per-target R²."""
    r2s = []
    for i in range(y_true.shape[1]):
        if np.std(y_true[:, i]) < 1e-8:
            r2s.append(0.0)
        else:
            r2s.append(r2_score(y_true[:, i], y_pred[:, i]))
    return np.array(r2s)


def target_mape(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Per-target MAPE (mean absolute percentage error)."""
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_pred - y_true) / denom), axis=0) * 100


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict:
    """Compute comprehensive metrics for all targets.

    Args:
        y_pred: (N, 5) predictions [Clover, Dead, Green, Total, GDM]
        y_true: (N, 5) targets in same order

    Returns:
        Dict with per-target and overall metrics
    """
    rmse = target_rmse(y_pred, y_true)
    mae = target_mae(y_pred, y_true)
    r2 = target_r2(y_pred, y_true)
    mape = target_mape(y_pred, y_true)

    metrics = {"overall_rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
               "overall_mae": float(np.mean(np.abs(y_pred - y_true))),
               "overall_r2": float(r2_score(y_true.ravel(), y_pred.ravel())), }

    for i, name in enumerate(TARGET_NAMES):
        metrics[f"{name}_rmse"] = float(rmse[i])
        metrics[f"{name}_mae"] = float(mae[i])
        metrics[f"{name}_r2"] = float(r2[i])
        metrics[f"{name}_mape"] = float(mape[i])

    return metrics


def compute_constraint_metrics(y_pred: np.ndarray) -> Dict:
    """Compute constraint violation metrics.

    Args:
        y_pred: (N, 5) [Clover, Dead, Green, Total, GDM]

    Returns:
        Dict with constraint violation statistics
    """
    clover, dead, green, total, gdm = [y_pred[:, i] for i in range(5)]

    # Constraint 1: Total ≈ Green + Dead + Clover
    c1_deviation = total - (green + dead + clover)
    c1_rmse = float(np.sqrt(np.mean(c1_deviation ** 2)))
    c1_mae = float(np.mean(np.abs(c1_deviation)))
    c1_max = float(np.max(np.abs(c1_deviation)))
    c1_violation_rate = float(np.mean(np.abs(c1_deviation) > 1.0))

    # Constraint 2: GDM ≈ Green + Clover
    c2_deviation = gdm - (green + clover)
    c2_rmse = float(np.sqrt(np.mean(c2_deviation ** 2)))
    c2_mae = float(np.mean(np.abs(c2_deviation)))
    c2_max = float(np.max(np.abs(c2_deviation)))
    c2_violation_rate = float(np.mean(np.abs(c2_deviation) > 1.0))

    return {
        "c1_total_equals_sum_rmse": c1_rmse,
        "c1_total_equals_sum_mae": c1_mae,
        "c1_total_equals_sum_max": c1_max,
        "c1_violation_rate": c1_violation_rate,
        "c2_gdm_equals_sum_rmse": c2_rmse,
        "c2_gdm_equals_sum_mae": c2_mae,
        "c2_gdm_equals_sum_max": c2_max,
        "c2_violation_rate": c2_violation_rate,
        "combined_constraint_rmse": float(np.sqrt((c1_rmse ** 2 + c2_rmse ** 2) / 2)),
    }
