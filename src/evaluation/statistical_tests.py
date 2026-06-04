"""Statistical tests for model comparison."""
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def diebold_mariano_test(errors_a: np.ndarray, errors_b: np.ndarray,
                         horizon: int = 1, alternative: str = "two_sided") -> Dict:
    """Diebold-Mariano test for predictive accuracy comparison.

    Tests whether model A has different predictive accuracy than model B.

    Args:
        errors_a: (N,) squared errors of model A
        errors_b: (N,) squared errors of model B
        horizon: forecast horizon (1 for one-step ahead)
        alternative: "two_sided", "greater" (A worse than B), "less" (A better than B)

    Returns:
        Dict with DM statistic, p-value, and conclusion
    """
    diff = errors_a - errors_b
    n = len(diff)

    # Simple DM test statistic
    mean_diff = np.mean(diff)

    # HAC standard error (Newey-West style for horizon > 1)
    if horizon > 1:
        from statsmodels.tsa.stattools import acf
        autocorr = acf(diff, nlags=horizon - 1, fft=False)
        var = (np.var(diff) + 2 * np.sum(autocorr[1:])) / n
    else:
        var = np.var(diff) / n

    if var < 1e-12:
        dm_stat = 0.0
        p_value = 1.0
    else:
        dm_stat = mean_diff / np.sqrt(var)
        if alternative == "two_sided":
            p_value = 2 * stats.norm.sf(np.abs(dm_stat))
        elif alternative == "greater":
            p_value = stats.norm.sf(dm_stat)
        else:  # "less"
            p_value = stats.norm.cdf(dm_stat)

    conclusion = "not significant"
    if p_value < 0.05:
        if mean_diff > 0:
            conclusion = "Model B is significantly better"
        else:
            conclusion = "Model A is significantly better"

    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "mean_diff": float(mean_diff),
        "conclusion": conclusion,
    }


def wilcoxon_test(errors_a: np.ndarray, errors_b: np.ndarray) -> Dict:
    """Wilcoxon signed-rank test for paired model comparison."""
    stat, p_value = stats.wilcoxon(errors_a, errors_b, alternative="two-sided")

    conclusion = "not significant"
    if p_value < 0.05:
        if np.mean(errors_a) > np.mean(errors_b):
            conclusion = "Model B is significantly better (Wilcoxon)"
        else:
            conclusion = "Model A is significantly better (Wilcoxon)"

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "conclusion": conclusion,
    }


def compare_models(model_results: Dict[str, Dict],
                   metric_key: str = "overall_rmse") -> Dict:
    """Compare multiple models using statistical tests.

    Args:
        model_results: {model_name: {"fold_metrics": [metric_dict_per_fold], ...}}
        metric_key: which metric to compare

    Returns:
        Comparison matrix
    """
    model_names = sorted(model_results.keys())
    n_models = len(model_names)
    comparisons = {}

    for i in range(n_models):
        for j in range(i + 1, n_models):
            name_a, name_b = model_names[i], model_names[j]
            # Extract fold-level errors
            # For DM test, we use per-fold RMSE (scalar per fold)
            folds_a = [m[metric_key] for m in model_results[name_a].get("fold_metrics", [])]
            folds_b = [m[metric_key] for m in model_results[name_b].get("fold_metrics", [])]

            if len(folds_a) >= 3 and len(folds_b) >= 3:
                # Use per-fold RMSE as the "error" measure
                # Lower RMSE = better, so we test if model with lower mean RMSE is significantly better
                errors_a = np.array(folds_a)
                errors_b = np.array(folds_b)

                dm_result = diebold_mariano_test(errors_a, errors_b)
                wilcoxon_result = wilcoxon_test(errors_a, errors_b)

                comparisons[f"{name_a}_vs_{name_b}"] = {
                    "dm": dm_result,
                    "wilcoxon": wilcoxon_result,
                }

    return comparisons
