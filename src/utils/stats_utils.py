"""Statistical helpers: Cohen's d, permutation testing, FDR correction."""

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Pooled-SD Cohen's d (signed: positive when group1 > group2)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_sd = np.sqrt(
        ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1))
        / (n1 + n2 - 2)
    )
    if pooled_sd == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_sd)


def permutation_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    observed_auc: float,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> dict:
    """Permutation test for AUC.

    Returns null AUC distribution, p-value, and whether observed AUC is significant.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(random_state)
    null_aucs = np.array([
        roc_auc_score(rng.permutation(y_true), y_score)
        for _ in range(n_permutations)
    ])
    p_value = float((null_aucs >= observed_auc).mean())
    return {
        "null_aucs": null_aucs,
        "observed_auc": observed_auc,
        "p_value": p_value,
        "significant_at_05": p_value < 0.05,
    }


def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Returns (rejected, corrected_p_values) where rejected is a boolean array.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = sorted_p <= thresholds
    if below.any():
        max_idx = np.where(below)[0].max()
        rejected_sorted = np.arange(n) <= max_idx
    else:
        rejected_sorted = np.zeros(n, dtype=bool)

    corrected_p = np.minimum(1.0, sorted_p * n / np.arange(1, n + 1))
    corrected_p = np.minimum.accumulate(corrected_p[::-1])[::-1]

    rejected = np.empty(n, dtype=bool)
    corrected = np.empty(n)
    rejected[sorted_idx] = rejected_sorted
    corrected[sorted_idx] = corrected_p
    return rejected, corrected


def mannwhitney_u(group1: np.ndarray, group2: np.ndarray) -> dict:
    """Mann-Whitney U test with effect size r = Z / sqrt(N)."""
    result = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    n = len(group1) + len(group2)
    z = stats.norm.ppf(1 - result.pvalue / 2)
    r = abs(z) / np.sqrt(n)
    return {"statistic": result.statistic, "p_value": result.pvalue, "effect_r": r}
