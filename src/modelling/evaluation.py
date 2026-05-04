"""Cross-validation, AUC computation, and dependency labelling."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from src.utils.stats_utils import permutation_auc


def label_high_dependency(
    late_features: pd.DataFrame,
    user_col: str = "hashed_ip",
    min_criteria: int = 2,
) -> pd.Series:
    """Apply the pre-specified composite rule to late-session features.

    A user is "high-dependency" if they meet ≥ 2 of:
    1. return_interval < median (faster returns)
    2. n_user_turns > 75th percentile
    3. refusal_resistance_rate > 0
    4. fpp_rate > cohort mean in late sessions

    Returns a boolean Series indexed by user_id.
    """
    df = late_features.set_index(user_col).copy()
    criteria = pd.DataFrame(index=df.index)

    if "return_interval_hours" in df.columns:
        median_ri = df["return_interval_hours"].median()
        criteria["fast_return"] = df["return_interval_hours"] < median_ri
    else:
        criteria["fast_return"] = False

    if "n_user_turns" in df.columns:
        p75 = df["n_user_turns"].quantile(0.75)
        criteria["long_session"] = df["n_user_turns"] > p75
    else:
        criteria["long_session"] = False

    if "refusal_resistance_rate" in df.columns:
        criteria["refusal_resistance"] = df["refusal_resistance_rate"] > 0
    else:
        criteria["refusal_resistance"] = False

    if "fpp_rate" in df.columns:
        mean_fpp = df["fpp_rate"].mean()
        criteria["high_fpp"] = df["fpp_rate"] > mean_fpp
    else:
        criteria["high_fpp"] = False

    return (criteria.sum(axis=1) >= min_criteria).rename("high_dependency")


def cross_validate_model(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    user_ids: np.ndarray,
    n_folds: int = 10,
    random_state: int = 42,
) -> dict:
    """User-level stratified K-fold cross-validation.

    Users are the unit of stratification so no user appears in both train and test.
    Returns per-fold and aggregate metrics.
    """
    unique_users = np.unique(user_ids)
    user_labels = np.array([y[user_ids == u][0] for u in unique_users])

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_aucs, fold_bal_accs, fold_precs, fold_recs = [], [], [], []
    all_y_true, all_y_prob = [], []

    for train_user_idx, test_user_idx in cv.split(unique_users, user_labels):
        train_users = unique_users[train_user_idx]
        test_users = unique_users[test_user_idx]

        train_mask = np.isin(user_ids, train_users)
        test_mask = np.isin(user_ids, test_users)

        pipeline.fit(X[train_mask], y[train_mask])
        y_prob = pipeline.predict_proba(X[test_mask])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        y_true = y[test_mask]

        fold_aucs.append(roc_auc_score(y_true, y_prob))
        fold_bal_accs.append(balanced_accuracy_score(y_true, y_pred))
        fold_precs.append(precision_score(y_true, y_pred, zero_division=0))
        fold_recs.append(recall_score(y_true, y_pred, zero_division=0))
        all_y_true.extend(y_true)
        all_y_prob.extend(y_prob)

    observed_auc = roc_auc_score(all_y_true, all_y_prob)
    return {
        "fold_aucs": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "mean_balanced_accuracy": float(np.mean(fold_bal_accs)),
        "mean_precision": float(np.mean(fold_precs)),
        "mean_recall": float(np.mean(fold_recs)),
        "observed_auc": observed_auc,
    }


def run_permutation_test(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    observed_auc: float,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> dict:
    """Fit model on shuffled labels and compare to observed AUC."""
    from sklearn.metrics import roc_auc_score as auc_fn
    rng = np.random.default_rng(random_state)
    null_aucs = []
    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y)
        pipeline.fit(X, y_shuffled)
        y_prob = pipeline.predict_proba(X)[:, 1]
        null_aucs.append(auc_fn(y_shuffled, y_prob))
    null_aucs = np.array(null_aucs)
    p_value = float((null_aucs >= observed_auc).mean())
    return {
        "null_aucs": null_aucs,
        "observed_auc": observed_auc,
        "p_value": p_value,
        "significant_at_05": p_value < 0.05,
    }
