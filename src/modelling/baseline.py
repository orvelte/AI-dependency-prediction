"""Baseline models: univariate logistic regression per feature."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.utils.stats_utils import cohens_d, fdr_correction


FEATURE_COLS = [
    "fpp_rate",
    "self_disclosure_rate",
    "hedging_rate",
    "filler_rate",
    "refusal_resistance_rate",
    "gratitude_rate",
    "question_to_statement_ratio",
    "mean_message_length",
    "type_token_ratio",
    "sentiment_slope",
    "log_n_user_turns",
    "log_return_interval_hours",
]


def run_univariate_baselines(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str] | None = None,
    n_folds: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit a logistic regression on each feature individually.

    Returns a DataFrame with AUC, Cohen's d, raw p-value, and BH-corrected p-value per feature.
    Uses StratifiedKFold cross-validation at the caller's responsibility to ensure user-level splits.
    """
    cols = feature_cols or [c for c in FEATURE_COLS if c in X.columns]
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    results = []
    raw_pvals = []

    for feat in cols:
        x_feat = X[[feat]].values
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_feat)

        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=random_state)
        y_prob = cross_val_predict(model, x_scaled, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, y_prob)

        high = X.loc[y == 1, feat].values
        low = X.loc[y == 0, feat].values
        d = cohens_d(high, low)

        from scipy.stats import ttest_ind
        _, p_raw = ttest_ind(high, low, equal_var=False)
        raw_pvals.append(p_raw)

        results.append({"feature": feat, "auc": auc, "cohens_d": d, "p_raw": p_raw})

    df_results = pd.DataFrame(results)
    _, corrected = fdr_correction(np.array(raw_pvals))
    df_results["p_corrected_bh"] = corrected
    df_results["significant_bh"] = corrected < 0.05
    return df_results.sort_values("auc", ascending=False).reset_index(drop=True)
