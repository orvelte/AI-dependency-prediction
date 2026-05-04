"""Primary predictive model: L1/L2 regularised logistic regression on the full feature set."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


NEGATIVE_CONTROL_FEATURE = "punctuation_rate"


def build_pipeline(
    penalty: str = "l1",
    C: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Return a sklearn Pipeline with StandardScaler + LogisticRegression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty=penalty,
            C=C,
            solver="liblinear" if penalty == "l1" else "lbfgs",
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        )),
    ])


def add_negative_control(X: pd.DataFrame, text_series: pd.Series | None = None) -> pd.DataFrame:
    """Add punctuation_rate as a theoretically unmotivated negative control feature.

    If `text_series` is None, fills with random noise so the column still exists.
    """
    if text_series is not None:
        import re
        X = X.copy()
        X[NEGATIVE_CONTROL_FEATURE] = text_series.apply(
            lambda t: len(re.findall(r"[.,;:!?]", str(t))) / max(len(str(t).split()), 1)
        )
    else:
        rng = np.random.default_rng(seed=0)
        X = X.copy()
        X[NEGATIVE_CONTROL_FEATURE] = rng.uniform(0, 0.1, size=len(X))
    return X


def get_feature_importances(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Extract coefficients from the fitted logistic regression."""
    coefs = pipeline.named_steps["clf"].coef_[0]
    return (
        pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )
