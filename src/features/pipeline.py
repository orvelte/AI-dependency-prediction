"""Unified feature extraction pipeline.

Runs lexical, pragmatic, and session-metric extractors across all sessions
for a cohort DataFrame and returns a user × session feature matrix.
"""

import pandas as pd
from tqdm import tqdm

from src.features.lexical import extract_lexical_features
from src.features.pragmatic import extract_pragmatic_features
from src.features.session_metrics import extract_session_metrics


def extract_features_for_session(session_df: pd.DataFrame, **kwargs) -> dict:
    """Run all extractors on a single session DataFrame and merge results."""
    features = {}
    features.update(extract_lexical_features(session_df, **kwargs.get("lexical_kwargs", {})))
    features.update(extract_pragmatic_features(session_df, **kwargs.get("pragmatic_kwargs", {})))
    features.update(extract_session_metrics(session_df, **kwargs.get("session_kwargs", {})))
    return features


def build_feature_matrix(
    df: pd.DataFrame,
    user_col: str = "user_id",
    session_col: str = "session_id",
    session_number_col: str = "session_number",
    **kwargs,
) -> pd.DataFrame:
    """Extract features for every (user, session) pair.

    Returns a DataFrame with one row per (user_id, session_id) and a column
    per feature, plus `session_number` for downstream train/test slicing.
    """
    records = []
    groups = list(df.groupby([user_col, session_col]))
    for (user_id, session_id), session_df in tqdm(groups, desc="Extracting features"):
        feats = extract_features_for_session(session_df, **kwargs)
        session_num = session_df[session_number_col].iloc[0] if session_number_col in session_df.columns else None
        records.append({
            user_col: user_id,
            session_col: session_id,
            "session_number": session_num,
            **feats,
        })
    return pd.DataFrame(records)


def get_early_session_features(
    feature_matrix: pd.DataFrame,
    user_col: str = "hashed_ip",
    early_sessions: tuple[int, int] = (1, 3),
) -> pd.DataFrame:
    """Average features across sessions 1–3 per user (predictors)."""
    lo, hi = early_sessions
    early = feature_matrix[feature_matrix["session_number"].between(lo, hi)]
    exclude = {user_col, "conversation_id", "session_number"}
    feature_cols = [c for c in early.columns if c not in exclude]
    return early.groupby(user_col)[feature_cols].mean().reset_index()


def get_late_session_features(
    feature_matrix: pd.DataFrame,
    user_col: str = "hashed_ip",
    late_session_min: int = 10,
) -> pd.DataFrame:
    """Average features across sessions 10+ per user (outcome variables)."""
    late = feature_matrix[feature_matrix["session_number"] >= late_session_min]
    exclude = {user_col, "conversation_id", "session_number"}
    feature_cols = [c for c in late.columns if c not in exclude]
    return late.groupby(user_col)[feature_cols].mean().reset_index()
