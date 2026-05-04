"""Filter functions for the WildChat longitudinal cohort."""

import pandas as pd


def exclude_redacted(df: pd.DataFrame) -> pd.DataFrame:
    """Remove conversations where redacted == True.

    Pre-specified exclusion — see .pre-analysis-plan.md. Redacted content (0.9%
    of conversations) may have distorted linguistic markers due to PII anonymisation.
    This must be the first filter applied after loading.
    """
    if "redacted" not in df.columns:
        return df
    before = df["conversation_id"].nunique() if "conversation_id" in df.columns else len(df)
    df = df[~df["redacted"]].copy()
    after = df["conversation_id"].nunique() if "conversation_id" in df.columns else len(df)
    print(f"exclude_redacted: removed {before - after} conversations ({(before - after) / max(before, 1):.1%})")
    return df


def filter_longitudinal_cohort(
    df: pd.DataFrame,
    user_col: str = "hashed_ip",
    conv_col: str = "conversation_id",
    min_conversations: int = 10,
) -> pd.DataFrame:
    """Return only rows belonging to users who have at least `min_conversations` conversations.

    Operates on turn-level or conversation-level DataFrames — counts distinct
    conversation_ids per user, not rows.
    """
    conv_counts = (
        df.groupby(user_col)[conv_col]
        .nunique()
        .rename("n_conversations")
        .reset_index()
    )
    qualifying = conv_counts.loc[
        conv_counts["n_conversations"] >= min_conversations, user_col
    ]
    return df[df[user_col].isin(qualifying)].copy()


def summary(
    df: pd.DataFrame,
    user_col: str = "hashed_ip",
    conv_col: str = "conversation_id",
) -> dict:
    """Return a brief cohort summary dict for logging."""
    result = {"n_users": df[user_col].nunique()}
    if conv_col in df.columns:
        result["n_conversations"] = df[conv_col].nunique()
    result["n_turns"] = len(df)
    return result
