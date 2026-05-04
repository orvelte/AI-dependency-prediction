"""Session-level metrics: length, duration, vocabulary richness, sentiment trajectory."""

import numpy as np
import pandas as pd


def mean_message_length(session_df: pd.DataFrame, text_col: str = "turn_text") -> float:
    """Mean user turn length in whitespace-split tokens."""
    user_texts = session_df[session_df["turn_role"] == "user"][text_col].fillna("")
    if user_texts.empty:
        return 0.0
    return user_texts.apply(lambda t: len(t.split())).mean()


def session_duration_minutes(session_df: pd.DataFrame, time_col: str = "timestamp") -> float:
    """Total session span in minutes (max - min timestamp)."""
    if time_col not in session_df.columns:
        return 0.0
    ts = pd.to_datetime(session_df[time_col], utc=True, errors="coerce").dropna()
    if ts.empty or ts.nunique() == 1:
        return 0.0
    return (ts.max() - ts.min()).total_seconds() / 60.0


def type_token_ratio(session_df: pd.DataFrame, text_col: str = "turn_text") -> float:
    """Type-token ratio (TTR) across all user turns in the session."""
    user_texts = session_df[session_df["turn_role"] == "user"][text_col].fillna("").tolist()
    tokens = " ".join(user_texts).lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def sentiment_trajectory_slope(
    session_df: pd.DataFrame,
    text_col: str = "turn_text",
    sentiment_fn=None,
) -> float:
    """Linear slope of sentiment scores across user turns (positive = improving sentiment).

    If no `sentiment_fn` is provided, falls back to a simple positive/negative word heuristic.
    `sentiment_fn` should accept a string and return a scalar in [-1, 1].
    """
    POSITIVE = {"good", "great", "thanks", "happy", "love", "excellent", "wonderful", "amazing"}
    NEGATIVE = {"bad", "hate", "terrible", "awful", "horrible", "angry", "sad", "frustrated"}

    def _simple_sentiment(text: str) -> float:
        words = set(text.lower().split())
        pos = len(words & POSITIVE)
        neg = len(words & NEGATIVE)
        total = max(pos + neg, 1)
        return (pos - neg) / total

    fn = sentiment_fn or _simple_sentiment
    user_turns = session_df[session_df["turn_role"] == "user"][text_col].fillna("").tolist()
    if len(user_turns) < 2:
        return 0.0

    scores = [fn(t) for t in user_turns]
    x = np.arange(len(scores), dtype=float)
    slope = np.polyfit(x, scores, 1)[0]
    return float(slope)


def n_user_turns(session_df: pd.DataFrame) -> int:
    """Number of user turns in the session."""
    return int((session_df["turn_role"] == "user").sum())


def extract_session_metrics(
    session_df: pd.DataFrame,
    text_col: str = "turn_text",
    time_col: str = "timestamp",
    sentiment_fn=None,
) -> dict:
    """Aggregate session-level metrics for a single session DataFrame."""
    return {
        "mean_message_length": mean_message_length(session_df, text_col),
        "session_duration_minutes": session_duration_minutes(session_df, time_col),
        "type_token_ratio": type_token_ratio(session_df, text_col),
        "sentiment_slope": sentiment_trajectory_slope(session_df, text_col, sentiment_fn),
        "n_user_turns": n_user_turns(session_df),
    }
