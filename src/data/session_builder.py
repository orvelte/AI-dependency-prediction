"""Assign session numbers to conversations.

For the WildChat interim dataset, each conversation IS a session.
Session number = chronological rank of the conversation within a user's history
(1 = earliest), determined by sorting on the top-level UTC timestamp.

The gap-based clustering function is retained for reference but is not used
in the primary pipeline for this dataset.
"""

import pandas as pd


def assign_session_numbers(
    df: pd.DataFrame,
    user_col: str = "hashed_ip",
    conv_col: str = "conversation_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Add `session_number` (chronological rank per user, 1 = earliest conversation).

    Sorts conversations by `time_col` within each user group, then assigns a
    sequential integer rank. Conversations with identical timestamps get the same
    rank (dense rank). The result is merged back onto the turn-level DataFrame.

    This is the primary session assignment function for the WildChat dataset,
    where one conversation = one session.
    """
    df = df.copy()

    # One row per (user, conversation) with its earliest turn timestamp
    conv_times = (
        df.groupby([user_col, conv_col])[time_col]
        .min()
        .reset_index()
        .rename(columns={time_col: "conv_start"})
    )

    # Chronological rank within each user — dense so tied timestamps share a number
    conv_times["session_number"] = (
        conv_times
        .sort_values([user_col, "conv_start"])
        .groupby(user_col)["conv_start"]
        .rank(method="dense")
        .astype(int)
    )

    df = df.merge(
        conv_times[[user_col, conv_col, "session_number"]],
        on=[user_col, conv_col],
        how="left",
    )
    return df


def get_session_summary(
    df: pd.DataFrame,
    user_col: str = "hashed_ip",
    conv_col: str = "conversation_id",
) -> pd.DataFrame:
    """Return one row per conversation with start time, end time, and user turn count."""
    return (
        df.groupby([user_col, conv_col, "session_number"])
        .agg(
            session_start=("timestamp", "min"),
            session_end=("timestamp", "max"),
            n_user_turns=(
                "turn_role",
                lambda s: (s == "user").sum(),
            ),
            n_turns=("turn_text", "count"),
        )
        .reset_index()
        .sort_values([user_col, "session_number"])
    )


# ---------------------------------------------------------------------------
# Gap-based clustering (not used for WildChat; kept for reference)
# ---------------------------------------------------------------------------

def assign_sessions_by_gap(
    df: pd.DataFrame,
    user_col: str = "hashed_ip",
    conv_col: str = "conversation_id",
    time_col: str = "timestamp",
    gap_hours: float = 4.0,
) -> pd.DataFrame:
    """Cluster conversations into sessions by time gap (gap_hours threshold).

    Not used in the primary WildChat pipeline — conversations are treated as
    sessions directly. Retained for datasets where temporal clustering is needed.
    """
    df = df.copy()
    conv_starts = (
        df.groupby([user_col, conv_col])[time_col]
        .min()
        .reset_index()
        .rename(columns={time_col: "conv_start"})
        .sort_values([user_col, "conv_start"])
    )
    conv_starts["gap"] = (
        conv_starts.groupby(user_col)["conv_start"].diff().dt.total_seconds() / 3600
    )
    conv_starts["new_session"] = conv_starts["gap"].isna() | (conv_starts["gap"] >= gap_hours)
    conv_starts["session_number"] = conv_starts.groupby(user_col)["new_session"].cumsum()
    df = df.merge(
        conv_starts[[user_col, conv_col, "session_number"]],
        on=[user_col, conv_col],
        how="left",
    )
    return df
