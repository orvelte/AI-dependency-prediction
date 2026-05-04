"""Dataset loading for the WildChat interim parquet file.

The raw WildChat-1M dataset (7 GB) was never downloaded locally; a filtered
parquet was produced via HuggingFace streaming in Colab and saved to
data/interim/wildchat_longitudinal.parquet (1.7 GB).

Actual schema of the parquet (one row per conversation):
  conversation_id  int64  — top-level unique conversation ID (added by Colab streaming step)
  hashed_ip        str    — top-level most-common IP; used as user_id
  timestamp        str    — UTC timestamp of the last turn in the conversation
  redacted         bool   — True if the conversation was redacted (pre-specified exclusion)
  conversation     str    — JSON-encoded list of turn dicts

Turn dict keys (only role and content survived the streaming filter):
  role     str  — "user" or "assistant"
  content  str  — message text

Canonical columns produced by this module:
  conversation_id  str   — str-cast of the top-level int64 conversation_id
  hashed_ip        str   — top-level user identifier
  timestamp        Timestamp (UTC-aware)
  redacted         bool
  turn_role        str   — "user" or "assistant"
  turn_text        str   — message content
"""

import json
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow.parquet as pq


CANONICAL_TURN_COLUMNS = [
    "conversation_id",
    "hashed_ip",
    "timestamp",
    "redacted",
    "turn_role",
    "turn_text",
]


def _parse_conversation(conv_raw) -> list[dict]:
    """Return conversation as a list of dicts, handling JSON-string storage."""
    if isinstance(conv_raw, str):
        return json.loads(conv_raw)
    return conv_raw  # already a list


def _expand_batch_to_turns(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Expand a conversation-level batch DataFrame to turn-level rows."""
    records = []
    for _, row in batch_df.iterrows():
        turns = _parse_conversation(row["conversation"])
        conv_id = str(row.get("conversation_id", ""))
        hashed_ip = str(row.get("hashed_ip", ""))
        ts = row.get("timestamp", pd.NaT)
        redacted = bool(row.get("redacted", False))
        for turn in turns:
            records.append({
                "conversation_id": conv_id,
                "hashed_ip": hashed_ip,
                "timestamp": ts,
                "redacted": redacted,
                "turn_role": turn.get("role", ""),
                "turn_text": turn.get("content", ""),
            })
    if not records:
        return pd.DataFrame(columns=CANONICAL_TURN_COLUMNS)
    df = pd.DataFrame(records, columns=CANONICAL_TURN_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def iter_wildchat_parquet(
    path: str | Path,
    batch_size: int = 500,
) -> Iterator[pd.DataFrame]:
    """Yield turn-level DataFrames in batches from the interim parquet file.

    Use this for all pipeline operations — do not load the full 1.7 GB file at once.

    Args:
        path: Path to wildchat_longitudinal.parquet
        batch_size: Number of conversations per batch (500 keeps RAM ~200 MB/batch)

    Yields:
        Turn-level DataFrames with CANONICAL_TURN_COLUMNS.
    """
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        batch_df = batch.to_pandas()
        yield _expand_batch_to_turns(batch_df)


def load_wildchat_parquet(path: str | Path, batch_size: int = 500) -> pd.DataFrame:
    """Load the full interim parquet into a turn-level DataFrame.

    WARNING: Only use this on machines with sufficient RAM (file expands to ~8 GB turn-level).
    For production pipeline use, prefer iter_wildchat_parquet().
    """
    return pd.concat(
        list(iter_wildchat_parquet(path, batch_size=batch_size)),
        ignore_index=True,
    )


def load_lmsys(path: str | Path) -> pd.DataFrame:
    """Load LMSYS-Chat-1M parquet/jsonl and return a turn-level DataFrame."""
    path = Path(path)
    if path.suffix == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_json(path, lines=True)

    records = []
    for _, row in raw.iterrows():
        conv_id = str(row.get("conversation_id", ""))
        ts = row.get("timestamp", pd.NaT)
        for turn in row.get("conversation", []):
            records.append({
                "conversation_id": conv_id,
                "hashed_ip": conv_id,
                "timestamp": ts,
                "redacted": False,
                "turn_role": turn.get("role", ""),
                "turn_text": turn.get("content", ""),
            })
    df = pd.DataFrame(records, columns=CANONICAL_TURN_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df
