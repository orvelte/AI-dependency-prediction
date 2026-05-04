"""Pragmatic feature extraction: refusal resistance, gratitude, question-to-statement ratio."""

import re

import pandas as pd


REFUSAL_PATTERNS = re.compile(
    r"\b(i('m| am) (sorry|unable|not able)|i (can't|cannot)|i don't think i (can|should)|"
    r"that('s| is) (not|outside)|i('m| am) afraid|i('m| am) not comfortable|"
    r"this (falls|goes) outside|i (must|have to) decline)\b",
    re.IGNORECASE,
)

REPHRASE_MIN_TOKENS = 4
GRATITUDE_RE = re.compile(
    r"\b(thank(s| you)|appreciate|grateful|cheers|thx)\b", re.IGNORECASE
)
DEFERENCE_RE = re.compile(
    r"\b(please|could you|would you|if you don't mind|kindly|sorry to bother)\b",
    re.IGNORECASE,
)

QUESTION_RE = re.compile(r"\?")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def _is_refusal(text: str) -> bool:
    return bool(REFUSAL_PATTERNS.search(text))


def _is_rephrase(prev_user_text: str, curr_user_text: str) -> bool:
    """Heuristic: non-identical user turn with ≥ REPHRASE_MIN_TOKENS words after a refusal."""
    if prev_user_text.strip() == curr_user_text.strip():
        return False
    return len(curr_user_text.split()) >= REPHRASE_MIN_TOKENS


def refusal_resistance_rate(session_df: pd.DataFrame, text_col: str = "turn_text") -> float:
    """Proportion of assistant refusals followed by a user rephrase in the same session."""
    turns = session_df[[text_col, "turn_role"]].reset_index(drop=True)
    refusal_count = 0
    rephrase_count = 0

    for i in range(len(turns) - 2):
        if turns.at[i, "turn_role"] == "user" and turns.at[i + 1, "turn_role"] == "assistant":
            if _is_refusal(turns.at[i + 1, text_col]):
                refusal_count += 1
                if i + 2 < len(turns) and turns.at[i + 2, "turn_role"] == "user":
                    if _is_rephrase(turns.at[i, text_col], turns.at[i + 2, text_col]):
                        rephrase_count += 1

    return rephrase_count / refusal_count if refusal_count > 0 else 0.0


def gratitude_rate(session_df: pd.DataFrame, text_col: str = "turn_text") -> float:
    """Gratitude/deference expression rate per user turn."""
    user_turns = session_df[session_df["turn_role"] == "user"][text_col].fillna("").tolist()
    if not user_turns:
        return 0.0
    combined = " ".join(user_turns)
    n_tokens = max(len(combined.split()), 1)
    return (len(GRATITUDE_RE.findall(combined)) + len(DEFERENCE_RE.findall(combined))) / n_tokens


def question_to_statement_ratio(session_df: pd.DataFrame, text_col: str = "turn_text") -> float:
    """Ratio of sentences ending in '?' to total sentences in user turns."""
    user_turns = session_df[session_df["turn_role"] == "user"][text_col].fillna("").tolist()
    if not user_turns:
        return 0.0
    combined = " ".join(user_turns)
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(combined) if s.strip()]
    if not sentences:
        return 0.0
    questions = sum(1 for s in sentences if s.endswith("?")) + len(QUESTION_RE.findall(combined))
    return min(questions / max(len(sentences), 1), 1.0)


def extract_pragmatic_features(session_df: pd.DataFrame, text_col: str = "turn_text") -> dict:
    """Aggregate pragmatic features for a single session DataFrame."""
    return {
        "refusal_resistance_rate": refusal_resistance_rate(session_df, text_col),
        "gratitude_rate": gratitude_rate(session_df, text_col),
        "question_to_statement_ratio": question_to_statement_ratio(session_df, text_col),
    }
