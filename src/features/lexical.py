"""Lexical feature extraction: first-person plural, self-disclosure, hedging, fillers."""

import re
from typing import Sequence

import pandas as pd


FPP_PRONOUNS = re.compile(r"\b(we|us|our|ours|ourselves)\b", re.IGNORECASE)

SELF_DISCLOSURE_MARKERS = [
    r"\bi feel\b",
    r"\bi felt\b",
    r"\bi('m| am) (scared|worried|anxious|sad|happy|excited|angry|lonely|depressed)\b",
    r"\bhonestly\b",
    r"\btruthfully\b",
    r"\bto be honest\b",
    r"\bconfess\b",
    r"\badmit\b",
    r"\bi('ve| have) never told\b",
    r"\bpersonally\b",
]
SELF_DISCLOSURE_RE = re.compile("|".join(SELF_DISCLOSURE_MARKERS), re.IGNORECASE)

HEDGING_MARKERS = [
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bprobably\b",
    r"\bpossibly\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\bI think\b",
    r"\bI guess\b",
    r"\bI suppose\b",
    r"\bI believe\b",
    r"\bseem(s)?\b",
    r"\bappear(s)?\b",
    r"\bmight\b",
    r"\bcould\b",
]
HEDGING_RE = re.compile("|".join(HEDGING_MARKERS), re.IGNORECASE)

FILLER_MARKERS = [
    r"\bum\b",
    r"\buh\b",
    r"\blike\b",
    r"\bactually\b",
    r"\bbasically\b",
    r"\bjust\b",
    r"\byou know\b",
    r"\bright\b",
    r"\bokay\b",
    r"\bok\b",
]
FILLER_RE = re.compile("|".join(FILLER_MARKERS), re.IGNORECASE)


def _token_count(text: str) -> int:
    return max(len(text.split()), 1)


def fpp_rate(text: str) -> float:
    """First-person plural pronoun rate per token."""
    return len(FPP_PRONOUNS.findall(text)) / _token_count(text)


def self_disclosure_rate(text: str) -> float:
    """Self-disclosure marker rate per token."""
    return len(SELF_DISCLOSURE_RE.findall(text)) / _token_count(text)


def hedging_rate(text: str) -> float:
    """Hedging language rate per token."""
    return len(HEDGING_RE.findall(text)) / _token_count(text)


def filler_rate(text: str) -> float:
    """Filler / phatic expression rate per token."""
    return len(FILLER_RE.findall(text)) / _token_count(text)


def extract_lexical_features(session_df: pd.DataFrame, text_col: str = "turn_text") -> dict:
    """Aggregate lexical features across all user turns in a single session DataFrame.

    `session_df` should contain only user turns for one session.
    Returns a dict of scalar feature values.
    """
    user_turns = session_df[session_df["turn_role"] == "user"][text_col].fillna("").tolist()
    if not user_turns:
        return {"fpp_rate": 0.0, "self_disclosure_rate": 0.0, "hedging_rate": 0.0, "filler_rate": 0.0}

    combined = " ".join(user_turns)
    return {
        "fpp_rate": fpp_rate(combined),
        "self_disclosure_rate": self_disclosure_rate(combined),
        "hedging_rate": hedging_rate(combined),
        "filler_rate": filler_rate(combined),
    }
