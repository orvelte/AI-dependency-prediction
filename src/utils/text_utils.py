"""Tokenisation and text normalisation helpers."""

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```|`[^`]+`")


def normalise(text: str, remove_urls: bool = True, remove_code: bool = False) -> str:
    """Unicode-normalise, collapse whitespace, optionally strip URLs and code blocks."""
    text = unicodedata.normalize("NFC", text)
    if remove_code:
        text = _CODE_BLOCK_RE.sub(" ", text)
    if remove_urls:
        text = _URL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenise(text: str) -> list[str]:
    """Lowercase whitespace tokeniser (no external dependency)."""
    return normalise(text).lower().split()


def token_count(text: str) -> int:
    return max(len(tokenise(text)), 1)


def sentence_split(text: str) -> list[str]:
    """Naive sentence splitter on '.', '!', '?'."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]
