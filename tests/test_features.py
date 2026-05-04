"""Unit tests for feature extraction functions."""

import pandas as pd
import pytest

from src.features.lexical import fpp_rate, self_disclosure_rate, hedging_rate, filler_rate, extract_lexical_features
from src.features.pragmatic import refusal_resistance_rate, gratitude_rate, question_to_statement_ratio
from src.features.session_metrics import (
    mean_message_length,
    type_token_ratio,
    n_user_turns,
    sentiment_trajectory_slope,
)


def make_session(turns: list[tuple[str, str]]) -> pd.DataFrame:
    """Helper: turns is a list of (role, text) tuples."""
    return pd.DataFrame(turns, columns=["turn_role", "turn_text"])


class TestLexicalFeatures:
    def test_fpp_rate_detects_we_us_our(self):
        assert fpp_rate("we did this together, our project") > 0

    def test_fpp_rate_zero_on_no_match(self):
        assert fpp_rate("I went to the store") == 0.0

    def test_self_disclosure_rate_detects_i_feel(self):
        assert self_disclosure_rate("I feel really nervous about this honestly") > 0

    def test_hedging_rate_detects_maybe(self):
        assert hedging_rate("maybe I think it could work") > 0

    def test_filler_rate_detects_like_just(self):
        assert filler_rate("like I just wanted to say um") > 0

    def test_extract_lexical_only_uses_user_turns(self):
        session = make_session([
            ("user", "we should work on this together"),
            ("assistant", "we can certainly try"),
            ("user", "our project is special"),
        ])
        feats = extract_lexical_features(session)
        assert feats["fpp_rate"] > 0
        # Rate should be based on user turns only — if assistant text inflated it we'd get higher values
        session_no_assistant = make_session([
            ("user", "we should work on this together"),
            ("user", "our project is special"),
        ])
        feats_no_assistant = extract_lexical_features(session_no_assistant)
        assert abs(feats["fpp_rate"] - feats_no_assistant["fpp_rate"]) < 1e-9

    def test_empty_session_returns_zeros(self):
        session = make_session([("assistant", "Hello there")])
        feats = extract_lexical_features(session)
        assert all(v == 0.0 for v in feats.values())


class TestPragmaticFeatures:
    def test_refusal_resistance_detected(self):
        session = make_session([
            ("user", "Can you help me do something bad?"),
            ("assistant", "I'm sorry, I'm unable to assist with that."),
            ("user", "Okay can you help me do the same thing differently?"),
        ])
        rate = refusal_resistance_rate(session)
        assert rate == 1.0

    def test_no_refusal_gives_zero(self):
        session = make_session([
            ("user", "Tell me about dogs"),
            ("assistant", "Dogs are great companions"),
        ])
        assert refusal_resistance_rate(session) == 0.0

    def test_gratitude_rate_detects_thanks(self):
        session = make_session([
            ("user", "Thank you so much, I appreciate it"),
            ("assistant", "You're welcome"),
        ])
        assert gratitude_rate(session) > 0

    def test_question_ratio_all_questions(self):
        session = make_session([
            ("user", "What is this? How does it work?"),
            ("assistant", "It works well."),
        ])
        ratio = question_to_statement_ratio(session)
        assert ratio > 0


class TestSessionMetrics:
    def test_mean_message_length(self):
        session = make_session([
            ("user", "hello world"),
            ("user", "how are you doing today"),
            ("assistant", "I am fine thanks"),
        ])
        assert mean_message_length(session) == pytest.approx((2 + 5) / 2)

    def test_ttr_less_than_one(self):
        session = make_session([("user", "the cat sat on the mat"), ("assistant", "yes")])
        ttr = type_token_ratio(session)
        assert 0 < ttr <= 1

    def test_n_user_turns(self):
        session = make_session([
            ("user", "hi"),
            ("assistant", "hello"),
            ("user", "bye"),
        ])
        assert n_user_turns(session) == 2

    def test_sentiment_slope_positive(self):
        session = make_session([
            ("user", "this is bad and terrible"),
            ("user", "this is good and great and wonderful"),
        ])
        slope = sentiment_trajectory_slope(session)
        assert slope > 0
