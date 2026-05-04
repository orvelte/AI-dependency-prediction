"""Integration tests for the full feature extraction pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.session_builder import assign_sessions
from src.features.pipeline import build_feature_matrix, get_early_session_features, get_late_session_features
from src.modelling.evaluation import label_high_dependency


def make_minimal_cohort() -> pd.DataFrame:
    """Create a synthetic cohort with 3 users, each having 12 sessions."""
    rng = np.random.default_rng(42)
    rows = []
    base_time = pd.Timestamp("2024-01-01", tz="UTC")
    for user_idx in range(3):
        user_id = f"user_{user_idx}"
        for session_num in range(1, 13):
            session_start = base_time + pd.Timedelta(days=(user_idx * 30 + session_num))
            for turn_idx in range(4):
                role = "user" if turn_idx % 2 == 0 else "assistant"
                text = "we should do this together honestly I feel good" if role == "user" else "Sure I can help"
                rows.append({
                    "user_id": user_id,
                    "conversation_id": f"{user_id}_conv_{session_num}",
                    "timestamp": session_start + pd.Timedelta(minutes=turn_idx * 5),
                    "turn_role": role,
                    "turn_text": text,
                    "session_id": f"{user_id}_s{session_num}",
                    "session_number": session_num,
                })
    return pd.DataFrame(rows)


class TestBuildFeatureMatrix:
    def test_returns_one_row_per_user_session(self):
        df = make_minimal_cohort()
        feat_matrix = build_feature_matrix(df)
        assert len(feat_matrix) == 3 * 12  # 3 users × 12 sessions

    def test_contains_expected_feature_columns(self):
        df = make_minimal_cohort()
        feat_matrix = build_feature_matrix(df)
        assert "fpp_rate" in feat_matrix.columns
        assert "refusal_resistance_rate" in feat_matrix.columns
        assert "n_user_turns" in feat_matrix.columns

    def test_no_nulls_in_numeric_features(self):
        df = make_minimal_cohort()
        feat_matrix = build_feature_matrix(df)
        numeric = feat_matrix.select_dtypes(include="number")
        assert not numeric.isnull().any().any()


class TestEarlyLateSlicing:
    def test_early_sessions_includes_only_1_to_3(self):
        df = make_minimal_cohort()
        feat_matrix = build_feature_matrix(df)
        early = get_early_session_features(feat_matrix)
        assert len(early) == 3  # one row per user

    def test_late_sessions_includes_session_10_plus(self):
        df = make_minimal_cohort()
        feat_matrix = build_feature_matrix(df)
        late = get_late_session_features(feat_matrix)
        assert len(late) == 3


class TestDependencyLabelling:
    def test_label_returns_bool_series(self):
        df = make_minimal_cohort()
        feat_matrix = build_feature_matrix(df)
        late = get_late_session_features(feat_matrix)
        labels = label_high_dependency(late)
        assert labels.dtype == bool or set(labels.unique()).issubset({True, False})

    def test_label_indexed_by_user(self):
        df = make_minimal_cohort()
        feat_matrix = build_feature_matrix(df)
        late = get_late_session_features(feat_matrix)
        labels = label_high_dependency(late)
        assert set(labels.index) == {"user_0", "user_1", "user_2"}
