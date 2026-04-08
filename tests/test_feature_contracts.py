"""
Tests for ml.contracts.feature_contracts
"""

import pytest
import pandas as pd

from app.ml.contracts.feature_contracts import get_feature_list, build_feature_frame


# ── get_feature_list ───────────────────────────────────────────────

class TestGetFeatureList:
    """Test ordered feature extraction from metadata dicts."""

    def test_returns_list_from_profiler(self, profiler_metadata):
        features = get_feature_list(profiler_metadata, key="feature_columns")
        assert isinstance(features, list)
        assert len(features) > 0
        assert features[0] == "total_transactions"

    def test_returns_list_from_anomaly(self, anomaly_metadata):
        features = get_feature_list(anomaly_metadata, key="feature_columns")
        assert isinstance(features, list)
        assert "total_expense" in features

    def test_returns_list_from_forecast(self, forecast_metadata):
        features = get_feature_list(forecast_metadata, key="feature_columns")
        assert isinstance(features, list)
        assert features[0] == "Year"

    def test_preserves_order(self, profiler_metadata):
        features = get_feature_list(profiler_metadata, key="feature_columns")
        expected_prefix = [
            "total_transactions",
            "total_expense",
            "total_income",
        ]
        assert features[:3] == expected_prefix

    def test_returns_new_list_not_reference(self, profiler_metadata):
        a = get_feature_list(profiler_metadata, key="feature_columns")
        b = get_feature_list(profiler_metadata, key="feature_columns")
        assert a == b
        assert a is not b  # must be a copy

    def test_missing_key_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty list"):
            get_feature_list({}, key="feature_columns")

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty list"):
            get_feature_list({"feature_columns": []}, key="feature_columns")

    def test_non_list_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty list"):
            get_feature_list({"feature_columns": "not_a_list"}, key="feature_columns")

    def test_default_key_is_feature_names(self):
        meta = {"feature_names": ["a", "b"]}
        assert get_feature_list(meta) == ["a", "b"]


# ── build_feature_frame ───────────────────────────────────────────

class TestBuildFeatureFrame:
    """Test single-row DataFrame construction."""

    def test_correct_shape(self, profiler_metadata):
        features = get_feature_list(profiler_metadata, key="feature_columns")
        input_dict = {f: 1.0 for f in features}
        df = build_feature_frame(input_dict, features)
        assert df.shape == (1, len(features))

    def test_column_order_matches_features(self, profiler_metadata):
        features = get_feature_list(profiler_metadata, key="feature_columns")
        input_dict = {f: float(i) for i, f in enumerate(features)}
        df = build_feature_frame(input_dict, features)
        assert list(df.columns) == features

    def test_values_match_input(self):
        features = ["a", "b", "c"]
        input_dict = {"a": 10, "b": 20, "c": 30}
        df = build_feature_frame(input_dict, features)
        assert df["a"].iloc[0] == 10
        assert df["b"].iloc[0] == 20
        assert df["c"].iloc[0] == 30

    def test_missing_feature_gets_fill_value(self):
        features = ["a", "b", "c"]
        input_dict = {"a": 1}
        df = build_feature_frame(input_dict, features)
        assert df["b"].iloc[0] == 0.0
        assert df["c"].iloc[0] == 0.0

    def test_custom_fill_missing(self):
        features = ["a", "b"]
        input_dict = {"a": 5}
        df = build_feature_frame(input_dict, features, fill_missing=-1.0)
        assert df["b"].iloc[0] == -1.0

    def test_extra_fields_in_input_are_ignored(self):
        features = ["a", "b"]
        input_dict = {"a": 1, "b": 2, "extra": 99}
        df = build_feature_frame(input_dict, features)
        assert list(df.columns) == ["a", "b"]
        assert "extra" not in df.columns

    def test_returns_dataframe(self):
        df = build_feature_frame({"x": 1}, ["x"])
        assert isinstance(df, pd.DataFrame)
