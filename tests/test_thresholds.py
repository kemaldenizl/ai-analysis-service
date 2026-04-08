"""
Tests for ml.contracts.thresholds
"""

import pytest

from app.ml.contracts.thresholds import anomaly_severity, risk_level_from_score


# ── anomaly_severity ──────────────────────────────────────────────

class TestAnomalySeverity:
    """Test anomaly score → severity classification."""

    def _meta(self, anomaly_th: float = 0.5, severe_th: float = 0.8):
        return {
            "thresholds": {
                "anomaly_score_threshold": anomaly_th,
                "severe_anomaly_score_threshold": severe_th,
            }
        }

    def test_low_score_is_normal(self):
        assert anomaly_severity(0.1, self._meta()) == "normal"

    def test_exactly_at_anomaly_threshold_is_normal(self):
        # score <= threshold → normal (uses strict >)
        assert anomaly_severity(0.5, self._meta()) == "normal"

    def test_above_anomaly_below_severe_is_anomaly(self):
        assert anomaly_severity(0.6, self._meta()) == "anomaly"

    def test_exactly_at_severe_threshold_is_anomaly(self):
        assert anomaly_severity(0.8, self._meta()) == "anomaly"

    def test_above_severe_is_severe_anomaly(self):
        assert anomaly_severity(0.9, self._meta()) == "severe_anomaly"

    def test_zero_score(self):
        assert anomaly_severity(0.0, self._meta()) == "normal"

    def test_negative_score(self):
        assert anomaly_severity(-1.0, self._meta()) == "normal"

    def test_very_high_score(self):
        assert anomaly_severity(100.0, self._meta()) == "severe_anomaly"

    def test_missing_thresholds_key_defaults_to_normal(self):
        # No thresholds → both default to 0.0
        # score > 0 → severe_anomaly (since severe_th defaults to anomaly_th = 0)
        assert anomaly_severity(0.0, {}) == "normal"

    def test_custom_thresholds_key(self):
        meta = {
            "custom": {
                "anomaly_score_threshold": 10,
                "severe_anomaly_score_threshold": 20,
            }
        }
        assert anomaly_severity(15, meta, thresholds_key="custom") == "anomaly"
        assert anomaly_severity(25, meta, thresholds_key="custom") == "severe_anomaly"


# ── risk_level_from_score ─────────────────────────────────────────

class TestRiskLevelFromScore:
    """Test risk score → level classification."""

    def _meta(self, low_th: float = 40.0, medium_th: float = 70.0):
        return {
            "risk_thresholds": {
                "low_threshold": low_th,
                "medium_threshold": medium_th,
            }
        }

    def test_low_risk(self):
        assert risk_level_from_score(10.0, self._meta()) == "LOW"

    def test_exactly_at_low_boundary_is_medium(self):
        # score >= low_th → not LOW (uses strict <)
        assert risk_level_from_score(40.0, self._meta()) == "MEDIUM"

    def test_medium_risk(self):
        assert risk_level_from_score(55.0, self._meta()) == "MEDIUM"

    def test_exactly_at_medium_boundary_is_high(self):
        assert risk_level_from_score(70.0, self._meta()) == "HIGH"

    def test_high_risk(self):
        assert risk_level_from_score(90.0, self._meta()) == "HIGH"

    def test_zero_score(self):
        assert risk_level_from_score(0.0, self._meta()) == "LOW"

    def test_max_score(self):
        assert risk_level_from_score(100.0, self._meta()) == "HIGH"

    def test_negative_score(self):
        assert risk_level_from_score(-5.0, self._meta()) == "LOW"

    def test_missing_thresholds_uses_defaults(self):
        # Defaults: low_threshold=40, medium_threshold=70
        assert risk_level_from_score(30.0, {}) == "LOW"
        assert risk_level_from_score(50.0, {}) == "MEDIUM"
        assert risk_level_from_score(80.0, {}) == "HIGH"

    def test_custom_thresholds_key(self):
        meta = {
            "my_thresholds": {
                "low_threshold": 20,
                "medium_threshold": 50,
            }
        }
        assert risk_level_from_score(10, meta, thresholds_key="my_thresholds") == "LOW"
        assert risk_level_from_score(30, meta, thresholds_key="my_thresholds") == "MEDIUM"
        assert risk_level_from_score(60, meta, thresholds_key="my_thresholds") == "HIGH"
