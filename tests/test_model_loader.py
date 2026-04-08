"""
Tests for ml.contracts.model_loader
"""

import os
import tempfile
import pytest
import joblib

from app.ml.contracts.model_loader import (
    resolve_model_path,
    load_model_artifact,
    load_scaler,
)


# ── resolve_model_path ─────────────────────────────────────────────

class TestResolveModelPath:
    """Verify model path resolution."""

    def test_resolves_existing_profiler_model(self, models_store_dir):
        path = resolve_model_path("spending_profiler.joblib", base_dir=models_store_dir)
        assert os.path.isfile(path)
        assert path.endswith("spending_profiler.joblib")

    def test_resolves_existing_anomaly_model(self, models_store_dir):
        path = resolve_model_path(
            "weekly_spending_anomaly_detector.joblib", base_dir=models_store_dir
        )
        assert os.path.isfile(path)

    def test_resolves_existing_forecast_model(self, models_store_dir):
        path = resolve_model_path(
            "monthly_forecaster.joblib", base_dir=models_store_dir
        )
        assert os.path.isfile(path)

    def test_missing_model_raises_file_not_found(self, models_store_dir):
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            resolve_model_path("does_not_exist.joblib", base_dir=models_store_dir)

    def test_returns_absolute_path(self, models_store_dir):
        path = resolve_model_path("spending_profiler.joblib", base_dir=models_store_dir)
        assert os.path.isabs(path)


# ── load_model_artifact ───────────────────────────────────────────

class TestLoadModelArtifact:
    """Verify joblib loading through the contract layer."""

    def test_loads_profiler_without_error(self, models_store_dir):
        artifact = load_model_artifact(
            "spending_profiler.joblib", base_dir=models_store_dir
        )
        assert artifact is not None

    def test_loads_anomaly_without_error(self, models_store_dir):
        artifact = load_model_artifact(
            "weekly_spending_anomaly_detector.joblib", base_dir=models_store_dir
        )
        assert artifact is not None

    def test_loads_forecast_without_error(self, models_store_dir):
        artifact = load_model_artifact(
            "monthly_forecaster.joblib", base_dir=models_store_dir
        )
        assert artifact is not None

    def test_missing_artifact_raises(self, models_store_dir):
        with pytest.raises(FileNotFoundError):
            load_model_artifact("nope.joblib", base_dir=models_store_dir)

    def test_round_trip_with_dummy_artifact(self, tmp_path):
        """Save & reload a dummy dict to confirm the load path works."""
        dummy = {"model": "dummy", "version": 1}
        path = tmp_path / "dummy.joblib"
        joblib.dump(dummy, path)
        loaded = load_model_artifact("dummy.joblib", base_dir=str(tmp_path))
        assert loaded == dummy


# ── load_scaler ───────────────────────────────────────────────────

class TestLoadScaler:
    """Verify scaler loading (same mechanism as model loading)."""

    def test_round_trip_with_dummy_scaler(self, tmp_path):
        dummy = {"mean": 0.0, "std": 1.0}
        path = tmp_path / "scaler.joblib"
        joblib.dump(dummy, path)
        loaded = load_scaler("scaler.joblib", base_dir=str(tmp_path))
        assert loaded == dummy

    def test_missing_scaler_raises(self, models_store_dir):
        with pytest.raises(FileNotFoundError):
            load_scaler("nonexistent_scaler.joblib", base_dir=models_store_dir)
