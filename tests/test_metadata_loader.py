"""
Tests for ml.contracts.metadata_loader
"""

import os
import pytest

from app.ml.contracts.metadata_loader import (
    load_metadata,
    validate_metadata_keys,
)


# ── load_metadata ──────────────────────────────────────────────────

class TestLoadMetadata:
    """Verify that every real metadata JSON loads correctly."""

    @pytest.mark.parametrize("name", ["profiler", "anomaly", "forecast", "risk"])
    def test_loads_each_metadata_successfully(self, name, models_store_dir):
        data = load_metadata(name, base_dir=models_store_dir)
        assert isinstance(data, dict)
        assert len(data) > 0

    @pytest.mark.parametrize("name", ["profiler", "anomaly", "forecast", "risk"])
    def test_metadata_contains_model_name(self, name, models_store_dir):
        data = load_metadata(name, base_dir=models_store_dir)
        assert "model_name" in data

    @pytest.mark.parametrize("name", ["profiler", "anomaly", "forecast", "risk"])
    def test_metadata_contains_model_version(self, name, models_store_dir):
        data = load_metadata(name, base_dir=models_store_dir)
        assert "model_version" in data

    def test_invalid_name_raises_file_not_found(self, models_store_dir):
        with pytest.raises(FileNotFoundError):
            load_metadata("nonexistent_model", base_dir=models_store_dir)

    def test_load_with_required_keys_passes(self, models_store_dir):
        data = load_metadata(
            "profiler",
            base_dir=models_store_dir,
            required_keys=["model_name", "feature_columns"],
        )
        assert "model_name" in data

    def test_load_with_missing_required_key_raises(self, models_store_dir):
        with pytest.raises(ValueError, match="missing required keys"):
            load_metadata(
                "profiler",
                base_dir=models_store_dir,
                required_keys=["model_name", "THIS_KEY_DOES_NOT_EXIST"],
            )


# ── validate_metadata_keys ─────────────────────────────────────────

class TestValidateMetadataKeys:
    """Verify the standalone key-validation helper."""

    def test_all_keys_present_passes(self):
        meta = {"a": 1, "b": 2, "c": 3}
        validate_metadata_keys(meta, ["a", "b"])  # should not raise

    def test_missing_key_raises_value_error(self):
        meta = {"a": 1}
        with pytest.raises(ValueError, match="missing required keys"):
            validate_metadata_keys(meta, ["a", "z"])

    def test_empty_required_keys_passes(self):
        validate_metadata_keys({"a": 1}, [])  # no keys required → no error

    def test_error_message_contains_source(self):
        meta = {"a": 1}
        with pytest.raises(ValueError, match="my_source"):
            validate_metadata_keys(meta, ["b"], source="my_source")
