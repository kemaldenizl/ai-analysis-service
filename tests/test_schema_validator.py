"""
Tests for ml.contracts.schema_validator
"""

import pytest

from app.ml.contracts.schema_validator import (
    validate_required_features,
    validate_no_extra_fields,
    validate_input,
)


# ── validate_required_features ─────────────────────────────────────

class TestValidateRequiredFeatures:
    """Verify required-feature checking."""

    def test_all_present_passes(self):
        validate_required_features({"a": 1, "b": 2}, ["a", "b"])

    def test_missing_feature_raises(self):
        with pytest.raises(ValueError, match="Missing required features"):
            validate_required_features({"a": 1}, ["a", "b"])

    def test_error_lists_missing_keys(self):
        with pytest.raises(ValueError, match="c"):
            validate_required_features({"a": 1}, ["a", "b", "c"])

    def test_extra_keys_are_fine(self):
        validate_required_features({"a": 1, "b": 2, "extra": 3}, ["a", "b"])

    def test_empty_required_passes(self):
        validate_required_features({"a": 1}, [])

    def test_empty_data_missing_required_raises(self):
        with pytest.raises(ValueError):
            validate_required_features({}, ["x"])


# ── validate_no_extra_fields ──────────────────────────────────────

class TestValidateNoExtraFields:
    """Verify extra-field rejection."""

    def test_exact_match_passes(self):
        validate_no_extra_fields({"a": 1, "b": 2}, ["a", "b"])

    def test_subset_passes(self):
        validate_no_extra_fields({"a": 1}, ["a", "b"])

    def test_extra_field_raises(self):
        with pytest.raises(ValueError, match="Unexpected extra fields"):
            validate_no_extra_fields({"a": 1, "b": 2, "z": 3}, ["a", "b"])

    def test_ignore_fields_allows_extras(self):
        validate_no_extra_fields(
            {"a": 1, "request_id": "abc"},
            ["a"],
            ignore_fields={"request_id"},
        )

    def test_ignore_fields_still_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unexpected extra fields"):
            validate_no_extra_fields(
                {"a": 1, "request_id": "abc", "unexpected": 0},
                ["a"],
                ignore_fields={"request_id"},
            )


# ── validate_input ────────────────────────────────────────────────

class TestValidateInput:
    """Verify the convenience wrapper."""

    def test_valid_input_passes(self):
        validate_input({"a": 1, "b": 2}, ["a", "b"])

    def test_missing_feature_raises(self):
        with pytest.raises(ValueError, match="Missing required features"):
            validate_input({"a": 1}, ["a", "b"])

    def test_extras_allowed_by_default(self):
        validate_input({"a": 1, "b": 2, "extra": 3}, ["a", "b"])

    def test_extras_rejected_when_flag_set(self):
        with pytest.raises(ValueError, match="Unexpected extra fields"):
            validate_input(
                {"a": 1, "b": 2, "extra": 3},
                ["a", "b"],
                reject_extras=True,
            )

    def test_extras_rejected_but_ignored_fields_pass(self):
        validate_input(
            {"a": 1, "b": 2, "request_id": "x"},
            ["a", "b"],
            reject_extras=True,
            ignore_fields={"request_id"},
        )

    def test_wrong_structure_raises(self):
        """A non-iterable input will fail at the 'f not in input_data' check."""
        with pytest.raises(TypeError):
            validate_input(42, ["a"])  # type: ignore[arg-type]
