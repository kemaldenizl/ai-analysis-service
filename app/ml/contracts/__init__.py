"""
ml.contracts - Inference contract layer.

Provides metadata-driven feature contracts, model/scaler loading,
schema validation, label mappings, and threshold helpers to
guarantee deterministic, reproducible inference across all predictors.
"""

from .metadata_loader import load_metadata, validate_metadata_keys
from .feature_contracts import get_feature_list, build_feature_frame
from .model_loader import load_model_artifact, load_scaler, resolve_model_path
from .schema_validator import validate_required_features, validate_no_extra_fields, validate_input
from .label_mappings import cluster_id_to_label, cluster_id_to_description
from .thresholds import anomaly_severity, risk_level_from_score

__all__ = [
    "load_metadata",
    "validate_metadata_keys",
    "get_feature_list",
    "build_feature_frame",
    "load_model_artifact",
    "load_scaler",
    "resolve_model_path",
    "validate_required_features",
    "validate_no_extra_fields",
    "validate_input",
    "cluster_id_to_label",
    "cluster_id_to_description",
    "anomaly_severity",
    "risk_level_from_score",
]
