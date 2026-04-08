"""
feature_contracts.py - Ordered feature list helpers.

Ensures every predictor feeds its model with columns in the exact order
declared in the metadata JSON, regardless of what order the caller
supplies them.
"""

from typing import Any, Dict, List

import pandas as pd


def get_feature_list(metadata: Dict[str, Any], *, key: str = "feature_names") -> List[str]:
    """
    Return the ordered feature list from a loaded metadata dict.

    Parameters
    ----------
    metadata : dict
        Parsed metadata (from ``metadata_loader.load_metadata``).
    key : str
        The dict key that holds the feature list (default ``"feature_names"``).

    Returns
    -------
    list[str]

    Raises
    ------
    ValueError
        If the key is missing or the value is not a non-empty list.
    """
    features = metadata.get(key)
    if not features or not isinstance(features, list):
        raise ValueError(
            f"Metadata key '{key}' must be a non-empty list of feature names. "
            f"Got: {type(features)}"
        )
    return list(features)


def build_feature_frame(
    input_dict: Dict[str, Any],
    feature_names: List[str],
    *,
    fill_missing: float = 0.0,
) -> pd.DataFrame:
    """
    Build a single-row ``DataFrame`` whose columns match *feature_names*
    exactly, in order.

    Parameters
    ----------
    input_dict : dict
        Raw key-value pairs (e.g. from an API request body).
    feature_names : list[str]
        Ordered feature names from metadata.
    fill_missing : float
        Value used for features present in metadata but absent from *input_dict*.

    Returns
    -------
    pd.DataFrame
        Shape ``(1, len(feature_names))``.
    """
    row = {col: input_dict.get(col, fill_missing) for col in feature_names}
    return pd.DataFrame([row], columns=feature_names)
