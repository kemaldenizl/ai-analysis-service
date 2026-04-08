"""
metadata_loader.py - Load and validate metadata JSON files for each model.

Each metadata JSON lives in ml/models_store/ and declares:
  - feature_names (ordered list)
  - model_file (joblib filename)
  - optional scaler_file, label_map, thresholds, etc.
"""

import os
import json
from typing import Any, Dict, List, Optional

_MODELS_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models_store",
)


def _resolve_metadata_path(name: str, base_dir: Optional[str] = None) -> str:
    """Return absolute path to ``<name>_metadata.json``."""
    base = base_dir or _MODELS_STORE_DIR
    filename = f"{name}_metadata.json" if not name.endswith(".json") else name
    return os.path.join(base, filename)


def load_metadata(
    name: str,
    *,
    base_dir: Optional[str] = None,
    required_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load a metadata JSON by logical name (e.g. ``"profiler"``).

    Parameters
    ----------
    name : str
        Logical name — maps to ``<name>_metadata.json`` inside *base_dir*.
    base_dir : str, optional
        Override the default models_store directory.
    required_keys : list[str], optional
        If supplied, ``validate_metadata_keys`` is called automatically.

    Returns
    -------
    dict
        Parsed metadata dictionary.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    ValueError
        If any *required_keys* are missing.
    """
    path = _resolve_metadata_path(name, base_dir=base_dir)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data: Dict[str, Any] = json.load(fh)

    if required_keys:
        validate_metadata_keys(data, required_keys, source=path)

    return data


def validate_metadata_keys(
    metadata: Dict[str, Any],
    required_keys: List[str],
    *,
    source: str = "<unknown>",
) -> None:
    """
    Raise ``ValueError`` if any *required_keys* are absent from *metadata*.
    """
    missing = [k for k in required_keys if k not in metadata]
    if missing:
        raise ValueError(
            f"Metadata from '{source}' is missing required keys: {missing}"
        )
