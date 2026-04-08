"""
model_loader.py - Centralised joblib model / scaler loading.

All path resolution goes through this module so predictors never hard-code
filesystem paths.
"""

import os
from typing import Any, Dict, Optional

import joblib

_MODELS_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models_store",
)


def resolve_model_path(filename: str, *, base_dir: Optional[str] = None) -> str:
    """
    Return the absolute path to a model file inside ``models_store/``.

    Parameters
    ----------
    filename : str
        Basename of the joblib file (e.g. ``"spending_profiler.joblib"``).
    base_dir : str, optional
        Override the default models_store directory.

    Returns
    -------
    str
        Absolute path.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    """
    base = base_dir or _MODELS_STORE_DIR
    path = os.path.join(base, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return path


def load_model_artifact(
    filename: str,
    *,
    base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load and return a joblib model artifact dict.

    Parameters
    ----------
    filename : str
        Basename of the joblib file.
    base_dir : str, optional
        Override the default models_store directory.

    Returns
    -------
    dict
        The deserialized artifact (model, scaler, selector, …).
    """
    path = resolve_model_path(filename, base_dir=base_dir)
    return joblib.load(path)


def load_scaler(
    filename: str,
    *,
    base_dir: Optional[str] = None,
) -> Any:
    """
    Load a standalone scaler (or any single sklearn object) from joblib.

    Parameters
    ----------
    filename : str
        Basename of the joblib file containing the scaler.
    base_dir : str, optional
        Override the default models_store directory.

    Returns
    -------
    object
        The deserialized scaler.
    """
    path = resolve_model_path(filename, base_dir=base_dir)
    return joblib.load(path)
