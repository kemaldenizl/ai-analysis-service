"""
Shared fixtures for the contract-layer test suite.
"""

import os
import json
import pytest
from typing import Any, Dict

# ── Paths ───────────────────────────────────────────────────────────
MODELS_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "app", "ml", "models_store",
)

METADATA_NAMES = ["profiler", "anomaly", "forecast", "risk"]


# ── Metadata fixtures ──────────────────────────────────────────────

def _load_json(name: str) -> Dict[str, Any]:
    path = os.path.join(MODELS_STORE_DIR, f"{name}_metadata.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def models_store_dir() -> str:
    """Absolute path to the real models_store directory."""
    return MODELS_STORE_DIR


@pytest.fixture
def profiler_metadata() -> Dict[str, Any]:
    return _load_json("profiler")


@pytest.fixture
def anomaly_metadata() -> Dict[str, Any]:
    return _load_json("anomaly")


@pytest.fixture
def forecast_metadata() -> Dict[str, Any]:
    return _load_json("forecast")


@pytest.fixture
def risk_metadata() -> Dict[str, Any]:
    return _load_json("risk")
