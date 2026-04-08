"""
thresholds.py - Anomaly severity and risk level helpers.

Each model's metadata JSON can declare threshold breakpoints; these
helpers apply those breakpoints uniformly.
"""

from typing import Any, Dict


def anomaly_severity(
    score: float,
    metadata: Dict[str, Any],
    *,
    thresholds_key: str = "thresholds",
) -> str:
    """
    Classify an anomaly score as ``"normal"``, ``"anomaly"``, or
    ``"severe_anomaly"`` using thresholds from metadata.

    Expected metadata structure under *thresholds_key*::

        {
            "anomaly_score_threshold": <float>,
            "severe_anomaly_score_threshold": <float>
        }

    Parameters
    ----------
    score : float
        The raw anomaly distance score.
    metadata : dict
        Loaded anomaly metadata.
    thresholds_key : str
        Key inside *metadata* holding the threshold dict.

    Returns
    -------
    str
        One of ``"normal"``, ``"anomaly"``, ``"severe_anomaly"``.
    """
    t = metadata.get(thresholds_key, {})
    anomaly_th = float(t.get("anomaly_score_threshold", 0.0))
    severe_th = float(t.get("severe_anomaly_score_threshold", anomaly_th))

    if score > severe_th:
        return "severe_anomaly"
    if score > anomaly_th:
        return "anomaly"
    return "normal"


def risk_level_from_score(
    risk_score: float,
    metadata: Dict[str, Any],
    *,
    thresholds_key: str = "risk_thresholds",
) -> str:
    """
    Classify a composite risk score (0–100) as ``"LOW"``, ``"MEDIUM"``,
    or ``"HIGH"`` using thresholds from metadata.

    Expected metadata structure under *thresholds_key*::

        {
            "low_threshold": <float>,
            "medium_threshold": <float>
        }

    Parameters
    ----------
    risk_score : float
        Composite risk score (0–100 scale).
    metadata : dict
        Loaded risk metadata.
    thresholds_key : str
        Key inside *metadata* holding the threshold dict.

    Returns
    -------
    str
        One of ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``.
    """
    t = metadata.get(thresholds_key, {})
    low_th = float(t.get("low_threshold", 40.0))
    medium_th = float(t.get("medium_threshold", 70.0))

    if risk_score < low_th:
        return "LOW"
    if risk_score < medium_th:
        return "MEDIUM"
    return "HIGH"
