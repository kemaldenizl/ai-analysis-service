"""
schema_validator.py - Input validation against feature contracts.

Provides helpers to ensure incoming data contains exactly the features
a model expects — no more, no less — and raises clear ``ValueError``
messages for any mismatch.
"""

from typing import Any, Dict, List, Optional, Set


def validate_required_features(
    input_data: Dict[str, Any],
    required_features: List[str],
) -> None:
    """
    Raise ``ValueError`` listing every *required_features* key absent from
    *input_data*.

    Parameters
    ----------
    input_data : dict
        The raw input dict to validate.
    required_features : list[str]
        Ordered feature names that must be present.
    """
    missing = [f for f in required_features if f not in input_data]
    if missing:
        raise ValueError(
            f"Missing required features ({len(missing)}): {missing}"
        )


def validate_no_extra_fields(
    input_data: Dict[str, Any],
    allowed_features: List[str],
    *,
    ignore_fields: Optional[Set[str]] = None,
) -> None:
    """
    Raise ``ValueError`` if *input_data* contains keys not in
    *allowed_features* (minus any *ignore_fields*).

    Parameters
    ----------
    input_data : dict
        The raw input dict to validate.
    allowed_features : list[str]
        Feature names that are allowed.
    ignore_fields : set[str], optional
        Extra keys to silently allow (e.g. ``{"request_id", "timestamp"}``).
    """
    allowed: Set[str] = set(allowed_features)
    if ignore_fields:
        allowed |= ignore_fields

    extras = [k for k in input_data if k not in allowed]
    if extras:
        raise ValueError(
            f"Unexpected extra fields ({len(extras)}): {extras}"
        )


def validate_input(
    input_data: Dict[str, Any],
    required_features: List[str],
    *,
    reject_extras: bool = False,
    ignore_fields: Optional[Set[str]] = None,
) -> None:
    """
    Convenience wrapper: validate required features exist and optionally
    reject extra fields, in a single call.

    Parameters
    ----------
    input_data : dict
        The raw input dict.
    required_features : list[str]
        Feature names that must all be present.
    reject_extras : bool
        If ``True``, also call ``validate_no_extra_fields``.
    ignore_fields : set[str], optional
        Passed to ``validate_no_extra_fields`` when *reject_extras* is True.
    """
    validate_required_features(input_data, required_features)
    if reject_extras:
        validate_no_extra_fields(
            input_data, required_features, ignore_fields=ignore_fields
        )
