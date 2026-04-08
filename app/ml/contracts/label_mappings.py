"""
label_mappings.py - Cluster-ID → human-readable label / description helpers.

Uses ``label_map`` and ``label_descriptions`` dicts stored in each model's
metadata JSON so that mappings are always explicit and version-controlled.
"""

from typing import Any, Dict, Optional


def cluster_id_to_label(
    cluster_id: int,
    metadata: Dict[str, Any],
    *,
    label_map_key: str = "label_map",
    default: str = "unknown_profile",
) -> str:
    """
    Map a numeric cluster id to its human-readable label.

    Parameters
    ----------
    cluster_id : int
        The cluster index returned by the model.
    metadata : dict
        Loaded metadata dict (must contain *label_map_key*).
    label_map_key : str
        Key inside *metadata* that holds ``{str(cluster_id): label}`` mapping.
    default : str
        Fallback label when *cluster_id* has no mapping.

    Returns
    -------
    str
    """
    label_map: Dict[str, str] = metadata.get(label_map_key, {})
    return label_map.get(str(cluster_id), default)


def cluster_id_to_description(
    cluster_id: int,
    metadata: Dict[str, Any],
    *,
    descriptions_key: str = "label_descriptions",
    default: str = "No description available.",
) -> str:
    """
    Map a numeric cluster id to a longer description string.

    Parameters
    ----------
    cluster_id : int
        The cluster index.
    metadata : dict
        Loaded metadata dict (must contain *descriptions_key*).
    descriptions_key : str
        Key inside *metadata* that holds ``{str(cluster_id): description}``
        mapping.
    default : str
        Fallback description.

    Returns
    -------
    str
    """
    descriptions: Dict[str, str] = metadata.get(descriptions_key, {})
    return descriptions.get(str(cluster_id), default)
