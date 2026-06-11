"""Utility helpers for KServe platform tests."""

from typing import Any

FAST_SUFFIXES = ("-fast-1", "-fast-2")


def extract_container_images(obj: Any) -> list[str]:
    """Recursively extract container image strings from a resource spec dict.

    Args:
        obj: A dict, list, or scalar from a deserialized Kubernetes resource spec.

    Returns:
        List of image strings found under any ``"image"`` key.
    """
    images: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "image" and isinstance(value, str):
                images.append(value)
            else:
                images.extend(extract_container_images(obj=value))
    elif isinstance(obj, list):
        for item in obj:
            images.extend(extract_container_images(obj=item))
    return images


def stable_name_for_fast(fast_name: str) -> str:
    """Derive the stable config name by stripping the fast suffix.

    Args:
        fast_name: Name of a fast overlay config (ending in ``-fast-1`` or ``-fast-2``).

    Returns:
        The corresponding stable config name.

    Raises:
        ValueError: If the name does not end with a known fast suffix.
    """
    for suffix in FAST_SUFFIXES:
        if fast_name.endswith(suffix):
            return fast_name[: -len(suffix)]
    raise ValueError(f"Not a fast config name: {fast_name}")
