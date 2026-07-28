"""Dict subclass with dot-access for Kubernetes resource data."""

from __future__ import annotations

from typing import Any


class ResourceDict(dict):
    """Dict that allows attribute-style access to nested keys.

    Nested dicts are wrapped recursively. Lists of dicts are wrapped too.
    Missing keys raise AttributeError (not KeyError) for clean dot-access.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = dict.__getitem__(self, name)
        except KeyError:
            return None
        return _wrap(value=value)

    def __getitem__(self, key: str) -> Any:
        return _wrap(value=dict.__getitem__(self, key))

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return _wrap(value=dict.__getitem__(self, key))
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict (recursively unwraps nested ResourceDicts)."""
        return _unwrap(value=self)

    def __repr__(self) -> str:
        return f"ResourceDict({dict.__repr__(self)})"


def _wrap(value: Any) -> Any:
    if isinstance(value, dict) and not isinstance(value, ResourceDict):
        return ResourceDict(value)  # noqa: FCN001 - dict subclass constructor
    if isinstance(value, list):
        return [_wrap(value=item) for item in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _unwrap(value=val) for key, val in value.items()}
    if isinstance(value, list):
        return [_unwrap(value=item) for item in value]
    return value
