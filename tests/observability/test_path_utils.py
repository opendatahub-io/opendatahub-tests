from pathlib import Path

import pytest

from utilities.path_utils import resolve_trusted_path

pytestmark = pytest.mark.tier1


def test_resolve_trusted_path_allows_new_directory_without_root_constraint(tmp_path: Path) -> None:
    """Given a new artifact directory, resolve it without requiring it to be under the repository."""
    destination = tmp_path / "evidence"

    resolved = resolve_trusted_path(source=destination)

    assert resolved == destination
    assert not resolved.exists()


def test_resolve_trusted_path_rejects_symlink_components(tmp_path: Path) -> None:
    """Given an artifact path containing a symlink, reject it before any evidence directory is created."""
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "link"
    symlink.symlink_to(target=target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        resolve_trusted_path(source=symlink / "evidence")


def test_resolve_trusted_path_rejects_malformed_values() -> None:
    """Given an empty or NUL-containing path, reject the malformed artifact destination."""
    with pytest.raises(ValueError, match="empty"):
        resolve_trusted_path(source="   ")
    with pytest.raises(ValueError, match="NUL"):
        resolve_trusted_path(source="/tmp/evidence\x00invalid")
