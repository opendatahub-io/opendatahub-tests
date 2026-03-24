"""Path resolution and validation utilities for repo-relative file access."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_repo_path(source: str | Path, repo_root: Path | None = None) -> Path:
    """Turn a repo-relative or absolute path into a safe, resolved absolute path.

    Ensures the final path lives inside the repository root. Symlinks are fully
    resolved before the check so that a symlink pointing outside the repo is rejected.

    Accepted (returns resolved absolute path)::

        resolve_repo_path("tests/data/sample.pdf")          # relative to repo root
        resolve_repo_path("tests/data/../data/sample.pdf")   # normalised, still under root
        resolve_repo_path("/home/user/repo/tests/data/f.txt")  # absolute, under root

    Rejected (raises ``ValueError``)::

        resolve_repo_path("../../etc/passwd")                # escapes repo root
        resolve_repo_path("/tmp/evil.txt")                   # absolute, outside root

    Args:
        source: A repo-relative string/path or an absolute path.
        repo_root: Repository root to validate against. Defaults to the detected
            repo root (parent of the ``utilities/`` package).

    Returns:
        The resolved absolute path, guaranteed to be under ``repo_root``.

    Raises:
        ValueError: If the resolved path falls outside the repo root.
    """
    repo_root_resolved = (repo_root or _REPO_ROOT).resolve()
    raw = Path(source)  # noqa: FCN001
    resolved = raw.resolve() if raw.is_absolute() else (repo_root_resolved / raw).resolve()
    if not resolved.is_relative_to(repo_root_resolved):
        raise ValueError(
            f"Path must be under repo root ({repo_root_resolved}): {source!r}",
        )
    return resolved
