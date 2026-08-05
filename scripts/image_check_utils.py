"""Shared utilities for container image check scripts."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_diff_lines(base: str) -> dict[str, set[int]]:
    """Get added lines per file from git diff against base.

    Returns {relative_path: set of added line numbers}.
    """
    result = subprocess.run(
        ["git", "diff", f"{base}...HEAD", "--unified=0", "--diff-filter=ACMR", "--", "*.py"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(f"git diff failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(2)

    file_lines: dict[str, set[int]] = {}
    current_file = None

    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            if current_file not in file_lines:
                file_lines[current_file] = set()
        elif line.startswith("@@ ") and current_file:
            hunk_header = line.split("@@")[1].strip()
            plus_part = hunk_header.split("+")[1].split(" ")[0]
            if "," in plus_part:
                start, count = plus_part.split(",")
                start, count = int(start), int(count)
            else:
                start, count = int(plus_part), 1
            for ln in range(start, start + count):
                file_lines[current_file].add(ln)

    return file_lines


_NOQA_RE = re.compile(r"#\s*noqa:\s*([A-Z0-9]+(?:\s*,\s*[A-Z0-9]+)*)")


def is_suppressed(line: str, code: str) -> bool:
    """Check if a line has a # noqa: marker suppressing the given exact code."""
    match = _NOQA_RE.search(string=line)
    if not match:
        return False
    codes = {c.strip() for c in match.group(1).split(",")}
    return code in codes


def build_image_regex() -> re.Pattern[str]:
    """Build regex matching quoted container image strings from known registries."""
    from scripts.generate_image_manifest import KNOWN_REGISTRIES

    return re.compile(
        r"""(['"])"""
        r"((?:oci://)?"
        r"(?:" + "|".join(re.escape(r) for r in KNOWN_REGISTRIES) + r")"
        r"/[^'\"]+)"
        r"\1"
    )


def scan_python_files() -> list[Path]:
    """Return sorted list of all Python files under tests/ and utilities/."""
    scan_dirs = [ROOT / "tests", ROOT / "utilities"]
    files = []
    for scan_dir in scan_dirs:
        if scan_dir.exists():
            files.extend(sorted(scan_dir.rglob("*.py")))
    return files


_READ_ERRORS = (OSError, UnicodeDecodeError)


def read_lines(path: Path) -> list[str] | None:
    """Read file lines, returning None on read errors."""
    try:
        return path.read_text().splitlines()
    except _READ_ERRORS:
        return None
