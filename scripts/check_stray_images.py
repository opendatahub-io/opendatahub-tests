#!/usr/bin/env python3
"""Check for container image strings not centralized in constants classes.

Scans Python files under tests/ and utilities/ for hardcoded image
references that are not defined in a registered IMAGE_SOURCES class.
Reports findings so developers can move images to the appropriate
constants file.

Modes:
  Full scan (default):
    python scripts/check_stray_images.py

  PR mode (only newly introduced strays in changed components):
    python scripts/check_stray_images.py --diff-base main

Exit codes:
  0  — no stray images found
  1  — stray images detected

Override: add '# noqa: IMG001' on the line to suppress the check.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUPPRESS_MARKER: str = "noqa: IMG001"


def _build_image_regex() -> re.Pattern[str]:
    from scripts.generate_image_manifest import KNOWN_REGISTRIES

    return re.compile(
        r"""(['"])"""  # opening quote
        r"((?:oci://)?"
        r"(?:" + "|".join(re.escape(r) for r in KNOWN_REGISTRIES) + r")"
        r"/[^'\"]+)"  # rest of image ref
        r"\1"  # closing quote
    )


def _get_constants_files() -> set[str]:
    from scripts.generate_image_manifest import IMAGE_SOURCES

    return {class_path.rsplit(".", 1)[0].replace(".", "/") + ".py" for class_path in IMAGE_SOURCES.values()}


def _collect_known_images() -> set[str]:
    from scripts.generate_image_manifest import generate_manifest

    manifest = generate_manifest()
    known = set()
    for images in manifest.values():
        known.update(images)
    return known


def _scan_file(path: Path, known: set[str], only_lines: set[int] | None = None) -> list[tuple[int, str]]:
    """Return (line_number, image_ref) for stray images.

    If only_lines is set, only check those line numbers (for diff mode).
    """
    findings = []
    try:
        lines = path.read_text().splitlines()
    except OSError, UnicodeDecodeError:
        return findings

    rel = str(path.relative_to(ROOT))
    if rel in _get_constants_files():
        return findings

    image_re = _build_image_regex()
    for i, line in enumerate(lines, 1):
        if only_lines is not None and i not in only_lines:
            continue
        if SUPPRESS_MARKER in line:
            continue
        for match in image_re.finditer(line):
            image = match.group(2)
            normalized = image.removeprefix("oci://")
            if normalized not in known:
                findings.append((i, image))
    return findings


def _file_to_component(rel_path: str) -> str | None:
    """Extract component name from a relative path, e.g. 'tests/ai_safety/foo.py' -> 'ai_safety'."""
    parts = Path(rel_path).parts
    if len(parts) >= 2 and parts[0] == "tests":
        return parts[1]
    if parts[0] == "utilities":
        return "utilities"
    return None


def _get_diff_info(base: str) -> dict[str, set[int]]:
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


def _full_scan(known: set[str]) -> list[tuple[str, int, str]]:
    scan_dirs = [ROOT / "tests", ROOT / "utilities"]
    all_findings: list[tuple[str, int, str]] = []
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for py_file in sorted(scan_dir.rglob("*.py")):
            rel = str(py_file.relative_to(ROOT))
            for line_no, image in _scan_file(path=py_file, known=known):
                all_findings.append((rel, line_no, image))
    return all_findings


def _diff_scan(known: set[str], base: str) -> list[tuple[str, int, str]]:
    diff_info = _get_diff_info(base=base)
    if not diff_info:
        return []

    components: set[str] = set()
    for rel_path in diff_info:
        comp = _file_to_component(rel_path=rel_path)
        if comp:
            components.add(comp)
    if components:
        print(f"Changed components: {', '.join(sorted(components))}", file=sys.stderr)

    all_findings: list[tuple[str, int, str]] = []
    for rel_path, added_lines in sorted(diff_info.items()):
        abs_path = ROOT / rel_path
        if not abs_path.exists():
            continue
        if _file_to_component(rel_path=rel_path) is None:
            continue
        for line_no, image in _scan_file(path=abs_path, known=known, only_lines=added_lines):
            all_findings.append((rel_path, line_no, image))
    return all_findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--diff-base",
        metavar="REF",
        help="Only check newly added lines in components modified since REF (e.g. main, origin/main)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output findings as JSON array (for CI integrations)",
    )
    args = parser.parse_args()

    known = _collect_known_images()

    if args.diff_base:
        all_findings = _diff_scan(known=known, base=args.diff_base)
        mode = f"PR mode (base: {args.diff_base})"
    else:
        all_findings = _full_scan(known=known)
        mode = "full scan"

    if args.json:
        findings_json = [
            {"file": rel, "line": line_no, "image": image, "component": _file_to_component(rel_path=rel) or "unknown"}
            for rel, line_no, image in all_findings
        ]
        json.dump(findings_json, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1 if all_findings else 0

    if not all_findings:
        print(f"No stray images found — {mode} ({len(known)} known images).")
        return 0

    print(f"[{mode}] Found {len(all_findings)} stray image(s) not in constants classes:\n")
    for rel, line_no, image in all_findings:
        print(f"  {rel}:{line_no}: {image}")
    print(f"\nMove these to a constants class or suppress with '# {SUPPRESS_MARKER}'")
    return 1


if __name__ == "__main__":
    sys.exit(main())
