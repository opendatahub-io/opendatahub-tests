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
  0  -- no stray images found
  1  -- stray images detected

Override: add '# noqa: IMG001' on the line to suppress the check.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.image_check_utils import (  # noqa: E402
    build_image_regex,
    get_diff_lines,
    is_suppressed,
    read_lines,
    scan_python_files,
)

SUPPRESS_CODE: str = "IMG001"


def _get_image_constants_files() -> set[str]:
    """Return paths to image_constants.py files — the only files exempt from stray checks."""
    from scripts.generate_image_manifest import IMAGE_SOURCES

    return {
        class_path.rsplit(".", 1)[0].replace(".", "/") + ".py"
        for class_path in IMAGE_SOURCES.values()
        if "image_constants" in class_path
    }


def _collect_known_images() -> set[str]:
    from scripts.generate_image_manifest import generate_manifest

    manifest = generate_manifest()
    known = set()
    for images in manifest.values():
        known.update(images)
    return known


def _scan_file(path: Path, known: set[str], only_lines: set[int] | None = None) -> list[tuple[int, str]]:
    findings = []
    lines = read_lines(path=path)
    if lines is None:
        return findings

    rel = str(path.relative_to(ROOT))
    if rel in _get_image_constants_files():
        return findings

    image_re = build_image_regex()
    for i, line in enumerate(lines, 1):
        if only_lines is not None and i not in only_lines:
            continue
        if is_suppressed(line=line, code=SUPPRESS_CODE):
            continue
        for match in image_re.finditer(line):
            image = match.group(2)
            normalized = image.removeprefix("oci://")
            if normalized not in known:
                findings.append((i, image))
    return findings


def _file_to_component(rel_path: str) -> str | None:
    parts = Path(rel_path).parts
    if len(parts) >= 2 and parts[0] == "tests":
        return parts[1]
    if parts[0] == "utilities":
        return "utilities"
    return None


def _full_scan(known: set[str]) -> list[tuple[str, int, str]]:
    all_findings: list[tuple[str, int, str]] = []
    for py_file in scan_python_files():
        rel = str(py_file.relative_to(ROOT))
        for line_no, image in _scan_file(path=py_file, known=known):
            all_findings.append((rel, line_no, image))
    return all_findings


def _diff_scan(known: set[str], base: str) -> list[tuple[str, int, str]]:
    diff_info = get_diff_lines(base=base)
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
        print(f"No stray images found -- {mode} ({len(known)} known images).")
        return 0

    print(f"[{mode}] Found {len(all_findings)} stray image(s) not in constants classes:\n")
    for rel, line_no, image in all_findings:
        print(f"  {rel}:{line_no}: {image}")
    print(f"\nMove these to a constants class or suppress with '# noqa: {SUPPRESS_CODE}'")
    return 1


if __name__ == "__main__":
    sys.exit(main())
