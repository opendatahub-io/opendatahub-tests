#!/usr/bin/env python3
"""Check container image references for digest pinning and registry policies.

Scans IMAGE_SOURCES classes and reports:
  - ERROR: images using mutable tag references instead of @sha256: digests

Scans all Python files under tests/ and utilities/ and reports:
  - WARNING: images pulled from DockerHub (docker.io) which has strict rate limits
    (including images suppressed with # noqa: IMG001)

Modes:
  Full scan (default):
    python scripts/check_image_digests.py

  PR mode (only newly introduced issues in changed files):
    python scripts/check_image_digests.py --diff-base main

Exit codes:
  0  -- no errors (warnings may still be present)
  1  -- one or more images use tag references

Suppress codes:
  # noqa: IMG002  -- suppress tag-without-digest error
  # noqa: IMG003  -- suppress DockerHub warning
"""

import argparse
import inspect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.image_check_utils import get_diff_lines, is_suppressed, read_lines, scan_python_files  # noqa: E402

SUPPRESS_DIGEST: str = "IMG002"
SUPPRESS_DOCKERHUB: str = "IMG003"
DIGEST_PATTERN: re.Pattern[str] = re.compile(r"@sha256:[0-9a-f]{64}$")
DOCKERHUB_RE: re.Pattern[str] = re.compile(
    r"""(['"])"""
    r"((?:oci://)?docker\.io/[^'\"]+)"
    r"\1"
)


def _find_image_line(source_lines: list[str], start_line: int, image: str) -> int | None:
    normalized = image.removeprefix("oci://")
    for i, line in enumerate(source_lines):
        if normalized in line or any(part in line for part in normalized.split("@sha256:")):
            return start_line + i
    return None


def _check_images(changed_lines: dict[str, set[int]] | None = None) -> tuple[list[dict], list[dict]]:
    from scripts.generate_image_manifest import IMAGE_SOURCES, REGISTRY_PATTERN, _normalize_image

    errors: list[dict] = []
    warnings: list[dict] = []

    for component, class_path in sorted(IMAGE_SOURCES.items()):
        if "image_constants" not in class_path:
            continue

        module_path, class_name = class_path.rsplit(".", 1)
        source_file = ROOT / module_path.replace(".", "/")
        source_file = source_file.with_suffix(suffix=".py")
        rel_path = str(source_file.relative_to(ROOT))

        if changed_lines is not None and rel_path not in changed_lines:
            continue

        source_lines = read_lines(path=source_file)
        if source_lines is None:
            continue

        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        only_lines = changed_lines.get(rel_path) if changed_lines is not None else None

        for name, value in inspect.getmembers(cls):
            if name.startswith("_"):
                continue
            if inspect.isclass(value):
                for sub_name, sub_value in inspect.getmembers(value):
                    if sub_name.startswith("_"):
                        continue
                    if isinstance(sub_value, str) and REGISTRY_PATTERN.match(sub_value):
                        _check_single(
                            image=_normalize_image(value=sub_value),
                            attr=f"{name}.{sub_name}",
                            component=component,
                            rel_path=rel_path,
                            source_lines=source_lines,
                            errors=errors,
                            only_lines=only_lines,
                        )
            elif isinstance(value, str) and REGISTRY_PATTERN.match(value):
                _check_single(
                    image=_normalize_image(value=value),
                    attr=name,
                    component=component,
                    rel_path=rel_path,
                    source_lines=source_lines,
                    errors=errors,
                    only_lines=only_lines,
                )

    return errors, warnings


def _check_single(
    image: str,
    attr: str,
    component: str,
    rel_path: str,
    source_lines: list[str],
    errors: list[dict],
    only_lines: set[int] | None = None,
) -> None:
    line_no = _find_image_line(source_lines=source_lines, start_line=1, image=image)

    if only_lines is not None and (line_no is None or line_no not in only_lines):
        return

    if DIGEST_PATTERN.search(image):
        return

    if line_no and is_suppressed(line=source_lines[line_no - 1], code=SUPPRESS_DIGEST):
        return

    errors.append({
        "file": rel_path,
        "line": line_no or 0,
        "attribute": attr,
        "image": image,
        "component": component,
        "severity": "error",
        "rule": "IMG002",
        "message": "image uses tag reference instead of digest",
    })


def _scan_dockerhub(warnings: list[dict], changed_lines: dict[str, set[int]] | None = None) -> None:
    for py_file in scan_python_files():
        rel = str(py_file.relative_to(ROOT))

        if changed_lines is not None and rel not in changed_lines:
            continue

        lines = read_lines(path=py_file)
        if lines is None:
            continue

        only_lines = changed_lines.get(rel) if changed_lines is not None else None

        for i, line in enumerate(lines, 1):
            if only_lines is not None and i not in only_lines:
                continue
            if is_suppressed(line=line, code=SUPPRESS_DOCKERHUB):
                continue
            for match in DOCKERHUB_RE.finditer(line):
                image = match.group(2).removeprefix("oci://")
                warnings.append({
                    "file": rel,
                    "line": i,
                    "attribute": "",
                    "image": image,
                    "component": "",
                    "severity": "warning",
                    "rule": "IMG003",
                    "message": "image sourced from DockerHub -- switch to quay.io or registry.redhat.io to avoid pull rate limits",
                })


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output all findings as JSON array",
    )
    parser.add_argument(
        "--diff-base",
        metavar="REF",
        help="Only check newly added lines in files modified since REF (e.g. main, origin/main)",
    )
    args = parser.parse_args()

    changed_lines = None
    if args.diff_base:
        changed_lines = get_diff_lines(base=args.diff_base)
        if not changed_lines:
            if args.json:
                print("[]")
            else:
                print(f"No relevant changes found (base: {args.diff_base}).")
            return 0

    errors, warnings = _check_images(changed_lines=changed_lines)
    _scan_dockerhub(warnings=warnings, changed_lines=changed_lines)

    mode = f"PR mode (base: {args.diff_base})" if args.diff_base else "full scan"

    if args.json:
        json.dump(errors + warnings, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1 if errors else 0

    if errors:
        print(f"[{mode}] Found {len(errors)} image(s) using tag references instead of digests:\n")
        for f in errors:
            loc = f"{f['file']}:{f['line']}" if f["line"] else f["file"]
            print(f"  {loc}: {f['attribute']}")
            print(f"    {f['image']}")
            print()
        print(f"Pin these images with @sha256:<digest> or suppress with '# noqa: {SUPPRESS_DIGEST}'")
        print()

    if warnings:
        print(f"[{mode}] Found {len(warnings)} image(s) sourced from DockerHub:\n")
        for f in warnings:
            loc = f"{f['file']}:{f['line']}" if f["line"] else f["file"]
            print(f"  {loc}: {f['attribute']}")
            print(f"    {f['image']}")
            print()
        print(
            f"Use an equivalent image from quay.io or registry.redhat.io, or suppress with '# noqa: {SUPPRESS_DOCKERHUB}'"
        )
        print()

    if not errors and not warnings:
        print(f"[{mode}] All images use digest references and approved registries.")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
