#!/usr/bin/env python3
"""Generate a JSON manifest of all container images required by tests.

Scans known constants classes for string attributes matching container
registry patterns. Output is used as an OCI label on the odh-tests
container image so disconnected environments can discover which images
to mirror.

To add a new component, add an entry to IMAGE_CLASS_MAP below.
"""

import argparse
import importlib
import inspect
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMAGE_CLASS_MAP: dict[str, str] = {
    "ai_hub": "tests.ai_hub.image_constants.AiHubImages",
    "ai_safety": "tests.ai_safety.image_constants.AiSafetyImages",
    "shared": "utilities.image_constants.SharedImages",
}

KNOWN_REGISTRIES: tuple[str, ...] = (
    "quay.io",
    "ghcr.io",
    "docker.io",
    "registry.redhat.io",
    "public.ecr.aws",
    "nvcr.io",
)

REGISTRY_PATTERN: re.Pattern[str] = re.compile(
    r"^(oci://)?"
    r"(" + "|".join(re.escape(r) for r in KNOWN_REGISTRIES) + r")"
    r"/"
)

VALID_IMAGE_RE: re.Pattern[str] = re.compile(
    r"^[a-zA-Z0-9._-]+"  # registry host
    r"(:[0-9]+)?"  # optional port
    r"(/[a-zA-Z0-9._-]+)+"  # repo path segments
    r"("
    r"@sha256:[0-9a-f]{64}"  # digest
    r"|:[a-zA-Z0-9._-]+"  # tag
    r")$"
)


def _is_image(value: str) -> bool:
    return bool(REGISTRY_PATTERN.match(value))


def _normalize_image(value: str) -> str:
    """Strip oci:// prefix — it's a KServe URI scheme, not part of the pullable ref."""
    return value.removeprefix("oci://")


def _extract_images(cls: type) -> list[str]:
    images = []
    for name, value in inspect.getmembers(cls):
        if name.startswith("_"):
            continue
        if isinstance(value, str) and _is_image(value):
            images.append(_normalize_image(value))
        elif inspect.isclass(value):
            images.extend(_extract_images(cls=value))
    return images


def validate_image_format(image: str) -> str | None:
    """Return an error message if the image ref is malformed, None if valid."""
    if not VALID_IMAGE_RE.match(image):
        return f"malformed image reference: {image}"
    return None


def generate_manifest() -> dict[str, list[str]]:
    manifest: dict[str, list[str]] = {}
    for component, class_path in sorted(IMAGE_CLASS_MAP.items()):
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(name=module_path)
        cls = getattr(module, class_name)
        images = sorted(set(_extract_images(cls=cls)))
        if images:
            manifest[component] = images
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true", help="Single-line JSON (for OCI labels)")
    parser.add_argument("--validate", action="store_true", help="Validate image format and exit non-zero on errors")
    args = parser.parse_args()

    manifest = generate_manifest()

    if args.validate:
        errors = []
        for component, images in manifest.items():
            for image in images:
                if err := validate_image_format(image=image):
                    errors.append(f"  [{component}] {err}")
        if errors:
            print("Image format validation failed:", file=sys.stderr)
            for e in errors:
                print(e, file=sys.stderr)
            sys.exit(1)
        print(f"All {sum(len(v) for v in manifest.values())} images valid.")
        sys.exit(0)

    indent = None if args.compact else 2
    json.dump(manifest, sys.stdout, indent=indent, separators=(",", ":") if args.compact else None)
    sys.stdout.write("\n")
