#!/usr/bin/env python3
"""Resolve the manifest-list digest for the current RHOAI_MCP_RHOAI_VERSION and update image_constants.py."""

import json
import re
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_CONSTANTS_FILE = _DIR / "constants.py"
_IMAGE_CONSTANTS_FILE = _DIR / "image_constants.py"

_VERSION_RE = re.compile(r'^RHOAI_MCP_RHOAI_VERSION:\s*str\s*=\s*["\'](.+?)["\']', re.MULTILINE)
_DIGEST_RE = re.compile(r'(RHOAI_MCP_RHOAI_DIGEST:\s*str\s*=\s*["\'])([^"\']+)(["\'])')


def _read_version() -> str:
    match = _VERSION_RE.search(_CONSTANTS_FILE.read_text())
    if not match:
        sys.exit("RHOAI_MCP_RHOAI_VERSION not found in constants.py")
    return match.group(1)


def _to_quay(image: str) -> str:
    return image.replace("registry.redhat.io", "quay.io", 1)


def _fetch_manifest_list_digest(image_ref: str) -> str:
    """Return the manifest-list digest for *image_ref* via ``skopeo inspect``.

    ``--override-arch/--override-os`` are required on non-linux hosts (e.g. macOS)
    but do not affect the returned Digest, which always refers to the top-level
    manifest list rather than a platform-specific manifest.
    """
    result = subprocess.run(
        [
            "skopeo",
            "inspect",
            "--no-tags",
            "--override-arch",
            "amd64",
            "--override-os",
            "linux",
            f"docker://{image_ref}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"skopeo inspect failed:\n{result.stderr}")
    return json.loads(result.stdout)["Digest"]


def main() -> None:
    version_image = _read_version()
    quay_image = _to_quay(version_image)
    print(f"Looking up digest for {quay_image} ...")

    digest = _fetch_manifest_list_digest(quay_image)
    repo = version_image.rsplit(":", 1)[0]
    new_ref = f"{repo}@{digest}"
    print(f"Resolved digest: {new_ref}")

    text = _IMAGE_CONSTANTS_FILE.read_text()
    updated, count = _DIGEST_RE.subn(rf"\g<1>{new_ref}\g<3>", text)
    if count == 0:
        sys.exit("RHOAI_MCP_RHOAI_DIGEST not found in image_constants.py")

    _IMAGE_CONSTANTS_FILE.write_text(updated)
    print(f"Updated {_IMAGE_CONSTANTS_FILE.name}")


if __name__ == "__main__":
    main()
