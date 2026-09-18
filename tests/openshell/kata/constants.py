from os import environ
from pathlib import Path

KATA_PATCH_MANIFESTS_DIR = Path(__file__).resolve().parents[3] / "utilities" / "manifests" / "kata_patches"

OPENSHELL_RUN_KATA_TESTS = environ.get("OPENSHELL_RUN_KATA_TESTS", "false").lower() == "true"
