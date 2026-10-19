import sys
from pathlib import Path

# Standalone test directory for utilities.upgrade_utils: deliberately outside
# tests/ so the repo-root conftest (cluster-dependent session fixtures) never
# loads. These tests run against a kind cluster in CI
# (.github/workflows/upgrade-utils-kind.yml).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
