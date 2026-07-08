#!/usr/bin/env bash
# Shift-left teardown entrypoint: verify full operator health after a component stage.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run pytest -m operator_health tests/cluster_health -svv "$@"
