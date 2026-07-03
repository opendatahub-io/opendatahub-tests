#!/usr/bin/env bash
# Shift-left teardown entrypoint: verify DSC/DSCI and operator namespaces after a component stage.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run pytest -m dsc_health tests/cluster_health -svv "$@"
