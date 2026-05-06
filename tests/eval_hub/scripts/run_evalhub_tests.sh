#!/bin/bash
set -euo pipefail

# Quick test runner for EvalHub Kueue integration tests.
#
# Required environment variables (set directly or via repo-root .env):
#   EVALHUB_BASE_URL, EVALHUB_MODEL_URL, EVALHUB_NAMESPACE
#
# Optional:
#   OC_BINARY_PATH — path to oc CLI (defaults to oc found on PATH)
#
# If "${REPO_ROOT}/.env" exists, it is sourced before validation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}" || {
  echo "error: failed to cd to repository root: ${REPO_ROOT}" >&2
  exit 1
}

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

if [[ -z "${OC_BINARY_PATH:-}" ]]; then
  if command -v oc >/dev/null 2>&1; then
    OC_BINARY_PATH="$(command -v oc)"
  else
    echo "error: OC_BINARY_PATH is unset and oc was not found in PATH" >&2
    exit 1
  fi
fi
export OC_BINARY_PATH

: "${EVALHUB_BASE_URL:?error: EVALHUB_BASE_URL must be set (export or add to ${REPO_ROOT}/.env)}"
: "${EVALHUB_MODEL_URL:?error: EVALHUB_MODEL_URL must be set (export or add to ${REPO_ROOT}/.env)}"
: "${EVALHUB_NAMESPACE:?error: EVALHUB_NAMESPACE must be set (export or add to ${REPO_ROOT}/.env)}"
export EVALHUB_BASE_URL EVALHUB_MODEL_URL EVALHUB_NAMESPACE

echo "Running EvalHub Kueue integration tests..."
echo "=========================================="

# Run all tests with summary
uv run pytest tests/eval_hub/evalhub_kueue_integration/ -v -m kueue --tb=short -q

echo ""
echo "=========================================="
echo "Test run complete!"
