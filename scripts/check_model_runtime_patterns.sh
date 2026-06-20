#!/bin/bash
# TC-GUARD-001: Pre-commit anti-pattern checker for model runtime tests.
# Greps for known anti-patterns that should not appear in the codebase.

set -euo pipefail

ERRORS=0
TARGET_DIR="tests/model_serving/model_runtime"

if [ ! -d "$TARGET_DIR" ]; then
    exit 0
fi

# Anti-pattern 1: CHECK_SNAPSHOT environment variable (removed, should not return)
if grep -rn "CHECK_SNAPSHOT" "$TARGET_DIR" --include="*.py" 2>/dev/null; then
    echo "ERROR: CHECK_SNAPSHOT references found. Use fuzzy validation instead."
    ERRORS=$((ERRORS + 1))
fi

# Anti-pattern 2: gRPC protocol usage in Triton tests (not supported on RHOAI)
if grep -rn "Protocols\.GRPC\|protocol.*grpc\|TRITON_GRPC" "$TARGET_DIR/triton/" --include="*.py" 2>/dev/null; then
    echo "ERROR: gRPC references found in Triton tests. RHOAI does not support gRPC."
    ERRORS=$((ERRORS + 1))
fi

# Anti-pattern 3: Hardcoded deployment_type string instead of KServeDeploymentType enum
if grep -rn 'deployment_mode.*=.*"RawDeployment"\|deployment_mode.*=.*"Serverless"\|deployment_type.*=.*"raw"' "$TARGET_DIR" --include="*.py" 2>/dev/null | grep -v "constant.py" | grep -v "RAW_DEPLOYMENT_TYPE" | grep -v "utils.py"; then
    echo "ERROR: Hardcoded deployment mode string found. Use KServeDeploymentType enum."
    ERRORS=$((ERRORS + 1))
fi

# Anti-pattern 4: Exact snapshot comparison for LLM/non-deterministic output
if grep -rn "assert.*==.*response_snapshot\|== snapshot" "$TARGET_DIR/vllm/" --include="*.py" 2>/dev/null; then
    echo "ERROR: Exact snapshot comparison found in vLLM tests. Use fuzzy validation."
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "Found $ERRORS anti-pattern(s) in model runtime tests."
    exit 1
fi
