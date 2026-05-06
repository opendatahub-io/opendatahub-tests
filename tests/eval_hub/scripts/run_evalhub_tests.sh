#!/bin/bash
# Quick test runner for EvalHub Kueue integration tests

export OC_BINARY_PATH="/opt/homebrew/bin/oc"
export EVALHUB_BASE_URL="https://evalhub-prabhu.apps.rosa.prabhu-comhub.xqmp.p3.openshiftapps.com"
export EVALHUB_MODEL_URL="http://granite-llm-metrics.prabhu.svc.cluster.local:8080/v1"
export EVALHUB_NAMESPACE="prabhu"

cd /Users/nbs/work/ai-engineering/code/opendatahub-tests

echo "Running EvalHub Kueue integration tests..."
echo "=========================================="

# Run all tests with summary
uv run pytest tests/eval_hub/evalhub_kueue_integration/ -v -m kueue --tb=short -q

echo ""
echo "=========================================="
echo "Test run complete!"
