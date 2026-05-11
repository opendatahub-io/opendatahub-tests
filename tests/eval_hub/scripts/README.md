# EvalHub Test Scripts

Helper scripts for running and managing EvalHub integration tests.

## Environment verification (pytest)

Preflight checks (EvalHub HTTP + Kueue) run automatically via the session fixture
``evalhub_preflight_verified`` when you execute tests under ``tests/eval_hub/``.

To run **only** those checks without the full suite:

```bash
uv run pytest tests/eval_hub/test_evalhub_preflight.py -q
```

Implementation: ``verify_evalhub_preflight`` in ``tests/eval_hub/evalhub_kueue_integration/utils.py``.

## Scripts

### cleanup_kueue_resources.sh

**Opt-in only.** By default this script **does not delete any cluster resources** (safe on shared clusters). Normal runs rely on fixture/context teardown.

To delete the legacy hard-coded evalhub test object names after you deliberately choose to:

```bash
export KUEUE_EVALHUB_FORCE_CLEANUP=1
./tests/eval_hub/scripts/cleanup_kueue_resources.sh
```

**When `KUEUE_EVALHUB_FORCE_CLEANUP=1`:**

- Removes evalhub ClusterQueues (evalhub-test-cq, team-a-cq, team-b-cq)
- Removes evalhub ResourceFlavors (evalhub-test-flavor, evalhub-multi-test-flavor)
- Removes WorkloadPriorityClass (evalhub-test-high-priority)

Does **not** delete namespaces; test fixtures manage namespace lifecycle.

### run_evalhub_tests.sh

Runs the complete EvalHub Kueue integration test suite with proper environment configuration.

```bash
./tests/eval_hub/scripts/run_evalhub_tests.sh
```

**What it does:**

- Sets required environment variables (OC_BINARY_PATH, EVALHUB_BASE_URL, etc.)
- Runs pytest with kueue marker
- Displays test summary

**Environment Variables:**

- `OC_BINARY_PATH`: Path to oc CLI binary
- `EVALHUB_BASE_URL`: EvalHub API endpoint
- `EVALHUB_MODEL_URL`: LLM model endpoint for evaluation
- `EVALHUB_NAMESPACE`: Namespace where EvalHub is deployed

## Recommended Workflow

```bash
# 1. Verify cluster + EvalHub + Kueue (preflight only)
uv run pytest tests/eval_hub/test_evalhub_preflight.py -q

# 2. (Optional) Force-delete legacy evalhub names after a crash — see cleanup_kueue_resources.sh

# 3. Run tests
./tests/eval_hub/scripts/run_evalhub_tests.sh
```

## Manual Test Execution

If you need to run tests manually with custom configuration:

```bash
export OC_BINARY_PATH="/path/to/oc"
export EVALHUB_BASE_URL="https://your-evalhub-url"
export EVALHUB_MODEL_URL="http://your-model-url"
export EVALHUB_NAMESPACE="your-namespace"

uv run pytest tests/eval_hub/evalhub_kueue_integration/ -v -m kueue
```
