# EvalHub Test Scripts

Helper scripts for running and managing EvalHub integration tests.

## Scripts

### verify_evalhub_setup.py

Verifies EvalHub and Kueue setup before running tests. **Run this first** to ensure the environment is properly configured.

```bash
uv run python tests/eval_hub/scripts/verify_evalhub_setup.py
```

**What it checks:**

- OpenShift authentication (oc token)
- EvalHub API connectivity and health endpoint
- EvalHub API responsiveness (404 for nonexistent job)
- Kueue CRDs installation

**Example output:**

```text
EvalHub Setup Verification
==================================================
✓ Test 1: Getting OpenShift token...
✓ Test 2: Testing EvalHub API health...
✓ Test 3: Testing EvalHub API (nonexistent job = 404)...
✓ Test 4: Checking Kueue CRDs...
==================================================
✓ All verification tests passed!
```

### cleanup_kueue_resources.sh

Cleans up leftover Kueue resources from previous test runs. **Run this before executing tests** to ensure a clean state.

```bash
./tests/eval_hub/scripts/cleanup_kueue_resources.sh
```

**What it does:**

- Deletes all evalhub test namespaces
- Removes evalhub ClusterQueues (evalhub-test-cq, team-a-cq, team-b-cq)
- Removes evalhub ResourceFlavors (evalhub-test-flavor, evalhub-multi-test-flavor)
- Removes WorkloadPriorityClass (evalhub-test-high-priority)

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
# 1. Verify environment setup
uv run python tests/eval_hub/scripts/verify_evalhub_setup.py

# 2. Clean up leftover resources
./tests/eval_hub/scripts/cleanup_kueue_resources.sh

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
