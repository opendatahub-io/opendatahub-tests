# EvalHub Kueue Integration

Integration of Kueue with EvalHub to enable production-ready LLM evaluation job management with fair resource sharing, priority-based scheduling, and automatic queueing.

## Links

- **Strategy**: [RHOAIENG-59092](https://issues.redhat.com/browse/RHOAIENG-59092)
- **Test Plan**: [TestPlan.md](TestPlan.md)

## Overview

This feature enables Kueue job queueing for EvalHub evaluation jobs, providing:

- Fair resource sharing across multiple teams with guaranteed quotas
- Priority-based job scheduling for critical production evaluations
- Automatic job queueing when resources are unavailable
- Resource quota enforcement to prevent cluster overload

## Test Cases

**28 test cases** generated across 10 categories: P0: 13 | P1: 11 | P2: 4

See [test_cases/INDEX.md](test_cases/INDEX.md) for the complete test case index.

## Automated Test Implementation

Automated tests for this feature are implemented in the component repository following the test plan specifications. These tests validate:

- API integration for job submission with queue specifications
- Job lifecycle management (queueing, admission, execution, completion)
- Resource quota enforcement and multi-tenancy isolation
- Preemption scenarios and job restart behavior
- Status reporting via EvalHub API and Kubernetes Workload resources

## Running Tests

### Quick Start

```bash
# 1. Verify environment setup
uv run python ../scripts/verify_evalhub_setup.py

# 2. (Optional) cleanup_kueue_resources.sh — does nothing unless KUEUE_EVALHUB_FORCE_CLEANUP=1
../scripts/cleanup_kueue_resources.sh

# 3. Run tests
../scripts/run_evalhub_tests.sh
```

### Manual Execution

**Authentication:** These tests do **not** read `EVALHUB_API_TOKEN`, `EVALHUB_AUTH_HEADER`, or similar. The shared [`current_client_token`](../../../tests/conftest.py) fixture supplies an OpenShift OAuth token from your kube context (`get_openshift_token`); EvalHub HTTP helpers send `Authorization: Bearer <token>` and, for job endpoints, `X-Tenant` (see [`utils.py`](utils.py)). Use a user who can reach the EvalHub route and APIs—otherwise you will see **401** (unauthenticated) or **403** (forbidden). Cluster login and kubeconfig are covered in [Getting started — Tests cluster](../../../docs/GETTING_STARTED.md#tests-cluster).

```bash
export KUBECONFIG="/path/to/kubeconfig"   # optional; defaults to ~/.kube/config
# oc login …  # ensure context matches the cluster where EvalHub runs (see Getting started)

export OC_BINARY_PATH="/path/to/oc"
export EVALHUB_BASE_URL="https://evalhub-url"
export EVALHUB_MODEL_URL="http://model-url"
export EVALHUB_NAMESPACE="namespace"

uv run pytest tests/eval_hub/evalhub_kueue_integration/ -v -m kueue
```

### Helper Scripts

Located in `tests/eval_hub/scripts/`:

- **verify_evalhub_setup.py**: Verifies EvalHub API connectivity and Kueue installation
- **cleanup_kueue_resources.sh**: Opt-in only (`KUEUE_EVALHUB_FORCE_CLEANUP=1`); deletes fixed legacy evalhub cluster-scoped names if you need manual cleanup after an abnormal run
- **run_evalhub_tests.sh**: Runs the full test suite with proper environment configuration

See [scripts/README.md](../scripts/README.md) for detailed documentation.

## Test Environment Setup

Each test scenario requires specific cluster state configuration including:

- Kueue operator installation and configuration
- ClusterQueue resources with specific resource quotas
- LocalQueue resources mapped to ClusterQueues
- Namespace labels for multi-tenancy
- Resource allocation (CPU, memory, GPU) for quota testing

Tests are designed with setup/teardown procedures to establish the required cluster state before execution and revert to the original state after completion.

## Test Implementation Notes

See [FIXES_APPLIED.md](FIXES_APPLIED.md) for detailed information about:

- Test fixes and improvements
- Test philosophy (validating Kueue behavior vs job execution)
- Common issues and solutions
- Resource naming conventions
