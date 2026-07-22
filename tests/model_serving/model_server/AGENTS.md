# Model Server Test Component

This directory contains integration tests for the **Serving Orchestration** (KServe) component of Red Hat OpenShift AI (RHOAI) and Open Data Hub (ODH). Tests validate InferenceService lifecycle, authentication, autoscaling, storage backends, observability, upgrade resilience, and LLM Deployment (LLMD) flows at the Kubernetes API level.

**Team**: Serving Orchestration (formerly Model Server / Model Serving)
**CODEOWNERS**: `@threcc @mwaykole`

## Tier Classification Rules

Every test function or test class should have a tier marker. Use the guide below to pick the right one. These are recommended defaults — use your judgment if a test doesn't fit neatly.

**Exceptions** (do NOT need tier markers):

- **GPU tests** — marked with `gpu` or `model_server_gpu`, run via the dedicated GPU Quality gate
- **Upgrade tests** — use `pre_upgrade` / `post_upgrade` markers instead

### Tier Definitions

- **smoke** — "Does it work at all?" Core flows that gate the build. If a smoke test fails, nothing else matters. Keep this set small.
- **tier1** — Core product functionality customers rely on. The main positive-path tests.
- **tier2** — Important but secondary functionality, or tests that depend on optional infrastructure (extra operators, GPU, multi-node).
- **tier3** — Negative paths, error handling, destructive scenarios, edge cases.

### Recommended Defaults by Area

| Area | Suggested Tier | Notes |
| --- | --- | --- |
| Authentication / auth flows | `smoke` or `tier1` | Basic auth → smoke; advanced auth scenarios → tier1 |
| Inference lifecycle (deploy, query, update, delete) | `tier1` | Core ISVC operations |
| Ingress / routing | `tier1` | Route visibility, reconciliation |
| Observability / metrics | `tier1` | Monitoring validation |
| Platform / DSC | `tier1` | Component health checks |
| Autoscaling (KEDA, Kueue) | `tier1` or `tier2` | Depends on how critical the flow is |
| Storage backends (S3, PVC, OCI) | `tier1` or `tier2` | Basic S3 → tier1; advanced PVC modes → tier2 |
| Canary rollout | `tier2` | Advanced deployment strategy |
| Model cache | `tier2` | LocalModelNamespaceCache flows |
| Multi-node / workerSpec | `tier2` | Requires special hardware |
| GPU (vLLM, NIM) | n/a | GPU tests use `gpu`/`model_server_gpu` marker; run via GPU Quality gate, not tier gating |
| Negative / error handling | `tier3` | Invalid input, auth rejection, overload |
| LLMD | `tier1` | LLM Deployment; smoke test → `smoke` |
| Upgrade | n/a | Uses `pre_upgrade` / `post_upgrade` markers instead of tier markers |

### Marker Placement

- Apply tier markers at the **class level** when all tests in the class share the same tier.
- Apply tier markers at the **function level** when tests within a class have different tiers (rare — prefer splitting into separate classes).
- Use `pytestmark = [pytest.mark.smoke]` at the **module level** only when the entire file is a single tier.
- Upgrade tests use `pre_upgrade` or `post_upgrade` markers instead of tier markers.

## Required Markers

Every test must have the following markers where applicable:

| Marker | When Required |
| --- | --- |
| `smoke` / `tier1` / `tier2` / `tier3` | **Always** — exactly one per test (except GPU and upgrade tests) |
| `pre_upgrade` / `post_upgrade` | Tests under `upgrade/` (replaces tier marker) |
| `rawdeployment` | Tests targeting KServe RawDeployment (Standard) mode |
| `gpu` / `model_server_gpu` | Tests requiring GPU nodes (replaces tier marker; runs via GPU Quality gate) |
| `multinode` | Tests requiring multiple nodes (workerSpec) |
| `slow` | Tests expected to take >10 minutes (used in model_cache, platform tests) |
| `llmd_cpu` / `llmd_gpu` | LLMD tests by resource requirement |
| `kueue` / `keda` | Tests depending on Kueue/KEDA operators |
| `minio` | Tests using MinIO storage |
| `tls` / `metrics` | Tests validating TLS or metrics specifically |

## Test Structure Standards

### Test Files

```python
import pytest

pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.tier1
@pytest.mark.parametrize(
    "unprivileged_model_namespace, some_fixture",
    [pytest.param({"name": "test-descriptive-namespace-name"}, {"model-dir": "test-dir"})],
    indirect=True,
)
class TestDescriptiveClassName:
    """One-line description of what this test class validates.

    Steps:
        1. Setup action
        2. Validation action
        3. Expected outcome
    """

    def test_specific_behavior(self, some_fixture):
        """Verify specific behavior under specific conditions."""
        # test body
```

### Rules

1. Every test class and standalone test function MUST have a docstring.
2. Class docstrings: describe what the class validates, list numbered steps.
3. Function docstrings: one line describing what is verified, or Given-When-Then for complex cases.
4. Use `pytest.mark.parametrize` instead of duplicating test logic.
5. Use `pytest.mark.dependency` when tests have ordering requirements.
6. Use bounded iteration — never `while True` or `time.sleep()`. Use `TimeoutSampler` or `wait_for_condition` with explicit timeouts.

### Fixture Conventions

1. Fixture names MUST be nouns: `storage_secret`, `inference_service`, `model_namespace` — not `create_secret`, `setup_namespace`.
2. Resource-creating fixtures MUST use context managers (`with Resource(...) as r: yield r`) to guarantee cleanup on failure.
3. `conftest.py` files contain ONLY fixtures — no helper functions, no constants, no classes. Helpers go in `utils.py`.
4. Use the narrowest scope: `function` > `class` > `module` > `session`.
5. Parameterized fixtures use `request.param` with dict structures.

### Kubernetes Resources

1. Use `openshift-python-wrapper` for all K8s API interactions.
2. Never use `subprocess`/`os.system` to call `oc`/`kubectl` directly — use the wrapper or `pyhelper_utils.shell.run_command`.
3. Resources MUST be managed via context managers to ensure cleanup.
4. Waits must use `TimeoutSampler` or resource `.wait_for_condition()` — never `time.sleep()`.

### Image References

1. All container images MUST be in a registered `image_constants.py` class.
2. All images MUST use `@sha256:` digest pinning — no mutable tags.
3. Prefer `quay.io` or `registry.redhat.io` over `docker.io`.

## Component Architecture Context

### KServe Deployment Modes

- **RawDeployment (Standard)**: Deployment + Route + HPA, no KNative. Annotation value: `serving.kserve.io/deploymentMode: Standard`. Auth via `kube-rbac-proxy` sidecar (annotation `security.opendatahub.io/enable-auth: "true"`). Bearer tokens validated against Kubernetes TokenReview API.
- **Serverless**: KNative Serving + Istio. Auth via Authorino external auth (AuthConfig CR + envoy sidecar).

### Key Resources

- `InferenceService` (ISVC): core serving primitive — model deployment, scaling, routing.
- `ServingRuntime`: runtime template (OVMS, Triton, vLLM, etc.).
- `LLMInferenceService` (LLMISVC): LLMD-specific CRD wrapping InferenceService with router, prefill, and scheduling config.
- `LocalModelNamespaceCache`: namespace-scoped model cache — pre-downloads model to node PVCs.
- `InferenceGraph`: DAG-based model routing (splitter, ensemble, sequence).

### Upgrade Tests

Upgrade tests use `@pytest.mark.pre_upgrade` and `@pytest.mark.post_upgrade` markers to indicate the test phase. CI runs use `--pre-upgrade` / `--post-upgrade` CLI flags to select which phase to execute. Upgrade tests do not require a separate tier marker.

1. **Pre-upgrade** (`@pytest.mark.pre_upgrade`): deploy resources, send inference, capture baseline to ConfigMap (`capture_upgrade_baseline` fixture).
2. **Operator upgrade happens externally** (Jenkins/CI).
3. **Post-upgrade** (`@pytest.mark.post_upgrade`): verify resources survived, compare against baseline, re-inference.

Baseline data goes in ConfigMaps. Bearer tokens go in Secrets (not ConfigMap data).

## Review Checklist

When reviewing PRs that touch this directory:

- [ ] Every non-exempt test has exactly one tier marker (smoke/tier1/tier2/tier3)
- [ ] Tier marker matches the subdirectory default from the classification table
- [ ] GPU tests use `gpu`/`model_server_gpu` marker (no tier marker needed)
- [ ] Every test has a docstring
- [ ] Fixtures use noun names and context managers
- [ ] No `time.sleep()` — uses `TimeoutSampler` or `.wait_for_condition()`
- [ ] No broad `except Exception:` — catches specific exceptions (CWE-396)
- [ ] Images use `@sha256:` digest pinning in `image_constants.py`
- [ ] Type annotations on all new code
- [ ] `openshift-python-wrapper` used for K8s API calls
- [ ] Upgrade tests have `pre_upgrade` or `post_upgrade` marker
