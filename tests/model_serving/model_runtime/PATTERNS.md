# Model Runtime Test Patterns

Reference guide for writing and maintaining tests under
`tests/model_serving/model_runtime/`.

## Inference Validation

### vLLM (LLM output)

Use `validate_text_inference_fuzzy()` from `model_runtime/utils.py`. Never
compare LLM output with exact snapshot assertions — output varies across GPU
types and runtime versions.

```python
validate_text_inference_fuzzy(
    completion_responses=wrapped_responses,
    queries=queries,
    model_info=model_info,
    require_keywords=False,
    allow_empty_responses=True,
    min_valid_responses=1,
)
```

### Triton / OpenVINO / MLServer (deterministic output)

Validate response structure (outputs list, data list, non-empty) without
comparing exact float values. Different hardware produces slightly different
numerical precision.

```python
assert response.get("outputs")
output = response["outputs"][0]
actual_data = output.get("data", [])
assert actual_data
assert isinstance(actual_data, list)
```

## Fixture Chain

Tests follow a strict fixture chain with `indirect=True` parametrization:

```
model_namespace → s3_models_storage_uri → serving_runtime → inference_service
```

All fixtures are `scope="class"`. The test class is decorated with
`@pytest.mark.parametrize((...), [...], indirect=True)`.

## Deployment Mode

Always use the `KServeDeploymentType` enum. Never use string literals
like `"RawDeployment"` or `"raw"` for deployment mode values.

```python
from utilities.constants import KServeDeploymentType

# Config dict key is "deployment_mode" (not "deployment_type")
BASE_RAW_DEPLOYMENT_CONFIG = {
    "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
    "min-replicas": 1,
}
```

Note: Some OpenVINO utility functions use `deployment_type` as a *dispatch
parameter* (accepting string `"raw"`) — this is an internal function parameter,
not a config dict key.

## External Route vs Port-Forward

Positive tests use `external_route=True` and `get_exposed_isvc_url(isvc)` to
reach the model server. Negative tests that intentionally break the ISVC may
need port-forwarding since the external route may not be available.

## Markers

| Marker | Use |
|---|---|
| `@pytest.mark.smoke` | Critical path tests |
| `@pytest.mark.tier1` | Core functionality |
| `@pytest.mark.tier2` | Extended coverage (probes, multi-model) |
| `@pytest.mark.tier3` | Negative, resilience, edge cases |
| `@pytest.mark.negative` | Tests that expect failures |
| `@pytest.mark.resilience` | Tests for recovery/degradation |
| `@pytest.mark.gpu` | Requires GPU (e.g., DALI) |
| `@pytest.mark.tested_verified` | Triton (not GA on RHOAI) |

## Protocol

REST only. gRPC is not supported on RHOAI. Do not add gRPC test parameters
or imports.

## Negative Test Pattern

For tests that expect deployment failures:

1. Create ISVC with `wait=False, wait_for_predictor_pods=False`
2. Use `TimeoutSampler` to poll ISVC status conditions
3. Assert on condition messages for descriptive errors

```python
from timeout_sampler import TimeoutSampler

for sample in TimeoutSampler(wait_timeout=300, sleep=10, func=get_isvc_condition_messages, isvc=isvc):
    if sample:
        messages_lower = " ".join(sample).lower()
        if any(keyword in messages_lower for keyword in ("error", "fail", ...)):
            return
```

## Probe Test Pattern

1. Create a `ServingRuntimeFromTemplate` with `containers=` parameter to inject
   readiness/liveness probe configs
2. Deploy the ISVC normally
3. Assert `pod_is_ready()`, `get_probe()` returns httpGet config
4. Run `exec_http_probe()` inside the pod via `pod.execute()` and assert HTTP 200
5. Verify no premature container restarts via `get_restart_counts()`

## Directory Structure

```
model_runtime/
├── vllm/
│   ├── s3/              # S3-backed vLLM tests
│   ├── probes/          # Probe health tests
│   ├── negative/        # Failure mode tests
│   └── resilience/      # Recovery/load tests
├── openvino/
│   ├── probes/          # OVMS probe tests
│   └── smoke/           # OVMS smoke scripts
├── mlserver/
│   ├── s3/              # S3-backed MLServer tests
│   ├── model_car/       # OCI model car tests
│   ├── probes/          # MLServer probe tests
│   └── negative/        # Failure mode tests
├── triton/
│   └── basic_model_deployment/  # REST-only Triton tests
└── image_validation/    # Runtime image validation
```
