# NeMo Guardrails Tests

Comprehensive test suite for NVIDIA NeMo Guardrails integration in OpenDataHub/RHOAI.

## Overview

This test suite validates the operational aspects of NeMo Guardrails deployments managed by the TrustyAI Service Operator. Tests focus on infrastructure, deployment, configuration, and integration rather than guardrails functionality.

## Test Structure

### Test Files

- **test_nemo_guardrails.py** - Main test suite covering deployment, authentication, multi-tenancy, and secret mounting
- **test_config_update.py** - Configuration update and rollout tests
- **conftest.py** - Pytest fixtures for test resources
- **utils.py** - Helper functions for configuration generation and API requests
- **constants.py** - Test constants and configuration

## Test Classes

### TestNemoGuardrailsLLMAsJudge (smoke, tier1)

Tests basic NeMo Guardrails deployment with LLM-as-a-judge configuration and authentication enabled.

**Tests:**

1. `test_nemo_llm_judge_validate_images` - Verify container images match TrustyAI operator ConfigMap
2. `test_nemo_llm_judge_deployment` - Verify successful deployment and route creation
3. `test_nemo_llm_judge_backend_communication` (×2 endpoints) - Verify communication with backend LLM
4. `test_nemo_llm_judge_with_authentication` (×2 endpoints) - Verify kube-rbac-proxy authentication enforcement

**Parameterization:** Tests 3-4 run against both `/v1/chat/completions` and `/v1/guardrail/checks` endpoints (4 total test runs)

### TestNemoGuardrailsMultiServer (tier2)

Tests multiple independent NeMo Guardrails servers in the same namespace.

**Tests:**

1. `test_nemo_multi_server_isolation` (×2 endpoints) - Verify servers operate independently
2. `test_nemo_multi_server_different_configs` - Verify servers have distinct configurations and ConfigMaps

**Parameterization:** Test 1 runs against both endpoints (2 total test runs)

### TestNemoGuardrailsMultiConfig (tier2)

Tests single NeMo Guardrails server with multiple named configurations.

**Tests:**

1. `test_nemo_multi_config_default_selection` (×2 endpoints) - Verify default config is used when unspecified
2. `test_nemo_multi_config` (×2 endpoints) - Verify explicit config selection works correctly
3. `test_nemo_multi_config_mount_paths` - Verify configs are mounted to correct paths (/app/config/{name})

**Parameterization:** Tests 1-2 run against both endpoints (4 total test runs)

### TestNemoGuardrailsSecretMounting (tier1)

Tests mounting model API tokens as secrets via environment variables.

**Tests:**

1. `test_nemo_secret_env_var_mounted` - Verify secret is mounted as environment variable with secretKeyRef
2. `test_nemo_secret_used_for_auth` (×2 endpoints) - Verify mounted secret is used for backend model authentication

**Parameterization:** Test 2 runs against both endpoints (2 total test runs)

### TestNemoGuardrailsConfigUpdate (tier1)

Tests configuration updates and deployment rollouts.

**Tests:**

1. `test_nemo_config_update_triggers_rollout` - Verify updating NemoGuardrails CR triggers deployment rollout with updated ConfigMap

## Running Tests

### Run All NeMo Guardrails Tests

```bash
pytest tests/model_explainability/nemo_guardrails/ -v
```

### Run Specific Test Class

```bash
pytest tests/model_explainability/nemo_guardrails/test_nemo_guardrails.py::TestNemoGuardrailsLLMAsJudge -v
```

### Run Specific Test

```bash
pytest tests/model_explainability/nemo_guardrails/test_nemo_guardrails.py::TestNemoGuardrailsLLMAsJudge::test_nemo_llm_judge_deployment -v
```

### Run Config Update Tests

```bash
pytest tests/model_explainability/nemo_guardrails/test_config_update.py -v
```

### Run by Marker

```bash
# Smoke tests only
pytest tests/model_explainability/nemo_guardrails/ -m smoke -v

# Tier1 tests
pytest tests/model_explainability/nemo_guardrails/ -m tier1 -v

# Tier2 tests
pytest tests/model_explainability/nemo_guardrails/ -m tier2 -v
```

## Test Count Summary

- **Total test methods:** 15
- **Total test runs (after parameterization):** 31
  - Smoke: 6 runs (1 class, 4 methods)
  - Tier1: 4 runs (2 classes, 3 methods)
  - Tier2: 9 runs (2 classes, 3 methods)

## Prerequisites

### Cluster Requirements

- OpenShift cluster with OpenDataHub/RHOAI installed
- KServe configured in Raw Deployment mode
- TrustyAI Service Operator deployed
- llm-d-inference-sim-isvc (Qwen2.5-1.5B-Instruct) deployed

### Namespace

All tests use the `test-nemo-guardrails` namespace (created automatically by test fixtures).

## Key Components

### Custom Resource

- **NemoGuardrails** (`trustyai.opendatahub.io/v1alpha1`) - Managed by TrustyAI Service Operator

### Configurations Tested

1. **LLM-as-a-Judge** - Self-check patterns using main LLM to validate inputs/outputs
2. **Presidio** - PII detection for email, SSN, credit card, person names

### Authentication

- **Enabled:** Uses kube-rbac-proxy sidecar (annotation: `security.opendatahub.io/enable-auth: true`)
- **Disabled:** Direct access to NeMo Guardrails server (no annotation)

### Endpoints Tested

- `/v1/chat/completions` - OpenAI-compatible chat endpoint
- `/v1/guardrail/checks` - Guardrails validation endpoint

## Architecture

```markdown
┌─────────────────────────────────────────────────────────────┐
│  Test Namespace: test-nemo-guardrails                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐      ┌──────────────────────┐    │
│  │  NemoGuardrails CR   │      │  ConfigMap           │    │
│  │  ├─ nemoConfigs[]    │─────▶│  ├─ config.yaml      │    │
│  │  ├─ replicas         │      │  ├─ prompts.yaml     │    │
│  │  ├─ env[]            │      │  └─ rails.co         │    │
│  │  └─ annotations      │      └──────────────────────┘    │
│  └──────────────────────┘                                   │
│            │                                                 │
│            ▼                                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Deployment                                           │  │
│  │  ├─ nemo-guardrails container                        │  │
│  │  │  └─ volumeMounts: /app/config/{name}              │  │
│  │  └─ kube-rbac-proxy (if auth enabled)                │  │
│  └──────────────────────────────────────────────────────┘  │
│            │                                                 │
│            ▼                                                 │
│  ┌──────────────────────┐                                   │
│  │  Route (HTTPS)       │                                   │
│  └──────────────────────┘                                   │
│            │                                                 │
│            ▼                                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Backend LLM (llm-d-inference-sim-isvc)              │  │
│  │  Model: Qwen2.5-1.5B-Instruct                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Fixtures

### Resource Fixtures

- `nemo_guardrails_llm_judge` - LLM-as-a-judge with auth enabled
- `nemo_guardrails_presidio` - Presidio PII detection with auth disabled
- `nemo_guardrails_multi_config` - Single server with multiple configurations
- `nemo_guardrails_second_server` - Second independent server for multi-tenancy tests
- `nemo_guardrails_config_update` - Server for configuration update tests

### ConfigMap Fixtures

- `nemo_llm_judge_configmap` - LLM-as-a-judge configuration
- `nemo_presidio_configmap` - Presidio PII detection configuration
- `nemo_multi_config_a` - First config for multi-config tests
- `nemo_multi_config_b` - Second config for multi-config tests
- `nemo_config_update_configmap` - ConfigMap for update tests

### Route Fixtures

- `nemo_guardrails_llm_judge_route`
- `nemo_guardrails_presidio_route`
- `nemo_guardrails_multi_config_route`
- `nemo_guardrails_second_server_route`
- `nemo_guardrails_config_update_route`

### Healthcheck Fixtures

All healthcheck fixtures use `wait_for_nemo_guardrails_health()` to verify the service is responsive before tests run.

### Helper Fixtures

- `openshift_ca_bundle_file` - CA bundle for HTTPS verification
- `nemo_api_token_secret` - Secret for model API authentication

## Configuration Examples

### LLM-as-a-Judge

```yaml
models:
  - type: main
    engine: openai
    parameters:
      openai_api_base: http://llm-d-inference-sim-isvc-predictor.test-nemo-guardrails.svc.cluster.local/v1
      model_name: Qwen2.5-1.5B-Instruct

rails:
  input:
    flows: ["self check input"]
  output:
    flows: ["self check output"]
```

### Presidio PII Detection

```yaml
models:
  - type: main
    engine: openai
    parameters:
      openai_api_base: http://llm-d-inference-sim-isvc-predictor.test-nemo-guardrails.svc.cluster.local/v1
      model_name: Qwen2.5-1.5B-Instruct

rails:
  config:
    sensitive_data_detection:
      input:
        entities: ["EMAIL_ADDRESS", "US_SSN", "CREDIT_CARD"]
      output:
        entities: ["PERSON", "EMAIL_ADDRESS"]
  input:
    flows: ["detect sensitive data on input"]
  output:
    flows: ["detect sensitive data on output"]
```

## Notes

- Tests are **operational**, not functional - they verify deployment/infrastructure, not guardrails logic
- Endpoint parameterization ensures both chat and check endpoints work correctly
- Health checks use retry logic with 300s timeout to handle slow startup
- ConfigMap updates trigger automatic deployment rollouts via operator watch
- Authentication tests verify kube-rbac-proxy integration, not guardrails authorization
