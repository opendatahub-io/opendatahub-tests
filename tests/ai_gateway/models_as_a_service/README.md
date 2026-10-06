# MaaS Billing Tests

This directory contains tests for MaaS (Model as a Service) in OpenDataHub/RHOAI. Tests cover API key management, subscriptions, OIDC authentication, external models, multitenancy, component health, and operator upgrade scenarios.

## Directory Structure

```text
models_as_a_service/
├── conftest.py                    # Root-level fixtures (gateway, tenant, model refs)
├── utils.py                       # Shared utilities
│
├── component_health/              # Health checks for MaaS components
│
├── external_model/                # External model discovery, egress, and auth tests
│
├── maas_api_key/                  # API key lifecycle and authorization tests
│
├── maas_cleanup/                  # Operator disable and cleanup tests
│
├── maas_subscription/             # Subscription enforcement and access control tests
│
├── multitenancy/                  # AITenant multitenancy tests
│   ├── conftest.py                # Shared AITenant bootstrap fixtures
│   ├── utils.py                   # Per-tenant maas-api verification helpers
│   ├── aitenant/                  # AITenant bootstrap and cleanup (scenario fixtures)
│   ├── isolation/                 # Tenant-scoped API key auth isolation
│   └── maas_api/                  # Per-tenant maas-api deployment and routing
│
├── oidc_tests/                    # OIDC authentication flow tests
│
├── upgrade/                       # Pre/post-upgrade tests
│
├── test_maas_endpoints.py         # /v1/models, /v1/chat/completions endpoints
├── test_maas_rbac_e2e.py          # Multi-tier user access control
├── test_maas_request_rate_limits.py
├── test_maas_token_rate_limits.py
└── test_maas_token_revoke.py
```

### Current Test Suites

- **`component_health/`** - Health checks for MaaS controller, API, and Tenant CR
- **`external_model/`** - External model discovery, egress routing, authentication, and cleanup
- **`maas_api_key/`** - API key CRUD, authorization, expiration, bulk operations, gateway rejection, and negative tests
- **`maas_cleanup/`** - Validates that disabling MaaS in DSC cleans up operator-managed resources
- **`maas_subscription/`** - Subscription enforcement, access control, filtering, rate limit exemptions, cascade deletion, multi-subscription and multi-auth-policy scenarios
- **`multitenancy/`** - AITenant bootstrap, per-tenant maas-api deployment/routing, auth isolation, and cross-gateway inference
- **`oidc_tests/`** - OIDC token flow, model access, multi-user, and header injection tests
- **`upgrade/`** - Pre/post-upgrade tests verifying that existing MaaS functionality, configuration, and customer workflows remain valid across platform/operator upgrades.
- **`test_maas_endpoints.py`** - Core MaaS API endpoint validation
- **`test_maas_rbac_e2e.py`** - End-to-end RBAC validation across user tiers
- **`test_maas_*_rate_limits.py`** - Request and token-based rate limiting
- **`test_maas_token_revoke.py`** - Token revocation behavior

## Test Markers

```python
@pytest.mark.smoke                 # Critical smoke tests
@pytest.mark.tier1                 # Tier 1 tests
@pytest.mark.tier2                 # Tier 2 tests
@pytest.mark.tier3                 # Tier 3 tests, includes negative tests
@pytest.mark.component_health      # Component health checks
@pytest.mark.pre_upgrade           # Pre-upgrade tests
@pytest.mark.post_upgrade          # Post-upgrade tests
```

## Running Tests

### Run All MaaS Tests

```bash
uv run pytest tests/ai_gateway/models_as_a_service/
```

### Run Tests by Component

```bash
# Run API key tests
uv run pytest tests/ai_gateway/models_as_a_service/maas_api_key/

# Run subscription tests
uv run pytest tests/ai_gateway/models_as_a_service/maas_subscription/

# Run OIDC tests
uv run pytest tests/ai_gateway/models_as_a_service/oidc_tests/

# Run component health tests
uv run pytest tests/ai_gateway/models_as_a_service/component_health/

# Run upgrade tests
uv run pytest tests/ai_gateway/models_as_a_service/upgrade/
```

### Run Tests with Markers

```bash
# Run smoke tests
uv run pytest -m smoke tests/ai_gateway/models_as_a_service/

# Run tier_1 tests
uv run pytest -m tier1 tests/ai_gateway/models_as_a_service/
```

## Upgrade Testing

Run pre-upgrade tests from the `opendatahub-tests` branch matching the source RHOAI version (for example, 3.5) to prepare resources and save a baseline. After upgrading the cluster, run post-upgrade tests from the branch matching the target version to validate the existing resources against that baseline.

### Running Upgrade Tests

```bash
# Run pre-upgrade tests, keeping resources for post-upgrade checks
uv run pytest tests/ai_gateway/models_as_a_service/ --pre-upgrade

# Run pre-upgrade tests with teardown for a standalone setup check
uv run pytest tests/ai_gateway/models_as_a_service/ --pre-upgrade --delete-pre-upgrade-resources

# Run post-upgrade tests
uv run pytest tests/ai_gateway/models_as_a_service/ --post-upgrade
```

### Upgrade Test Modules

Coverage reflects the tests on this branch; individual checks are documented in the test modules.

- [test_maas_upgrade.py](upgrade/test_maas_upgrade.py) — Checks MaaS resource survival, subscription spec preservation, component health, gateway reachability, and creation of new model references after upgrade.
- [test_inference_with_llmd.py](upgrade/test_inference_with_llmd.py) — Checks that an llm-d workload, its routing and MaaS configuration survive the upgrade, and that inference still succeeds with the existing API key.
- [test_external_model_legacy_migration.py](upgrade/test_external_model_legacy_migration.py) — Checks ExternalModel migration, removal of legacy networking, and preservation of model references, auth policies, and subscriptions.

## Additional Resources

- [MaaS Documentation](https://opendatahub-io.github.io/models-as-a-service/)
