# MaaS Billing Tests

This directory contains tests for MaaS (Model as a Service) in OpenDataHub/RHOAI. Tests cover API key management, subscriptions, OIDC authentication, external models, multitenancy, component health, and operator upgrade scenarios.

## Directory Structure

```text
maas_billing/
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
- **`upgrade/`** - Pre/post-upgrade tests validating MaaS control plane survival across operator upgrades
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
uv run pytest tests/model_serving/maas_billing/
```

### Run Tests by Component

```bash
# Run API key tests
uv run pytest tests/model_serving/maas_billing/maas_api_key/

# Run subscription tests
uv run pytest tests/model_serving/maas_billing/maas_subscription/

# Run OIDC tests
uv run pytest tests/model_serving/maas_billing/oidc_tests/

# Run component health tests
uv run pytest tests/model_serving/maas_billing/component_health/

# Run upgrade tests
uv run pytest tests/model_serving/maas_billing/upgrade/
```

### Run Tests with Markers

```bash
# Run smoke tests
uv run pytest -m smoke tests/model_serving/maas_billing/

# Run pre-upgrade tests
uv run pytest -m pre_upgrade tests/model_serving/maas_billing/

# Run post-upgrade tests
uv run pytest -m post_upgrade tests/model_serving/maas_billing/
```

## Upgrade Testing

### Running Upgrade Tests

```bash
# Pre-upgrade: deploy control plane resources and capture state snapshot
uv run pytest tests/model_serving/maas_billing/upgrade/test_maas_upgrade.py \
  --pre-upgrade -v

# ... perform the actual cluster upgrade ...

# Post-upgrade: validate CR survival, deployments, and API compatibility
uv run pytest tests/model_serving/maas_billing/upgrade/test_maas_upgrade.py \
  --post-upgrade -v
```

### Test Execution Flow

- **Pre-upgrade** deploys MaaS control plane resources (MaaSModelRef, MaaSAuthPolicy, MaaSSubscription) and captures a state snapshot into a ConfigMap. Resources are left in place for post-upgrade validation.
- **Post-upgrade** loads the snapshot and validates that all CRs, deployments, and CRDs survived the upgrade, then cleans up.

### Coverage Matrix

| Component | Pre-upgrade | Post-upgrade | Upgrade Paths |
| --- | --- | --- | --- |
| MaaS Gateway | Verify Programmed | Verify still Programmed | 3.4 → 3.5 |
| MaasTenantConfig CR | Verify Ready | Verify survives | 3.4 → 3.5 |
| MaaSModelRef | Create and verify | Verify survives | 3.4 → 3.5 |
| MaaSAuthPolicy | Create and verify | Verify survives | 3.4 → 3.5 |
| MaaSSubscription | Create and verify | Verify survives, spec not mutated | 3.4 → 3.5 |
| MaaS Deployments | — | Verify Available | 3.4 → 3.5 |
| MaaS CRDs | — | Verify all present | 3.4 → 3.5 |
| AIGateway CR | Verify absent | Verify bootstrapped | 3.4 → 3.5 |
| MaaS Config CR | Verify absent | Verify bootstrapped | 3.4 → 3.5 |
| Gateway probe | — | Verify reachable | 3.4 → 3.5 |
| API compatibility | — | Create new MaaSModelRef | 3.4 → 3.5 |

### Known Limitations

No known limitations.

### Maintenance Ownership

- **Who updates tests when APIs change:** MaaS QE
- **When to update fixtures:** When new MaaS CRs are introduced or existing CR specs change between versions
- **When to update upgrade assertions:** When new operator-bootstrapped resources are added in a new release
- **When to update baseline snapshot:** When new fields need to be tracked for mutation detection across upgrade

## Additional Resources

- [MaaS Documentation](https://opendatahub-io.github.io/models-as-a-service/)
