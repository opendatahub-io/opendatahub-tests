# Model Registry Tests

This directory contains tests for OpenDataHub's Model Registry and Model Catalog components. These tests validate model registration, versioning, metadata management, catalog discovery, RBAC, and multi-backend database support.

## Directory Structure

```text
model_registry/
├── conftest.py                    # Root-level fixtures
├── constants.py                   # Shared constants
├── utils.py                       # Shared utilities
├── test_security.py               # Security tests (unauthenticated access)
│
├── component_health/              # Health checks for MR components
│
├── image_validation/              # Container image validation
│
├── mcp_servers/                   # MCP (Model Context Protocol) server tests
│   ├── config/                    # MCP configuration tests
│   └── search/                    # MCP search and filtering
│
├── model_catalog/                 # Model Catalog tests
│   ├── catalog_config/            # Catalog configuration management
│   ├── db_check/                  # Database validation
│   ├── huggingface/               # HuggingFace model integration
│   ├── metadata/                  # Metadata endpoint tests
│   ├── rbac/                      # Catalog RBAC tests
│   ├── search/                    # Search functionality
│   ├── sorting/                   # Sorting functionality
│   └── upgrade/                   # Catalog upgrade tests
│
├── model_registry/                # Core Model Registry tests
│   ├── async_job/                 # Asynchronous model upload tests
│   ├── negative_tests/            # Error scenario validation
│   ├── python_client/             # Python client tests
│   │   └── signing/               # Model signing tests
│   ├── rbac/                      # User/group permission tests
│   ├── rest_api/                  # REST API operation tests
│   └── upgrade/                   # Registry upgrade tests
│
├── scc/                           # Security Context Constraints validation
│
└── upgrade/                       # Combined upgrade scenarios
    ├── model_catalog/
    └── model_registry/
```

### Current Test Suites

- **`component_health/`** - Basic health checks for Model Registry components, namespace existence, and DSC status
- **`image_validation/`** - Container image validation (SHA256 digests, registry sources)
- **`mcp_servers/`** - MCP server integration tests covering data loading, filtering, keyword search, named queries, and multi-source support
- **`model_catalog/`** - Model Catalog tests including HuggingFace model integration, custom/default catalog sources, source merging, inclusion/exclusion filtering, search, sorting, metadata endpoints, and lifecycle management
- **`model_registry/`** - Core registry tests for model registration and versioning via REST API and Python client, RBAC with users/groups/ServiceAccounts, async model uploads, model signing, negative testing, and deployment with InferenceService
- **`scc/`** - Security Context Constraints validation for registry and catalog pods
- **`test_security.py`** - Validates unauthenticated access is denied (401 responses)

## Test Markers

```python
@pytest.mark.smoke                 # Critical smoke tests
@pytest.mark.tier1                 # Tier 1 tests
@pytest.mark.tier2                 # Tier 2 tests
@pytest.mark.tier3                 # Tier 3 tests, includes negative tests
@pytest.mark.custom_namespace      # Custom namespace tests
@pytest.mark.component_health      # Component health checks
@pytest.mark.skip_on_disconnected  # Requires internet connectivity
@pytest.mark.skip_must_gather      # Skip must-gather collection
@pytest.mark.pre_upgrade           # Pre-upgrade tests
@pytest.mark.post_upgrade          # Post-upgrade tests
```

## Database Backends

Tests are parametrized across multiple database backends:

| Backend    | Description               |
| ---------- | ------------------------- |
| MySQL      | MySQL database            |
| PostgreSQL | PostgreSQL database       |
| MariaDB    | MariaDB database          |
| Default    | Default embedded database |

## Running Tests

### Run All Model Registry Tests

```bash
uv run pytest tests/ai_hub/
```

### Run Tests by Component

```bash
# Run core registry tests
uv run pytest tests/ai_hub/model_registry/

# Run catalog tests
uv run pytest tests/ai_hub/model_catalog/

# Run MCP server tests
uv run pytest tests/ai_hub/mcp_servers/

# Run HuggingFace integration tests
uv run pytest tests/ai_hub/model_catalog/huggingface/
```

### Run Tests with Markers

```bash
# Run smoke tests
uv run pytest -m smoke tests/ai_hub/

# Run RBAC tests
uv run pytest tests/ai_hub/model_registry/rbac/

# Run upgrade tests
uv run pytest -m pre_upgrade tests/ai_hub/
```

## Upgrade Testing

### Running Upgrade Tests

```bash
# Pre-upgrade: patches ConfigMaps, registers models
uv run pytest tests/model_registry/ --pre-upgrade -v

# ... perform the actual cluster upgrade ...

# Post-upgrade: validates persistence, then cleans up
uv run pytest tests/model_registry/ --post-upgrade -v
```

### Test Execution Flow

- **Pre-upgrade** patches ConfigMaps with custom sources and registers models. Resources are left in place for post-upgrade validation.
- **Post-upgrade** validates that custom sources, models, and MCP servers persisted across the upgrade, then cleans up.
- Fixtures like `mcp_servers_configmap_patch` and `updated_catalog_config_map` become no-ops during upgrade runs to avoid overwriting the state set up by the pre-upgrade session.

### Coverage Matrix

| Component | Pre-upgrade | Post-upgrade | Upgrade Paths |
| --- | --- | --- | --- |
| Model Registry (MySQL) | Register model | Retrieve model, validate spec | All (via release branches) |
| Model Registry (PostgreSQL) | Register model | Retrieve model | All |
| Model Catalog ConfigMap | Patch custom source | Validate persistence | All |
| Model Catalog API | Custom catalog tests (reused) | Custom catalog tests (reused) | All |
| MCP Catalog ConfigMap | Patch custom source | Validate persistence | All |
| MCP Catalog API | Default server tests | Default + custom server tests | All |
| MCP Default Sources | ConfigMap entries, labels, API | Same | All |

### Known Limitations

No known limitations.

### Maintenance Ownership

- **Who updates tests when APIs change:** AI Hub QE
- **When to update constants:** When default MCP catalogs/labels change upstream (new partner servers, label renames)
- **When to update upgrade fixtures:** When new ConfigMaps are introduced (e.g., new catalog source types)

## Additional Resources

- [Model Registry Documentation](https://github.com/kubeflow/model-registry)
