# RHOAI MCP Server Tests

End-to-end tests for the RHOAI MCP server deployed as a standalone workload.

## Directory structure

```text
tests/rhoai_mcp/
├── conftest.py          # Deployment fixtures (namespace, RBAC, configmap, deployment, service, route, health)
├── constants.py         # Resource names, port, namespace
├── image_constants.py   # Container image constant
├── utils.py             # Helper functions (deployment template, health probing)
├── test_deployment.py   # Smoke and E2E tests
└── README.md
```

## Markers

- `smoke` — deployment readiness and health checks, authentication tests, endpoint tests

No component marker is needed. Jenkins uses `tests/<component_name>` as the entry point
already. Unless there will be tests that would be part of another component, there is otherwise
no need for a component marker.

## Running

```bash
# Collect without running (verify structure)
uv run pytest tests/rhoai_mcp/ --collect-only

# Run smoke tests
uv run pytest tests/rhoai_mcp/ -m smoke -v

# Run all rhoai-mcp tests
uv run pytest tests/rhoai_mcp/ -v
```

## Development

```bash
OC_BINARY_PATH=$(which oc) uv run pytest tests/rhoai_mcp/test_deployment.py -s -v --cluster-sanity-skip-check
```
