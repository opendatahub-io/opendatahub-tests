# RHOAI MCP Server Tests

End-to-end tests for the RHOAI MCP server deployed as a standalone workload.

## Directory structure

```text
tests/rhoai_mcp/
├── conftest.py          # Deployment fixtures (namespace, RBAC, configmap, deployment, service, route, health)
├── constants.py         # Resource names, port, namespace
├── image_constants.py   # Container image constant
├── test_deployment.py   # Smoke and E2E tests
└── README.md
```

## Markers

- `rhoai_mcp` — component marker for all rhoai-mcp tests
- `smoke` — deployment readiness and health checks
- `tier2` — authentication tests, endpoint tests

## Running

```bash
# Collect without running (verify structure)
uv run pytest tests/rhoai_mcp/ --collect-only

# Run smoke tests
uv run pytest tests/rhoai_mcp/ -m smoke -v

# Run all rhoai-mcp tests
uv run pytest tests/rhoai_mcp/ -v

# Run with partially ready DSC
uv run pytest tests/rhoai_mcp/ -m rhoai_mcp -v --cluster-sanity-skip-check
```
