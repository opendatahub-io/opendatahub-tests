# Overview

This is a testing repo for OpenDataHub and OpenShift AI, which are MLOps platforms for OpenShift.
The tests are high-level integration tests at the Kubernetes API level.

You are an expert QE engineer writing maintainable pytest tests that other engineers can understand without deep domain knowledge.

## Commands

### Validation (run before committing)

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Run tox (CI validation)
tox
```

### Test Execution

```bash
# Collect tests without running (verify structure)
uv run pytest --collect-only

# Run specific marker
uv run pytest -m smoke
uv run pytest -m "model_serving and tier1"

# Run with setup plan (debug fixtures)
uv run pytest --setup-plan tests/model_serving/
```

## Project Structure

```text
tests/                    # Test modules by component
├── conftest.py           # All shared fixtures
├── <component>/          # Component test directories
│   ├── conftest.py       # Component-scoped fixtures
│   └── test_*.py         # Test files
|   └── utils.py          # Component-specific utility functions
utilities/                # Shared utility functions
└── <topic>_utils.py      # Topic-specific utility functions
```

## Essential Patterns

### Tests

- Every test MUST have a docstring explaining what it tests (see `tests/cluster_health/test_cluster_health.py`)
- Apply relevant markers from `pytest.ini`: tier (`smoke`, `sanity`, `tier1`, `tier2`, `tier3`), and infrastructure (`gpu`, `parallel`, `slow`). Use component markers (`model_explainability`, `llama_stack`, `rag`) as needed for cross-directory ownership (e.g., `tests/llama_stack`) — see `pytest.ini` for the full list
- Use Given-When-Then format in docstrings for behavioral clarity

### Fixtures

- Fixture names MUST be nouns: `storage_secret` not `create_secret`
- Use context managers for resource lifecycle (see `cluster_monitoring_config` fixture in `tests/conftest.py` for pattern)
- Fixtures do one thing only—compose them rather than nesting
- Use narrowest scope that meets the need: function > class > module > session

### Kubernetes Resources

- Use [openshift-python-wrapper](https://github.com/RedHatQE/openshift-python-wrapper) for all K8s API calls
- Resource lifecycle MUST use context managers to ensure cleanup
- Use `oc` CLI only when wrapper is not relevant (e.g., must-gather)

## Common Pitfalls

- **ERROR vs FAILED**: Pytest reports fixture failures as ERROR, test failures as FAILED
- **Heavy imports**: Don't import heavy resources at module level; defer to fixture scope
- **Flaky tests**: Use `pytest.skip()` with `@pytest.mark.jira("PROJ-123")`, never delete
- **Fixture scope**: Session fixtures in `tests/conftest.py` run once for entire suite—modify carefully

## Boundaries

### ✅ Always

- Use meaningful variable names: `index`/`idx` instead of `i`, `server` instead of `s`, `item` instead of `x`
- Use `test_` prefix in `pytest.param` id values (e.g., `id="test_without_order_by"`)
- Use `pytest.mark.parametrize` when possible instead of duplicating test logic
- Use bounded iteration instead of `while True` loops
- Avoid code duplication by creating meaningful utilities
- Follow existing patterns before introducing new approaches
- Add type annotations (mypy strict enforced)
- Write Google-format docstrings for utility functions
- Tests should have a concise docstring (Given-When-Then for tests, one-line for fixtures)
- Run `pre-commit run --all-files` before suggesting changes

### ⚠️ Ask First

- Adding new dependencies to `pyproject.toml`
- Creating new `conftest.py` files
- Moving fixtures to shared locations
- Adding new markers to `pytest.ini`
- Modifying session-scoped fixtures
- Adding new binaries or system-level dependencies (must also update `Dockerfile` and verify with `make build`)

### 🚫 Never

- Remove or modify existing tests without explicit request
- Add code that isn't immediately used (YAGNI)
- Log secrets, tokens, or credentials
- Skip pre-commit or type checking
- Create abstractions for single-use code

## Documentation Reference

Consult these for detailed guidance:

- [Constitution](./CONSTITUTION.md) - Non-negotiable principles (supersedes all other docs)
- [Developer Guide](./docs/DEVELOPER_GUIDE.md) - Contribution workflow, fixture examples
- [Style Guide](./docs/STYLE_GUIDE.md) - Naming, typing, docstrings
