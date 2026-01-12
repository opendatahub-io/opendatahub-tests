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

```
tests/                    # Test modules by component
├── conftest.py           # Shared fixtures (session/class scope)
├── fixtures/             # Extracted fixture modules
├── <component>/          # Component test directories
│   ├── conftest.py       # Component-scoped fixtures
│   └── test_*.py         # Test files
utilities/                # Shared utility functions
├── manifests/            # Runtime manifests and configs
└── <topic>_utils.py      # Topic-specific utilities
```

## Essential Patterns

### Tests
- Every test MUST have a docstring explaining what it tests (see `tests/cluster_health/test_cluster_health.py`)
- Apply relevant markers from `pytest.ini`: tier (`smoke`, `sanity`, `tier1`, `tier2`), component (`model_serving`, `model_registry`, `llama_stack`), infrastructure (`gpu`, `parallel`, `slow`)
- Use Given-When-Then format in docstrings for behavioral clarity

### Fixtures
- Fixture names MUST be nouns: `storage_secret` not `create_secret`
- Use context managers for resource lifecycle (see `tests/conftest.py:544-550` for pattern)
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
- Follow existing patterns before introducing new approaches
- Add type annotations (mypy strict enforced)
- Write Google-format docstrings for tests and fixtures
- Run `pre-commit run --all-files` before suggesting changes

### ⚠️ Ask First
- Adding new dependencies to `pyproject.toml`
- Creating new `conftest.py` files
- Moving fixtures to shared locations
- Adding new markers to `pytest.ini`
- Modifying session-scoped fixtures

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
