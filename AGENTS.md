# Overview

This is a testing repo for OpenDataHub and OpenShift AI, which are MLOps platforms for OpenShift.
The tests are high-level integration tests at the Kubernetes API level.

You are an expert QE engineer writing maintainable pytest tests that other engineers can understand without deep domain knowledge.

## Setup

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- Access to an OpenShift cluster with OpenDataHub or OpenShift AI installed

### Install and Configure

```bash
# 1. Log in to the target OpenShift cluster
oc login --server=<cluster-api-url> --token=<token>

# 2. Install dependencies
uv sync
```

All test commands should be run with `uv run` to use the project's virtual environment.

### Run Prerequisites

Tests run against a live OpenShift cluster. Before running tests:

- **Cluster access**: `oc login` must be completed (or `KUBECONFIG` env var set)
- **Jira connectivity** (for xfail/skip of known bugs): `PYTEST_JIRA_URL`, `PYTEST_JIRA_USERNAME`, `PYTEST_JIRA_TOKEN`
- **S3 credentials** (model serving tests): `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `CI_S3_BUCKET_NAME`, `CI_S3_BUCKET_REGION`, `CI_S3_BUCKET_ENDPOINT`
- **Model storage** (model serving tests): `MODELS_S3_BUCKET_NAME`, `MODELS_S3_BUCKET_REGION`, `MODELS_S3_BUCKET_ENDPOINT`
- **Optional overrides**: `VLLM_RUNTIME_IMAGE`, `OVMS_RUNTIME_IMAGE`, `TRITON_RUNTIME_IMAGE`, `MLSERVER_RUNTIME_IMAGE`

These can also be passed as pytest command-line options (see `conftest.py` for `addoption` definitions).

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

### Debugging

```bash
# Drop into pdb on failure
uv run pytest --pdb tests/model_serving/

# Drop into pdb at a specific point (add to code)
import pdb; pdb.set_trace()

# Verbose output with full tracebacks
uv run pytest -vvv --tb=long tests/model_serving/
```

### Make Targets

```bash
make check                  # Install tox and run tox (CI validation)
make build                  # Build container image (podman/docker)
make push                   # Push container image to registry
make build-and-push-container  # Build and push
```

To run tests in a container: `make build` then run the image via `podman run`.

Image defaults: `quay.io/opendatahub/opendatahub-tests:latest` (override with `IMAGE_REGISTRY`, `REGISTRY_NAMESPACE`, `IMAGE_TAG`).

### Tox Targets

```bash
tox -e unused-code          # Check for unused code
tox -e pytest               # Collect tests and verify setup plan
tox                         # Run all targets
```

## Architecture

### Component Relationships

Tests are organized by product component. Each component maps to a feature area of OpenDataHub/OpenShift AI:

- `model_serving` — Model deployment and inference (KServe, ModelMesh, vLLM, OVMS, Triton)
- `model_registry` — Model registration, catalog, RBAC, REST API
- `model_explainability` — TrustyAI, LM-Eval, guardrails
- `llama_stack` — Llama Stack inference and safety
- `workbenches` — Jupyter notebook controllers and images
- `cluster_health` — Operator and cluster health checks

### Fixture Flow

Fixtures follow a hierarchical scope with composition:

1. **Session fixtures** (`tests/conftest.py`) — cluster-wide resources (admin client, DSC, namespaces)
2. **Component fixtures** (`tests/<component>/conftest.py`) — component-scoped resources
3. **Test fixtures** (local `conftest.py`) — per-test resources

Resources flow top-down: session fixtures create shared state, component fixtures build on them, test fixtures add test-specific resources. All use context managers for cleanup.

### Key Conventions

- `conftest.py` files contain **fixtures only** — no utility functions or constants
- Utility functions go in `utils.py` (component-level) or `utilities/<topic>_utils.py` (shared)
- All K8s resources use [openshift-python-wrapper](https://github.com/RedHatQE/openshift-python-wrapper); for missing resources, generate with `class_generator` and contribute upstream

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
- Apply relevant markers from `pytest.ini`: tier (`smoke`, `tier1`, `tier2`), component (`model_serving`, `model_registry`, `llama_stack`), infrastructure (`gpu`, `parallel`, `slow`)
- Use Given-When-Then format in docstrings for behavioral clarity

### Fixtures

- Every fixture MUST have a one-line docstring explaining what it does
- Fixture names MUST be nouns: `storage_secret` not `create_secret`
- Use context managers for resource lifecycle (see `tests/conftest.py:544-550` for pattern)
- Fixtures do one thing only—compose them rather than nesting
- Use narrowest scope that meets the need: function > class > module > session

### Utilities

- Every utility function MUST have a full Google-format docstring including description, Args, Returns, and Raises sections

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

## Key Rules (from Constitution)

The [Constitution](./CONSTITUTION.md) supersedes all other docs. Critical rules:

- **Simplicity first**: Favor the simplest solution; no YAGNI; every function/variable must be used or removed
- **Consistency**: Follow existing patterns; use absolute imports; import specific functions not modules
- **Security**: Never log secrets; use detect-secrets and gitleaks hooks; do not reference internal-only resources
- **Conftest discipline**: `conftest.py` files contain fixtures only — no utility functions or constants
- **Fixture ordering**: Call pytest native fixtures first, then session-scoped, then others

## Style Summary (from Style Guide)

- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Use descriptive names; meaningful names over short names; no single-letter names
- Use [Google-format](https://google.github.io/styleguide/pyguide.html#381-docstrings) docstrings
- Add type annotations to all new code; enforced by mypy (rules in `pyproject.toml`)

## Documentation Reference

Consult these for detailed guidance:

- [Constitution](./CONSTITUTION.md) - Non-negotiable principles (supersedes all other docs)
- [Developer Guide](./docs/DEVELOPER_GUIDE.md) - Contribution workflow, fixture examples
- [Style Guide](./docs/STYLE_GUIDE.md) - Naming, typing, docstrings
