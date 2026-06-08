# Workbenches Tests

This directory contains tests for Jupyter notebook workbenches in OpenDataHub/RHOAI. These tests validate notebook spawning and lifecycle management and resource customization.

## Directory Structure

```text
workbenches/
├── notebooks_server/
│   └── controller/
│       ├── conftest.py                   # Pytest fixtures (PVC, notebook image, notebook CR, pod)
│       ├── utils.py                      # Shared utilities (username retrieval)
│       └── test_spawning.py              # Basic notebook spawning tests
└── notebook_images/                      # Notebook container image tests (placeholder)
```

### Current Test Suites

- **`notebooks_server/controller/test_spawning.py`** - Tests basic notebook creation via Notebook CR and validates pod creation. Also tests OAuth proxy container resource customization via annotations

## Test Markers

```python
@pytest.mark.smoke         # Quick validation tests (basic spawning)
```

## Running Tests

### Run All Workbenches Tests

```bash
uv run pytest tests/workbenches/
```

### Run Tests by Component

```bash
# Run notebook spawning tests
uv run pytest tests/workbenches/notebooks_server/controller/test_spawning.py
```

### Run Tests with Markers

```bash
# Run smoke tests only
uv run pytest -m smoke tests/workbenches/
```

## Additional Resources

- [Kubeflow Notebook Controller](https://github.com/kubeflow/kubeflow/tree/master/components/notebook-controller)
