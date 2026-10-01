# Workbenches Tests

This directory contains tests for Jupyter notebook workbenches in OpenDataHub/RHOAI. These tests validate notebook spawning and lifecycle management and resource customization.

## Directory Structure

```text
workbenches/
|-- notebooks_server/
|   |-- controller/
|   |   |-- conftest.py                   # Pytest fixtures (PVC, notebook image, notebook CR, pod)
|   |   |-- utils.py                      # Shared utilities (image resolution, notebook CR building)
|   |   |-- test_spawning.py              # Basic notebook spawning tests
|   |   |-- test_custom_images.py         # Custom image package verification tests
|   |   +-- upgrade/
|   |       |-- conftest.py               # Session-scoped fixtures for upgrade lifecycle
|   |       +-- test_upgrade.py           # Pre/post upgrade notebook survival tests
|   +-- operator/
|       +-- test_imagestream_health.py    # ImageStream validation tests
+-- notebook_images/                      # N-1 workbench image upgrade survival tests
    |-- utils.py                          # Image resolution, log/HTTP validation helpers
    +-- upgrade/
        |-- conftest.py                   # Session-scoped parametrized upgrade fixtures
        |-- elyra_utils.py                # Utilities for interacting with Elyra
        |-- survival_checks.py            # Shared pre/post-upgrade validation steps
        |-- test_upgrade_workbench.py     # Parametrized N-1 survival checks (all IDEs)
        |-- test_upgrade_jupyter_elyra.py # Elyra survival tests
        +-- test_bump_jupyterlab.py       # Dashboard-driven N-1 to N image bump test
```

### Current Test Suites

- **`notebooks_server/controller/test_spawning.py`** - Tests basic notebook creation via Notebook CR and validates pod creation. Also tests OAuth proxy container resource customization via annotations
- **`notebooks_server/controller/upgrade/test_upgrade.py`** - Upgrade survival tests. Pre-upgrade creates a notebook and captures its pod creation timestamp to a ConfigMap. Post-upgrade verifies the pod was not restarted by comparing timestamps
- **`notebooks_server/controller/upgrade/test_upgrade_routing.py`** - Upgrade tests for OpenShift Routes. Pre-upgrade verifies the Route exists and targets the correct service. Post-upgrade verifies the Route survived unchanged
- **`notebooks_server/controller/upgrade/test_upgrade_stopped.py`** - Upgrade tests for stopped notebooks. Pre-upgrade stops a notebook via annotation and verifies scale-down. Post-upgrade verifies it remains stopped
- **`notebook_images/upgrade/`** - Parametrized N-1 workbench image survival tests for JupyterLab, Code Server, RStudio (EUS), and Elyra. Pre-upgrade launches dashboard-faithful workbenches and captures baselines; post-upgrade verifies pod continuity, image invariants, StatefulSet health, PVC data, kernel state (JupyterLab), Elyra extension preservation, logs, and HTTP health

## Test Markers

```python
@pytest.mark.smoke         # Quick validation tests (basic spawning)
@pytest.mark.pre_upgrade   # Tests to run before platform upgrade
@pytest.mark.post_upgrade  # Tests to run after platform upgrade
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

# Run upgrade tests (pre-upgrade phase)
uv run pytest --pre-upgrade tests/workbenches/notebooks_server/controller/upgrade/

# Run upgrade tests (post-upgrade phase)
uv run pytest --post-upgrade tests/workbenches/notebooks_server/controller/upgrade/

# Run N-1 image upgrade tests (pre-upgrade phase)
uv run pytest --pre-upgrade tests/workbenches/notebook_images/upgrade/

# Run N-1 image upgrade tests (post-upgrade phase)
uv run pytest --post-upgrade tests/workbenches/notebook_images/upgrade/
```

### Run Tests with Markers

```bash
# Run smoke tests only
uv run pytest -m smoke tests/workbenches/
```

## Additional Resources

- [Kubeflow Notebook Controller](https://github.com/kubeflow/kubeflow/tree/master/components/notebook-controller)
