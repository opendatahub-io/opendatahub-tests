# Workbenches Tests

This directory contains tests for Jupyter notebook workbenches in OpenDataHub/RHOAI. These tests validate notebook spawning and lifecycle management, ImageStream health, custom workbench image validation, and resource customization.

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
        |-- conftest.py                   # Session-scoped upgrade fixtures per IDE
        +-- elyra_utils.py                # Utilities for interacting with Elyra
        |-- survival_checks.py            # Shared pre/post-upgrade validation steps
        |-- test_upgrade_jupyterlab.py    # JupyterLab survival tests
        |-- test_upgrade_codeserver.py    # Code Server survival tests
        +-- test_upgrade_rstudio.py       # RStudio survival tests (EUS only)
        +-- test_upgrade_jupyter_elyra.py # Elyra survival tests
```

### Current Test Suites

- **`notebooks_server/operator/test_imagestream_health.py`** - Validates that ImageStreams are properly imported and resolved. Uses compound label selectors (`opendatahub.io/notebook-image` or `opendatahub.io/runtime-image` combined with `platform.opendatahub.io/part-of`) to scope checks per component. Validates correct ImageStream counts, tag digest references (`@sha256:`), and `ImportSuccess` conditions for workbench notebook images (11 expected), workbench runtime images (7 expected), and trainer images (3 expected)
- **`notebooks_server/controller/test_spawning.py`** - Tests basic notebook creation via Notebook CR and validates pod creation. Also tests Auth proxy container resource customization via annotations
- **`notebooks_server/controller/test_custom_images.py`** - Validates custom workbench images contain required Python packages by spawning a workbench, installing any missing packages, and executing import verification
- **`notebooks_server/controller/upgrade/test_upgrade.py`** - Upgrade survival tests. Pre-upgrade creates a notebook and captures its pod creation timestamp to a ConfigMap. Post-upgrade verifies the pod was not restarted by comparing timestamps
- **`notebook_images/upgrade/`** - Per-IDE N-1 workbench image survival tests for JupyterLab, Code Server, and RStudio (EUS). Pre-upgrade launches dashboard-faithful workbenches and captures baselines; post-upgrade verifies pod continuity, image invariants, StatefulSet health, PVC data, logs, and HTTP health. This test suite also includes a test to verify that the Elyra extension survives the upgrade on a Jupyter Data Science image.

## Test Markers

```python
@pytest.mark.smoke         # Quick validation tests (imagestream health, basic spawning)
@pytest.mark.tier1         # Comprehensive validation tests
@pytest.mark.slow          # Long-running tests (custom image validation)
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
# Run ImageStream health tests
uv run pytest tests/workbenches/notebooks_server/operator/test_imagestream_health.py

# Run notebook spawning tests
uv run pytest tests/workbenches/notebooks_server/controller/test_spawning.py

# Run custom image validation tests
uv run pytest tests/workbenches/notebooks_server/controller/test_custom_images.py

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
