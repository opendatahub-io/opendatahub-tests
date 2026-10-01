# Workbenches Tests

This directory contains tests for Jupyter notebook workbenches in OpenDataHub/RHOAI. These tests validate notebook spawning and lifecycle management and resource customization.

## Directory Structure

```text
workbenches/
├── notebooks_server/
│   └── controller/
│       ├── conftest.py                   # Pytest fixtures (PVC, notebook image, notebook CR, pod)
│       ├── utils.py                      # Shared utilities (image resolution, notebook CR building, username retrieval)
│       ├── test_spawning.py              # Basic notebook spawning tests
│       └── upgrade/
│           ├── conftest.py               # Session-scoped fixtures for upgrade lifecycle
│           ├── test_upgrade.py           # Pre/post upgrade notebook survival tests
│           ├── test_upgrade_routing.py   # Pre/post upgrade Route survival tests
│           └── test_upgrade_stopped.py   # Pre/post upgrade stopped notebook tests
└── notebook_images/
    ├── utils.py                          # Image resolution, Notebook CR, oauth-proxy wait
    └── upgrade/
        ├── conftest.py                   # Session fixtures for image survival and bump
        ├── survival_checks.py            # Shared post-upgrade survival assertions
        ├── elyra_utils.py                # Elyra extension and runtime-config helpers
        ├── test_upgrade_workbench.py     # JupyterLab, Elyra, and Code Server image survival
        ├── test_upgrade_jupyter_elyra.py # Elyra extension survival
        └── test_bump_jupyterlab.py       # Dashboard image bump after upgrade
```

### Current Test Suites

- **`notebooks_server/controller/test_spawning.py`** - Tests basic notebook creation via Notebook CR and validates pod creation. Also tests OAuth proxy container resource customization via annotations
- **`notebooks_server/controller/upgrade/test_upgrade.py`** - Upgrade survival tests. Pre-upgrade creates a notebook and captures its pod creation timestamp to a ConfigMap. Post-upgrade verifies the pod was not restarted by comparing timestamps
- **`notebooks_server/controller/upgrade/test_upgrade_routing.py`** - Upgrade tests for OpenShift Routes. Pre-upgrade verifies the Route exists and targets the correct service. Post-upgrade verifies the Route survived unchanged
- **`notebooks_server/controller/upgrade/test_upgrade_stopped.py`** - Upgrade tests for stopped notebooks. Pre-upgrade stops a notebook via annotation and verifies scale-down. Post-upgrade verifies it remains stopped
- **`notebook_images/upgrade/test_upgrade_workbench.py`** - Workbench image survival. Pre-upgrade creates JupyterLab, Elyra, and Code Server workbenches on the source image and records pod identity, digest, and a PVC marker. Post-upgrade verifies the running workbench was not restarted
- **`notebook_images/upgrade/test_upgrade_jupyter_elyra.py`** - Elyra extension and runtime-config survival on `s2i-generic-data-science-notebook`. Extension checks run when the workbench still exists
- **`notebook_images/upgrade/test_bump_jupyterlab.py`** - After upgrade, applies the Dashboard image patch. Skips when the source and current images are the same URL, which is expected on a 2.x z-stream upgrade

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
```

### Run Tests with Markers

```bash
# Run smoke tests only
uv run pytest -m smoke tests/workbenches/
```

## Additional Resources

- [Kubeflow Notebook Controller](https://github.com/kubeflow/kubeflow/tree/master/components/notebook-controller)
