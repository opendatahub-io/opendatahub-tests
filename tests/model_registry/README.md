# Model Registry

This section outlines specific behaviour of Model Registry testing users should be aware of.
The following behaviour applies when the user specifies `--model-registry-upstream` as a CLI option.

## Upstream tests

In the [conftest.py](conftest.py) file in this subfolder there are 3 hook functions defined

  - [pytest_sessionstart](conftest.py#L322)
  - [pytest_collection_modifyitems](conftest.py#L335)
  - [pytest_sessionfinish](conftest.py#L348)

These functions will download, discover and run, and remove test cases defined in the
[upstream model registry repository](https://github.com/kubeflow/model-registry/blob/main/clients/python/tests) dynamically at runtime.

Be aware of these when launching pytest in such a way that this subfolder's conftest.py is loaded
(e.g. uv run pytest tests/model_registry/).
