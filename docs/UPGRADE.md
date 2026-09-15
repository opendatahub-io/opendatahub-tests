# How to run upgrade tests

Note: product upgrade is out of scope for this project and should be done by the user.

## Run pre-upgrade tests

`SKIP_RESOURCE_TEARDOWN` environment variable is set to skip resources teardown.

```bash
uv run pytest --pre-upgrade

```

To run pre-upgrade tests and delete the resources at the end of the run (useful for debugging pre-upgrade tests)

```bash
uv run pytest --pre-upgrade --delete-pre-upgrade-resources
```

## Run post-upgrade tests

`REUSE_IF_RESOURCE_EXISTS` environment variable is set to reuse resources if they already exist.

```bash
uv run pytest --post-upgrade
```

## Run pre-upgrade and post-upgrade tests

```bash
uv run pytest --pre-upgrade --post-upgrade
```

## To run only specific deployment tests, pass --upgrade-deployment-modes with requested mode(s), for example

```bash
uv run pytest --pre-upgrade --post-upgrade --upgrade-deployment-modes=servelerss
```

```bash
uv run pytest --pre-upgrade --post-upgrade --upgrade-deployment-modes=servelerss,rawdeployment
```

## Workbench image survival (N-1)

Run the workbench image survival suite separately from controller upgrade tests:

```bash
# Pre-upgrade on the source cluster
uv run pytest --pre-upgrade tests/workbenches/notebook_images/upgrade/

# Post-upgrade on the upgraded cluster
uv run pytest --post-upgrade tests/workbenches/notebook_images/upgrade/

# Target a single IDE
uv run pytest --post-upgrade tests/workbenches/notebook_images/upgrade/test_upgrade_jupyterlab.py

# Target Jupyter workbenches with Elyra
uv run pytest tests/workbenches/notebook_images/upgrade/test_upgrade_jupyter_elyra.py
```

Override ImageStream tag selection when needed:

```bash
uv run pytest --pre-upgrade tests/workbenches/notebook_images/upgrade/ --tc workbench_image_tag=3.4
uv run pytest --pre-upgrade tests/workbenches/notebook_images/upgrade/ --tc workbench_upgrade_track=eus
```

See [tests/workbenches/notebook_images/README.md](../tests/workbenches/notebook_images/README.md) for coverage details.
