How to run upgrade tests
==========================

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

## To run only specific deployment tests, pass --upgrade-deployment-modes with requested mode(s), for example:

```bash
uv run pytest --pre-upgrade --post-upgrade --upgrade-deployment-modes=servelerss
```

```bash
uv run pytest --pre-upgrade --post-upgrade --upgrade-deployment-modes=servelerss,rawdeployment
```

## Workbench image survival

`tests/workbenches/notebook_images/upgrade/test_upgrade_workbench.py` creates JupyterLab and Code Server workbenches before the upgrade and asserts they are still the same pods afterwards. A running workbench is not rolled by a z-stream upgrade or by a 2.x to 3.x upgrade until auth migration restarts it.

```bash
uv run pytest --pre-upgrade tests/workbenches/notebook_images/upgrade/
uv run pytest --post-upgrade tests/workbenches/notebook_images/upgrade/
```

On 2.x the default source image is the newest `YYYY.N` ImageStream tag. To pin an older source tag (so the post-upgrade bump in `test_bump_jupyterlab.py` has a newer image to apply), pass:

```bash
uv run pytest --pre-upgrade tests/workbenches/notebook_images/upgrade/ --tc workbench_image_tag=2025.1
```

`--tc workbench_upgrade_track=eus` forces legacy year-based tag selection. That is already the default when the product major is less than 3.
