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
