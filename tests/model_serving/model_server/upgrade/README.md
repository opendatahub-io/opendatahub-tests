How to run upgrade tests
==========================

Note: product upgrade is out of scope for this project and should be done by the user.

## Run pre-upgrade tests

```bash
uv run pytest --pre-upgrade

```

To run pre-upgrade tests delete the resources (useful for debugging pre-upgrade tests)

```bash
uv run pytest --pre-upgrade --delete-pre-upgrade-resources
```

## Run post-upgrade tests

```bash
uv run pytest --post-upgrade
```


## Run pre-upgrade and post-upgrade tests

```bash
uv run pytest --pre-upgrade --post-upgrade
```
