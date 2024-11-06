# opendatahub-tests

This repository contains ODH / Red Hat Openshift AI (RHOAI) tests.

## Contribute to opendatahub-tests
Please follow the [Contributing Guide](CONTRIBUTING.md)

# Getting started
## Installation

Install [uv](https://github.com/astral-sh/uv)

## Tests cluster

These tests can be executed against arbitrary cluster with ODH / RHOAI installed.

You can log in into such cluster via:

```bash
oc login -u user -p password
```

Or by setting `KUBECONFIG` variable:

```bash
KUBECONFIG=<kubeconfig file>
```

or by saving the kubeconfig file under `~/.kube/config`

## Running the tests
### Basic run of all tests

```bash
uv run pytest
```

### To overwrite a pytest config argument

```bash
uv run pytest --tc=<arg name>:<arg value>
```

For example:

```bash
uv run pytest --tc=ci_s3_bucket_name:my-bucket
```
