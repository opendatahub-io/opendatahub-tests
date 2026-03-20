# opendatahub-tests

This repository contains ODH / Red Hat Openshift AI (RHOAI) tests.  
The tests are written in Python and use [pytest](https://docs.pytest.org/en/stable/) as a test framework.

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

To see optional CLI arguments run:

```bash
uv run pytest --help
```

### Using CLI arguments

CLI arguments can be passed to pytest by setting them in [pytest.ini](pytest.ini).  
You can either use the default pytest.ini file and pass CLI arguments or create a custom one.  
For example, add the below under the `addopts` section:
```code
    --ci-s3-bucket-name=name
    --ci-s3-bucket-endpoint=endpoint-path
    --ci-s3-bucket-region=region
```

Then pass the path to the custom pytest.ini file to pytest:

```bash
uv run pytest -c custom-pytest.ini

```


### Running specific tests
```bash
uv run pytest -k test_name
```

### Running on different distributions
Bt default, RHOAI distribution is set.  
To run on ODH, pass `--tc=distribution:upstream` to pytest.
