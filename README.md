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

To see optional cli arguments run:

```bash
uv run pytest --help
```

### Running on different distributions
Bt default, RHOAI distribution is set.  
To run on ODH, pass `--tc=distribution:upstream` to pytest.

### jira integration
To skip running tests which have open bugs, [pytest_jira](https://github.com/rhevm-qe-automation/pytest_jira) plugin is used.
To run tests with jira integration, you need to set `PYTEST_JIRA_URL` and `PYTEST_JIRA_TOKEN` environment variables.
To make a test with jira marker, add: `@pytest.mark.jira(jira_id="RHOAIENG-0000", run=False)` to the test.
