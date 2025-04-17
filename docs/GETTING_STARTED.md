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

CLI arguments can be passed to pytest by setting them in [pytest.ini](../pytest.ini).  
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

### Skip cluster sanity checks
By default, cluster sanity checks are run to make cluster ready for tests.
To skip cluster sanity checks, pass `--cluster-sanity-skip-check` to skip all tests.
To skip RHOAI/ODH-related tests (for example when running in upstream), pass `--cluster-sanity-skip-rhoai-check`.

### Check dependent operators
By default, `Serveless` (serverless-operator), `Authorino` (authorino-operator) and `Service Mesh` (servicemeshoperator) operators are checked to be installed when  
running model registry or model server Serverless tests.
To check only specific operator, pass `--tc=dependent_operators:<operator_name>` to pytest.
For example, to check only `Serveless` and `Service Mesh` operators, pass `--tc=dependent_operators:serverless-operator,servicemeshoperator`.


### jira integration
To skip running tests which have open bugs, [pytest_jira](https://github.com/rhevm-qe-automation/pytest_jira) plugin is used.
To run tests with jira integration, you need to set `PYTEST_JIRA_URL` and `PYTEST_JIRA_TOKEN` environment variables.
To make a test with jira marker, add: `@pytest.mark.jira(jira_id="RHOAIENG-0000", run=False)` to the test.


### Running containerized tests
Save kubeconfig file to a local directory, for example: `$HOME/kubeconfig`
To run tests in containerized environment:

```bash
podman run  -v $HOME:/mnt/host:Z  -e KUBECONFIG=/mnt/host/kubeconfig quay.io/opendatahub/opendatahub-tests
```
