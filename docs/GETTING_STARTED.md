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

## OpenShift CLI (oc) Binary

By default, the test framework automatically downloads the OpenShift CLI binary from the target cluster's console CLI download service. This ensures compatibility between the client and cluster versions.

### Using a Local oc Binary

If you already have the `oc` binary installed locally, you can avoid the download by setting the `OC_BINARY_PATH` environment variable:

```bash
export OC_BINARY_PATH=/usr/local/bin/oc
```

Or run tests with the variable:

```bash
OC_BINARY_PATH=/usr/local/bin/oc uv run pytest
```

**Note:** Ensure your local `oc` binary is executable and compatible with your target cluster version.

## Must gather

In order to collect must-gather on failure point one may use `--collect-must-gather` to the pytest command. e.g.

```bash
uv run pytest tests/<your component> --collect-must-gather
```

By default, the collected must-gather would be archived. To skip archiving, please set environment variable
ARCHIVE_MUST_GATHER to any value other than "true". e.g.

```bash
export ARCHIVE_MUST_GATHER="false"
```

### Benefits of Using Local Binary

- Faster test startup (no download time)
- Consistent tooling across different test runs
- Useful in air-gapped environments or when internet access is limited

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

### Turning off console logging

By default, pytest will output logging reports in the console. You can disable this behavior with `-o log_cli=false`

```bash
uv run pytest -o log_cli=false
```

### Running specific tests

```bash
uv run pytest -k test_name
```

### LlamaStack Integration Tests

For more information about LlamaStack integration tests, see [/tests/llama_stack/README.md](../tests/llama_stack/README.md).

### Running on different distributions

Bt default, RHOAI distribution is set.  
To run on ODH, pass `--tc=distribution:upstream` to pytest.

### Skip cluster sanity checks

By default, cluster sanity checks are run to make cluster ready for tests.
To skip cluster sanity checks, pass `--cluster-sanity-skip-check` to skip all tests.
To skip RHOAI/ODH-related tests (for example when running in upstream), pass `--cluster-sanity-skip-rhoai-check`.

### Running tests with admin client instead of unprivileged client

To run tests with admin client only, pass `--tc=use_unprivileged_client:False` to pytest.


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
