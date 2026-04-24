# Pipelines Components Smoke Tests

Smoke tests for reusable Kubeflow Pipelines assets from
[pipelines-components](https://github.com/opendatahub-io/pipelines-components).
Each test submits a pipeline run to a DataSciencePipelinesApplication (DSPA) and asserts successful completion.

## Test Suites

- **`automl/`** - AutoGluon Tabular Training pipeline smoke test

## Prerequisites

- OpenShift cluster with RHOAI/ODH installed
- `aipipelines` component enabled (the test enables it automatically via DSC patching)

## Pipeline YAMLs

Pre-compiled pipeline YAMLs are included in the repo under `automl/pipelines/`.
They are sourced from
[red-hat-data-services/pipelines-components](https://github.com/red-hat-data-services/pipelines-components).
To use a different version, set the `AUTOML_PIPELINE_YAML` env var to point to your own YAML.

## Optional Environment Variables

| Variable                     | Description                            | Default                                                                     |
| ---------------------------- | -------------------------------------- | --------------------------------------------------------------------------- |
| `AUTOML_PIPELINE_YAML`       | Override path to pipeline YAML         | `automl/pipelines/autogluon_tabular_training.yaml` (bundled)                |
| `AUTOML_TRAIN_DATA_FILE_KEY` | S3 key for training CSV                | `automl-smoke/train.csv`                                                    |
| `AUTOML_LABEL_COLUMN`        | Target column name in CSV              | `target`                                                                    |
| `AUTOML_TASK_TYPE`           | AutoGluon task type                    | `binary`                                                                    |
| `AUTOML_TOP_N`               | Number of top models to refit          | `1`                                                                         |
| `AUTOML_PIPELINE_TIMEOUT`    | Max wait for pipeline completion (sec) | `1800`                                                                      |
| `PIPELINE_POLL_INTERVAL`     | Seconds between status polls           | `30`                                                                        |
| `MINIO_MC_IMAGE`             | MinIO client image for data upload     | `quay.io/opendatahub/minio:RELEASE.2019-08-14T20-37-41Z-license-compliance` |

## Running Tests

### Run AutoML smoke test

```bash
uv run pytest tests/pipelines_components/automl/ -m smoke -v -s --junitxml=results_xunit.xml
```

### Run all pipelines-components tests

```bash
uv run pytest -m pipelines_components tests/pipelines_components/ --junitxml=results_xunit.xml
```

### Run from container image (CI / Jenkins)

The repo ships a container image with entrypoint `uv run pytest`.
Mount a valid `kubeconfig` and pass pytest arguments:

```bash
podman run --rm \
  -v "${KUBECONFIG}:/home/odh/.kube/config:z" \
  -e KUBECONFIG=/home/odh/.kube/config \
  quay.io/opendatahub/opendatahub-tests:latest \
  tests/pipelines_components/automl/ -m smoke -v -s \
  --junitxml=/home/odh/opendatahub-tests/results_xunit.xml
```

To extract the xUnit report after the run:

```bash
podman cp <container_id>:/home/odh/opendatahub-tests/results_xunit.xml .
```

Or use a volume mount for the results directory:

```bash
podman run --rm \
  -v "${KUBECONFIG}:/home/odh/.kube/config:z" \
  -v "$(pwd)/results:/home/odh/opendatahub-tests/results:z" \
  -e KUBECONFIG=/home/odh/.kube/config \
  quay.io/opendatahub/opendatahub-tests:latest \
  tests/pipelines_components/automl/ -m smoke -v -s \
  --junitxml=/home/odh/opendatahub-tests/results/results_xunit.xml
```

### Monitoring a run

The pipeline run appears in the RHOAI/ODH dashboard and CLI as
`automl-smoke-<namespace>`, where `<namespace>` is the randomly generated
test namespace (e.g. `automl-smoke-pipelines-smoke-abc123def`).

To find it via CLI while the test is running:

```bash
oc get projects | grep pipelines-smoke
oc get pipelineruns -n <namespace>
```

## Test Markers

- `@pytest.mark.smoke` - Quality gate marker
- `@pytest.mark.pipelines_components` - Component ownership marker

## Infrastructure

The tests automatically provision:

- A dedicated OpenShift namespace (cleaned up after test)
- DSPA with built-in MinIO for object storage
- Synthetic training data uploaded to MinIO
- Pipeline submitted via DSPA REST API through the external Route

No external S3 or pre-existing pipeline server is required.

## Pipeline Sources

| Test   | Source                                                                                               |
| ------ | ---------------------------------------------------------------------------------------------------- |
| AutoML | `pipelines/training/automl/autogluon_tabular_training_pipeline/pipeline.yaml` (pipelines-components) |
