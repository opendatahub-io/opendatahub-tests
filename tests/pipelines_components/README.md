# Pipelines Components Smoke Tests

Smoke tests for reusable Kubeflow Pipelines assets from
[pipelines-components](https://github.com/opendatahub-io/pipelines-components).
Each test submits a pipeline run to a DataSciencePipelinesApplication (DSPA) and verifies
that the run is registered and its workflow starts.

## Test Suites

- **`automl/`** -- AutoGluon tabular (regression, binary and multiclass classification) and timeseries smoke tests
- **`autorag/`** -- Managed Documents RAG Optimization pipeline smoke test

Both suites create a dedicated namespace and DSPA with MinIO. Test-owned resources
are cleaned up on teardown.

The DSPA MinIO server and the CSV uploader use the same digest-pinned image,
`PipelinesComponentsImages.MINIO`, which includes the `mc` client. It is listed in
the image manifest for mirroring into disconnected clusters. Set `DSPA_MINIO_IMAGE`
and/or `MINIO_MC_IMAGE` to override it when a cluster requires a different image.

## Prerequisites

- OpenShift cluster with RHOAI/ODH installed
- External S3 bucket with training data and AWS credentials (AutoML upgrade tests only)

## Configuration via `.env` file

To run tests from your IDE or locally, create a `.env` file at `tests/pipelines_components/.env`
with the required environment variables (see tables below).
The file is loaded automatically and is in `.gitignore`.
Environment variables set in the shell take precedence over `.env` values.
Sensitive values (API keys, tokens) are masked in test logs.

## Pipeline Modes

### Managed pipelines (default)

The DSPA operator auto-registers pipelines in KFP by display name.
AutoML and AutoRAG smoke tests discover their managed pipelines by display name.

### Legacy YAML upload (fallback)

AutoML regression upgrade tests support `AUTOML_PIPELINE_YAML` for manual upload;
this setting does not affect smoke tests. URLs are downloaded automatically at test startup.

## Environment Variables

### External S3 credentials (AutoML upgrade tests only)

| Variable | Description | Default |
| --- | --- | --- |
| `AWS_ACCESS_KEY_ID` | AWS access key for external S3 | _(required)_ |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for external S3 | _(required)_ |

### AutoML

The AutoML smoke tests generate 150 unique CSV rows per task in
[`automl/data/`](automl/data/README.md). A temporary ConfigMap mounts each CSV into
an uploader Pod, which copies it to the test DSPA's MinIO. They do not need an external
S3 bucket or credentials. The tabular test is parametrized by `task_type`
(regression, classification, multiclass); timeseries uses the same upload fixture.

The tabular pipeline accepts `task_type` values: `"binary"`, `"multiclass"`, `"regression"`.

```bash
# Run all AutoML smoke tests
pytest tests/pipelines_components/automl/ -m smoke -v

# Run only one task type
pytest tests/pipelines_components/automl/ -k regression
pytest tests/pipelines_components/automl/ -k classification
pytest tests/pipelines_components/automl/ -k multiclass
pytest tests/pipelines_components/automl/ -k timeseries
```

| Variable | Description | Default |
| --- | --- | --- |
| `AUTOML_TRAIN_DATA_FILE_KEY` | Base destination key in DSPA MinIO; prefixed with the task type | `automl-smoke/train.csv` |
| `AUTOML_PIPELINE_START_TIMEOUT` | Max wait for the smoke workflow to start (sec) | `300` |

AutoML **upgrade** tests retain their external-S3 upload fixtures. They require
`AUTOML_S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AUTOML_REGRESSION_S3_TRAIN_DATA_KEY`, and `AUTOML_TIMESERIES_S3_TRAIN_DATA_KEY`.
`AUTOML_PIPELINE_YAML` controls legacy upload for regression upgrade tests;
`AUTOML_PIPELINE_TIMEOUT` defaults to 1800 seconds for upgrade runs.

### AutoRAG configuration

The AutoRAG smoke test uploads a local document and a matching benchmark record
from [`autorag/data/`](autorag/data/README.md). It creates the required connection
Secrets and supplies model identifiers for pipeline submission. No MaaS models,
database, or external S3 credentials are required. This verifies pipeline
registration and startup only, not completed RAG optimization.

| Variable | Description | Default |
| --- | --- | --- |
| `AUTORAG_INPUT_DATA_KEY` | Destination prefix for the sample document | `autorag-smoke/input_data` |
| `AUTORAG_TEST_DATA_KEY` | Destination key for generated benchmark JSON | `autorag-smoke/benchmark_data.json` |
| `AUTORAG_MAX_RAG_PATTERNS` | Maximum RAG patterns to evaluate | `4` |
| `AUTORAG_OPTIMIZATION_METRIC` | Optimization metric | `unitxt:faithfulness` |
| `AUTORAG_PIPELINE_START_TIMEOUT` | Max wait for workflow progression (sec) | `300` |

### Managed pipeline overrides (optional)

| Variable | Description | Default |
| --- | --- | --- |
| `MANAGED_PIPELINE_AUTOML_TABULAR` | Display name of AutoML pipeline | `autogluon-tabular-training-pipeline` |
| `MANAGED_PIPELINE_AUTOML_TIMESERIES` | Display name of timeseries pipeline | `autogluon-timeseries-training-pipeline` |
| `MANAGED_PIPELINE_AUTORAG` | Display name of AutoRAG pipeline | `documents-rag-optimization-pipeline` |
| `MANAGED_PIPELINES_IMAGE` | Pin a specific pipelines-components image | _(empty)_ |
| `DSPA_READY_BUFFER_SECONDS` | Buffer after DSPA ready before polling | `30` |
| `MANAGED_PIPELINE_WAIT_TIMEOUT` | Timeout for pipeline discovery (sec) | `300` |

### Debug

| Variable | Description | Default |
| --- | --- | --- |
| `SKIP_TEARDOWN` | Keep all resources after test for debugging | `false` |

## Running Tests

### Run AutoML smoke tests (tabular and timeseries)

```bash
uv run pytest tests/pipelines_components/automl/ -m smoke -v -s

# Run only one task type
uv run pytest tests/pipelines_components/automl/ -k regression -v -s
uv run pytest tests/pipelines_components/automl/ -k classification -v -s
```

### Run AutoRAG smoke test

```bash
uv run pytest tests/pipelines_components/autorag/ -m smoke -v -s
```

### Run all pipelines-components smoke tests

```bash
uv run pytest tests/pipelines_components/ -m smoke -v -s
```

## Infrastructure

### AutoML (fully self-contained)

- Creates a dedicated namespace (`automl-aqa-<hash>`)
- Deploys DSPA with built-in MinIO for object storage
- Generates each 150-row training dataset and uploads it to a separate DSPA MinIO key
- Discovers the managed tabular and timeseries pipelines
- Submits a run per task type via DSPA REST API and verifies its workflow reaches Running
  (or Succeeded if it finishes before polling); does not wait for training to complete
- Cleans up namespace on teardown

### AutoRAG (startup only)

- Creates a dedicated namespace (`autorag-aqa-<hash>`)
- Deploys DSPA with built-in MinIO
- Creates the required connection Secrets
- Uploads a sample document and benchmark JSON from the repo to DSPA MinIO
- Discovers the managed pipeline and submits a run via the DSPA REST API
- Retries with `vector_db_secret_name` if the managed pipeline rejects `db_secret_name`
- Verifies the run is registered and its workflow starts, without waiting for optimization to finish
- Cleans up namespace on teardown
