# Pipelines Components Smoke Tests

Smoke tests for reusable Kubeflow Pipelines assets from
[pipelines-components](https://github.com/opendatahub-io/pipelines-components).
Each test submits a pipeline run to a DataSciencePipelinesApplication (DSPA) and verifies
that the run is registered and its workflow starts.

## Test Suites

- **`automl/`** -- AutoGluon tabular (regression, binary and multiclass classification) and timeseries smoke tests
- **`autorag/`** -- Managed Documents RAG Optimization pipeline smoke test

Both suites create a dedicated namespace and DSPA with MinIO. AutoRAG uses existing MaaS
and Milvus services; test-owned resources are cleaned up on teardown.

## Prerequisites

- OpenShift cluster with RHOAI/ODH installed
- External S3 bucket with training/test data (AWS credentials required)
- Existing MaaS endpoint with generation and embedding models, and a reachable Milvus database (for AutoRAG)

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

### Shared S3 credentials

| Variable | Description | Default |
| --- | --- | --- |
| `AWS_ACCESS_KEY_ID` | AWS access key for external S3 | _(required)_ |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for external S3 | _(required)_ |

### AutoML

The AutoML tabular test is parametrized by `task_type` (regression, classification, multiclass).

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
| `AUTOML_S3_BUCKET` | External S3 bucket with training data | _(required)_ |
| `AUTOML_REGRESSION_S3_TRAIN_DATA_KEY` | S3 key for regression training CSV | `datasets/regression/regression.csv` |
| `AUTOML_CLASSIFICATION_S3_TRAIN_DATA_KEY` | S3 key for binary classification training CSV | `datasets/classification/classification.csv` |
| `AUTOML_MULTICLASS_S3_TRAIN_DATA_KEY` | S3 key for multiclass classification training CSV | `datasets/classification/multiclass.csv` |
| `AUTOML_TIMESERIES_S3_TRAIN_DATA_KEY` | S3 key for timeseries training CSV | _(required)_ |
| `AUTOML_TRAIN_DATA_FILE_KEY` | Base destination key in DSPA MinIO; prefixed with the task type | `automl-smoke/train.csv` |
| `AUTOML_PIPELINE_START_TIMEOUT` | Max wait for the smoke workflow to start (sec) | `300` |
| `AUTOML_PIPELINE_YAML` | Legacy YAML for regression upgrade tests only | _(empty = managed mode)_ |
| `AUTOML_PIPELINE_TIMEOUT` | Max wait for upgrade pipeline completion (sec) | `1800` |

### AutoRAG configuration

| Variable | Description | Default |
| --- | --- | --- |
| `AUTORAG_S3_BUCKET` | External S3 bucket with test data | `mlpipeline` |
| `AUTORAG_INPUT_DATA_KEY` | S3 key prefix for input documents | `autorag-smoke/input_data` |
| `AUTORAG_TEST_DATA_KEY` | S3 key for benchmark JSON | `autorag-smoke/benchmark_data.json` |
| `AUTORAG_INFERENCE_MODEL_NAME` | Generation model ID available through MaaS | _(required)_ |
| `AUTORAG_EMBEDDING_MODEL_NAME` | Embedding model ID available through MaaS | _(required)_ |
| `AUTORAG_MAAS_BASE_URL` | URL of the existing MaaS endpoint | _(required)_ |
| `AUTORAG_MAAS_API_KEY` | MaaS API key, stored in a namespace Secret | _(required)_ |
| `AUTORAG_MILVUS_URI` | URI of an existing Milvus database | _(required)_ |
| `AUTORAG_MILVUS_TOKEN` | Milvus access token, stored in a namespace Secret | _(optional)_ |
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
- Downloads each training dataset from external S3 into a separate DSPA MinIO key
- Discovers the managed tabular and timeseries pipelines
- Submits a run per task type via DSPA REST API and verifies its workflow reaches Running
  (or Succeeded if it finishes before polling); does not wait for training to complete
- Cleans up namespace on teardown

### AutoRAG (external MaaS and Milvus)

- Creates a dedicated namespace (`autorag-aqa-<hash>`)
- Deploys DSPA with built-in MinIO
- Creates connection Secrets for the existing MaaS and Milvus services
- Downloads test data from external S3 into DSPA MinIO
- Discovers the managed pipeline and submits a run via the DSPA REST API
- Verifies the run is registered and its workflow starts, without waiting for optimization to finish
- Cleans up namespace on teardown
