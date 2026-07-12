# Pipelines Components Smoke Tests

Smoke tests for reusable Kubeflow Pipelines assets from
[pipelines-components](https://github.com/opendatahub-io/pipelines-components).
Each test submits a pipeline run to a DataSciencePipelinesApplication (DSPA) and asserts successful completion.

## Test Suites

- **`automl/`** -- AutoGluon Tabular Training pipeline smoke tests (regression + classification)
- **`autorag/`** -- Documents RAG Optimization pipeline smoke test

Both suites are **fully self-contained**: they create a dedicated namespace, deploy all required infrastructure, run the pipeline, and clean up on teardown.

## Prerequisites

- OpenShift cluster with RHOAI/ODH installed
- External S3 bucket with training/test data (AWS credentials required)
- HuggingFace token (for AutoRAG model downloads)

## Configuration via `.env` file

To run tests from your IDE or locally, create a `.env` file at `tests/pipelines_components/.env`
with the required environment variables (see tables below).
The file is loaded automatically and is in `.gitignore`.
Environment variables set in the shell take precedence over `.env` values.
Sensitive values (API keys, tokens) are masked in test logs.

## Pipeline Modes

### Managed pipelines (default)

The DSPA operator auto-registers pipelines in KFP by display name.
Tests discover them automatically -- no YAML files needed.
Leave `AUTOML_PIPELINE_YAML` and `AUTORAG_PIPELINE_YAML` empty or unset.

### Legacy YAML upload (fallback)

Set `AUTOML_PIPELINE_YAML` or `AUTORAG_PIPELINE_YAML` to a local path or URL
to fall back to manual YAML upload. URLs are downloaded automatically at test startup.

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
# Run all AutoML tests (regression + classification + multiclass)
pytest tests/pipelines_components/automl/ -v

# Run only one task type
pytest tests/pipelines_components/automl/ -k regression
pytest tests/pipelines_components/automl/ -k classification
pytest tests/pipelines_components/automl/ -k multiclass
```

| Variable | Description | Default |
| --- | --- | --- |
| `AUTOML_S3_BUCKET` | External S3 bucket with training data | _(required)_ |
| `AUTOML_REGRESSION_S3_TRAIN_DATA_KEY` | S3 key for regression training CSV | `datasets/regression/regression.csv` |
| `AUTOML_CLASSIFICATION_S3_TRAIN_DATA_KEY` | S3 key for binary classification training CSV | _(required)_ |
| `AUTOML_MULTICLASS_S3_TRAIN_DATA_KEY` | S3 key for multiclass classification training CSV | _(required)_ |
| `AUTOML_TRAIN_DATA_FILE_KEY` | Destination key in DSPA MinIO | `automl-smoke/train.csv` |
| `AUTOML_PIPELINE_YAML` | Legacy: path or URL to pipeline YAML | _(empty = managed mode)_ |
| `AUTOML_PIPELINE_TIMEOUT` | Max wait for pipeline completion (sec) | `1800` |

### AutoRAG

| Variable | Description | Default |
| --- | --- | --- |
| `AUTORAG_S3_BUCKET` | External S3 bucket with test data | _(required)_ |
| `AUTORAG_INPUT_DATA_KEY` | S3 key prefix for input documents | `autorag-smoke/input_data` |
| `AUTORAG_TEST_DATA_KEY` | S3 key for benchmark JSON | `autorag-smoke/benchmark_data.json` |
| `HF_TOKEN` | HuggingFace token for model downloads | _(required)_ |
| `AUTORAG_INFERENCE_MODEL_URI` | Storage URI for inference model | _(required)_ |
| `AUTORAG_INFERENCE_MODEL_NAME` | Inference model name | _(required)_ |
| `AUTORAG_EMBEDDING_MODEL_URI` | Storage URI for embedding model | _(required)_ |
| `AUTORAG_EMBEDDING_MODEL_NAME` | Embedding model name | _(required)_ |
| `AUTORAG_EMBEDDING_MAX_MODEL_LEN` | Embedding model max sequence length | `512` |
| `AUTORAG_PIPELINE_YAML` | Legacy: path or URL to pipeline YAML | _(empty = managed mode)_ |
| `AUTORAG_MAX_RAG_PATTERNS` | Maximum RAG patterns to evaluate | `4` |
| `AUTORAG_OPTIMIZATION_METRIC` | Optimization metric | `faithfulness` |
| `AUTORAG_PIPELINE_TIMEOUT` | Max wait for pipeline completion (sec) | `3600` |

### Managed pipeline overrides (optional)

| Variable | Description | Default |
| --- | --- | --- |
| `MANAGED_PIPELINE_AUTOML_TABULAR` | Display name of AutoML pipeline | `autogluon-tabular-training-pipeline` |
| `MANAGED_PIPELINE_AUTORAG` | Display name of AutoRAG pipeline | `documents-rag-optimization-pipeline` |
| `MANAGED_PIPELINES_IMAGE` | Pin a specific pipelines-components image | _(empty)_ |
| `DSPA_READY_BUFFER_SECONDS` | Buffer after DSPA ready before polling | `30` |
| `MANAGED_PIPELINE_WAIT_TIMEOUT` | Timeout for pipeline discovery (sec) | `300` |

### Debug

| Variable | Description | Default |
| --- | --- | --- |
| `SKIP_TEARDOWN` | Keep all resources after test for debugging | `false` |

## Running Tests

### Run AutoML smoke tests (regression + classification)

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
- Downloads training data from external S3 into DSPA MinIO (dataset per task type)
- Discovers managed pipeline or uploads YAML (legacy mode)
- Submits pipeline run via DSPA REST API (parametrized: regression + classification)
- Cleans up namespace on teardown

### AutoRAG (fully self-contained)

- Creates a dedicated namespace (`autorag-aqa-<hash>`)
- Deploys DSPA with built-in MinIO
- Deploys PostgreSQL, etcd, and Milvus for vector storage
- Deploys vLLM CPU inference and embedding models via KServe InferenceService
- Deploys OGX server (formerly LlamaStack) with rh-dev distribution
- Downloads test data from external S3 into DSPA MinIO
- Discovers managed pipeline or uploads YAML (legacy mode)
- Submits pipeline run via DSPA REST API
- Cleans up namespace on teardown
