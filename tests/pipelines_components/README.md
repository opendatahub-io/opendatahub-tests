# Pipelines Components Smoke Tests

Smoke tests for reusable Kubeflow Pipelines assets from
[pipelines-components](https://github.com/opendatahub-io/pipelines-components).
Each test submits a pipeline run to a DataSciencePipelinesApplication (DSPA) and asserts successful completion.

## Test Suites

- **`automl/`** - AutoGluon Tabular Training pipeline smoke test
- **`autorag/`** - Documents RAG Optimization pipeline smoke test

## Prerequisites

### Common (all suites)

- OpenShift cluster with RHOAI/ODH installed

### AutoML

- `aipipelines` component enabled (the test enables it automatically via DSC patching)

### AutoRAG

AutoRAG uses a **pre-existing DSPA** with pre-loaded test data. Before running:

1. **Deploy a DSPA** in a namespace with test data and benchmark JSON in S3
2. **Deploy a Llama Stack server** with RAG capabilities (chat model, embedding model, vector store)
3. **Register models and vector database** in Llama Stack
4. **Configure `.env`** with connection details (see below)

## Configuration via `.env` file

AutoRAG tests load environment variables from `tests/pipelines_components/.env` automatically.
Copy the example and fill in your values:

```bash
cp tests/pipelines_components/.env.example tests/pipelines_components/.env
```

The `.env` file is in `.gitignore` and will not be committed.
Environment variables set in the shell take precedence over `.env` values.
Sensitive values (API keys, tokens) are masked in test logs.

## Pipeline YAMLs

Compiled KFP pipeline YAMLs are **not bundled** in the repo. Provide a local file path
or a URL via the `.env` file. URLs are downloaded automatically at test startup.

Sources:

- **AutoML**: [red-hat-data-services/pipelines-components](https://github.com/red-hat-data-services/pipelines-components)
  (`pipelines/training/automl/autogluon_tabular_training_pipeline/pipeline.yaml`)
- **AutoRAG**: [red-hat-data-services/red-hat-ai-examples](https://github.com/red-hat-data-services/red-hat-ai-examples/tree/main/examples/autorag)
  (`examples/autorag/pipelines/pipeline.yaml`)

Example `.env` entries using raw GitHub URLs:

```shell
AUTOML_PIPELINE_YAML=https://raw.githubusercontent.com/red-hat-data-services/pipelines-components/main/pipelines/training/automl/autogluon_tabular_training_pipeline/pipeline.yaml
AUTORAG_PIPELINE_YAML=https://raw.githubusercontent.com/red-hat-data-services/red-hat-ai-examples/main/examples/autorag/pipelines/pipeline.yaml
```

## Environment Variables

### AutoML (required env var marked with *)

| Variable                     | Description                            | Default                                |
| ---------------------------- | -------------------------------------- | -------------------------------------- |
| `AUTOML_PIPELINE_YAML` *     | Path or URL to compiled pipeline YAML  | _(none — test skips if unset)_         |
| `AUTOML_TRAIN_DATA_FILE_KEY` | S3 key for training CSV                | `automl-smoke/train.csv`               |
| `AUTOML_LABEL_COLUMN`        | Target column name in CSV              | `target`                               |
| `AUTOML_TASK_TYPE`           | AutoGluon task type                    | `binary`                               |
| `AUTOML_TOP_N`               | Number of top models to refit          | `1`                                    |
| `AUTOML_PIPELINE_TIMEOUT`    | Max wait for pipeline completion (sec) | `1800`                                 |
| `PIPELINE_POLL_INTERVAL`     | Seconds between status polls           | `30`                                   |
| `DSPA_MINIO_IMAGE`           | MinIO server image for DSPA            | `quay.io/opendatahub/minio:RELEASE...` |
| `MINIO_MC_IMAGE`             | MinIO client image for data upload     | `quay.io/minio/mc@sha256:...`          |

### AutoRAG (required env vars marked with *)

| Variable | Description | Default |
| --- | --- | --- |
| `AUTORAG_PIPELINE_YAML` * | Path or URL to compiled pipeline YAML | _(none — test skips if unset)_ |
| `AUTORAG_DSPA_NAMESPACE` * | Namespace with pre-existing DSPA | _(none — test skips if unset)_ |
| `AUTORAG_DSPA_NAME` | DSPA resource name | `dspa` |
| `AUTORAG_S3_SECRET_NAME` * | S3 credentials secret name in the DSPA namespace | _(none — test skips if unset)_ |
| `AUTORAG_S3_BUCKET` | S3 bucket name | `mlpipeline` |
| `AUTORAG_LLAMA_STACK_URL` * | Llama Stack server base URL | _(none — test skips if unset)_ |
| `AUTORAG_LLAMA_STACK_API_KEY` * | Llama Stack API key / bearer token | _(none — test skips if unset)_ |
| `AUTORAG_EMBEDDINGS_MODEL` * | Embedding model ID | _(none — test skips if unset)_ |
| `AUTORAG_GENERATION_MODEL` * | Generation model ID | _(none — test skips if unset)_ |
| `AUTORAG_VECTOR_DB_ID` | Vector database ID registered in Llama Stack | `""` (omitted if empty) |
| `AUTORAG_INPUT_DATA_KEY` | S3 key prefix for input documents | `autorag-smoke/input_data` |
| `AUTORAG_TEST_DATA_KEY` | S3 key for benchmark JSON | `autorag-smoke/benchmark_data.json` |
| `AUTORAG_MAX_RAG_PATTERNS` | Maximum RAG patterns to evaluate | `1` |
| `AUTORAG_OPTIMIZATION_METRIC` | Optimization metric | `faithfulness` |
| `AUTORAG_PIPELINE_TIMEOUT` | Max wait for pipeline completion (sec) | `3600` |

## Running Tests

### Run AutoML smoke test

```bash
uv run pytest tests/pipelines_components/automl/ -m smoke -v -s --junitxml=results_xunit.xml
```

### Run AutoRAG smoke test

```bash
uv run pytest tests/pipelines_components/autorag/ -m smoke -v -s --junitxml=results_xunit.xml
```

### Run all pipelines-components smoke tests

```bash
uv run pytest tests/pipelines_components/ -m smoke -v -s --junitxml=results_xunit.xml
```

## Infrastructure

### AutoML (fully self-contained)

- A dedicated OpenShift namespace (cleaned up after test)
- DSPA with built-in MinIO for object storage
- Synthetic training data uploaded to MinIO
- Pipeline submitted via DSPA REST API through the external Route

No external S3 or pre-existing pipeline server is required.

### AutoRAG (uses pre-existing DSPA)

- Connects to a pre-deployed DSPA in an existing namespace
- Test data (documents + benchmark JSON) must be pre-loaded in S3
- Creates a Kubernetes secret for Llama Stack credentials (cleaned up after test)
- Pipeline submitted via DSPA REST API through the external Route

Requires an external Llama Stack server with LLM, embedding models, and vector DB.
The test skips automatically if required env vars are not set.
