# Shared Test Fixtures

This directory contains shared pytest fixtures that are used across multiple test modules. These fixtures are automatically loaded via pytest's plugin mechanism, registered in `/tests/conftest.py`.

## Directory Structure

```text
fixtures/
├── files.py           # File storage provider fixtures
├── guardrails.py      # Guardrails orchestrator infrastructure fixtures
├── inference.py       # Inference service and serving runtime fixtures
├── trustyai.py        # TrustyAI operator and DSC configuration fixtures
└── vector_io.py       # Vector database provider deployment fixtures
```

### Fixture Modules

- **`files.py`** - Factory fixture for configuring file storage providers (local, S3/MinIO)
- **`guardrails.py`** - Fixtures for deploying and configuring the Guardrails Orchestrator, including pods, routes, health checks, and gateway configuration
- **`inference.py`** - Fixtures for vLLM CPU serving runtimes, InferenceServices (Qwen), LLM-d inference simulator, and KServe controller configuration
- **`trustyai.py`** - Fixtures for TrustyAI operator deployment and DataScienceCluster LMEval configuration
- **`vector_io.py`** - Factory fixture for deploying vector database providers (Milvus, Faiss, PGVector, Qdrant) with their backing services and configuration

## Registration

All fixture modules are registered as pytest plugins in `/tests/conftest.py`:

```python
pytest_plugins = [
    "tests.fixtures.inference",
    "tests.fixtures.guardrails",
    "tests.fixtures.trustyai",
    "tests.fixtures.vector_io",
    "tests.fixtures.files",
]
```

## Usage

Fixtures are automatically available to all tests. Factory fixtures accept parameters via `pytest.mark.parametrize` with `indirect=True`.

### Vector I/O Provider Example

```python
@pytest.mark.parametrize(
    "vector_io_provider_deployment_config_factory",
    ["milvus", "pgvector", "qdrant-remote"],
    indirect=True,
)
def test_with_vector_db(vector_io_provider_deployment_config_factory):
    # Fixture deploys the provider and returns env var configuration
    ...
```

### Supported Vector I/O Providers

| Provider        | Type   | Description                                 |
| --------------- | ------ | ------------------------------------------- |
| `milvus`        | Local  | In-memory Milvus (no external dependencies) |
| `milvus-remote` | Remote | Milvus standalone with etcd backend         |
| `faiss`         | Local  | Facebook AI Similarity Search (in-memory)   |
| `pgvector`      | Local  | PostgreSQL with pgvector extension          |
| `qdrant-remote` | Remote | Qdrant vector database                      |

### Supported File Providers

| Provider | Description                        |
| -------- | ---------------------------------- |
| `local`  | Local filesystem storage (default) |
| `s3`     | S3/MinIO remote object storage     |

## Adding New Fixtures

When adding shared fixtures, place them in the appropriate module file (or create a new one), and register the new module in `/tests/conftest.py` under `pytest_plugins`. Follow the project's fixture conventions: use noun-based names, narrowest appropriate scope, and context managers for resource lifecycle.
