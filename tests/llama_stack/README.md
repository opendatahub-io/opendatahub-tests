# Llama Stack Integration Tests

This directory contains OpenShift AI integration tests for Llama Stack components. These tests validate the functionality of Llama Stack APIs and providers when deployed on OpenShift AI using the [Red Hat LlamaStack Distribution](https://github.com/opendatahub-io/llama-stack-distribution).

## Directory Structure

The folder structure is based on the upstream Llama Stack integration tests, available at [llamastack/llama-stack/tests/integration](https://github.com/llamastack/llama-stack/tree/main/tests/integration). Each subfolder maps to an endpoint in the Llama Stack API. For more information about the available endpoints, see the [Llama Stack API documentation](https://llamastack.github.io/docs/concepts/apis) and the [Python SDK Reference](https://llamastack.github.io/docs/references/python_sdk_reference).

### Current Test Suites

- **`eval/`** - Evaluation provider tests (LM Eval)
- **`inference/`** - Inference functionality tests
- **`models/`** - Model management and catalog tests
- **`operator/`** - Tests for the llama-stack-k8s-operator and Red Hat LlamaStack Distribution image
- **`responses/`** - Response handling and validation tests
- **`safety/`** - Safety and guardrails tests (TrustyAI FMS provider)
- **`vector_io/`** - Vector store and I/O tests

## Test Markers

Each test suite should have a marker indicating the component/team name. The marker format is `@pytest.mark.<component_name>`. For example:

```python
@pytest.mark.rag
def test_vector_stores_functionality():
    # Test implementation
```

## Adding Support for New API Providers

To add support for testing new LlamaStack API providers (e.g., a new vector_io provider), create deployment fixtures in the appropriate `/tests/fixtures/` file, update the corresponding provider factory function to return the required environment variables, and add the new provider as a test parameter in the relevant test files. For example, to add a new vector_io provider, add deployment fixtures in `/tests/fixtures/vector_io.py`, update the `vector_io_provider_deployment_config_factory` function, and add a new `pytest.param` entry in `/tests/llama_stack/vector_io/test_vector_stores.py`.

### Available Team Markers  (to be expanded)

- `@pytest.mark.llama_stack` - LlamaStack Core team tests
- `@pytest.mark.model_explainability` - AI Safety team tests
- `@pytest.mark.rag` - RAG team tests

## Running Tests

### Required environment variables

LlamaStack tests require setting the following environment variables (for example in a `.env` file at the root folder).

> **Note:** Most of these environment variables are added as `env_vars` in the LlamaStackDistribution CR, as they are required to configure the Red Hat LlamaStack Distribution's [run.yaml](https://github.com/opendatahub-io/llama-stack-distribution/blob/main/distribution/run.yaml).

```bash
OC_BINARY_PATH=/usr/local/sbin/oc                 # Optional
LLS_CLIENT_VERIFY_SSL=false                       # Optional

# Core Inference Configuration
LLS_CORE_VLLM_URL=<LLAMA-3.2-3b-ENDPOINT>/v1  (ends with /v1)
LLS_CORE_INFERENCE_MODEL=<LLAMA-3.2-3b-MODEL_NAME>
LLS_CORE_VLLM_API_TOKEN=<LLAMA-3.2-3b-TOKEN>
LLS_CORE_VLLM_MAX_TOKENS=16384                   # Optional
LLS_CORE_VLLM_TLS_VERIFY=true                    # Optional

# Core Embedding Configuration
LLS_CORE_EMBEDDING_MODEL=nomic-embed-text-v1-5    # Optional
LLS_CORE_EMBEDDING_PROVIDER_MODEL_ID=nomic-embed-text-v1-5  # Optional
LLS_CORE_VLLM_EMBEDDING_URL=<EMBEDDING-ENDPOINT>/v1  # Optional
LLS_CORE_VLLM_EMBEDDING_API_TOKEN=<EMBEDDING-TOKEN>  # Optional
LLS_CORE_VLLM_EMBEDDING_MAX_TOKENS=8192          # Optional
LLS_CORE_VLLM_EMBEDDING_TLS_VERIFY=true          # Optional

# Vector I/O Configuration
LLS_VECTOR_IO_MILVUS_IMAGE=<CUSTOM-MILVUS-IMAGE>  # Optional
LLS_VECTOR_IO_MILVUS_TOKEN=<CUSTOM-MILVUS-TOKEN>  # Optional
LLS_VECTOR_IO_ETCD_IMAGE=<CUSTOM-ETCD-IMAGE>      # Optional
LLS_VECTOR_IO_PGVECTOR_IMAGE=<CUSTOM-PGVECTOR-IMAGE> # Optional
LLS_VECTOR_IO_PGVECTOR_USER=<CUSTOM-PGVECTOR-USER> # Optional
LLS_VECTOR_IO_PGVECTOR_PASSWORD=<CUSTOM-PGVECTOR-PASSWORD> # Optional
LLS_VECTOR_IO_QDRANT_IMAGE=<CUSTOM-QDRANT-IMAGE> # Optional
LLS_VECTOR_IO_QDRANT_API_KEY=<CUSTOM-QDRANT-API-KEY> # Optional
LLS_VECTOR_IO_QDRANT_URL=<QDRANT_URL_WITH_PROTOCOL> # Optional

# Red Hat Llama Stack Distribution requires PostgreSQL (replacing SQLite)
LLS_VECTOR_IO_POSTGRES_IMAGE=<CUSTOM-POSTGRES-IMAGE> # Optional
LLS_VECTOR_IO_POSTGRESQL_USER=ps_user            # Optional
LLS_VECTOR_IO_POSTGRESQL_PASSWORD=ps_password    # Optional

# Files Provider Configuration
LLS_FILES_S3_AUTO_CREATE_BUCKET=true             # Optional
```

### Run All Llama Stack Tests

To run all tests in the `/tests/llama_stack` directory:

```bash
pytest tests/llama_stack/
```

### Run Tests by Component/Team

To run tests for a specific team (e.g. rag):

```bash
pytest -m rag tests/llama_stack/
```

### Run Tests for a llama-stack API

To run tests for a specific API (e.g., vector_io):

```bash
pytest tests/llama_stack/vector_io
```

### Run Tests with Additional Markers

You can combine team markers with other pytest markers:

```bash
# Run only smoke tests for rag
pytest -m "rag and smoke" tests/llama_stack/

# Run all rag tests except the ones requiring a GPU
pytest -m "rag and not gpu" tests/llama_stack/
```

## Related Testing Repositories

### Llama Stack K8s Operator

The `operator/` folder contains tests specifically for the llama-stack-k8s-operator and the Red Hat LlamaStack Distribution image. These tests validate the operator's functionality and the distribution image when deployed on OpenShift AI.

There is also a separate operator repository with additional tests related to llama-stack-operator verifications. The main end-to-end (e2e) tests for the operator are implemented in the [llama-stack-k8s-operator repository](https://github.com/llamastack/llama-stack-k8s-operator/tree/main/tests/e2e).

### Test Scope Guidelines

Tests in this repository should be specific to OpenDataHub and OpenShift AI, such as:

- Verifying that LlamaStack components included in the builds work as expected
- Testing particular scenarios like ODH/RHOAI upgrades
- Validating OpenShift AI-specific configurations and integrations
- Testing Red Hat LlamaStack Distribution-specific features

For generic llama-stack testing, it is preferred to contribute to the upstream llama-stack [unit](https://github.com/llamastack/llama-stack/tree/main/tests/unit) and [integration](https://github.com/llamastack/llama-stack/tree/main/tests/integration) tests.

## Red Hat LlamaStack Distribution

For information about the APIs and Providers available in the Red Hat LlamaStack Distribution image, see the [distribution documentation](https://github.com/opendatahub-io/llama-stack-distribution/tree/main/distribution).

## Additional Resources

- [Llama Stack Documentation](https://llamastack.github.io/docs/)
- [OpenDataHub Documentation](https://opendatahub.io/docs)
- [OpenShift AI Documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed)
