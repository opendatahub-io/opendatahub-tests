# Llama Stack Integration Tests

This directory contains OpenShift AI integration tests for Llama Stack components. These tests validate the functionality of Llama Stack APIs and providers when deployed on OpenShift AI using the [Red Hat LlamaStack Distribution](https://github.com/opendatahub-io/llama-stack-distribution).

## Directory Structure

The folder structure is based on the upstream Llama Stack integration tests, available at [llamastack/llama-stack/tests/integration](https://github.com/llamastack/llama-stack/tree/main/tests/integration). Each subfolder maps to an endpoint in the Llama Stack API. For more information about the available endpoints, see the [Llama Stack API documentation](https://llamastack.github.io/docs/concepts/apis) and the [Python SDK Reference](https://llamastack.github.io/docs/references/python_sdk_reference).

### Current Test Suites

- **`agents/`** - Agent functionality tests
- **`eval/`** - Evaluation provider tests (LM Eval)
- **`inference/`** - Inference functionality tests
- **`models/`** - Model management and catalog tests
- **`operator/`** - Tests for the llama-stack-k8s-operator and Red Hat LlamaStack Distribution image
- **`responses/`** - Response handling and validation tests
- **`safety/`** - Safety and guardrails tests (TrustyAI FMS provider)
- **`vector_io/`** - Vector store and I/O tests

## Test Markers

Each test suite should have a marker indicating the team that owns it. The marker format is `@pytest.mark.team_<team_name>`. For example:

```python
@pytest.mark.team_rag
def test_vector_stores_functionality():
    # Test implementation
```

### Available Team Markers  (to be expanded)

- `@pytest.mark.team_ai_safety` - Evaluation team tests
- `@pytest.mark.team_llama_stack` - LlamaStack Core team tests
- `@pytest.mark.team_rag` - RAG team tests


## Running Tests

### Run All Llama Stack Tests

To run all tests in the `/tests/llama_stack` directory:

```bash
pytest tests/llama_stack/
```

### Run Tests by Team Marker

To run tests for a specific team (e.g., agents team):

```bash
pytest -m team_rag tests/llama_stack/
```

### Run Tests with Additional Markers

You can combine team markers with other pytest markers:

```bash
# Run only smoke tests for the rag team
pytest -m "team_rag and smoke" tests/llama_stack/

# Run all team-rag tests except slow ones
pytest -m "team_rag and not slow" tests/llama_stack/
```

## Related Testing Repositories

### Llama Stack K8s Operator

The `operator/` folder contains tests specifically for the llama-stack-k8s-operator and the Red Hat LlamaStack Distribution image. These tests validate the operator's functionality and the distribution image when deployed on OpenShift AI.

There is also a separate operator repository with additional tests related to llama-stack-operator verifications. The main end-to-end (e2e) tests for the operator are implemented in the [llama-stack-k8s-operator repository](https://github.com/llamastack/llama-stack-k8s-operator/tree/main/tests/e2e).

## Shift-Left Testing Practices

To follow shift-left practices, we encourage contributing new unit and integration tests to the upstream [llama-stack integration tests](https://github.com/llamastack/llama-stack/tree/main/tests/integration) because:

- They are run to verify llama-stack releases
- They are run when building the Red Hat LlamaStack Distribution (see [run_integration_tests.sh](https://github.com/opendatahub-io/llama-stack-distribution/blob/main/tests/run_integration_tests.sh))

### Test Scope Guidelines

Tests in this repository should be specific to OpenDataHub and OpenShift AI, such as:

- Verifying that LlamaStack components included in the builds work as expected
- Testing particular scenarios like ODH/RHOAI upgrades
- Validating OpenShift AI-specific configurations and integrations
- Testing Red Hat LlamaStack Distribution-specific features

## Red Hat LlamaStack Distribution

For information about the APIs and Providers available in the Red Hat LlamaStack Distribution image, see the [distribution documentation](https://github.com/opendatahub-io/llama-stack-distribution/tree/main/distribution).

## Additional Resources

- [Llama Stack Documentation](https://llamastack.github.io/docs/)
- [OpenDataHub Documentation](https://opendatahub.io/docs)
- [OpenShift AI Documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed)
