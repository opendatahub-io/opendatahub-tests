import os
import uuid
from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.secret import Secret

import tests.pipelines_components.constants as _constants
from tests.pipelines_components.constants import (
    AUTORAG_DSPA_NAME,
    AUTORAG_DSPA_NAMESPACE,
    AUTORAG_EMBEDDINGS_MODEL,
    AUTORAG_GENERATION_MODEL,
    AUTORAG_INPUT_DATA_KEY,
    AUTORAG_LLAMA_STACK_API_KEY,
    AUTORAG_LLAMA_STACK_URL,
    AUTORAG_MAX_RAG_PATTERNS,
    AUTORAG_OPTIMIZATION_METRIC,
    AUTORAG_PIPELINE_YAML,
    AUTORAG_S3_BUCKET,
    AUTORAG_S3_SECRET_NAME,
    AUTORAG_TEST_DATA_KEY,
    AUTORAG_VECTOR_DB_ID,
)
from tests.pipelines_components.utils import (
    create_pipeline_run,
    delete_pipeline,
    delete_pipeline_run,
    resolve_pipeline_yaml,
    upload_pipeline,
)
from utilities.certificates_utils import create_ca_bundle_file

AUTORAG_LLAMA_STACK_SECRET_NAME: str = "autorag-llama-stack"

_AUTORAG_REQUIRED_ENV = {
    "AUTORAG_PIPELINE_YAML": "Path to compiled AutoRAG pipeline YAML",
    "AUTORAG_DSPA_NAMESPACE": "Namespace with pre-existing DSPA",
    "AUTORAG_S3_SECRET_NAME": "S3 credentials secret name in the DSPA namespace",  # pragma: allowlist secret
    "AUTORAG_LLAMA_STACK_URL": "Llama Stack server base URL",
    "AUTORAG_LLAMA_STACK_API_KEY": "Llama Stack API key / bearer token",  # pragma: allowlist secret
    "AUTORAG_EMBEDDINGS_MODEL": "Embedding model ID",
    "AUTORAG_GENERATION_MODEL": "Generation model ID",
}


@pytest.fixture(scope="session", autouse=True)
def _validate_autorag_env() -> None:
    missing = [f"  {var}: {desc}" for var, desc in _AUTORAG_REQUIRED_ENV.items() if not os.environ.get(var)]
    if missing:
        pytest.skip("AutoRAG smoke test requires environment variables:\n" + "\n".join(missing))

    _constants.AUTORAG_PIPELINE_YAML = resolve_pipeline_yaml(value=AUTORAG_PIPELINE_YAML)


@pytest.fixture(scope="class")
def autorag_dspa(admin_client: DynamicClient) -> DataSciencePipelinesApplication:
    """Pre-existing DataSciencePipelinesApplication for AutoRAG tests."""
    dspa = DataSciencePipelinesApplication(
        client=admin_client,
        name=AUTORAG_DSPA_NAME,
        namespace=AUTORAG_DSPA_NAMESPACE,
    )
    assert dspa.exists, (
        f"DSPA '{AUTORAG_DSPA_NAME}' not found in namespace '{AUTORAG_DSPA_NAMESPACE}'. "
        f"Ensure the DSPA is deployed before running AutoRAG tests."
    )
    return dspa


@pytest.fixture(scope="class")
def dspa_api_url(autorag_dspa: DataSciencePipelinesApplication) -> str:
    """Base URL for the DSPA v2 REST API (from pre-existing DSPA status)."""
    try:
        url = autorag_dspa.instance.status.components.apiServer.externalUrl
    except AttributeError:
        url = None
    assert url, (
        f"DSPA '{AUTORAG_DSPA_NAME}' in namespace '{AUTORAG_DSPA_NAMESPACE}' has no external URL. "
        f"Ensure the DSPA is Ready before running AutoRAG tests."
    )
    return url


@pytest.fixture(scope="class")
def dspa_auth_headers(current_client_token: str) -> dict[str, str]:
    """Authorization headers for DSPA API requests."""
    return {"Authorization": f"Bearer {current_client_token}"}


@pytest.fixture(scope="class")
def dspa_ca_bundle_file(admin_client: DynamicClient) -> str:
    """CA bundle file for TLS verification against the DSPA Route."""
    return create_ca_bundle_file(client=admin_client)


@pytest.fixture(scope="class")
def autorag_llama_stack_secret(
    admin_client: DynamicClient,
) -> Generator[Secret, Any, Any]:
    """Kubernetes secret with Llama Stack connection details for AutoRAG pipeline components."""
    with Secret(
        client=admin_client,
        name=AUTORAG_LLAMA_STACK_SECRET_NAME,
        namespace=AUTORAG_DSPA_NAMESPACE,
        string_data={
            "LLAMA_STACK_CLIENT_BASE_URL": AUTORAG_LLAMA_STACK_URL,
            "LLAMA_STACK_CLIENT_API_KEY": str(AUTORAG_LLAMA_STACK_API_KEY),
        },
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def autorag_pipeline_id(
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
) -> Generator[str, Any, Any]:
    """Upload the AutoRAG pipeline YAML and yield the pipeline ID. Deletes the pipeline on teardown."""
    run_suffix = uuid.uuid4().hex[:8]
    pipeline_id = upload_pipeline(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        pipeline_yaml_path=AUTORAG_PIPELINE_YAML,
        pipeline_name=f"autorag-smoke-{run_suffix}",
        ca_bundle=dspa_ca_bundle_file,
    )
    yield pipeline_id
    delete_pipeline(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        pipeline_id=pipeline_id,
        ca_bundle=dspa_ca_bundle_file,
    )


@pytest.fixture(scope="class")
def autorag_run_id(
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
    autorag_pipeline_id: str,
    autorag_llama_stack_secret: Secret,
) -> Generator[str, Any, Any]:
    """Create a pipeline run and yield the run ID. Deletes the run on teardown."""
    parameters: dict = {
        "input_data_secret_name": AUTORAG_S3_SECRET_NAME,
        "input_data_bucket_name": AUTORAG_S3_BUCKET,
        "input_data_key": AUTORAG_INPUT_DATA_KEY,
        "test_data_secret_name": AUTORAG_S3_SECRET_NAME,
        "test_data_bucket_name": AUTORAG_S3_BUCKET,
        "test_data_key": AUTORAG_TEST_DATA_KEY,
        "llama_stack_secret_name": autorag_llama_stack_secret.name,
        "optimization_max_rag_patterns": AUTORAG_MAX_RAG_PATTERNS,
        "optimization_metric": AUTORAG_OPTIMIZATION_METRIC,
    }

    if AUTORAG_EMBEDDINGS_MODEL:
        parameters["embeddings_models"] = [AUTORAG_EMBEDDINGS_MODEL]
    if AUTORAG_GENERATION_MODEL:
        parameters["generation_models"] = [AUTORAG_GENERATION_MODEL]
    if AUTORAG_VECTOR_DB_ID:
        parameters["llama_stack_vector_database_id"] = AUTORAG_VECTOR_DB_ID

    run_id = create_pipeline_run(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        pipeline_id=autorag_pipeline_id,
        run_name=f"autorag-smoke-{uuid.uuid4().hex[:8]}",
        parameters=parameters,
        ca_bundle=dspa_ca_bundle_file,
    )
    yield run_id
    delete_pipeline_run(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        run_id=run_id,
        ca_bundle=dspa_ca_bundle_file,
    )
