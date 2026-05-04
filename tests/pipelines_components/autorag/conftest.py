import os
from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.secret import Secret

from tests.pipelines_components.constants import (
    AUTORAG_DSPA_NAME,
    AUTORAG_DSPA_NAMESPACE,
    AUTORAG_LLAMA_STACK_API_KEY,
    AUTORAG_LLAMA_STACK_URL,
    AUTORAG_PIPELINE_YAML,
)
from tests.pipelines_components.utils import resolve_pipeline_yaml
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

    import tests.pipelines_components.constants as _constants

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
