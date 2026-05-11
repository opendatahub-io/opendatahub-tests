import os

import pytest
import structlog
from kubernetes.dynamic import DynamicClient

from tests.eval_hub.evalhub_kueue_integration.utils import (
    EvalHubSetupError,
    get_evalhub_route_url,
    verify_evalhub_preflight,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="session", autouse=True)
def evalhub_preflight_verified(
    admin_client: DynamicClient,
    evalhub_base_url: str,
) -> None:
    """Run EvalHub API + Kueue checks before any test under ``tests/eval_hub/``.

    Calls ``verify_evalhub_preflight`` (token, health probe, 404 on fake job id,
    Kueue CRDs/controller). For a standalone run, use ``tests/eval_hub/test_evalhub_preflight.py``.
    """
    try:
        verify_evalhub_preflight(admin_client=admin_client, base_url=evalhub_base_url)
    except EvalHubSetupError as exc:
        pytest.fail(str(exc))


@pytest.fixture(scope="session")
def evalhub_namespace() -> str:
    """Namespace where EvalHub is deployed."""
    return os.environ.get("EVALHUB_NAMESPACE", "redhat-ods-applications")


@pytest.fixture(scope="session")
def evalhub_base_url(admin_client: DynamicClient, evalhub_namespace: str) -> str:
    """Base URL for the EvalHub API, discovered from OpenShift Route."""
    url_override = os.environ.get("EVALHUB_BASE_URL")
    if url_override:
        return url_override
    return get_evalhub_route_url(client=admin_client, namespace=evalhub_namespace)


@pytest.fixture(scope="session")
def evalhub_model_url() -> str:
    """URL of the LLM model endpoint for evaluation jobs."""
    return os.environ.get("EVALHUB_MODEL_URL", "http://llm-service.default.svc.cluster.local:8080/v1")


@pytest.fixture(scope="session")
def evalhub_model_name() -> str:
    """Name of the LLM model for evaluation jobs."""
    return os.environ.get("EVALHUB_MODEL_NAME", "granite-3.1-8b")
