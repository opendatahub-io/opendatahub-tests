"""Fixtures shared by the remote::gemini provider test suite.

The Gemini tests reuse the standard OgxServer stack from ``tests/ogx/conftest.py``
(namespace, secret, server, route, client). They enable the provider by
parametrizing the ``ogx_server`` fixture indirectly with ``{"enable_gemini": True}``,
which injects ``GEMINI_API_KEY`` into the pod (see ``build_ogx_server_config``).

The fixtures here add Gemini-specific concerns on top of that stack: skipping the
suite when no API key is configured, resolving Gemini model ids from the running
distribution, and exposing the OgxServer pod for environment/log inspection.
"""

from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ogx_client import OgxClient

from tests.ogx.constants import GEMINI_API_KEY, OGX_CORE_POD_FILTER
from tests.ogx.gemini.utils import resolve_gemini_model_id
from utilities.general import wait_for_pods_by_labels

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="session", autouse=True)
def skip_if_no_gemini_api_key() -> None:
    """Skip the entire Gemini suite when no Gemini API key is configured.

    Every remote::gemini test requires a real key so the provider activates and
    can authenticate to the Gemini API. Running without one would only produce
    misleading failures, so the whole suite is skipped instead.
    """
    if not GEMINI_API_KEY:
        pytest.skip(
            reason="No Gemini API key configured; set OGX_CORE_GEMINI_API_KEY (or GEMINI_API_KEY) to run these tests"
        )


@pytest.fixture(scope="class")
def gemini_model_id(ogx_client: OgxClient) -> str:
    """The id of a Gemini LLM model served via the remote::gemini provider.

    Resolved dynamically from ``GET /v1/models`` (or an explicit override in
    constants). Skips the requesting test if the Gemini provider registered no
    LLM model.
    """
    model_id = resolve_gemini_model_id(ogx_client=ogx_client, model_type="llm")
    if not model_id:
        pytest.skip(reason="No Gemini LLM model is registered by the remote::gemini provider")
    LOGGER.info(f"Resolved Gemini LLM model: {model_id}")
    return model_id


@pytest.fixture(scope="class")
def gemini_embedding_model_id(ogx_client: OgxClient) -> str:
    """The id of a Gemini embedding model served via the remote::gemini provider.

    Resolved dynamically from ``GET /v1/models`` (or an explicit override in
    constants). Skips the requesting test if the Gemini provider registered no
    embedding model.
    """
    model_id = resolve_gemini_model_id(ogx_client=ogx_client, model_type="embedding")
    if not model_id:
        pytest.skip(reason="No Gemini embedding model is registered by the remote::gemini provider")
    LOGGER.info(f"Resolved Gemini embedding model: {model_id}")
    return model_id


@pytest.fixture(scope="class")
def ogx_gemini_pod(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ogx_server: Any,
) -> Generator[Pod, Any, Any]:
    """The single OgxServer pod, used for environment-variable and log inspection."""
    pod = wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=unprivileged_model_namespace.name,
        label_selector=OGX_CORE_POD_FILTER,
        expected_num_pods=1,
    )[0]
    yield pod
