from __future__ import annotations

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_model_ref import MaaSModelRef

from tests.ai_gateway.models_as_a_service.maas_api_key.utils import (
    MAAS_GATEWAY_AUTH_POLICY_NAME,
    wait_for_auth_policy_accepted,
)
from utilities.constants import MAAS_GATEWAY_NAMESPACE
from utilities.plugins.constant import OpenAIEnpoints

LOGGER = structlog.get_logger(name=__name__)

CHAT_COMPLETIONS = OpenAIEnpoints.CHAT_COMPLETIONS


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
)
class TestGatewayDenyByDefault:
    """Verify maas-gateway-auth denies access to unconfigured models."""

    def test_gateway_default_auth_in_gateway_namespace_and_accepted(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Given MaaS gateway auth is reconciled, when reading maas-gateway-auth in the gateway namespace,
        then the AuthPolicy is Accepted (post-#912 singleton; legacy gateway-default-auth is not used).
        """
        gateway_namespace = MAAS_GATEWAY_NAMESPACE

        wait_for_auth_policy_accepted(
            admin_client=admin_client,
            policy_name=MAAS_GATEWAY_AUTH_POLICY_NAME,
            namespace=gateway_namespace,
            reconciliation_hint=(
                "Ensure MaaS is enabled and a MaaSAuthPolicy exists to bootstrap "
                f"{MAAS_GATEWAY_AUTH_POLICY_NAME} in {gateway_namespace}."
            ),
        )

        LOGGER.info(
            f"{MAAS_GATEWAY_AUTH_POLICY_NAME} deployed to '{gateway_namespace}' "
            "and Accepted/Enforced"
        )

    def test_unconfigured_model_denies_unauthenticated_request(
        self,
        request_session_http: requests.Session,
        maas_scheme: str,
        maas_host: str,
        unconfigured_model_ref: MaaSModelRef,
    ) -> None:
        """Verify a model without MaaSAuthPolicy rejects unauthenticated requests with 403."""
        inference_url = f"{maas_scheme}://{maas_host}/llm/{unconfigured_model_ref.name}{CHAT_COMPLETIONS}"

        response = request_session_http.post(
            url=inference_url,
            headers={"Content-Type": "application/json"},
            json={
                "model": "any",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
            },
            timeout=60,
        )

        assert response.status_code in (403, 404), (
            f"Unconfigured model accepted unauthenticated "
            f"request. Expected 403 (deny-by-default) or 404 (no route), "
            f"got {response.status_code}: {response.text[:200]}"
        )

        LOGGER.info(
            f"Unconfigured model '{unconfigured_model_ref.name}' correctly denied "
            f"unauthenticated request with {response.status_code}"
        )
