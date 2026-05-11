from __future__ import annotations

import pytest
import requests
import structlog
from ocp_resources.maas_model_ref import MaaSModelRef
from pytest_testconfig import config as py_config

from tests.model_serving.maas_billing.maas_api_key.utils import get_auth_policy_condition
from utilities.constants import MAAS_GATEWAY_NAMESPACE
from utilities.plugins.constant import OpenAIEnpoints

LOGGER = structlog.get_logger(name=__name__)

GATEWAY_DEFAULT_AUTH_NAME = "gateway-default-auth"
CHAT_COMPLETIONS = OpenAIEnpoints.CHAT_COMPLETIONS


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
)
class TestGatewayDenyByDefault:
    """Verify gateway-default-auth denies access to unconfigured models."""

    def test_gateway_default_auth_in_gateway_namespace_and_accepted(
        self,
        admin_client,
    ) -> None:
        """Verify gateway-default-auth exists in the gateway namespace and is Accepted."""
        gateway_namespace = MAAS_GATEWAY_NAMESPACE
        applications_namespace: str = py_config["applications_namespace"]

        accepted_condition = get_auth_policy_condition(
            admin_client=admin_client,
            policy_name=GATEWAY_DEFAULT_AUTH_NAME,
            namespace=gateway_namespace,
            condition_type="Accepted",
        )

        assert accepted_condition is not None, (
            f"{GATEWAY_DEFAULT_AUTH_NAME} AuthPolicy not found in "
            f"namespace '{gateway_namespace}' or has no 'Accepted' condition. "
            f"It may be deployed to '{applications_namespace}' instead."
        )
        assert accepted_condition.get("status") == "True", (
            f"{GATEWAY_DEFAULT_AUTH_NAME} is not Accepted: "
            f"reason={accepted_condition.get('reason')}, "
            f"message={accepted_condition.get('message')}"
        )

        LOGGER.info(f"{GATEWAY_DEFAULT_AUTH_NAME} correctly deployed to '{gateway_namespace}' and Accepted")

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

        assert response.status_code == 403, (
            f"Unconfigured model accepted unauthenticated "
            f"request. Expected 403, got {response.status_code}: "
            f"{response.text[:200]}"
        )

        LOGGER.info(
            f"Unconfigured model '{unconfigured_model_ref.name}' correctly denied unauthenticated request with 403"
        )
