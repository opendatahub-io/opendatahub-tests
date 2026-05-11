from __future__ import annotations

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.external_model.utils import EXTERNAL_MODEL_NAME
from tests.model_serving.maas_billing.utils import build_maas_headers

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "external_model_cr",
    "external_model_ref",
    "external_model_auth_policy",
    "external_model_subscription",
)
class TestExternalModelEgress:
    """Verify requests with a valid API key are forwarded to the external endpoint."""

    @pytest.mark.tier1
    @pytest.mark.skip_on_disconnected
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_request_forwarded_to_external_endpoint(
        self,
        request_session_http: requests.Session,
        external_model_inference_url: str,
        external_model_api_key: str,
    ) -> None:
        """Given a valid API key, when a chat request is sent, then it reaches the external endpoint."""
        headers = build_maas_headers(token=external_model_api_key)
        response = request_session_http.post(
            url=external_model_inference_url,
            headers=headers,
            json={
                "model": EXTERNAL_MODEL_NAME,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
            },
            timeout=60,
        )

        assert response.status_code not in (401, 403), (
            f"Request was blocked by auth (HTTP {response.status_code}). "
            f"Expected the request to reach the external endpoint."
        )
        LOGGER.info(f"Egress connectivity confirmed: external endpoint returned HTTP {response.status_code}")
