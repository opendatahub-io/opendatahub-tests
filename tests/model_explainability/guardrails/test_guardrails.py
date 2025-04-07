import http

import pytest
import requests
from timeout_sampler import retry

from utilities.constants import Timeout


@pytest.mark.parametrize(
    "model_namespace, minio_data_connection",
    [
        pytest.param(
            {"name": "test-guardrails"},
            {"bucket": "llms"},
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
class TestGuardrails:
    def test_guardrails_health_endpoint(self, admin_client, qwen_llm_model, guardrails_orchestrator_health_route):
        # It takes a bit for the endpoint to come online, so we retry for a brief period of time
        @retry(wait_timeout=Timeout.TIMEOUT_1MIN, sleep=1)
        def check_health_endpoint():
            response = requests.get(url=f"https://{guardrails_orchestrator_health_route.host}/health", verify=False)
            if response.status_code == http.HTTPStatus.OK:
                return response
            return False

        response = check_health_endpoint()
        assert "fms-guardrails-orchestr8" in response.text

    def test_guardrails_info_endpoint(self, admin_client, qwen_llm_model, guardrails_orchestrator_health_route):
        response = requests.get(url=f"https://{guardrails_orchestrator_health_route.host}/info", verify=False)
        assert response.status_code == http.HTTPStatus.OK

        healthy_status = "HEALTHY"
        response_data = response.json()
        assert response_data["services"]["chat_generation"]["status"] == healthy_status
        assert response_data["services"]["regex"]["status"] == healthy_status
