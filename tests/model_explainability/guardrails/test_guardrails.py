import http

import pytest
import requests
from simple_logger.logger import get_logger
from timeout_sampler import retry

from tests.model_explainability.guardrails.utils import (
    verify_builtin_detector_unsuitable_input_response,
    verify_negative_detection_response,
    verify_builtin_detector_unsuitable_output_response,
    get_auth_headers,
    get_chat_payload,
)
from tests.model_explainability.utils import validate_tai_component_images
from utilities.constants import Timeout, MinIo
from utilities.plugins.constant import OpenAIEnpoints

LOGGER = get_logger(name=__name__)
EXAMPLE_EMAIL_ADDRESS: str = "myemail@domain.com"
INPUT_WITH_EMAIL_ADDRESS: str = f"This is my email address: {EXAMPLE_EMAIL_ADDRESS}, just answer ACK."
PII_ENDPOINT: str = "/pii"


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-guardrails-image"},
        )
    ],
    indirect=True,
)
@pytest.mark.smoke
def test_validate_guardrails_orchestrator_images(guardrails_orchestrator_pod, trustyai_operator_configmap):
    """Test to verify Guardrails pod images.
    Checks if the image tag from the ConfigMap is used within the Pod and if it's pinned using a sha256 digest.
    """
    validate_tai_component_images(pod=guardrails_orchestrator_pod, tai_operator_configmap=trustyai_operator_configmap)


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection",
    [
        pytest.param(
            {"name": "test-guardrails"},
            MinIo.PodConfig.QWEN_MINIO_CONFIG,
            {"bucket": "llms"},
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.smoke
class TestGuardrailsOrchestratorWithBuiltInDetectors:
    """
    Tests that the basic functionality of the GuardrailsOrchestrator work properly with the built-in (regex) detectors.
        1. Deploy an LLM using vLLM as a SR.
        2. Deploy the Guardrails Orchestrator.
        3. Check that the Orchestrator is healthy by querying the health and info endpoints of its /health route.
        4. Check that the built-in regex detectors work as expected:
         4.1. Unsuitable input detection.
         4.2. Unsuitable output detection.
         4.3. No detection.
        5. Check that the /passthrough endpoint forwards the
         query directly to the model without performing any detection.
    """

    def test_guardrails_health_endpoint(self, qwen_isvc, guardrails_orchestrator_health_route):
        # It takes a bit for the endpoint to come online, so we retry for a brief period of time
        @retry(wait_timeout=Timeout.TIMEOUT_1MIN, sleep=1)
        def check_health_endpoint():
            response = requests.get(url=f"https://{guardrails_orchestrator_health_route.host}/health", verify=False)
            if response.status_code == http.HTTPStatus.OK:
                return response
            return False

        response = check_health_endpoint()
        assert "fms-guardrails-orchestr8" in response.text

    def test_guardrails_info_endpoint(self, qwen_isvc, guardrails_orchestrator_health_route):
        response = requests.get(url=f"https://{guardrails_orchestrator_health_route.host}/info", verify=False)
        assert response.status_code == http.HTTPStatus.OK

        healthy_status = "HEALTHY"
        response_data = response.json()
        assert response_data["services"]["chat_generation"]["status"] == healthy_status
        assert response_data["services"]["regex"]["status"] == healthy_status

    def test_guardrails_builtin_detectors_unsuitable_input(
        self, current_client_token, openshift_ca_bundle_file, guardrails_orchestrator_route
    ):
        response = requests.post(
            url=f"https://{guardrails_orchestrator_route.host}{PII_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            headers=get_auth_headers(token=current_client_token),
            json=get_chat_payload(content=INPUT_WITH_EMAIL_ADDRESS),
            verify=openshift_ca_bundle_file,
        )

        verify_builtin_detector_unsuitable_input_response(
            response=response,
            detector_id="regex",
            detection_name="EmailAddress",
            detection_type="pii",
            detection_text=EXAMPLE_EMAIL_ADDRESS,
        )

    def test_guardrails_builtin_detectors_unsuitable_output(
        self, current_client_token, openshift_ca_bundle_file, guardrails_orchestrator_route
    ):
        response = requests.post(
            url=f"https://{guardrails_orchestrator_route.host}{PII_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            headers=get_auth_headers(token=current_client_token),
            json=get_chat_payload(
                content="Hi, write three and only three examples of email adresses "
                "that I can use to create an account for an online service."
                "Don't provide any additional explanation."
            ),
            verify=openshift_ca_bundle_file,
        )

        verify_builtin_detector_unsuitable_output_response(
            response=response, detector_id="regex", detection_name="EmailAddress", detection_type="pii"
        )

    @pytest.mark.parametrize(
        "message, url_path",
        [
            pytest.param(
                "What is the opposite of up?",
                PII_ENDPOINT,
                id="harmless_input",
            ),
            pytest.param(INPUT_WITH_EMAIL_ADDRESS, "/passthrough", id="pastthrough_endpoint"),
        ],
    )
    def test_guardrails_builtin_detectors_negative_detection(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        guardrails_orchestrator_route,
        message,
        url_path,
    ):
        response = requests.post(
            url=f"https://{guardrails_orchestrator_route.host}{url_path}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            headers=get_auth_headers(token=current_client_token),
            json=get_chat_payload(content=str(message)),
            verify=openshift_ca_bundle_file,
        )

        verify_negative_detection_response(response=response)
