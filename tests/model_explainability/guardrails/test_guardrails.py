import http
from typing import Dict, Any

import pytest
import requests
import yaml
from simple_logger.logger import get_logger
from timeout_sampler import retry

from tests.model_explainability.guardrails.constants import (
    AUTOCONFIG_DETECTOR_LABEL,
    PII_INPUT_DETECTION_PROMPT,
    PII_OUTPUT_DETECTION_PROMPT,
    PROMPT_INJECTION_INPUT_DETECTION_PROMPT,
    HAP_INPUT_DETECTION_PROMPT,
)
from tests.model_explainability.guardrails.utils import (
    verify_builtin_detector_unsuitable_input_response,
    verify_negative_detection_response,
    verify_builtin_detector_unsuitable_output_response,
    get_auth_headers,
    get_chat_detections_payload,
)
from tests.model_explainability.utils import validate_tai_component_images
from utilities.constants import (
    Timeout,
    CHAT_GENERATION_CONFIG,
    BUILTIN_DETECTOR_CONFIG,
    MinIo,
    QWEN_MODEL_NAME,
)
from utilities.plugins.constant import OpenAIEnpoints

LOGGER = get_logger(name=__name__)


HARMLESS_PROMPT: str = "What is the opposite of up?"

CHAT_COMPLETIONS_DETECTION_ENDPOINT: str = "api/v2/chat/completions-detection"
PII_ENDPOINT: str = "/pii"
AUTOCONFIG_GATEWAY_ENDPOINT: str = "/all"
STANDALONE_DETECTION_ENDPOINT: str = "api/v2/text/detection/content"

PROMPT_INJECTION_DETECTOR: str = "prompt-injection-detector"
HAP_DETECTOR: str = "hap-detector"


def create_detector_config(*detector_names: str) -> Dict[str, Dict[str, Any]]:
    detectors_dict = {name: {} for name in detector_names}
    return {
        "input": detectors_dict.copy(),
        "output": detectors_dict.copy(),
    }


@retry(exceptions_dict={TimeoutError: []}, wait_timeout=Timeout.TIMEOUT_1MIN, sleep=1)
def check_guardrails_health_endpoint(
    host,
    token,
    ca_bundle_file,
):
    response = requests.get(url=f"https://{host}/health", headers=get_auth_headers(token=token), verify=ca_bundle_file)
    if response.status_code == http.HTTPStatus.OK:
        return response
    raise TimeoutError(
        f"Timeout waiting GuardrailsOrchestrator to be healthy. Response status code: {response.status_code}"
    )


def verify_health_info_response(host, token, ca_bundle_file):
    response = requests.get(url=f"https://{host}/info", headers=get_auth_headers(token=token), verify=ca_bundle_file)
    assert response.status_code == http.HTTPStatus.OK

    healthy_status = "HEALTHY"
    response_data = response.json()
    mismatches = []
    for service_name, service_info in response_data["services"].items():
        if service_info["status"] != healthy_status:
            mismatches.append(f"Service {service_name} is not healthy: {service_info['status']}")

    assert not mismatches, f"GuardrailsOrchestrator service failures: {mismatches}"


@pytest.mark.parametrize(
    "model_namespace, orchestrator_config, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-image"},
            {
                "orchestrator_config_data": {
                    "config.yaml": yaml.dump({
                        "openai": CHAT_GENERATION_CONFIG,
                        "detectors": BUILTIN_DETECTOR_CONFIG,
                    })
                },
            },
            {"orchestrator_config": True, "enable_built_in_detectors": False, "enable_guardrails_gateway": False},
        )
    ],
    indirect=True,
)
@pytest.mark.smoke
def test_validate_guardrails_orchestrator_images(
    orchestrator_config, guardrails_orchestrator_pod, trustyai_operator_configmap
):
    """Test to verify Guardrails pod images.
    Checks if the image tag from the ConfigMap is used within the Pod and if it's pinned using a sha256 digest.
    """
    validate_tai_component_images(pod=guardrails_orchestrator_pod, tai_operator_configmap=trustyai_operator_configmap)


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, "
    "orchestrator_config, guardrails_gateway_config, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-builtin"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "orchestrator_config_data": {
                    "config.yaml": yaml.dump({
                        "openai": CHAT_GENERATION_CONFIG,
                        "detectors": BUILTIN_DETECTOR_CONFIG,
                    })
                },
            },
            {
                "guardrails_gateway_config_data": {
                    "config.yaml": yaml.dump({
                        "orchestrator": {
                            "host": "localhost",
                            "port": 8032,
                        },
                        "detectors": [
                            {
                                "name": "regex",
                                "input": True,
                                "output": True,
                                "detector_params": {"regex": ["email", "ssn"]},
                            },
                        ],
                        "routes": [
                            {"name": "pii", "detectors": ["regex"]},
                            {"name": "passthrough", "detectors": []},
                        ],
                    })
                },
            },
            {
                "orchestrator_config": True,
                "enable_built_in_detectors": True,
                "enable_guardrails_gateway": True,
                "guardrails_gateway_config": True,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.smoke
@pytest.mark.rawdeployment
@pytest.mark.usefixtures("guardrails_gateway_config")
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

    def test_guardrails_health_endpoint(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        qwen_isvc,
        orchestrator_config,
        guardrails_orchestrator_health_route,
    ):
        response = check_guardrails_health_endpoint(
            host=guardrails_orchestrator_health_route.host,
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
        )
        assert "fms-guardrails-orchestr8" in response.text

    def test_guardrails_info_endpoint(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        qwen_isvc,
        orchestrator_config,
        guardrails_orchestrator_health_route,
    ):
        verify_health_info_response(
            host=guardrails_orchestrator_health_route.host,
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
        )

    def test_guardrails_builtin_detectors_unsuitable_input(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        qwen_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
    ):
        response = requests.post(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{PII_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            headers=get_auth_headers(token=current_client_token),
            json=get_chat_detections_payload(
                content=PII_INPUT_DETECTION_PROMPT.content,
                model=QWEN_MODEL_NAME,
            ),
            verify=openshift_ca_bundle_file,
        )

        verify_builtin_detector_unsuitable_input_response(
            response=response,
            detector_id=PII_INPUT_DETECTION_PROMPT.detector_id,
            detection_name=PII_INPUT_DETECTION_PROMPT.detection_name,
            detection_type=PII_INPUT_DETECTION_PROMPT.detection_type,
            detection_text=PII_INPUT_DETECTION_PROMPT.detection_text,
        )

    def test_guardrails_builtin_detectors_unsuitable_output(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        qwen_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
    ):
        response = requests.post(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{PII_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            headers=get_auth_headers(token=current_client_token),
            json=get_chat_detections_payload(
                content=PII_OUTPUT_DETECTION_PROMPT.content,
                model=QWEN_MODEL_NAME,
            ),
            verify=openshift_ca_bundle_file,
        )

        verify_builtin_detector_unsuitable_output_response(
            response=response,
            detector_id=PII_OUTPUT_DETECTION_PROMPT.detector_id,
            detection_name=PII_OUTPUT_DETECTION_PROMPT.detection_name,
            detection_type=PII_OUTPUT_DETECTION_PROMPT.detection_type,
        )

    @pytest.mark.parametrize(
        "message, url_path",
        [
            pytest.param(
                HARMLESS_PROMPT,
                PII_ENDPOINT,
                id="harmless_input",
            ),
            pytest.param(PII_INPUT_DETECTION_PROMPT.content, "/passthrough", id="passthrough_endpoint"),
        ],
    )
    def test_guardrails_builtin_detectors_negative_detection(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        qwen_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
        message,
        url_path,
    ):
        response = requests.post(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{url_path}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            headers=get_auth_headers(token=current_client_token),
            json=get_chat_detections_payload(
                content=str(message),
                model=QWEN_MODEL_NAME,
            ),
            verify=openshift_ca_bundle_file,
        )

        verify_negative_detection_response(response=response)


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, orchestrator_config, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-huggingface"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "orchestrator_config_data": {
                    "config.yaml": yaml.dump({
                        "openai": {
                            "service": {
                                "hostname": f"{QWEN_MODEL_NAME}-predictor",
                                "port": 8032,
                            }
                        },
                        "detectors": {
                            PROMPT_INJECTION_DETECTOR: {
                                "type": "text_contents",
                                "service": {
                                    "hostname": f"{PROMPT_INJECTION_DETECTOR}-predictor",
                                    "port": 8000,
                                },
                                "chunker_id": "whole_doc_chunker",
                                "default_threshold": 0.5,
                            },
                            HAP_DETECTOR: {
                                "type": "text_contents",
                                "service": {
                                    "hostname": f"{HAP_DETECTOR}-predictor",
                                    "port": 8000,
                                },
                                "chunker_id": "whole_doc_chunker",
                                "default_threshold": 0.5,
                            },
                        },
                    })
                },
            },
            {"orchestrator_config": True, "enable_built_in_detectors": False, "enable_guardrails_gateway": False},
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
class TestGuardrailsOrchestratorWithHuggingFaceDetectors:
    """
    These tests verify that the GuardrailsOrchestrator works as expected when using HuggingFace detectors
    Steps:
        - Deploy an LLM (Qwen2.5-0.5B-Instruct) using the vLLM SR.
        - Deploy the GuardrailsOrchestrator.
        - Deploy a prompt injection detector using the HuggingFace SR.
        - Check that the detector works when we have an unsuitable input.
        - Check that the detector works when we have a harmless input (no detection).
         - Check the standalone detections by querying its /text/detection/content endpoint, verifying that an input
           detection is correctly performed.
    """

    def test_guardrails_hf_detector_unsuitable_input(
        self,
        current_client_token,
        minio_pod,
        minio_data_connection,
        qwen_isvc,
        orchestrator_config,
        guardrails_orchestrator_route,
        prompt_injection_detector_route,
        openshift_ca_bundle_file,
    ):
        response = requests.post(
            url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
            headers=get_auth_headers(token=current_client_token),
            json=get_chat_detections_payload(
                content=PROMPT_INJECTION_INPUT_DETECTION_PROMPT.content,
                model=QWEN_MODEL_NAME,
                detectors=create_detector_config(PROMPT_INJECTION_DETECTOR),
            ),
            verify=openshift_ca_bundle_file,
        )

        verify_builtin_detector_unsuitable_input_response(
            response=response,
            detector_id=PROMPT_INJECTION_INPUT_DETECTION_PROMPT.detector_id,
            detection_name=PROMPT_INJECTION_INPUT_DETECTION_PROMPT.detection_name,
            detection_type=PROMPT_INJECTION_INPUT_DETECTION_PROMPT.detection_type,
            detection_text=PROMPT_INJECTION_INPUT_DETECTION_PROMPT.detection_text,
        )

    def test_guardrails_hf_detector_negative_detection(
        self,
        current_client_token,
        minio_pod,
        minio_data_connection,
        qwen_isvc,
        orchestrator_config,
        guardrails_orchestrator_route,
        prompt_injection_detector_route,
        openshift_ca_bundle_file,
    ):
        response = requests.post(
            url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
            headers=get_auth_headers(token=current_client_token),
            json=get_chat_detections_payload(
                content=HARMLESS_PROMPT,
                model=QWEN_MODEL_NAME,
                detectors=create_detector_config(PROMPT_INJECTION_DETECTOR),
            ),
            verify=openshift_ca_bundle_file,
        )

        verify_negative_detection_response(response=response)

    def test_guardrails_standalone_detector_endpoint(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        qwen_isvc,
        orchestrator_config,
        guardrails_orchestrator_route,
        hap_detector_route,
    ):
        url = f"https://{guardrails_orchestrator_route.host}/{STANDALONE_DETECTION_ENDPOINT}"
        headers = get_auth_headers(token=current_client_token)
        payload = {"detectors": {HAP_DETECTOR: {}}, "content": HAP_INPUT_DETECTION_PROMPT.content}

        response = requests.post(
            url=url,
            headers=headers,
            json=payload,
            verify=openshift_ca_bundle_file,
        )

        assert response.status_code == http.HTTPStatus.OK, (
            f"Unexpected status code: {response.status_code}, body: {response.text}"
        )
        data = response.json()

        assert "detections" in data

        score = data["detections"][0]["score"]
        assert score > 0.9, f"Expected score > 0.9, got {score}"


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, orchestrator_config, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-huggingface"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "orchestrator_config_data": {
                    "config.yaml": yaml.dump({
                        "openai": {
                            "service": {
                                "hostname": f"{QWEN_MODEL_NAME}-predictor",
                                "port": 8032,
                            }
                        },
                        "detectors": {
                            PROMPT_INJECTION_DETECTOR: {
                                "type": "text_contents",
                                "service": {
                                    "hostname": f"{PROMPT_INJECTION_DETECTOR}-predictor",
                                    "port": 8000,
                                },
                                "chunker_id": "whole_doc_chunker",
                                "default_threshold": 0.5,
                            },
                            HAP_DETECTOR: {
                                "type": "text_contents",
                                "service": {
                                    "hostname": f"{HAP_DETECTOR}-predictor",
                                    "port": 8000,
                                },
                                "chunker_id": "whole_doc_chunker",
                                "default_threshold": 0.5,
                            },
                        },
                    })
                },
            },
            {"orchestrator_config": True, "enable_built_in_detectors": False, "enable_guardrails_gateway": False},
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
class TestGuardrailsOrchestratorWithMultipleDetectors:
    """
    These tests verify that the GuardrailsOrchestrator works as expected when using two HuggingFace detectors
    (prompt injection and hap).
    Steps:
        - Deploy an LLM (Qwen2.5-0.5B-Instruct) using the vLLM SR.
        - Deploy the GuardrailsOrchestrator.
        - Deploy a prompt injection detector and HAP detectors using the HuggingFace SR.
        - Check that the detectors works when we have an unsuitable input.
        - Check that the detector works when we have a harmless input (no detection).
    """

    def test_guardrails_multi_detector_unsuitable_input(
        self,
        current_client_token,
        minio_pod,
        minio_data_connection,
        qwen_isvc,
        guardrails_orchestrator_route,
        prompt_injection_detector_route,
        hap_detector_route,
        openshift_ca_bundle_file,
        orchestrator_config,
        guardrails_orchestrator,
    ):
        for prompt in [PROMPT_INJECTION_INPUT_DETECTION_PROMPT, HAP_INPUT_DETECTION_PROMPT]:
            response = requests.post(
                url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
                headers=get_auth_headers(token=current_client_token),
                json=get_chat_detections_payload(
                    content=prompt.content,
                    model=QWEN_MODEL_NAME,
                    detectors=create_detector_config(PROMPT_INJECTION_DETECTOR, HAP_DETECTOR),
                ),
                verify=openshift_ca_bundle_file,
            )

            verify_builtin_detector_unsuitable_input_response(
                response=response,
                detector_id=prompt.detector_id,
                detection_name=prompt.detection_name,
                detection_type=prompt.detection_type,
                detection_text=prompt.detection_text,
            )

    def test_guardrails_multi_detector_negative_detection(
        self,
        current_client_token,
        minio_pod,
        minio_data_connection,
        qwen_isvc,
        orchestrator_config,
        guardrails_orchestrator_route,
        hap_detector_route,
        prompt_injection_detector_route,
        openshift_ca_bundle_file,
    ):
        response = requests.post(
            url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
            headers=get_auth_headers(token=current_client_token),
            json=get_chat_detections_payload(
                content=HARMLESS_PROMPT,
                model=QWEN_MODEL_NAME,
                detectors=create_detector_config(PROMPT_INJECTION_DETECTOR, HAP_DETECTOR),
            ),
            verify=openshift_ca_bundle_file,
        )

        verify_negative_detection_response(response=response)


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-autoconfig"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "auto_config": {
                    "inferenceServiceToGuardrail": QWEN_MODEL_NAME,
                    "detectorServiceLabelToMatch": AUTOCONFIG_DETECTOR_LABEL,
                },
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
class TestGuardrailsOrchestratorAutoConfig:
    """
    These tests verify that the GuardrailsOrchestrator works as expected when configured through the AutoConfig feature.
    """

    def test_guardrails_gateway_health_endpoint(
        self,
        current_client_token,
        minio_pod,
        minio_data_connection,
        qwen_isvc,
        prompt_injection_detector_route,
        hap_detector_route,
        openshift_ca_bundle_file,
        guardrails_orchestrator,
        guardrails_orchestrator_health_route,
    ):
        response = check_guardrails_health_endpoint(
            host=guardrails_orchestrator_health_route.host,
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
        )
        assert "fms-guardrails-orchestr8" in response.text

    def test_guardrails_gateway_info_endpoint(
        self, current_client_token, openshift_ca_bundle_file, qwen_isvc, guardrails_orchestrator_health_route
    ):
        verify_health_info_response(
            host=guardrails_orchestrator_health_route.host,
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
        )

    def test_guardrails_autoconfig_unsuitable_input(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        qwen_isvc,
        guardrails_orchestrator_route,
    ):
        for prompt in [HAP_INPUT_DETECTION_PROMPT, PROMPT_INJECTION_INPUT_DETECTION_PROMPT]:
            response = requests.post(
                url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
                headers=get_auth_headers(token=current_client_token),
                json=get_chat_detections_payload(
                    content=prompt.content,
                    model=QWEN_MODEL_NAME,
                    detectors=create_detector_config(PROMPT_INJECTION_DETECTOR, HAP_DETECTOR),
                ),
                verify=openshift_ca_bundle_file,
            )

            verify_builtin_detector_unsuitable_input_response(
                response=response,
                detector_id=prompt.detector_id,
                detection_name=prompt.detection_name,
                detection_type=prompt.detection_type,
                detection_text=prompt.detection_text,
            )

    def test_guardrails_autoconfig_negative_detection(
        self,
        current_client_token,
        qwen_isvc,
        guardrails_orchestrator_route,
        openshift_ca_bundle_file,
    ):
        response = requests.post(
            url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
            headers=get_auth_headers(token=current_client_token),
            json=get_chat_detections_payload(
                content=str(HARMLESS_PROMPT),
                model=QWEN_MODEL_NAME,
                detectors=create_detector_config(PROMPT_INJECTION_DETECTOR, HAP_DETECTOR),
            ),
            verify=openshift_ca_bundle_file,
        )

        verify_negative_detection_response(response=response)


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-autoconfig-gateway"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "auto_config": {
                    "inferenceServiceToGuardrail": QWEN_MODEL_NAME,
                    "detectorServiceLabelToMatch": AUTOCONFIG_DETECTOR_LABEL,
                },
                "enable_built_in_detectors": True,
                "enable_guardrails_gateway": True,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
class TestGuardrailsOrchestratorAutoConfigWithGateway:
    """
    These tests verify that the GuardrailsOrchestrator works as expected when configured
    through the AutoConfig feature in order to use the gateway route.
    """

    def test_guardrails_autoconfig_gateway_health_endpoint(
        self,
        current_client_token,
        minio_pod,
        minio_data_connection,
        qwen_isvc,
        prompt_injection_detector_route,
        hap_detector_route,
        openshift_ca_bundle_file,
        guardrails_orchestrator,
        guardrails_orchestrator_health_route,
    ):
        response = check_guardrails_health_endpoint(
            host=guardrails_orchestrator_health_route.host,
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
        )
        assert "fms-guardrails-orchestr8" in response.text

    def test_guardrails_autoconfig_gateway_info_endpoint(
        self, current_client_token, openshift_ca_bundle_file, qwen_isvc, guardrails_orchestrator_health_route
    ):
        verify_health_info_response(
            host=guardrails_orchestrator_health_route.host,
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
        )

    def test_guardrails_autoconfig_gateway_unsuitable_input(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        qwen_isvc,
        guardrails_orchestrator_gateway_route,
    ):
        for prompt in [HAP_INPUT_DETECTION_PROMPT, PROMPT_INJECTION_INPUT_DETECTION_PROMPT]:
            response = requests.post(
                url=f"https://{guardrails_orchestrator_gateway_route.host}"
                f"{AUTOCONFIG_GATEWAY_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
                headers=get_auth_headers(token=current_client_token),
                json=get_chat_detections_payload(
                    content=prompt.content,
                    model=QWEN_MODEL_NAME,
                ),
                verify=openshift_ca_bundle_file,
            )

            verify_builtin_detector_unsuitable_input_response(
                response=response,
                detector_id=prompt.detector_id,
                detection_name=prompt.detection_name,
                detection_type=prompt.detection_type,
                detection_text=prompt.detection_text,
            )

    @pytest.mark.parametrize(
        "message, url_path",
        [
            pytest.param(
                HARMLESS_PROMPT,
                AUTOCONFIG_GATEWAY_ENDPOINT,
                id="harmless_input",
            ),
            pytest.param(PII_INPUT_DETECTION_PROMPT.content, "/passthrough", id="passthrough_endpoint"),
        ],
    )
    def test_guardrails_autoconfig_gateway_negative_detection(
        self,
        current_client_token,
        qwen_isvc,
        guardrails_orchestrator_gateway_route,
        openshift_ca_bundle_file,
        url_path,
        message,
    ):
        response = requests.post(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{url_path}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            headers=get_auth_headers(token=current_client_token),
            json=get_chat_detections_payload(
                content=str(message),
                model=QWEN_MODEL_NAME,
            ),
            verify=openshift_ca_bundle_file,
        )

        verify_negative_detection_response(response=response)
