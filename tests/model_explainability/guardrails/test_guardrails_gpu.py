import pytest
import yaml

from tests.model_explainability.guardrails.constants import (
    CHAT_COMPLETIONS_DETECTION_ENDPOINT,
    HAP_INPUT_DETECTION_PROMPT,
    HARMLESS_PROMPT,
    PII_ENDPOINT,
    PII_INPUT_DETECTION_PROMPT,
    PII_OUTPUT_DETECTION_PROMPT_QWEN,
    PROMPT_INJECTION_INPUT_DETECTION_PROMPT,
    STANDALONE_DETECTION_ENDPOINT,
)
from tests.model_explainability.guardrails.utils import (
    create_detector_config,
    send_and_verify_negative_detection,
    send_and_verify_standalone_detection,
    send_and_verify_unsuitable_input_detection,
    send_and_verify_unsuitable_output_detection,
    verify_health_info_response,
)
from utilities.constants import (
    HAP_DETECTOR,
    PROMPT_INJECTION_DETECTOR,
    VLLMGPUConfig,
)
from utilities.plugins.constant import OpenAIEnpoints


@pytest.mark.parametrize(
    "model_namespace, orchestrator_config_gpu, guardrails_gateway_config, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-builtin-gpu"},
            {"use_builtin_detectors": True},
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
                "orchestrator_config_gpu": True,
                "enable_built_in_detectors": True,
                "enable_guardrails_gateway": True,
                "guardrails_gateway_config": True,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.gpu
@pytest.mark.rawdeployment
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestGuardrailsOrchestratorWithBuiltInDetectors:
    """
    Tests if basic functions of the GuardrailsOrchestrator are working properly with the built-in (regex) detectors.
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

    def test_guardrails_info_endpoint(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        qwen_gpu_isvc,
        guardrails_gateway_config,
        orchestrator_config_gpu,
        guardrails_orchestrator_health_route,
        guardrails_healthcheck,
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
        qwen_gpu_isvc,
        guardrails_gateway_config,
        orchestrator_config_gpu,
        guardrails_orchestrator_gateway_route,
        guardrails_healthcheck,
    ):
        send_and_verify_unsuitable_input_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{PII_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            prompt=PII_INPUT_DETECTION_PROMPT,
            model=VLLMGPUConfig.model_name,
        )

    def test_guardrails_builtin_detectors_unsuitable_output(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        qwen_gpu_isvc,
        guardrails_gateway_config,
        orchestrator_config_gpu,
        guardrails_orchestrator_gateway_route,
        guardrails_healthcheck,
    ):
        send_and_verify_unsuitable_output_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{PII_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            prompt=PII_OUTPUT_DETECTION_PROMPT_QWEN,
            model=VLLMGPUConfig.model_name,
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
        qwen_gpu_isvc,
        guardrails_gateway_config,
        orchestrator_config_gpu,
        guardrails_orchestrator_gateway_route,
        message,
        url_path,
        guardrails_healthcheck,
    ):
        send_and_verify_negative_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{url_path}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            content=str(message),
            model=VLLMGPUConfig.model_name,
        )


@pytest.mark.gpu
@pytest.mark.rawdeployment
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
@pytest.mark.parametrize(
    "model_namespace, orchestrator_config_gpu, guardrails_gateway_config, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-huggingface-gpu"},
            {"orchestrator_config_data": None},
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
                }
            },
            {
                "orchestrator_config_gpu": True,
                "enable_built_in_detectors": False,
                "enable_guardrails_gateway": True,
                "guardrails_gateway_config": True,
            },
        )
    ],
    indirect=True,
)
class TestGuardrailsOrchestratorHuggingFaceGPU:
    """
    These tests verify that the GuardrailsOrchestrator works as expected when using HuggingFace detectors
    Steps:
        - Deploy an LLM (Qwen2.5-3B-Instruct) using the vLLM SR.
        - Deploy the GuardrailsOrchestrator.
        - Deploy a prompt injection detector using the HuggingFace SR.
        - Check that the detector works when we have an unsuitable input.
        - Check that the detector works when we have a harmless input (no detection).
         - Check the standalone detections by querying its /text/detection/content endpoint, verifying that input
           detection is correctly performed.
    """

    def test_guardrails_multi_detector_unsuitable_input(
        self,
        current_client_token,
        qwen_gpu_isvc,
        guardrails_orchestrator_route,
        prompt_injection_detector_route,
        hap_detector_route,
        openshift_ca_bundle_file,
        orchestrator_config_gpu,
        guardrails_orchestrator,
        guardrails_gateway_config,
        guardrails_healthcheck,
    ):
        for prompt in [PROMPT_INJECTION_INPUT_DETECTION_PROMPT, HAP_INPUT_DETECTION_PROMPT]:
            send_and_verify_unsuitable_input_detection(
                url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
                token=current_client_token,
                ca_bundle_file=openshift_ca_bundle_file,
                prompt=prompt,
                model=VLLMGPUConfig.model_name,
                detectors=create_detector_config(PROMPT_INJECTION_DETECTOR, HAP_DETECTOR),
            )

    def test_guardrails_multi_detector_negative_detection(
        self,
        current_client_token,
        qwen_gpu_isvc,
        orchestrator_config_gpu,
        guardrails_orchestrator_route,
        hap_detector_route,
        prompt_injection_detector_route,
        openshift_ca_bundle_file,
        guardrails_gateway_config,
        guardrails_healthcheck,
    ):
        send_and_verify_negative_detection(
            url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            content=HARMLESS_PROMPT,
            model=VLLMGPUConfig.model_name,
            detectors=create_detector_config(PROMPT_INJECTION_DETECTOR, HAP_DETECTOR),
        )

    def test_guardrails_standalone_detector_endpoint(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        qwen_gpu_isvc,
        orchestrator_config_gpu,
        guardrails_orchestrator_route,
        hap_detector_route,
        guardrails_gateway_config,
        guardrails_healthcheck,
    ):
        send_and_verify_standalone_detection(
            url=f"https://{guardrails_orchestrator_route.host}/{STANDALONE_DETECTION_ENDPOINT}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            detector_name=HAP_DETECTOR,
            content=HAP_INPUT_DETECTION_PROMPT.content,
            expected_min_score=0.9,
        )
