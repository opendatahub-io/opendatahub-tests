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
    create_detector_config,
    verify_health_info_response,
    send_and_verify_unsuitable_input_detection,
    send_and_verify_unsuitable_output_detection,
    send_and_verify_negative_detection,
    send_and_verify_standalone_detection,
)
from tests.model_explainability.utils import validate_tai_component_images
from utilities.constants import (
    LLM_D_CHAT_GENERATION_CONFIG,
    BUILTIN_DETECTOR_CONFIG,
    LLMdInferenceSimConfig,
    Timeout,
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


@pytest.mark.parametrize(
    "model_namespace, orchestrator_config, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-image"},
            {
                "orchestrator_config_data": {
                    "config.yaml": yaml.dump({
                        "openai": LLM_D_CHAT_GENERATION_CONFIG,
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
    model_namespace,
    orchestrator_config,
    guardrails_orchestrator,
    guardrails_orchestrator_pod,
    trustyai_operator_configmap,
):
    """Test to verify Guardrails pod images.
    Checks if the image tag from the ConfigMap is used within the Pod and if it's pinned using a sha256 digest.
    """
    validate_tai_component_images(pod=guardrails_orchestrator_pod, tai_operator_configmap=trustyai_operator_configmap)


@pytest.mark.parametrize(
    "model_namespace, orchestrator_config, guardrails_gateway_config, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-builtin"},
            {
                "orchestrator_config_data": {
                    "config.yaml": yaml.dump({
                        "openai": LLM_D_CHAT_GENERATION_CONFIG,
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
@pytest.mark.usefixtures("patched_dsc_kserve_headed", "guardrails_gateway_config")
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
        llm_d_inference_sim_isvc,
        orchestrator_config,
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
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
        guardrails_healthcheck,
    ):
        send_and_verify_unsuitable_input_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{PII_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            prompt=PII_INPUT_DETECTION_PROMPT,
            model=LLMdInferenceSimConfig.model_name,
        )

    def test_guardrails_builtin_detectors_unsuitable_output(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
        guardrails_healthcheck,
    ):
        send_and_verify_unsuitable_output_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{PII_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            prompt=PII_OUTPUT_DETECTION_PROMPT,
            model=LLMdInferenceSimConfig.model_name,
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
        llm_d_inference_sim_isvc,
        orchestrator_config,
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
            model=LLMdInferenceSimConfig.model_name,
        )


@pytest.mark.parametrize(
    "model_namespace, orchestrator_config, guardrails_gateway_config,guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-huggingface"},
            {
                "orchestrator_config_data": {
                    "config.yaml": yaml.dump({
                        "openai": LLM_D_CHAT_GENERATION_CONFIG,
                        "detectors": {
                            PROMPT_INJECTION_DETECTOR: {
                                "type": "text_contents",
                                "service": {
                                    "hostname": f"{PROMPT_INJECTION_DETECTOR}-predictor",
                                    "port": 80,
                                },
                                "chunker_id": "whole_doc_chunker",
                                "default_threshold": 0.5,
                            },
                            HAP_DETECTOR: {
                                "type": "text_contents",
                                "service": {
                                    "hostname": f"{HAP_DETECTOR}-predictor",
                                    "port": 80,
                                },
                                "chunker_id": "whole_doc_chunker",
                                "default_threshold": 0.5,
                            },
                        },
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
                "enable_built_in_detectors": False,
                "enable_guardrails_gateway": True,
                "guardrails_gateway_config": True,
                "otel_exporter_config": True,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.usefixtures(
    "patched_dsc_kserve_headed",
    "guardrails_gateway_config",
    "minio_pvc_otel",
    "minio_deployment_otel",
    "minio_service_otel",
    "minio_secret_otel",
    "installed_tempo_operator",
    "installed_opentelemetry_operator",
    "tempo_stack",
    "otel_collector",
)
class TestGuardrailsOrchestratorWithHuggingFaceDetectors:
    """
    These tests verify that the GuardrailsOrchestrator works as expected when using HuggingFace detectors
    Steps:
        - Deploy an LLM (Qwen2.5-0.5B-Instruct) using the vLLM SR.
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
        llm_d_inference_sim_isvc,
        guardrails_orchestrator_route,
        prompt_injection_detector_route,
        hap_detector_route,
        openshift_ca_bundle_file,
        orchestrator_config,
        guardrails_orchestrator,
        otel_collector,
        tempo_stack,
        guardrails_healthcheck,
    ):
        for prompt in [PROMPT_INJECTION_INPUT_DETECTION_PROMPT, HAP_INPUT_DETECTION_PROMPT]:
            send_and_verify_unsuitable_input_detection(
                url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
                token=current_client_token,
                ca_bundle_file=openshift_ca_bundle_file,
                prompt=prompt,
                model=LLMdInferenceSimConfig.model_name,
                detectors=create_detector_config(PROMPT_INJECTION_DETECTOR, HAP_DETECTOR),
            )

    def test_guardrails_multi_detector_negative_detection(
        self,
        current_client_token,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_route,
        hap_detector_route,
        prompt_injection_detector_route,
        openshift_ca_bundle_file,
        otel_collector,
        tempo_stack,
        guardrails_healthcheck,
    ):
        send_and_verify_negative_detection(
            url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            content=HARMLESS_PROMPT,
            model=LLMdInferenceSimConfig.model_name,
            detectors=create_detector_config(PROMPT_INJECTION_DETECTOR, HAP_DETECTOR),
        )

    def test_guardrails_standalone_detector_endpoint(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_route,
        hap_detector_route,
        otel_collector,
        tempo_stack,
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

    def test_guardrails_traces_in_tempo(
        self,
        admin_client,
        model_namespace,
        orchestrator_config,
        guardrails_orchestrator,
        guardrails_gateway_config,
        otel_collector,
        tempo_stack,
        tempo_traces_service_portforward,
        guardrails_healthcheck,
    ):
        """
        Ensure that OpenTelemetry traces from Guardrails Orchestrator are collected in Tempo.
        Equivalent to clicking 'Find Traces' in the Tempo UI.
        """

        @retry(wait_timeout=Timeout.TIMEOUT_1MIN, sleep=5)
        def check_traces():
            services = requests.get(f"{tempo_traces_service_portforward}/api/services").json().get("data", [])

            guardrails_services = [s for s in services if "guardrails" in s]
            if not guardrails_services:
                return False

            svc = guardrails_services[0]

            traces = requests.get(f"{tempo_traces_service_portforward}/api/traces?service={svc}").json()

            if traces.get("data"):
                return traces


@pytest.mark.parametrize(
    "model_namespace, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-autoconfig"},
            {
                "auto_config": {
                    "inferenceServiceToGuardrail": LLMdInferenceSimConfig.isvc_name,
                    "detectorServiceLabelToMatch": AUTOCONFIG_DETECTOR_LABEL,
                },
            },
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
@pytest.mark.rawdeployment
class TestGuardrailsOrchestratorAutoConfig:
    """
    These tests verify that the GuardrailsOrchestrator works as expected when configured through the AutoConfig feature.
    """

    def test_guardrails_gateway_info_endpoint(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        guardrails_orchestrator_health_route,
        guardrails_healthcheck,
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
        llm_d_inference_sim_isvc,
        guardrails_orchestrator_route,
        guardrails_healthcheck,
    ):
        for prompt in [HAP_INPUT_DETECTION_PROMPT, PROMPT_INJECTION_INPUT_DETECTION_PROMPT]:
            send_and_verify_unsuitable_input_detection(
                url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
                token=current_client_token,
                ca_bundle_file=openshift_ca_bundle_file,
                prompt=prompt,
                model=LLMdInferenceSimConfig.model_name,
                detectors=create_detector_config(PROMPT_INJECTION_DETECTOR, HAP_DETECTOR),
            )

    def test_guardrails_autoconfig_negative_detection(
        self,
        current_client_token,
        llm_d_inference_sim_isvc,
        guardrails_orchestrator_route,
        openshift_ca_bundle_file,
        guardrails_healthcheck,
    ):
        send_and_verify_negative_detection(
            url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            content=str(HARMLESS_PROMPT),
            model=LLMdInferenceSimConfig.model_name,
            detectors=create_detector_config(PROMPT_INJECTION_DETECTOR, HAP_DETECTOR),
        )


@pytest.mark.parametrize(
    "model_namespace, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-autoconfig-gateway"},
            {
                "auto_config": {
                    "inferenceServiceToGuardrail": LLMdInferenceSimConfig.isvc_name,
                    "detectorServiceLabelToMatch": AUTOCONFIG_DETECTOR_LABEL,
                },
                "enable_built_in_detectors": True,
                "enable_guardrails_gateway": True,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
@pytest.mark.rawdeployment
class TestGuardrailsOrchestratorAutoConfigWithGateway:
    """
    These tests verify that the GuardrailsOrchestrator works as expected when configured
    through the AutoConfig feature to use the gateway route.
    """

    def test_guardrails_autoconfig_gateway_info_endpoint(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        hap_detector_isvc,
        prompt_injection_detector_isvc,
        guardrails_orchestrator_health_route,
        guardrails_healthcheck,
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
        llm_d_inference_sim_isvc,
        prompt_injection_detector_isvc,
        hap_detector_isvc,
        guardrails_orchestrator_gateway_route,
        guardrails_healthcheck,
    ):
        for prompt in [HAP_INPUT_DETECTION_PROMPT, PROMPT_INJECTION_INPUT_DETECTION_PROMPT]:
            send_and_verify_unsuitable_input_detection(
                url=f"https://{guardrails_orchestrator_gateway_route.host}"
                f"{AUTOCONFIG_GATEWAY_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
                token=current_client_token,
                ca_bundle_file=openshift_ca_bundle_file,
                prompt=prompt,
                model=LLMdInferenceSimConfig.model_name,
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
        llm_d_inference_sim_isvc,
        prompt_injection_detector_isvc,
        hap_detector_isvc,
        guardrails_orchestrator_gateway_route,
        openshift_ca_bundle_file,
        url_path,
        message,
        guardrails_healthcheck,
    ):
        send_and_verify_negative_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{url_path}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            content=str(message),
            model=LLMdInferenceSimConfig.model_name,
        )
