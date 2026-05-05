import pytest
import yaml
from simple_logger.logger import get_logger

from tests.model_explainability.guardrails.constants import (
    PROMPT_INJECTION_INPUT_DETECTION_PROMPT,
    HAP_INPUT_DETECTION_PROMPT,
)
from tests.model_explainability.guardrails.utils import (
    create_detector_config,
    send_and_verify_unsuitable_input_detection,
    send_and_verify_negative_detection,
    send_and_verify_standalone_detection,
    check_guardrails_traces_in_tempo,
)
from utilities.constants import MinIo, QWEN_MODEL_NAME

LOGGER = get_logger(name=__name__)


HARMLESS_PROMPT: str = "What is the opposite of up?"

CHAT_COMPLETIONS_DETECTION_ENDPOINT: str = "api/v2/chat/completions-detection"
STANDALONE_DETECTION_ENDPOINT: str = "api/v2/text/detection/content"

PROMPT_INJECTION_DETECTOR: str = "prompt-injection-detector"
HAP_DETECTOR: str = "hap-detector"


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, orchestrator_config, guardrails_gateway_config,"
    "guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-hf-upgrade"},
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
            {
                "guardrails_gateway_config_data": {
                    "config.yaml": yaml.dump({
                        "orchestrator": {
                            "host": "localhost",
                            "port": 8032,
                        },
                        "detectors": [],
                        "routes": [],
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
class TestPreUpgradeGuardrailsOrchestratorWithHuggingFaceDetectors:
    """
    Pre-upgrade tests to verify that the GuardrailsOrchestrator works as expected when using HuggingFace detectors.
    Steps:
        - Deploy an LLM (Qwen2.5-0.5B-Instruct) using the vLLM SR.
        - Deploy the GuardrailsOrchestrator.
        - Deploy a prompt injection detector using the HuggingFace SR.
        - Check that the detector works when we have an unsuitable input.
        - Check that the detector works when we have a harmless input (no detection).
        - Check the standalone detections by querying its /text/detection/content endpoint.
        - Verify OpenTelemetry traces are collected in Tempo.
    """

    @pytest.mark.pre_upgrade
    def test_guardrails_hf_detector_unsuitable_input_pre_upgrade(
        self,
        current_client_token,
        minio_pod,
        minio_data_connection,
        qwen_isvc,
        orchestrator_config,
        guardrails_orchestrator_route,
        prompt_injection_detector_route,
        openshift_ca_bundle_file,
        otel_collector,
        tempo_stack,
    ):
        """Verify HuggingFace detector detects unsuitable input before upgrade."""
        send_and_verify_unsuitable_input_detection(
            url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            prompt=PROMPT_INJECTION_INPUT_DETECTION_PROMPT,
            model=QWEN_MODEL_NAME,
            detectors=create_detector_config(PROMPT_INJECTION_DETECTOR),
        )

    @pytest.mark.pre_upgrade
    def test_guardrails_hf_detector_negative_detection_pre_upgrade(
        self,
        current_client_token,
        minio_pod,
        minio_data_connection,
        qwen_isvc,
        orchestrator_config,
        guardrails_orchestrator_route,
        prompt_injection_detector_route,
        openshift_ca_bundle_file,
        otel_collector,
        tempo_stack,
    ):
        """Verify HuggingFace detector does not detect harmless input before upgrade."""
        send_and_verify_negative_detection(
            url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            content=HARMLESS_PROMPT,
            model=QWEN_MODEL_NAME,
            detectors=create_detector_config(PROMPT_INJECTION_DETECTOR),
        )

    @pytest.mark.pre_upgrade
    def test_guardrails_standalone_detector_endpoint_pre_upgrade(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        qwen_isvc,
        orchestrator_config,
        guardrails_orchestrator_route,
        hap_detector_route,
        otel_collector,
        tempo_stack,
    ):
        """Verify standalone detector endpoint works before upgrade."""
        send_and_verify_standalone_detection(
            url=f"https://{guardrails_orchestrator_route.host}/{STANDALONE_DETECTION_ENDPOINT}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            detector_name=HAP_DETECTOR,
            content=HAP_INPUT_DETECTION_PROMPT.content,
            expected_min_score=0.9,
        )

    @pytest.mark.pre_upgrade
    def test_guardrails_traces_in_tempo_pre_upgrade(
        self,
        admin_client,
        minio_pod,
        minio_data_connection,
        orchestrator_config,
        guardrails_orchestrator,
        guardrails_gateway_config,
        otel_collector,
        tempo_stack,
        tempo_traces_service_portforward,
    ):
        """Verify OpenTelemetry traces from Guardrails Orchestrator are collected in Tempo before upgrade."""
        check_guardrails_traces_in_tempo(tempo_traces_service_portforward=tempo_traces_service_portforward)


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection",
    [
        pytest.param(
            {"name": "test-guardrails-hf-upgrade"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures(
    "minio_pvc_otel",
    "minio_deployment_otel",
    "minio_service_otel",
    "minio_secret_otel",
    "installed_tempo_operator",
    "installed_opentelemetry_operator",
    "tempo_stack",
    "otel_collector",
)
class TestPostUpgradeGuardrailsOrchestratorWithHuggingFaceDetectors:
    """
    Post-upgrade tests to verify that the GuardrailsOrchestrator continues to work after upgrade.
    These tests verify that all functionality that worked before upgrade still works after upgrade.
    """

    @pytest.mark.post_upgrade
    def test_guardrails_hf_detector_unsuitable_input_post_upgrade(
        self,
        current_client_token,
        minio_pod,
        minio_data_connection,
        qwen_isvc,
        orchestrator_config,
        guardrails_orchestrator_route,
        prompt_injection_detector_route,
        openshift_ca_bundle_file,
        otel_collector,
        tempo_stack,
    ):
        """Verify HuggingFace detector detects unsuitable input after upgrade."""
        send_and_verify_unsuitable_input_detection(
            url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            prompt=PROMPT_INJECTION_INPUT_DETECTION_PROMPT,
            model=QWEN_MODEL_NAME,
            detectors=create_detector_config(PROMPT_INJECTION_DETECTOR),
        )

    @pytest.mark.post_upgrade
    def test_guardrails_hf_detector_negative_detection_post_upgrade(
        self,
        current_client_token,
        minio_pod,
        minio_data_connection,
        qwen_isvc,
        orchestrator_config,
        guardrails_orchestrator_route,
        prompt_injection_detector_route,
        openshift_ca_bundle_file,
        otel_collector,
        tempo_stack,
    ):
        """Verify HuggingFace detector does not detect harmless input after upgrade."""
        send_and_verify_negative_detection(
            url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            content=HARMLESS_PROMPT,
            model=QWEN_MODEL_NAME,
            detectors=create_detector_config(PROMPT_INJECTION_DETECTOR),
        )

    @pytest.mark.post_upgrade
    def test_guardrails_standalone_detector_endpoint_post_upgrade(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        qwen_isvc,
        orchestrator_config,
        guardrails_orchestrator_route,
        hap_detector_route,
        otel_collector,
        tempo_stack,
    ):
        """Verify standalone detector endpoint works after upgrade."""
        send_and_verify_standalone_detection(
            url=f"https://{guardrails_orchestrator_route.host}/{STANDALONE_DETECTION_ENDPOINT}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            detector_name=HAP_DETECTOR,
            content=HAP_INPUT_DETECTION_PROMPT.content,
            expected_min_score=0.9,
        )

    @pytest.mark.post_upgrade
    def test_guardrails_traces_in_tempo_post_upgrade(
        self,
        admin_client,
        minio_pod,
        minio_data_connection,
        orchestrator_config,
        guardrails_orchestrator,
        guardrails_gateway_config,
        otel_collector,
        tempo_stack,
        tempo_traces_service_portforward,
    ):
        """Verify OpenTelemetry traces from Guardrails Orchestrator are collected in Tempo after upgrade."""
        check_guardrails_traces_in_tempo(tempo_traces_service_portforward=tempo_traces_service_portforward)
