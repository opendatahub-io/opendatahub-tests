import pytest

from tests.ai_safety.guardrails.constants import (
    CHAT_COMPLETIONS_DETECTION_ENDPOINT,
    HAP_DETECTOR,
    HAP_INPUT_DETECTION_PROMPT,
    HARMLESS_PROMPT,
    PII_ENDPOINT,
    PII_INPUT_DETECTION_PROMPT,
    PII_OUTPUT_DETECTION_PROMPT,
    PROMPT_INJECTION_DETECTOR,
    PROMPT_INJECTION_INPUT_DETECTION_PROMPT,
    STANDALONE_DETECTION_ENDPOINT,
)
from tests.ai_safety.guardrails.utils import (
    check_guardrails_traces_in_tempo,
    create_detector_config,
    send_and_verify_negative_detection,
    send_and_verify_standalone_detection,
    send_and_verify_unsuitable_input_detection,
    send_and_verify_unsuitable_output_detection,
    verify_health_info_response,
)
from utilities.constants import (
    QWEN_MODEL_NAME,
    LLMdInferenceSimConfig,
    MinIo,
)
from utilities.plugins.constant import OpenAIEnpoints


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-guardrails-upgrade"},
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.usefixtures("patched_dsc_kserve_headed", "guardrails_gateway_config")
class TestGuardrailsOrchestratorWithBuiltInDetectorsPostUpgrade:
    """
    Tests that the GuardrailsOrchestrator functionality persists after an ODH upgrade.

    Validates that pre-existing guardrails deployments continue to function correctly
    after the platform upgrade, ensuring:
        1. The orchestrator health endpoints remain responsive.
        2. Built-in regex detectors still detect unsuitable input.
        3. Built-in regex detectors still detect unsuitable output.
        4. Passthrough and harmless prompt handling is preserved.
    """

    @pytest.mark.post_upgrade
    def test_guardrails_info_endpoint(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_health_route,
        guardrails_healthcheck,
    ):
        """Verify guardrails orchestrator health/info endpoint is responsive after upgrade.

        Given: A guardrails orchestrator deployed before the ODH upgrade.
        When: The health info endpoint is queried after upgrade.
        Then: A valid health/info response is returned, confirming the service survived the upgrade.
        """
        verify_health_info_response(
            host=guardrails_orchestrator_health_route.host,
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
        )

    @pytest.mark.post_upgrade
    def test_guardrails_builtin_detectors_unsuitable_input(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
        guardrails_healthcheck,
    ):
        """Verify built-in regex detectors block unsuitable input after upgrade.

        Given: A guardrails orchestrator with regex detectors deployed before the ODH upgrade.
        When: A prompt containing PII patterns is sent as input after upgrade.
        Then: The orchestrator detects and blocks the unsuitable input.
        """
        send_and_verify_unsuitable_input_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{PII_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            prompt=PII_INPUT_DETECTION_PROMPT,
            model=LLMdInferenceSimConfig.model_name,
        )

    @pytest.mark.post_upgrade
    def test_guardrails_builtin_detectors_unsuitable_output(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
        guardrails_healthcheck,
    ):
        """Verify built-in regex detectors block unsuitable output after upgrade.

        Given: A guardrails orchestrator with regex detectors deployed before the ODH upgrade.
        When: A prompt triggers the model to generate output containing PII patterns after upgrade.
        Then: The orchestrator detects and blocks the unsuitable output.
        """
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
    @pytest.mark.post_upgrade
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
        """Verify harmless prompts pass through without detection after upgrade.

        Given: A guardrails orchestrator with regex detectors deployed before the ODH upgrade.
        When: A harmless prompt (no PII) is sent via the PII endpoint after upgrade,
              or a PII-containing prompt is sent via the passthrough endpoint after upgrade.
        Then: The request is forwarded to the model and a response is returned.

        Test cases:
            - harmless_input: Safe content through the PII detection route.
            - passthrough_endpoint: PII content through the passthrough route (no detection).
        """
        send_and_verify_negative_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{url_path}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            content=str(message),
            model=LLMdInferenceSimConfig.model_name,
        )


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
@pytest.mark.rawdeployment
@pytest.mark.usefixtures(
    "seaweedfs_pvc_otel",
    "seaweedfs_deployment_otel",
    "seaweedfs_service_otel",
    "seaweedfs_secret_otel",
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
        guardrails_healthcheck,
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
        guardrails_healthcheck,
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
        guardrails_healthcheck,
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
