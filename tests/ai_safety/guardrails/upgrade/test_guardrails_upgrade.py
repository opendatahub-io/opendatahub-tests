import pytest
import yaml

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
    BUILTIN_DETECTOR_CONFIG,
    LLM_D_CHAT_GENERATION_CONFIG,
    QWEN_MODEL_NAME,
    LLMdInferenceSimConfig,
    MinIo,
)
from utilities.plugins.constant import OpenAIEnpoints


@pytest.mark.parametrize(
    "model_namespace, orchestrator_config, guardrails_gateway_config, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-upgrade"},
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
@pytest.mark.rawdeployment
@pytest.mark.usefixtures("patched_dsc_kserve_headed", "guardrails_gateway_config")
class TestGuardrailsOrchestratorWithBuiltInDetectorsPreUpgrade:
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

    @pytest.mark.pre_upgrade
    def test_guardrails_info_endpoint(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_health_route,
        guardrails_healthcheck,
    ):
        """Verify guardrails orchestrator health/info endpoint is responsive before upgrade.

        Given: A guardrails orchestrator is deployed with built-in detectors.
        When: The health info endpoint is queried.
        Then: A valid health/info response is returned.
        """

        verify_health_info_response(
            host=guardrails_orchestrator_health_route.host,
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
        )

    @pytest.mark.pre_upgrade
    def test_guardrails_builtin_detectors_unsuitable_input(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
        guardrails_healthcheck,
    ):
        """Verify built-in regex detectors block unsuitable input before upgrade.

        Given: A guardrails orchestrator with regex detectors for PII (email, ssn).
        When: A prompt containing PII patterns is sent as input.
        Then: The orchestrator detects and blocks the unsuitable input.
        """
        send_and_verify_unsuitable_input_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{PII_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            prompt=PII_INPUT_DETECTION_PROMPT,
            model=LLMdInferenceSimConfig.model_name,
        )

    @pytest.mark.pre_upgrade
    def test_guardrails_builtin_detectors_unsuitable_output(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
        guardrails_healthcheck,
    ):
        """Verify built-in regex detectors block unsuitable output before upgrade.

        Given: A guardrails orchestrator with regex detectors for PII (email, ssn).
        When: A prompt triggers the model to generate output containing PII patterns.
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
    @pytest.mark.pre_upgrade
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
        """Verify harmless prompts pass through without detection before upgrade.

        Given: A guardrails orchestrator with regex detectors for PII.
        When: A harmless prompt (no PII) is sent via the PII endpoint,
              or a PII-containing prompt is sent via the passthrough endpoint.
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
@pytest.mark.rawdeployment
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
