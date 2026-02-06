import pytest
import yaml
from tests.model_explainability.guardrails.constants import (
    PII_INPUT_DETECTION_PROMPT,
    PII_OUTPUT_DETECTION_PROMPT,
    HARMLESS_PROMPT,
    PII_ENDPOINT,
)
from tests.model_explainability.guardrails.utils import (
    verify_health_info_response,
    send_and_verify_unsuitable_input_detection,
    send_and_verify_unsuitable_output_detection,
    send_and_verify_negative_detection,
)
from utilities.constants import LLM_D_CHAT_GENERATION_CONFIG, BUILTIN_DETECTOR_CONFIG, LLMdInferenceSimConfig
from utilities.plugins.constant import OpenAIEnpoints


@pytest.mark.parametrize(
    "model_namespace, orchestrator_config, guardrails_gateway_config, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-builtin-upgrade"},
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
            {"name": "test-guardrails-builtin-upgrade"},
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.usefixtures("guardrails_gateway_config")
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
