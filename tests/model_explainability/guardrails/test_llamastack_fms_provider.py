import pytest
import yaml
from simple_logger.logger import get_logger

from tests.model_explainability.guardrails.constants import (
    CHAT_GENERATION_CONFIG,
    BUILTIN_DETECTOR_CONFIG,
    PROMPT_WITH_PII,
)
from tests.model_explainability.constants import MNT_MODELS
from utilities.constants import MinIo

LOGGER = get_logger(name=__name__)
PII_REGEX_SHIELD_ID = "regex"


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, "
    "orchestrator_config, guardrails_orchestrator, llamastack_distribution",
    [
        pytest.param(
            {"name": "test-guardrails-lls"},
            MinIo.PodConfig.QWEN_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "orchestrator_config_data": {
                    "config.yaml": yaml.dump({
                        "chat_generation": CHAT_GENERATION_CONFIG,
                        "detectors": BUILTIN_DETECTOR_CONFIG,
                    })
                },
            },
            {"enable_built_in_detectors": True, "enable_guardrails_gateway": False},
            {"guardrails_orchestrator_route_fixture": "guardrails_orchestrator_route"},
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.usefixtures("orchestrator_config", "guardrails_orchestrator")
class TestLlamaStackFMSGuardrailsProvider:
    """
    Adds basic tests for the LlamaStack FMS Guardrails provider.

    Given a basic guardrails setup (generator model + detectors),
    and a llama-stack distribution and client:

    1. Register the generator model via lls client
    2. Test that we can run inferences on said model via lls client
    3. Register the shields (detectors)
    4. Test a basic detection (PII) by using run_shields
    """

    def test_fms_guardrails_register_model(self, qwen_isvc, llamastack_client):
        provider_id = "vllm-inference"
        model_type = "llm"
        llamastack_client.models.register(provider_id=provider_id, model_type=model_type, model_id=MNT_MODELS)
        models = llamastack_client.models.list()

        # We only need to check the first model;
        # second is a granite embedding model present by default
        assert len(models) == 2
        assert models[0].identifier == MNT_MODELS
        assert models[0].provider_id == "vllm-inference"
        assert models[0].model_type == "llm"

    def test_fms_guardrails_inference(self, minio_pod, qwen_isvc, llamastack_client):
        chat_completion_response = llamastack_client.inference.chat_completion(
            messages=[
                {"role": "system", "content": "You are a friendly assistant."},
                {"role": "user", "content": "Only respond with ack"},
            ],
            model_id="/mnt/models",
        )

        assert chat_completion_response.completion_message.content != ""

    def test_fms_guardrails_register_shield(
        self, current_client_token, qwen_isvc, patched_llamastack_deployment_tls_certs, llamastack_client
    ):
        trustyai_fms_provider_id = "trustyai_fms"
        shield_params = {
            "type": "content",
            "confidence_threshold": 0.5,
            "detectors": {"regex": {"detector_params": {"regex": ["email", "ssn"]}}},
            "auth_token": current_client_token,
            "verify_ssl": True,
            "ssl_cert_path": "/etc/llama/certs/orch-certificate.crt",
        }
        llamastack_client.shields.register(
            shield_id=PII_REGEX_SHIELD_ID,
            provider_shield_id=PII_REGEX_SHIELD_ID,
            provider_id=trustyai_fms_provider_id,
            params=shield_params,
            timeout=120,
        )
        shields = llamastack_client.shields.list()

        assert len(shields) == 1
        assert shields[0].identifier == PII_REGEX_SHIELD_ID
        assert shields[0].provider_id == trustyai_fms_provider_id
        assert shields[0].params == shield_params

    def test_fms_guardrails_run_shield(self, llamastack_client):
        run_shields_response = llamastack_client.safety.run_shield(
            shield_id=PII_REGEX_SHIELD_ID,
            messages=[
                {
                    "content": PROMPT_WITH_PII,
                    "role": "system",
                },
            ],
            params={},
        )

        assert run_shields_response.violation is not None, "Expected shield violation to be present"
        assert run_shields_response.violation.metadata["status"] == "violation", (
            "Expected run shields response status to be 'violation'"
        )
        assert run_shields_response.violation.metadata["results"][0]["detection_type"] == "pii", (
            "Expected detection type to be 'pii'"
        )
        assert run_shields_response.violation.metadata["shield_id"] == "regex", "Expected shield_id to be 'regex'"
