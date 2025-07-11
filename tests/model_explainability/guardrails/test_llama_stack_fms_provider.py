import pytest
from simple_logger.logger import get_logger

from tests.model_explainability.guardrails.test_guardrails import MNT_MODELS
from utilities.constants import MinIo

LOGGER = get_logger(name=__name__)
PII_REGEX_SHIELD_ID = "regex"


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection",
    [
        pytest.param(
            {"name": "test-guardrails-lls"},
            MinIo.PodConfig.QWEN_MINIO_CONFIG,
            {"bucket": "llms"},
        )
    ],
    indirect=True,
)
class TestLlamaStackFMSGuardrailsProvider:
    """
    Adds basic tests for the LlamaStack FMS Guardrails provider.

    Given a basic guardrails setup (generator model + detectors),
    and a llama-stack distribution and client:

    1. Register the generator model via lls client
    2. Test that we can run inferences on said model via lls client
    3. Register the shields (detectors)
    4. TODO: Add tests for run_shields
    """

    def test_fms_guardrails_register_model(self, lls_client):
        provider_id = "vllm-inference"
        model_type = "llm"
        lls_client.models.register(provider_id=provider_id, model_type=model_type, model_id=MNT_MODELS)
        models = lls_client.models.list()

        # We only need to check the first model;
        # second is a granite embedding model present by default
        assert len(models) == 2
        assert models[0].identifier == MNT_MODELS
        assert models[0].provider_id == "vllm-inference"
        assert models[0].model_type == "llm"

    def test_fms_guardrails_inference(self, lls_client):
        chat_completion_response = lls_client.inference.chat_completion(
            messages=[
                {"role": "system", "content": "You are a friendly assistant."},
                {"role": "user", "content": "Only respond with ack"},
            ],
            model_id="/mnt/models",
        )

        assert chat_completion_response.completion_message.content != ""

    def test_fms_guardrails_register_shield(self, lls_client):
        trustyai_fms_provider_id = "trustyai_fms"
        shield_params = {
            "type": "content",
            "confidence_threshold": 0.5,
            "detectors": {"regex": {"detector_params": {"regex": ["email", "ssn"]}}},
        }
        lls_client.shields.register(
            shield_id=PII_REGEX_SHIELD_ID,
            provider_shield_id=PII_REGEX_SHIELD_ID,
            provider_id=trustyai_fms_provider_id,
            params=shield_params,
        )
        shields = lls_client.shields.list()

        assert len(shields) == 1
        assert shields[0].identifier == PII_REGEX_SHIELD_ID
        assert shields[0].provider_id == trustyai_fms_provider_id
        assert shields[0].params == shield_params
