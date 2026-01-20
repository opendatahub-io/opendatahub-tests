import pytest
import yaml
from simple_logger.logger import get_logger

from tests.llama_stack.constants import LlamaStackProviders

from utilities.constants import MinIo, CHAT_GENERATION_CONFIG, BUILTIN_DETECTOR_CONFIG, QWEN_MODEL_NAME

LOGGER = get_logger(name=__name__)
SECURE_SHIELD_ID: str = "secure_shield"


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, "
    "orchestrator_config, guardrails_orchestrator, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-llamastack-gorch"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "orchestrator_config_data": {
                    "config.yaml": yaml.dump({
                        "chat_generation": CHAT_GENERATION_CONFIG,
                        "detectors": BUILTIN_DETECTOR_CONFIG,
                    })
                },
            },
            {"orchestrator_config": True, "enable_built_in_detectors": True, "enable_guardrails_gateway": False},
            {
                "vllm_url_fixture": "qwen_isvc_url",
                "inference_model": QWEN_MODEL_NAME,
                "fms_orchestrator_url_fixture": "guardrails_orchestrator_url",
                "embedding_provider": "sentence-transformers",
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.usefixtures("patched_dsc_kserve_headed", "orchestrator_config", "guardrails_orchestrator")
@pytest.mark.model_explainability
class TestLlamaStackFMSGuardrailsProvider:
    """
    Adds basic tests for the LlamaStack FMS Guardrails provider.

    Given a basic guardrails setup (generator model + detectors),
    and a llama-stack distribution and client:

    1. Register a shield (detectors)
    2. Test a basic detection (PII) by using run_shields
    """

    def test_fms_guardrails_register_shield(
        self,
        current_client_token,
        minio_pod,
        minio_data_connection,
        patched_llamastack_deployment_tls_certs,
        llama_stack_client,
    ):
        shield_params = {
            "type": "content",
            "confidence_threshold": 0.5,
            "message_types": ["system", "user"],
            "auth_token": current_client_token,
            "verify_ssl": True,
            "ssl_cert_path": "/etc/llama/certs/orch-certificate.crt",
            "detectors": {"regex": {"detector_params": {"regex": ["email", "ssn", "credit-card", "^hello$"]}}},
        }

        llama_stack_client.shields.register(
            shield_id=SECURE_SHIELD_ID,
            provider_shield_id=SECURE_SHIELD_ID,
            provider_id=LlamaStackProviders.Safety.TRUSTYAI_FMS,
            params=shield_params,
            timeout=120,
        )
        shields = llama_stack_client.shields.list()

        assert len(shields) == 1
        assert shields[0].identifier == SECURE_SHIELD_ID
        assert shields[0].provider_id == LlamaStackProviders.Safety.TRUSTYAI_FMS
        assert shields[0].params == shield_params

    def test_fms_guardrails_run_shield(self, minio_pod, minio_data_connection, llama_stack_client):
        run_shields_response = llama_stack_client.safety.run_shield(
            shield_id=SECURE_SHIELD_ID,
            messages=[
                {
                    "content": "My email is johndoe@example.com",
                    "role": "user",
                }
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
        assert run_shields_response.violation.metadata["shield_id"] == SECURE_SHIELD_ID, (
            f"Expected shield_id to be {SECURE_SHIELD_ID}"
        )

    def test_fms_moderations(self, minio_pod, minio_data_connection, llama_stack_client):
        """Test to check if moderations API works with the registered shield above.
        refer: https://github.com/m-misiura/demos/tree/main/fms_safety_provider_lllamastack
        """
        moderations_response = llama_stack_client.moderations.create(
            input="My email is juandoe@example.com", model=SECURE_SHIELD_ID
        )
        assert len(moderations_response.results) > 0, "Moderation response results was empty."
        assert moderations_response.model == SECURE_SHIELD_ID, "Moderation shield_id did not match the model."
        assert moderations_response.results[0].categories["pii"], "The pii moderation category was not triggered."
        assert moderations_response.results[0].flagged, "The moderation was not flagged."
        assert moderations_response.results[0].metadata["status"] == "violation"
        assert moderations_response.results[0].metadata["detection_type"] == "pii"
        assert moderations_response.results[0].metadata["text"] == "My email is juandoe@example.com"
