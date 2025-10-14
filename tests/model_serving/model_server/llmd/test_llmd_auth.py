import pytest

from tests.model_serving.model_server.llmd.utils import (
    verify_llm_service_status,
    verify_gateway_status,
)
from utilities.constants import Protocols
from utilities.llmd_utils import verify_inference_response_llmd
from utilities.manifests.tinyllama import TINYLLAMA_INFERENCE_CONFIG
from utilities.llmd_constants import ModelStorage, ContainerImages

pytestmark = [
    pytest.mark.llmd_cpu,
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [({"name": "llmd-auth-test"})],
    indirect=True,
)
class TestLLMISVCAuth:
    """Authentication testing for LLMD."""

    def test_llmisvc_auth(self, llmd_gateway, llmisvc_auth, llmisvc_auth_token):
        """Test LLMD inference with S3 storage for two users using authentication tokens."""

        llmisvc_auth_prefix = "llmisvc-auth-user-"
        sa_prefix = "llmisvc-auth-sa-"

        # Create LLMInferenceService instances using the factory fixture
        llmisvc_user_a, sa_user_a = llmisvc_auth(
            service_name=llmisvc_auth_prefix + "a",
            service_account_name=sa_prefix + "a",
            storage_uri=ModelStorage.HF_TINYLLAMA,
            container_image=ContainerImages.VLLM_CPU,
        )
        llmisvc_user_b, sa_user_b = llmisvc_auth(
            service_name=llmisvc_auth_prefix + "b",
            service_account_name=sa_prefix + "b",
            storage_uri=ModelStorage.HF_TINYLLAMA,
            container_image=ContainerImages.VLLM_CPU,
        )

        # Create tokens with all RBAC resources
        token_user_a = llmisvc_auth_token(service_account=sa_user_a, llmisvc=llmisvc_user_a)
        token_user_b = llmisvc_auth_token(service_account=sa_user_b, llmisvc=llmisvc_user_b)

        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llmisvc_user_a), "LLMInferenceService user A should be ready"
        assert verify_llm_service_status(llmisvc_user_b), "LLMInferenceService user B should be ready"

        # Verify inference for user A with user A's token (should succeed)
        verify_inference_response_llmd(
            llm_service=llmisvc_user_a,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
            model_name=llmisvc_user_a.name,
            token=token_user_a,
            authorized_user=True,
        )

        # Verify inference for user B with user B's token (should succeed)
        verify_inference_response_llmd(
            llm_service=llmisvc_user_b,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
            model_name=llmisvc_user_b.name,
            token=token_user_b,
            authorized_user=True,
        )

        # Verify that user B's token cannot access user A's service (should fail)
        verify_inference_response_llmd(
            llm_service=llmisvc_user_a,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
            model_name=llmisvc_user_a.name,
            token=token_user_b,
            authorized_user=False,
        )

        # Verify that accessing user A's service without a token fails
        verify_inference_response_llmd(
            llm_service=llmisvc_user_a,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=False,
            authorized_user=False,
        )
