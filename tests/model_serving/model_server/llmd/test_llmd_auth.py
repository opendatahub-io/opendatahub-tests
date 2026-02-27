import pytest

from tests.model_serving.model_server.llmd.utils import (
    verify_gateway_status,
    verify_llm_service_status,
)
from utilities.constants import Protocols
from utilities.llmd_utils import verify_inference_response_llmd
from utilities.manifests.tinyllama import TINYLLAMA_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.llmd_cpu,
    pytest.mark.smoke,
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [({"name": "llmd-auth-test"})],
    indirect=True,
)
class TestLLMISVCAuth:
    """Authentication testing for LLMD."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_auth_resources(
        self,
        llmd_gateway,
        llmisvc_auth,
        llmisvc_auth_token,
        llmisvc_auth_view_role,
        llmisvc_auth_role_binding,
    ):
        """Set up gateway, LLMInferenceServices, and tokens once for all tests."""
        llmisvc_auth_prefix = "llmisvc-auth-user-"
        sa_prefix = "llmisvc-auth-sa-"

        # Create LLMInferenceService instances using the factory fixture
        llmisvc_user_a, sa_user_a = llmisvc_auth(
            service_name=llmisvc_auth_prefix + "a",
            service_account_name=sa_prefix + "a",
        )
        llmisvc_user_b, sa_user_b = llmisvc_auth(
            service_name=llmisvc_auth_prefix + "b",
            service_account_name=sa_prefix + "b",
        )

        # Create tokens with all RBAC resources
        token_user_a = llmisvc_auth_token(
            service_account=sa_user_a,
            llmisvc=llmisvc_user_a,
            view_role_factory=llmisvc_auth_view_role,
            role_binding_factory=llmisvc_auth_role_binding,
        )
        token_user_b = llmisvc_auth_token(
            service_account=sa_user_b,
            llmisvc=llmisvc_user_b,
            view_role_factory=llmisvc_auth_view_role,
            role_binding_factory=llmisvc_auth_role_binding,
        )

        # Verify all resources are ready
        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llmisvc_user_a), "LLMInferenceService user A should be ready"
        assert verify_llm_service_status(llmisvc_user_b), "LLMInferenceService user B should be ready"

        # Store resources as class attributes for use in tests
        TestLLMISVCAuth.llmisvc_user_a = llmisvc_user_a
        TestLLMISVCAuth.llmisvc_user_b = llmisvc_user_b
        TestLLMISVCAuth.token_user_a = token_user_a
        TestLLMISVCAuth.token_user_b = token_user_b

    def test_llmisvc_authorized(self):
        """Test that authorized users can access their own LLMInferenceServices."""
        # Verify inference for user A with user A's token (should succeed)
        verify_inference_response_llmd(
            llm_service=self.llmisvc_user_a,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=False,
            model_name=self.llmisvc_user_a.name,
            token=self.token_user_a,
            authorized_user=True,
        )

        # Verify inference for user B with user B's token (should succeed)
        verify_inference_response_llmd(
            llm_service=self.llmisvc_user_b,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=False,
            model_name=self.llmisvc_user_b.name,
            token=self.token_user_b,
            authorized_user=True,
        )

    def test_llmisvc_unauthorized(self):
        """Test that unauthorized access to LLMInferenceServices is properly blocked."""
        # Verify that user B's token cannot access user A's service (should fail)
        verify_inference_response_llmd(
            llm_service=self.llmisvc_user_a,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=False,
            model_name=self.llmisvc_user_a.name,
            token=self.token_user_b,
            authorized_user=False,
        )

        # Verify that accessing user A's service without a token fails
        verify_inference_response_llmd(
            llm_service=self.llmisvc_user_a,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=False,
            authorized_user=False,
        )
