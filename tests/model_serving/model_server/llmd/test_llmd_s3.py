import pytest

from tests.model_serving.model_server.llmd.utils import (
    verify_llm_service_status,
    verify_gateway_status,
    verify_llmd_pods_not_restarted,
)
from utilities.constants import Protocols
from utilities.llmd_utils import verify_inference_response_llmd
from utilities.manifests.opt125m_cpu import OPT125M_CPU_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.llmd_cpu,
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmd_gateway, llmd_inference_service_s3",
    [({"name": "llmd-s3-test"}, "openshift-default", {"storage_path": "opt-125m/"})],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config")
class TestLLMDS3Inference:
    """LLMD inference testing with S3 storage."""

    def test_llmd_s3(self, admin_client, llmd_gateway, llmd_inference_service_s3):
        """Test LLMD inference with S3 storage."""
        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llmd_inference_service_s3), "LLMInferenceService should be ready"

        verify_inference_response_llmd(
            llm_service=llmd_inference_service_s3,
            inference_config=OPT125M_CPU_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
        )

        verify_llmd_pods_not_restarted(client=admin_client, llm_service=llmd_inference_service_s3)
