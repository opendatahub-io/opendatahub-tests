import pytest

from tests.model_serving.model_server.llmd.utils import (
    verify_gateway_status,
    verify_llm_service_status,
    verify_llmd_no_failed_pods,
)
from utilities.constants import Protocols
from utilities.llmd_utils import verify_inference_response_llmd
from utilities.manifests.tinyllama import TINYLLAMA_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.llmd_cpu,
    pytest.mark.smoke,
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmd_inference_service_s3",
    [pytest.param({"name": "llmd-s3-test"}, {"name_suffix": "s3"}, id="s3-cpu-basic")],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config")
class TestLLMDS3Inference:
    """LLMD inference testing with S3 storage and CPU runtime using vLLM."""

    def test_llmd_s3_cpu(self, unprivileged_client, llmd_gateway, llmd_inference_service_s3):
        """Test LLMD inference with S3 storage and CPU runtime."""
        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llmd_inference_service_s3), "LLMInferenceService should be ready"

        verify_inference_response_llmd(
            llm_service=llmd_inference_service_s3,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
            model_name=llmd_inference_service_s3.name,
        )

        verify_llmd_no_failed_pods(client=unprivileged_client, llm_service=llmd_inference_service_s3)
