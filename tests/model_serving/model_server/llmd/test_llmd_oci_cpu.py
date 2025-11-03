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

BASIC_LLMD_PARAMS = [({"name": "llmd-comprehensive-test"}, "basic")]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmd_inference_service",
    BASIC_LLMD_PARAMS,
    indirect=True,
)
class TestLLMDOCICPUInference:
    """LLMD inference testing with OCI storage and CPU runtime using vLLM.

    Tests CPU-based LLMD inference using OCI container registry for model storage.
    This test validates the basic LLMD functionality with CPU resources and
    ensures proper integration with the TinyLlama model from OCI storage.
    """

    def test_llmd_oci_cpu(self, unprivileged_client, llmd_gateway, llmd_inference_service):
        """Test LLMD inference with OCI storage and CPU runtime."""
        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llmd_inference_service), "LLMInferenceService should be ready"

        verify_inference_response_llmd(
            llm_service=llmd_inference_service,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
            model_name=llmd_inference_service.name,
        )

        verify_llmd_no_failed_pods(client=unprivileged_client, llm_service=llmd_inference_service)
