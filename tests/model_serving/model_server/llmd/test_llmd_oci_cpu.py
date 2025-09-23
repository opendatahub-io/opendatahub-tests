import pytest

from tests.model_serving.model_server.llmd.utils import (
    verify_llm_service_status,
    verify_gateway_status,
    verify_llmd_pods_not_restarted,
)
from utilities.constants import Protocols
from utilities.llmd_utils import verify_inference_response_llmd
from utilities.llmd_constants import BASIC_LLMD_PARAMS
from utilities.manifests.opt125m_cpu import OPT125M_CPU_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.llmd_cpu,
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmd_gateway, llmd_inference_service",
    BASIC_LLMD_PARAMS,
    indirect=True,
)
class TestLLMDOCICPUInference:
    """LLMD inference testing with OCI storage and CPU runtime using vLLM."""

    def test_llmd_oci(self, admin_client, llmd_gateway, llmd_inference_service):
        """Test LLMD inference with OCI storage and CPU runtime."""
        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llmd_inference_service), "LLMInferenceService should be ready"

        verify_inference_response_llmd(
            llm_service=llmd_inference_service,
            inference_config=OPT125M_CPU_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
        )

        verify_llmd_pods_not_restarted(client=admin_client, llm_service=llmd_inference_service)
