import pytest
import structlog

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import Protocols
from utilities.manifests.vllm import VLLM_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.tier2,
    pytest.mark.rawdeployment,
    pytest.mark.usefixtures("skip_if_no_gpu_nodes"),
    pytest.mark.multinode,
    pytest.mark.model_server_gpu,
    pytest.mark.gpu,
]

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, multi_node_oci_inference_service",
    [
        pytest.param(
            {"name": "gpu-oci-multi-node"},
            {"name": "multi-oci-vllm"},
        )
    ],
    indirect=True,
)
class TestOciMultiNode:
    """Validate multi-node GPU inference using OCI-based model storage on KServe.

    Steps:
        1. Deploy a multi-node vLLM inference service using an OCI model image.
        2. Send an inference request over the external HTTPS route.
        3. Verify the model returns a successful completion response.
    """

    def test_oci_multi_node_basic_external_inference(self, multi_node_oci_inference_service):
        """Test multi node basic inference"""
        verify_inference_response(
            inference_service=multi_node_oci_inference_service,
            inference_config=VLLM_INFERENCE_CONFIG,
            inference_type="completions",
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
