import pytest

from tests.model_serving.model_server.utils import (
    verify_inference_response,
)
from utilities.constants import Protocols
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG


@pytest.mark.parametrize(
    "unprivileged_model_namespace, unprivileged_s3_ovms_raw_inference_service",
    [
        pytest.param(
            {"name": "test-non-admin-raw"},
            {"model-dir": "/test-dir/"},
        )
    ],
    indirect=True,
)
@pytest.mark.sanity
@pytest.mark.rawdeployment
class TestRawUnprivilegedUser:
    @pytest.mark.polarion("ODS-2611")
    def test_non_admin_deploy_raw_and_query_model(
        self,
        unprivileged_s3_ovms_raw_inference_service,
    ):
        """Verify non admin can deploy a Raw model and query using REST"""
        verify_inference_response(
            inference_service=unprivileged_s3_ovms_raw_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )
