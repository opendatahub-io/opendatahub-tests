import pytest

from tests.model_serving.model_server.authentication.utils import verify_inference_response
from utilities.constants import ModelFormat, ModelStoragePath, Protocols, RuntimeQueryKeys
from utilities.inference_utils import Inference


@pytest.mark.parametrize(
    "model_namespace, ci_s3_storage_uri",
    [
        pytest.param(
            {"name": "kserve-serverless-openvino"},
            {"model-dir": ModelStoragePath.KSERVE_OPENVINO_EXAMPLE_MODEL},
        )
    ],
    indirect=True,
)
class TestOpenVINO:
    @pytest.mark.smoke
    @pytest.mark.polarion("ODS-2626")
    @pytest.mark.jira("RHOAIENG-9045")
    def test_serverless_openvino_rest_inference(self, http_openvino_serverless_inference_service):
        verify_inference_response(
            inference_service=http_openvino_serverless_inference_service,
            runtime=RuntimeQueryKeys.OPENVINO_KSERVE_RUNTIME,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.OPENVINO,
            use_default_query=True,
        )
