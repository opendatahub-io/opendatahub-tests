import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    Annotations,
    KServeDeploymentType,
    ModelAndFormat,
    ModelFormat,
    ModelInferenceRuntime,
    ModelStoragePath,
    ModelVersion,
    Protocols,
)
from utilities.inference_utils import Inference
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG

pytestmark = [pytest.mark.sanity, pytest.mark.rawdeployment]

RUNTIME_PARAMS = {
    "runtime-name": ModelInferenceRuntime.OPENVINO_KSERVE_RUNTIME,
    "model-format": {ModelAndFormat.OPENVINO_IR: ModelVersion.OPSET1},
}
INFERENCE_SERVICE_PARAMS = {
    "model-version": ModelVersion.OPSET1,
    "model-dir": ModelStoragePath.KSERVE_OPENVINO_EXAMPLE_MODEL,
}


@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "default_deployment_mode_in_dsc, unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_inference_service",
    [
        pytest.param(
            {"default-deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT},
            {"name": "dsc-raw"},
            RUNTIME_PARAMS,
            {
                "name": f"{ModelFormat.OPENVINO}-{KServeDeploymentType.RAW_DEPLOYMENT.lower()}",
                **INFERENCE_SERVICE_PARAMS,
            },
        )
    ],
    indirect=True,
)
class TestKServeDSCRawDefaultDeploymentMode:
    def test_isvc_contains_raw_deployment_mode(self, default_deployment_mode_in_dsc, ovms_inference_service):
        """Verify that default deployment mode is set to raw in inference service."""
        assert (
            ovms_inference_service.instance.metadata.annotations[Annotations.KserveIo.DEPLOYMENT_MODE]
            == KServeDeploymentType.RAW_DEPLOYMENT
        )

    def test_kserve_dsc_raw_default_deployment_mode(self, default_deployment_mode_in_dsc, ovms_inference_service):
        """Verify that Raw deployment model can be deployed without specifying in isvc"""
        verify_inference_response(
            inference_service=ovms_inference_service,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
