import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    Annotations,
    KServeDeploymentType,
    ModelFormat,
    ModelInferenceRuntime,
    ModelStoragePath,
    ModelVersion,
    Protocols,
)
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [pytest.mark.tier1, pytest.mark.rawdeployment]

RUNTIME_PARAMS = {
    "runtime-name": ModelInferenceRuntime.MLSERVER_RUNTIME,
    "model-format": {ModelFormat.ONNX: ModelVersion.V1},
}
INFERENCE_SERVICE_PARAMS = {
    "model-version": ModelVersion.V1,
    "model-dir": ModelStoragePath.MNIST_ONNX,
}


@pytest.mark.parametrize(
    "default_deployment_mode_in_dsc, unprivileged_model_namespace, "
    "mlserver_kserve_serving_runtime, dsc_mlserver_inference_service",
    [
        pytest.param(
            {"default-deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT},
            {"name": "dsc-raw"},
            RUNTIME_PARAMS,
            {
                "name": f"{ModelFormat.ONNX}-{KServeDeploymentType.RAW_DEPLOYMENT.lower()}",
                **INFERENCE_SERVICE_PARAMS,
            },
        )
    ],
    indirect=True,
)
class TestKServeDSCRawDefaultDeploymentMode:
    """Validate that the DSC default deployment mode propagates to inference services.

    Steps:
        1. Set the DSC default deployment mode to RawDeployment.
        2. Deploy an MLServer inference service without explicitly specifying a deployment mode.
        3. Verify the ISVC annotation reflects RawDeployment as the deployment mode.
        4. Send an inference request and verify a successful response over HTTPS.
    """

    def test_isvc_contains_raw_deployment_mode(self, default_deployment_mode_in_dsc, dsc_mlserver_inference_service):
        """Verify that default deployment mode is set to raw in inference service."""
        actual_mode = dsc_mlserver_inference_service.instance.metadata.annotations[Annotations.KserveIo.DEPLOYMENT_MODE]
        assert actual_mode in KServeDeploymentType.RAW_DEPLOYMENT_MODES, (
            f"Expected one of {KServeDeploymentType.RAW_DEPLOYMENT_MODES}, got {actual_mode!r}"
        )

    def test_kserve_dsc_raw_default_deployment_mode(
        self, default_deployment_mode_in_dsc, dsc_mlserver_inference_service
    ):
        """Verify that Raw deployment model can be deployed without specifying in isvc"""
        verify_inference_response(
            inference_service=dsc_mlserver_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
