"""
Test module for GPU-based MLServer model serving using OCI Model Car images.

This module validates GPU accelerated inference with MLServer using models
served from OCI container images.
"""

import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    ModelCarImage,
    ModelFormat,
    ModelName,
    Protocols,
    RuntimeTemplates,
)
from utilities.inference_utils import Inference
from utilities.infra import get_pods_by_isvc_label
from utilities.manifests.onnx import ONNX_RESNET50_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.model_server_gpu,
    pytest.mark.rawdeployment,
    pytest.mark.usefixtures("skip_if_no_gpu_available"),
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template, gpu_model_car_inference_service",
    [
        pytest.param(
            {"name": f"{ModelFormat.ONNX}-gpu-model-car"},
            {
                "name": f"{ModelName.MNIST}-gpu-runtime",
                "template-name": RuntimeTemplates.MLSERVER_CUDA,
                "multi-model": False,
            },
            {
                "storage-uri": ModelCarImage.MLSERVER_ONNX,
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "gpu-count": 1,
            },
            id="mlserver-gpu-raw-deployment",
        ),
    ],
    indirect=True,
)
class TestKserveGpuModelCar:
    """Test GPU accelerated MLServer model serving with OCI Model Car images."""

    def test_gpu_model_car_no_restarts(self, gpu_model_car_inference_service):
        """Verify that GPU model pod doesn't restart"""
        pod = get_pods_by_isvc_label(
            client=gpu_model_car_inference_service.client,
            isvc=gpu_model_car_inference_service,
        )[0]
        restarted_containers = [
            container.name for container in pod.instance.status.containerStatuses if container.restartCount > 2
        ]
        assert not restarted_containers, f"Containers {restarted_containers} restarted"

    def test_gpu_model_car_using_rest(self, gpu_model_car_inference_service):
        """Verify GPU model query with token using REST"""
        verify_inference_response(
            inference_service=gpu_model_car_inference_service,
            inference_config=ONNX_RESNET50_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
