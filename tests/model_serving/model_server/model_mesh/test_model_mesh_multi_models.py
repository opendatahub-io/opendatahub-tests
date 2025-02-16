import re

import pytest
from semver import Version

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    ModelAndFormat,
    ModelStoragePath,
    ModelVersion,
    Protocols,
)
from utilities.inference_utils import Inference
from utilities.infra import get_pods_by_isvc_label, get_product_version
from utilities.manifests.openvino import OPENVINO_INFERENCE_CONFIG

pytestmark = [pytest.mark.modelmesh, pytest.mark.sanity]


@pytest.mark.parametrize(
    "model_namespace, http_s3_openvino_model_mesh_inference_service, "
    "http_s3_openvino_second_model_mesh_inference_service",
    [
        pytest.param(
            {"name": "model-mesh-multi-model", "modelmesh-enabled": True},
            {"model-path": ModelStoragePath.OPENVINO_EXAMPLE_MODEL},
            {
                "model-path": ModelStoragePath.OPENVINO_VEHICLE_DETECTION,
                "runtime-fixture-name": "http_s3_ovms_model_mesh_serving_runtime",
                "model-format": ModelAndFormat.OPENVINO_IR,
                "model-version": ModelVersion.OPSET1,
            },
        )
    ],
    indirect=True,
)
class TestOpenVINOModelMeshMultiModels:
    def test_model_mesh_openvino_inference_with_tensorflow(
        self,
        http_s3_openvino_model_mesh_inference_service,
        http_s3_openvino_second_model_mesh_inference_service,
    ):
        """Test inference with OpenVINO model when multiple models in the same server"""
        verify_inference_response(
            inference_service=http_s3_openvino_model_mesh_inference_service,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    def test_model_mesh_tensorflow_with_openvino(
        self,
        http_s3_openvino_second_model_mesh_inference_service,
        http_s3_openvino_model_mesh_inference_service,
    ):
        """Test inference with Tensorflow model when multiple models in the same server"""
        verify_inference_response(
            inference_service=http_s3_openvino_second_model_mesh_inference_service,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type="infer-vehicle-detection",
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    def test_product_version_in_model_mesh_container(
        self,
        admin_client,
        http_s3_ovms_model_mesh_serving_runtime,
        http_s3_openvino_model_mesh_inference_service,
        http_s3_openvino_second_model_mesh_inference_service,
    ):
        """Verify model mesh container contains the correct product version"""
        pod = get_pods_by_isvc_label(
            client=http_s3_openvino_model_mesh_inference_service.client,
            isvc=http_s3_openvino_model_mesh_inference_service,
            runtime_name=http_s3_ovms_model_mesh_serving_runtime.name,
        )[0]

        mm_log = pod.log(container="mm")

        if version_match := re.search(r"Registering ModelMesh Service version as \"v(\d\.\d+\.\d+)", mm_log):
            assert Version.parse(version_match.group(1)) == get_product_version(admin_client=admin_client)

        else:
            raise ValueError("Model mesh container does not contain a product version")
