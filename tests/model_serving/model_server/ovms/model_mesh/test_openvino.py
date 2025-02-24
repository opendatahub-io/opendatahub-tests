import re
from pprint import pprint

import pytest
from semver import Version

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    ModelStoragePath,
    Protocols,
)
from utilities.inference_utils import Inference
from utilities.infra import get_pods_by_isvc_label, get_product_version
from utilities.manifests.openvino import OPENVINO_INFERENCE_CONFIG

pytestmark = [pytest.mark.modelmesh]


@pytest.mark.parametrize(
    "model_namespace, http_s3_openvino_model_mesh_inference_service",
    [
        pytest.param(
            {"name": "model-mesh-openvino", "modelmesh-enabled": True},
            {"model-path": ModelStoragePath.OPENVINO_EXAMPLE_MODEL},
        )
    ],
    indirect=True,
)
class TestOpenVINOModelMesh:
    @pytest.mark.smoke
    @pytest.mark.dependency(name="test_product_version_in_model_mesh_container")
    def test_product_version_in_model_mesh_container(
        self,
        admin_client,
        http_s3_openvino_model_mesh_inference_service,
        http_s3_ovms_model_mesh_serving_runtime,
    ):
        """Verify model mesh container contains the correct product version"""
        pod = get_pods_by_isvc_label(
            client=http_s3_openvino_model_mesh_inference_service.client,
            isvc=http_s3_openvino_model_mesh_inference_service,
            runtime_name=http_s3_ovms_model_mesh_serving_runtime.name,
        )[0]

        mm_log = pod.log(container="mm")

        if version_match := re.search(r"Registering ModelMesh Service version as \"v(\d\.\d+\.\d+)", mm_log):
            truncated_mm_log = re.match(r"^(.*?)service starting", mm_log, re.DOTALL)
            product_version = get_product_version(admin_client=admin_client)
            mm_version = version_match.group(1)
            assert Version.parse(mm_version) == product_version, (
                f"mm container version {mm_version} does not "
                f"match product version {product_version}. Log: {pprint(truncated_mm_log.group(1))}"
            )

        else:
            raise ValueError(f"Model mesh container does not contain a product version, mm log: {mm_log}")

    @pytest.mark.smoke
    @pytest.mark.dependency(depends=["test_product_version_in_model_mesh_container"])
    @pytest.mark.polarion("ODS-2053", "ODS-2054")
    def test_model_mesh_openvino_rest_inference_internal_route(self, http_s3_openvino_model_mesh_inference_service):
        """Test OpenVINO ModelMesh inference with internal route"""
        verify_inference_response(
            inference_service=http_s3_openvino_model_mesh_inference_service,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    @pytest.mark.sanity
    @pytest.mark.polarion("ODS-1920")
    def test_model_mesh_openvino_inference_with_token(
        self,
        patched_model_mesh_sr_with_authentication,
        http_s3_openvino_model_mesh_inference_service,
        model_mesh_inference_token,
    ):
        """Test OpenVINO ModelMesh inference with token"""
        verify_inference_response(
            inference_service=http_s3_openvino_model_mesh_inference_service,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
            token=model_mesh_inference_token,
        )
