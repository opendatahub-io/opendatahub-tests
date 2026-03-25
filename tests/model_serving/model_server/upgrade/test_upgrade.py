import pytest

from tests.model_serving.model_server.upgrade.utils import (
    verify_inference_generation,
    verify_serving_runtime_generation,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import Protocols
from utilities.inference_utils import Inference
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG

pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


class TestPreUpgradeModelServer:
    """Validate raw deployment model inference before an operator upgrade.

    Steps:
        1. Deploy an OVMS inference service as a raw deployment.
        2. Send an inference request via the internal HTTP route and verify a successful response.
    """

    @pytest.mark.pre_upgrade
    def test_raw_deployment_pre_upgrade_inference(self, inference_service_fixture):
        """Test raw deployment model inference using internal route before upgrade"""
        verify_inference_response(
            inference_service=inference_service_fixture,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )


class TestPostUpgradeModelServer:
    """Validate raw deployment model integrity and inference after an operator upgrade.

    Steps:
        1. Verify the inference service still exists after the upgrade.
        2. Verify the inference service was not modified during the upgrade (generation=1).
        3. Verify the serving runtime was not modified during the upgrade (generation=1).
        4. Send an inference request via the internal HTTP route and verify a successful response.
    """

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="isvc_exists")
    def test_raw_deployment_post_upgrade_inference_exists(self, inference_service_fixture):
        """Test that raw deployment inference service exists after upgrade"""
        assert inference_service_fixture.exists

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["isvc_exists"])
    def test_raw_deployment_post_upgrade_inference_not_modified(self, inference_service_fixture):
        """Test that the raw deployment inference service is not modified in upgrade"""
        verify_inference_generation(isvc=inference_service_fixture, expected_generation=1)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["isvc_exists"])
    def test_raw_deployment_post_upgrade_runtime_not_modified(self, inference_service_fixture):
        """Test that the raw deployment runtime is not modified in upgrade"""
        verify_serving_runtime_generation(isvc=inference_service_fixture, expected_generation=1)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["isvc_exists"])
    def test_raw_deployment_post_upgrade_inference(self, inference_service_fixture):
        """Test raw deployment model inference using internal route after upgrade"""
        verify_inference_response(
            inference_service=inference_service_fixture,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )
