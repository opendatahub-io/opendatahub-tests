import pytest

from tests.model_serving.model_server.upgrade.utils import (
    verify_inference_generation,
    verify_isvc_pods_not_restarted_against_baseline,
    verify_model_status_loaded,
    verify_serving_runtime_generation,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import Protocols
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG

pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.usefixtures("capture_upgrade_baseline")
class TestPreUpgradeModelServer:
    """Validate raw deployment model inference before an operator upgrade.

    Steps:
        1. Deploy an OVMS inference service as a raw deployment.
        2. Send an inference request via the internal HTTP route and verify a successful response.
        3. Capture baseline values (generation, restart counts) to ConfigMap.
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
        2. Verify the inference service was not modified during the upgrade.
        3. Verify the serving runtime was not modified during the upgrade.
        4. Verify pods have not restarted beyond pre-upgrade baseline.
        5. Send an inference request via the internal HTTP route and verify a successful response.
    """

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="isvc_exists")
    def test_raw_deployment_post_upgrade_inference_exists(self, inference_service_fixture):
        """Test that raw deployment inference service exists after upgrade"""
        assert inference_service_fixture.exists

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["isvc_exists"])
    def test_raw_deployment_post_upgrade_inference_not_modified(
        self, inference_service_fixture, upgrade_baseline_fixture
    ):
        """Test that the raw deployment inference service is not modified in upgrade"""
        baseline = upgrade_baseline_fixture.get(inference_service_fixture.name, {})
        expected = baseline.get("isvc_observed_generation", 1)
        verify_inference_generation(isvc=inference_service_fixture, expected_generation=expected)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["isvc_exists"])
    def test_raw_deployment_post_upgrade_runtime_not_modified(
        self, inference_service_fixture, upgrade_baseline_fixture
    ):
        """Test that the raw deployment runtime is not modified in upgrade"""
        baseline = upgrade_baseline_fixture.get(inference_service_fixture.name, {})
        expected = baseline.get("runtime_generation", 1)
        verify_serving_runtime_generation(isvc=inference_service_fixture, expected_generation=expected)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["isvc_exists"])
    def test_raw_deployment_post_upgrade_pods_not_restarted(
        self,
        admin_client,
        inference_service_fixture,
        upgrade_baseline_fixture,
    ):
        """Verify InferenceService pods have not restarted beyond pre-upgrade baseline"""
        baseline = upgrade_baseline_fixture.get(inference_service_fixture.name, {})
        verify_isvc_pods_not_restarted_against_baseline(
            client=admin_client,
            isvc=inference_service_fixture,
            baseline_restart_counts=baseline.get("pod_restart_counts", {}),
        )

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


class TestPostUpgradeNewIsvcCreation:
    """Verify that the upgraded control plane can create new InferenceServices.

    Creates a fresh ISVC on the upgraded kserve-controller/webhook to validate
    that the creation path works, not just preservation of pre-existing resources.
    """

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="new_isvc_created")
    def test_create_new_isvc_post_upgrade(self, new_isvc_inference_service_fixture):
        """Verify a new InferenceService can be created on the upgraded control plane"""
        assert new_isvc_inference_service_fixture is not None, "Fixture returned None; only runs post-upgrade"
        assert new_isvc_inference_service_fixture.exists, (
            f"Newly created InferenceService {new_isvc_inference_service_fixture.name} does not exist"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["new_isvc_created"])
    def test_new_isvc_post_upgrade_model_loaded(self, new_isvc_inference_service_fixture):
        """Verify newly created ISVC reaches Loaded state on upgraded control plane"""
        verify_model_status_loaded(isvc=new_isvc_inference_service_fixture)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["new_isvc_created"])
    def test_new_isvc_post_upgrade_generation(self, new_isvc_inference_service_fixture):
        """Verify newly created ISVC has generation=1 (fresh resource)"""
        verify_inference_generation(isvc=new_isvc_inference_service_fixture, expected_generation=1)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["new_isvc_created"])
    def test_new_isvc_post_upgrade_inference(self, new_isvc_inference_service_fixture):
        """Verify inference works on a freshly created ISVC post-upgrade"""
        verify_inference_response(
            inference_service=new_isvc_inference_service_fixture,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
