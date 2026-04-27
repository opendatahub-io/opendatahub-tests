import pytest

from tests.model_serving.model_server.upgrade.utils import (
    verify_inference_generation,
    verify_isvc_pods_not_restarted_against_baseline,
    verify_model_status_loaded,
    verify_serving_runtime_generation,
    verify_storage_uri_unchanged,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import ModelCarImage, Protocols
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = pytest.mark.rawdeployment


@pytest.mark.usefixtures("capture_model_car_upgrade_baseline")
class TestPreUpgradeModelCarServer:
    """Pre-upgrade tests for Model Car (OCI image) based model serving."""

    @pytest.mark.pre_upgrade
    def test_model_car_pre_upgrade_exists(self, model_car_inference_service_fixture):
        """Verify Model Car InferenceService exists before upgrade"""
        assert model_car_inference_service_fixture.exists, (
            f"InferenceService {model_car_inference_service_fixture.name} does not exist"
        )

    @pytest.mark.pre_upgrade
    def test_model_car_pre_upgrade_model_loaded(self, model_car_inference_service_fixture):
        """Verify Model Car model is in Loaded state before upgrade"""
        verify_model_status_loaded(isvc=model_car_inference_service_fixture)

    @pytest.mark.pre_upgrade
    def test_model_car_pre_upgrade_inference(self, model_car_inference_service_fixture):
        """Verify Model Car inference before upgrade"""
        verify_inference_response(
            inference_service=model_car_inference_service_fixture,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )


class TestPostUpgradeModelCarServer:
    """Post-upgrade tests for Model Car (OCI image) based model serving."""

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="model_car_isvc_exists")
    def test_model_car_post_upgrade_exists(self, model_car_inference_service_fixture):
        """Verify Model Car InferenceService exists after upgrade"""
        assert model_car_inference_service_fixture.exists, (
            f"InferenceService {model_car_inference_service_fixture.name} does not exist after upgrade"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["model_car_isvc_exists"])
    def test_model_car_post_upgrade_model_loaded(self, model_car_inference_service_fixture):
        """Verify Model Car model status remains Loaded after upgrade"""
        verify_model_status_loaded(isvc=model_car_inference_service_fixture)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["model_car_isvc_exists"])
    def test_model_car_post_upgrade_not_modified(
        self, model_car_inference_service_fixture, upgrade_baseline_fixture
    ):
        """Verify Model Car InferenceService is not modified during upgrade"""
        baseline = upgrade_baseline_fixture.get(model_car_inference_service_fixture.name, {})
        expected = baseline.get("isvc_observed_generation", 1)
        verify_inference_generation(isvc=model_car_inference_service_fixture, expected_generation=expected)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["model_car_isvc_exists"])
    def test_model_car_post_upgrade_runtime_not_modified(
        self, model_car_inference_service_fixture, upgrade_baseline_fixture
    ):
        """Verify ServingRuntime is not modified during upgrade"""
        baseline = upgrade_baseline_fixture.get(model_car_inference_service_fixture.name, {})
        expected = baseline.get("runtime_generation", 1)
        verify_serving_runtime_generation(isvc=model_car_inference_service_fixture, expected_generation=expected)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["model_car_isvc_exists"])
    def test_model_car_post_upgrade_storage_uri_unchanged(self, model_car_inference_service_fixture):
        """Verify OCI storage URI is unchanged after upgrade"""
        verify_storage_uri_unchanged(
            isvc=model_car_inference_service_fixture,
            expected_uri=ModelCarImage.MNIST_8_1,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["model_car_isvc_exists"])
    def test_model_car_post_upgrade_pods_not_restarted(
        self,
        admin_client,
        model_car_inference_service_fixture,
        upgrade_baseline_fixture,
    ):
        """Verify Model Car pods have not restarted beyond pre-upgrade baseline"""
        baseline = upgrade_baseline_fixture.get(model_car_inference_service_fixture.name, {})
        verify_isvc_pods_not_restarted_against_baseline(
            client=admin_client,
            isvc=model_car_inference_service_fixture,
            baseline_restart_counts=baseline.get("pod_restart_counts", {}),
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["model_car_isvc_exists"])
    def test_model_car_post_upgrade_inference(self, model_car_inference_service_fixture):
        """Verify Model Car inference after upgrade"""
        verify_inference_response(
            inference_service=model_car_inference_service_fixture,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
