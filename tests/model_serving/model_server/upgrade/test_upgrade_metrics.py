import pytest

from tests.model_serving.model_server.upgrade.utils import (
    get_metrics_value,
    verify_inference_generation,
    verify_isvc_pods_not_restarted,
    verify_metrics_configmap_exists,
    verify_metrics_retained,
    verify_model_status_loaded,
)
from tests.model_serving.model_server.utils import (
    run_inference_multiple_times,
    verify_inference_response,
)
from utilities.constants import Protocols
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.rawdeployment,
    pytest.mark.metrics,
    pytest.mark.usefixtures("upgrade_user_workload_monitoring_config_map"),
]


class TestPreUpgradeMetricsServer:
    """Pre-upgrade tests for metrics persistence during model serving."""

    @pytest.mark.pre_upgrade
    def test_metrics_pre_upgrade_isvc_exists(self, metrics_inference_service_fixture):
        """Verify metrics InferenceService exists before upgrade"""
        assert metrics_inference_service_fixture.exists, (
            f"InferenceService {metrics_inference_service_fixture.name} does not exist"
        )

    @pytest.mark.pre_upgrade
    def test_metrics_pre_upgrade_model_loaded(self, metrics_inference_service_fixture):
        """Verify metrics model is in Loaded state before upgrade"""
        verify_model_status_loaded(isvc=metrics_inference_service_fixture)

    @pytest.mark.pre_upgrade
    def test_metrics_pre_upgrade_configmap_exists(self, metrics_inference_service_fixture):
        """Verify metrics dashboard ConfigMap exists before upgrade"""
        verify_metrics_configmap_exists(isvc=metrics_inference_service_fixture)

    @pytest.mark.pre_upgrade
    def test_metrics_pre_upgrade_inference(self, metrics_inference_service_fixture):
        """Verify inference works and metrics are captured before upgrade"""
        verify_inference_response(
            inference_service=metrics_inference_service_fixture,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.pre_upgrade
    def test_metrics_pre_upgrade_multiple_requests(self, metrics_inference_service_fixture, prometheus):
        """Run multiple inference requests to generate metrics data before upgrade"""
        total_runs = 5

        run_inference_multiple_times(
            isvc=metrics_inference_service_fixture,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            iterations=total_runs,
            run_in_parallel=True,
        )

        metrics_query = (
            f'ovms_requests_success{{namespace="{metrics_inference_service_fixture.namespace}", '
            f'name="{metrics_inference_service_fixture.name}"}}'
        )

        verify_metrics_retained(
            prometheus=prometheus,
            query=metrics_query,
            min_value=total_runs,
        )


class TestPostUpgradeMetricsServer:
    """Post-upgrade tests for metrics persistence during model serving."""

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="metrics_isvc_exists")
    def test_metrics_post_upgrade_isvc_exists(self, metrics_inference_service_fixture):
        """Verify metrics InferenceService exists after upgrade"""
        assert metrics_inference_service_fixture.exists, (
            f"InferenceService {metrics_inference_service_fixture.name} does not exist after upgrade"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["metrics_isvc_exists"])
    def test_metrics_post_upgrade_configmap_exists(self, metrics_inference_service_fixture):
        """Verify metrics dashboard ConfigMap exists after upgrade"""
        verify_metrics_configmap_exists(isvc=metrics_inference_service_fixture)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["metrics_isvc_exists"])
    def test_metrics_post_upgrade_not_modified(self, metrics_inference_service_fixture):
        """Verify metrics InferenceService is not modified during upgrade"""
        verify_inference_generation(isvc=metrics_inference_service_fixture, expected_generation=1)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["metrics_isvc_exists"])
    def test_metrics_post_upgrade_historical_data_retained(self, metrics_inference_service_fixture, prometheus):
        """Verify historical metrics data is retained after upgrade"""
        metrics_query = (
            f'ovms_requests_success{{namespace="{metrics_inference_service_fixture.namespace}", '
            f'name="{metrics_inference_service_fixture.name}"}}'
        )

        verify_metrics_retained(
            prometheus=prometheus,
            query=metrics_query,
            min_value=5,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["metrics_isvc_exists"])
    def test_metrics_post_upgrade_new_requests_captured(self, metrics_inference_service_fixture, prometheus):
        """Verify new inference requests are captured in metrics after upgrade"""
        metrics_query = (
            f'ovms_requests_success{{namespace="{metrics_inference_service_fixture.namespace}", '
            f'name="{metrics_inference_service_fixture.name}"}}'
        )

        pre_request_value = get_metrics_value(prometheus=prometheus, query=metrics_query) or 0

        verify_inference_response(
            inference_service=metrics_inference_service_fixture,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

        verify_metrics_retained(
            prometheus=prometheus,
            query=metrics_query,
            min_value=pre_request_value + 1,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["metrics_isvc_exists"])
    def test_metrics_post_upgrade_pods_not_restarted(
        self,
        admin_client,
        metrics_inference_service_fixture,
    ):
        """Verify metrics pods have not restarted during upgrade"""
        verify_isvc_pods_not_restarted(
            client=admin_client,
            isvc=metrics_inference_service_fixture,
            max_restarts=2,
        )
