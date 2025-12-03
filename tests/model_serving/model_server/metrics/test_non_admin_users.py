import pytest

from tests.model_serving.model_server.utils import (
    run_inference_multiple_times,
    verify_inference_response,
)
from utilities.constants import Protocols
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG
from utilities.monitoring import validate_metrics_field


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [
        pytest.param(
            {"name": "non-admin-serverless"},
        )
    ],
    indirect=True,
)
@pytest.mark.smoke
@pytest.mark.serverless
class TestServerlessUnprivilegedUser:
    @pytest.mark.polarion("ODS-2552")
    def test_non_admin_deploy_serverless_and_query_metrics(self, unprivileged_ovms_serverless_inference_service):
        """Verify non admin can deploy a model and query using REST"""
        verify_inference_response(
            inference_service=unprivileged_ovms_serverless_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [
        pytest.param(
            {"name": "non-admin-metrics"},
        )
    ],
    indirect=True,
)
@pytest.mark.sanity
@pytest.mark.rawdeployment
class TestRawUnprivilegedUserMetrics:
    @pytest.mark.metrics
    def test_non_admin_raw_metrics(
        self,
        unprivileged_ovms_raw_inference_service,
        prometheus,
        user_workload_monitoring_config_map,
    ):
        """Verify number of total model requests in OpenShift monitoring system (UserWorkloadMonitoring) metrics"""
        total_runs = 5

        run_inference_multiple_times(
            isvc=unprivileged_ovms_raw_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            iterations=total_runs,
        )
        validate_metrics_field(
            prometheus=prometheus,
            metrics_query="ovms_requests_success",
            expected_value=str(total_runs),
        )
