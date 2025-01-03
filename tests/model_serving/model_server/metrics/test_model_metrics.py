import pytest

from tests.model_serving.model_server.metrics.utils import run_inference_multiple_times
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    ModelFormat,
    ModelInferenceRuntime,
    ModelStoragePath,
    Protocols,
)
from utilities.inference_utils import Inference
from utilities.monitoring import validate_metrics_value

pytestmark = pytest.mark.usefixtures("skip_if_no_deployed_openshift_serverless", "valid_aws_config", "deleted_metrics")


@pytest.mark.serverless
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri",
    [
        pytest.param(
            {"name": "kserve-tgis-metrics"},
            {"model-dir": ModelStoragePath.FLAN_T5_SMALL},
        )
    ],
    indirect=True,
)
class TestModelMetrics:
    @pytest.mark.smoke
    @pytest.mark.polarion("ODS-2555")
    @pytest.mark.dependency(name="test_model_metrics_num_success_requests")
    def test_model_metrics_num_success_requests(
        self, http_s3_caikit_serverless_inference_service_auth_disabled, prometheus
    ):
        """Verify number of successful model requests in OpenShift monitoring system (UserWorkloadMonitoring)metrics"""
        verify_inference_response(
            inference_service=http_s3_caikit_serverless_inference_service_auth_disabled,
            runtime=ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )
        validate_metrics_value(
            prometheus=prometheus,
            metric_name="tgi_request_success",
            expected_value="1",
        )

    @pytest.mark.smoke
    @pytest.mark.polarion("ODS-2555")
    @pytest.mark.dependency(depends=["test_model_metrics_num_success_requests"])
    def test_model_metrics_num_total_requests(
        self, http_s3_caikit_serverless_inference_service_auth_disabled, prometheus
    ):
        """Verify number of total model requests in OpenShift monitoring system (UserWorkloadMonitoring)metrics"""
        total_runs = 5

        run_inference_multiple_times(
            isvc=http_s3_caikit_serverless_inference_service_auth_disabled,
            runtime=ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.CAIKIT,
            iterations=total_runs,
            run_in_parallel=True,
        )
        validate_metrics_value(
            prometheus=prometheus,
            metric_name="tgi_request_count",
            expected_value=str(total_runs + 1),
        )
