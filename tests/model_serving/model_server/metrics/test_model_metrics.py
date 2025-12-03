import pytest

from tests.model_serving.model_server.utils import (
    run_inference_multiple_times,
    verify_inference_response,
)
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelVersion,
    Protocols,
    RuntimeTemplates,
)
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG
from utilities.monitoring import get_metrics_value, validate_metrics_field

pytestmark = [
    pytest.mark.serverless,
    pytest.mark.usefixtures("valid_aws_config", "user_workload_monitoring_config_map"),
    pytest.mark.metrics,
]


@pytest.mark.serverless
@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template, s3_models_inference_service",
    [
        pytest.param(
            {"name": "kserve-ovms-metrics"},
            {
                "name": f"{Protocols.HTTP}-{ModelFormat.ONNX}",
                "template-name": RuntimeTemplates.OVMS_KSERVE,
                "multi-model": False,
                "enable-http": True,
            },
            {
                "name": f"{Protocols.HTTP}-{ModelFormat.ONNX}",
                "deployment-mode": KServeDeploymentType.SERVERLESS,
                "model-dir": "test-dir",
                "model-version": ModelVersion.OPSET13,
            },
        )
    ],
    indirect=True,
)
class TestModelMetrics:
    @pytest.mark.smoke
    @pytest.mark.polarion("ODS-2555")
    def test_model_metrics_num_success_requests(self, s3_models_inference_service, prometheus):
        """Verify number of successful model requests in OpenShift monitoring system (UserWorkloadMonitoring) metrics"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
        validate_metrics_field(
            prometheus=prometheus,
            metrics_query="ovms_requests_success",
            expected_value="1",
        )

    @pytest.mark.smoke
    @pytest.mark.polarion("ODS-2555")
    def test_model_metrics_num_total_requests(self, s3_models_inference_service, prometheus):
        """Verify number of total model requests in OpenShift monitoring system (UserWorkloadMonitoring) metrics"""
        total_runs = 5

        run_inference_multiple_times(
            isvc=s3_models_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            iterations=total_runs,
            run_in_parallel=True,
        )
        validate_metrics_field(
            prometheus=prometheus,
            metrics_query="ovms_requests_success",
            expected_value=str(total_runs + 1),
        )

    @pytest.mark.smoke
    @pytest.mark.polarion("ODS-2555")
    def test_model_metrics_cpu_utilization(self, s3_models_inference_service, prometheus):
        """Verify CPU utilization data in OpenShift monitoring system (UserWorkloadMonitoring) metrics"""
        assert get_metrics_value(
            prometheus=prometheus,
            metrics_query=f"pod:container_cpu_usage:sum{{namespace='{s3_models_inference_service.namespace}'}}",
        )
