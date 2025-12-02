import pytest

from tests.model_serving.model_server.metrics.utils import validate_metrics_configuration
from tests.model_serving.model_server.utils import (
    run_inference_multiple_times,
    verify_inference_response,
)
from utilities.constants import (
    KServeDeploymentType,
    ModelAndFormat,
    ModelInferenceRuntime,
    ModelStoragePath,
    ModelVersion,
    Protocols,
)
from timeout_sampler import TimeoutSampler
from utilities.inference_utils import Inference
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG
from utilities.monitoring import get_metrics_value, validate_metrics_field

pytestmark = [
    pytest.mark.usefixtures("valid_aws_config", "user_workload_monitoring_config_map"),
    pytest.mark.metrics,
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "test-ovms-metrics"},
            {
                "runtime-name": ModelInferenceRuntime.OPENVINO_KSERVE_RUNTIME,
                "model-format": {ModelAndFormat.OPENVINO_IR: ModelVersion.OPSET1},
            },
            {
                "name": "ovms-metrics",
                "model-dir": ModelStoragePath.KSERVE_OPENVINO_EXAMPLE_MODEL,
                "model-version": ModelVersion.OPSET1,
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
            },
        )
    ],
    indirect=True,
)
class TestModelMetrics:
    @pytest.mark.smoke
    @pytest.mark.polarion("ODS-2555")
    def test_model_metrics_num_success_requests(self, ovms_kserve_inference_service, prometheus):
        """Verify number of successful model requests in OpenShift monitoring system (UserWorkloadMonitoring) metrics"""
        validate_metrics_configuration(inference_service=ovms_kserve_inference_service)

        verify_inference_response(
            inference_service=ovms_kserve_inference_service,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

        metrics_query = (
            f'ovms_requests_success{{namespace="{ovms_kserve_inference_service.namespace}", '
            f'name="{ovms_kserve_inference_service.name}"}}'
        )

        validate_metrics_field(
            prometheus=prometheus,
            metrics_query=metrics_query,
            expected_value="1",
        )

    @pytest.mark.smoke
    @pytest.mark.polarion("ODS-2555")
    def test_model_metrics_num_total_requests(self, ovms_kserve_inference_service, prometheus):
        """Verify number of total model requests in OpenShift monitoring system (UserWorkloadMonitoring) metrics"""
        validate_metrics_configuration(inference_service=ovms_kserve_inference_service)

        total_runs = 5

        run_inference_multiple_times(
            isvc=ovms_kserve_inference_service,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            iterations=total_runs,
            run_in_parallel=True,
        )

        metrics_query = (
            f'ovms_requests_success{{namespace="{ovms_kserve_inference_service.namespace}", '
            f'name="{ovms_kserve_inference_service.name}"}}'
        )

        validate_metrics_field(
            prometheus=prometheus,
            metrics_query=metrics_query,
            expected_value=str(total_runs),
            greater_than=True,
        )

    @pytest.mark.smoke
    @pytest.mark.polarion("ODS-2555")
    def test_model_metrics_cpu_utilization(self, ovms_kserve_inference_service, prometheus):
        """Verify CPU utilization data in OpenShift monitoring system (UserWorkloadMonitoring) metrics"""
        validate_metrics_configuration(inference_service=ovms_kserve_inference_service)

        metrics_query = f"pod:container_cpu_usage:sum{{namespace='{ovms_kserve_inference_service.namespace}'}}"

        for cpu_value in TimeoutSampler(
            wait_timeout=120,
            sleep=10,
            func=get_metrics_value,
            prometheus=prometheus,
            metrics_query=metrics_query,
        ):
            if cpu_value is not None:
                break
