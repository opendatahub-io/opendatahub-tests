import pytest
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_server.kserve.metrics.utils import validate_metrics_configuration
from tests.model_serving.model_server.utils import (
    run_inference_multiple_times,
    verify_inference_response,
)
from utilities.constants import (
    KServeDeploymentType,
    ModelName,
    ModelStorage,
    Protocols,
    RuntimeTemplates,
)
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG
from utilities.monitoring import get_metrics_value, validate_metrics_field

pytestmark = [
    pytest.mark.usefixtures("user_workload_monitoring_config_map"),
    pytest.mark.metrics,
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template, model_car_inference_service",
    [
        pytest.param(
            {"name": "test-ovms-metrics"},
            {
                "name": f"{ModelName.MNIST}-runtime",
                "template-name": RuntimeTemplates.OVMS_KSERVE,
                "multi-model": False,
            },
            {
                "storage-uri": ModelStorage.OCI.MNIST_8_1,
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
            },
        )
    ],
    indirect=True,
)
class TestModelMetrics:
    @pytest.mark.smoke
    @pytest.mark.polarion("ODS-2555")
    def test_model_metrics_num_success_requests(self, model_car_inference_service, prometheus):
        """Verify number of successful model requests in OpenShift monitoring system (UserWorkloadMonitoring) metrics"""
        # validate cm values is true for metrics dashboard
        validate_metrics_configuration(model_car_inference_service=model_car_inference_service)

        verify_inference_response(
            inference_service=model_car_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

        metrics_query = (
            f'ovms_requests_success{{namespace="{model_car_inference_service.namespace}", '
            f'name="{model_car_inference_service.name}"}}'
        )

        validate_metrics_field(
            prometheus=prometheus,
            metrics_query=metrics_query,
            expected_value="1",
        )

    @pytest.mark.smoke
    @pytest.mark.polarion("ODS-2555")
    def test_model_metrics_num_total_requests(self, model_car_inference_service, prometheus):
        """Verify number of total model requests in OpenShift monitoring system (UserWorkloadMonitoring) metrics"""
        validate_metrics_configuration(model_car_inference_service=model_car_inference_service)

        total_runs = 5

        run_inference_multiple_times(
            isvc=model_car_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            iterations=total_runs,
            run_in_parallel=True,
        )

        metrics_query = (
            f'ovms_requests_success{{namespace="{model_car_inference_service.namespace}", '
            f'name="{model_car_inference_service.name}"}}'
        )

        validate_metrics_field(
            prometheus=prometheus,
            metrics_query=metrics_query,
            expected_value=str(total_runs),
            greater_than=True,
        )

    @pytest.mark.smoke
    @pytest.mark.polarion("ODS-2555")
    def test_model_metrics_cpu_utilization(self, model_car_inference_service, prometheus):
        """Verify CPU utilization data in OpenShift monitoring system (UserWorkloadMonitoring) metrics"""
        validate_metrics_configuration(model_car_inference_service=model_car_inference_service)

        metrics_query = f"pod:container_cpu_usage:sum{{namespace='{model_car_inference_service.namespace}'}}"

        for cpu_value in TimeoutSampler(
            wait_timeout=120,
            sleep=10,
            func=get_metrics_value,
            prometheus=prometheus,
            metrics_query=metrics_query,
        ):
            if cpu_value is not None:
                break
