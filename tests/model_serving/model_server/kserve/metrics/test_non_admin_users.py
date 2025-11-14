import pytest

from tests.model_serving.model_server.kserve.metrics.utils import validate_metrics_configuration
from tests.model_serving.model_server.utils import (
    run_inference_multiple_times,
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
from utilities.monitoring import validate_metrics_field


@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template, model_car_inference_service",
    [
        pytest.param(
            {"name": "test-non-admin-metrics"},
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
@pytest.mark.sanity
@pytest.mark.rawdeployment
class TestRawUnprivilegedUserMetrics:
    @pytest.mark.metrics
    def test_non_admin_raw_metrics(
        self,
        model_car_inference_service,
        prometheus,
        user_workload_monitoring_config_map,
    ):
        """Verify number of total model requests in OpenShift monitoring system (UserWorkloadMonitoring) metrics"""
        validate_metrics_configuration(model_car_inference_service=model_car_inference_service)

        total_runs = 5

        run_inference_multiple_times(
            isvc=model_car_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            iterations=total_runs,
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
