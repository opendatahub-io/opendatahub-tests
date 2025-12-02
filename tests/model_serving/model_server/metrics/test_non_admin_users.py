import pytest

from tests.model_serving.model_server.metrics.utils import validate_metrics_configuration
from tests.model_serving.model_server.utils import (
    run_inference_multiple_times,
)
from utilities.constants import (
    KServeDeploymentType,
    ModelAndFormat,
    ModelInferenceRuntime,
    ModelStoragePath,
    ModelVersion,
    Protocols,
)
from utilities.inference_utils import Inference
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG
from utilities.monitoring import validate_metrics_field


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "test-non-admin-metrics"},
            {
                "runtime-name": ModelInferenceRuntime.OPENVINO_KSERVE_RUNTIME,
                "model-format": {ModelAndFormat.OPENVINO_IR: ModelVersion.OPSET1},
            },
            {
                "name": "ovms-non-admin",
                "model-dir": ModelStoragePath.KSERVE_OPENVINO_EXAMPLE_MODEL,
                "model-version": ModelVersion.OPSET1,
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
        ovms_kserve_inference_service,
        prometheus,
        user_workload_monitoring_config_map,
    ):
        """Verify number of total model requests in OpenShift monitoring system (UserWorkloadMonitoring) metrics"""
        validate_metrics_configuration(inference_service=ovms_kserve_inference_service)

        total_runs = 5

        run_inference_multiple_times(
            isvc=ovms_kserve_inference_service,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            iterations=total_runs,
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
