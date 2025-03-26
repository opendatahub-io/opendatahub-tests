import pytest

from tests.model_serving.model_server.serverless.constants import ONNX_SERVERLESS_INFERENCE_SERVICE_CONFIG
from tests.model_serving.model_server.serverless.utils import (
    inference_service_pods_sampler,
)
from utilities.constants import (
    RunTimeConfigs,
    Timeout,
)

pytestmark = [
    pytest.mark.serverless,
    pytest.mark.sanity,
    pytest.mark.usefixtures("valid_aws_config"),
]


@pytest.mark.parametrize(
    "model_namespace, openvino_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "serverless-auto-scale"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {
                **ONNX_SERVERLESS_INFERENCE_SERVICE_CONFIG,
                "scale-metric": "concurrency",
                "scale-target": 1,
            },
        )
    ],
    indirect=True,
)
class TestConcurrencyAutoScale:
    @pytest.mark.dependency(name="test_auto_scale_using_concurrency")
    def test_auto_scale_using_concurrency(
        self,
        admin_client,
        ovms_kserve_inference_service,
        multiple_onnx_inference_requests,
    ):
        """Verify model is successfully scaled up based on concurrency metrics (KPA)"""
        for pods in inference_service_pods_sampler(
            client=admin_client,
            isvc=ovms_kserve_inference_service,
            timeout=Timeout.TIMEOUT_1MIN,
        ):
            if pods:
                if len(pods) > 1 and all([pod.status == pod.Status.RUNNING for pod in pods]):
                    return

    @pytest.mark.dependency(requires=["test_auto_scale_using_concurrency"])
    def test_pods_scaled_down_when_no_requests(self, admin_client, ovms_kserve_inference_service):
        """Verify auto-scaled pods are deleted when there are no inference requests"""
        for pods in inference_service_pods_sampler(
            client=admin_client,
            isvc=ovms_kserve_inference_service,
            timeout=Timeout.TIMEOUT_4MIN,
        ):
            if pods and len(pods) == 1:
                return
