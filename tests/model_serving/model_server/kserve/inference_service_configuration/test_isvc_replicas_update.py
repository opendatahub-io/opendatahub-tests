import pytest
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_server.kserve.inference_service_configuration.constants import (
    BASE_ISVC_CONFIG,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    Protocols,
    RunTimeConfigs,
    Timeout,
)
from utilities.inference_utils import Inference
from utilities.infra import get_pods_by_isvc_label
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

LOGGER = get_logger(name=__name__)

pytestmark = [pytest.mark.sanity, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "test-raw-isvc-replicas"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {
                **BASE_ISVC_CONFIG,
                "min-replicas": 2,
                "max-replicas": 4,
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
            },
        )
    ],
    indirect=True,
)
class TestRawISVCReplicasUpdates:
    @pytest.mark.dependency(name="test_raw_increase_isvc_replicas")
    def test_raw_increase_isvc_replicas(self, isvc_pods, ovms_kserve_inference_service):
        """Test replicas increase"""
        assert len(isvc_pods) == 2, "Expected 2 inference pods, existing pods: {pod.name for pod in isvc_pods}"

    @pytest.mark.dependency(depends=["test_raw_increase_isvc_replicas"])
    def test_raw_increase_isvc_replicas_inference(self, ovms_kserve_inference_service):
        """Verify inference after replicas increase"""
        verify_inference_response(
            inference_service=ovms_kserve_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "patched_isvc_replicas",
        [
            pytest.param({"min-replicas": 1, "max-replicas": 1, "wait-for-new-pods": False}),
        ],
        indirect=True,
    )
    @pytest.mark.dependency(name="test_raw_decrease_isvc_replicas")
    def test_raw_decrease_isvc_replicas(self, unprivileged_client, isvc_pods, patched_isvc_replicas):
        """Test replicas decrease"""
        orig_pod_names = [pod.name for pod in isvc_pods]
        pods = []

        try:
            for pods in TimeoutSampler(
                wait_timeout=Timeout.TIMEOUT_2MIN,
                sleep=1,
                func=get_pods_by_isvc_label,
                client=unprivileged_client,
                isvc=patched_isvc_replicas,
            ):
                if len(pods) == 1 and pods[0].name in orig_pod_names:
                    return

        except TimeoutError:
            LOGGER.error(f"Expected 1 pod to be running, but got {[_pod.name for _pod in pods]}")

    @pytest.mark.dependency(depends=["test_raw_decrease_isvc_replicas"])
    def test_raw_decrease_isvc_replicas_inference(self, ovms_kserve_inference_service):
        """Verify inference after replicas decrease"""
        verify_inference_response(
            inference_service=ovms_kserve_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )
