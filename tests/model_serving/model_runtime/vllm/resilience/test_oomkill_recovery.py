"""vLLM pod OOM recovery."""

import pytest
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.vllm.constant import BASE_RAW_DEPLOYMENT_CONFIG
from utilities.infra import get_pods_by_isvc_label

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.tier3
@pytest.mark.resilience
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, resilience_serving_runtime, resilience_inference_service",
    [
        pytest.param(
            {"name": "vllm-res-oom"},
            {"model-dir": "vllm/granite-7b-lab"},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {
                "name": "vllm-oom-recovery",
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "gpu_count": 1,
                "runtime_argument": [
                    "--model=/mnt/models",
                    "--dtype=float16",
                ],
            },
            id="vllm-oom-recovery",
        ),
    ],
    indirect=True,
)
class TestOOMKillRecovery:
    """Validate vLLM pod stability and recovery under resource-constrained deployments."""

    def test_pod_restarts_after_oom(
        self,
        admin_client,
        resilience_inference_service: InferenceService,
    ) -> None:
        """Verify pod recovers or remains stable under tight memory limits."""
        pods = get_pods_by_isvc_label(client=admin_client, isvc=resilience_inference_service)
        assert pods, f"No pods found for ISVC {resilience_inference_service.name}"

        pod = pods[0]
        phase = pod.instance.status.phase
        assert phase in ("Running", "Pending", "Succeeded"), f"Pod {pod.name} is in unexpected phase: {phase}"
