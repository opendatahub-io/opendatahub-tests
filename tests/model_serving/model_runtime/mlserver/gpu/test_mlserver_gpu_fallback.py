"""MLServer CUDA runtime GPU fallback behavior validation.

Validates two GPU fallback scenarios for the mlserver-cuda-runtime:
- TC-FALLBACK-001: When deployed without GPU resources, CUDA init fails silently and
  ONNX Runtime falls back to CPU execution provider — inference still succeeds.
- TC-FALLBACK-002: When GPU resources are requested on a cluster without GPU nodes,
  the predictor pod remains Pending and the inference endpoint is inaccessible.
"""

from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.event import Event
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.mlserver.constant import MODEL_CONFIGS
from tests.model_serving.model_runtime.mlserver.utils import get_model_storage_uri_dict, validate_inference_request
from utilities.constants import (
    Containers,
    KServeDeploymentType,
    Labels,
    ModelFormat,
    Protocols,
    Timeout,
)
from utilities.infra import get_pods_by_isvc_label

pytestmark = [pytest.mark.usefixtures("valid_aws_config")]

LOGGER = structlog.get_logger(name=__name__)

_CUDA_INIT_FAILURE_PATTERN: str = "CUDAExecutionProvider"
_ONNX_MODEL_CONFIG: dict[str, Any] = MODEL_CONFIGS[ModelFormat.ONNX]


@pytest.mark.parametrize(
    ("model_namespace", "s3_models_storage_uri", "mlserver_serving_runtime", "mlserver_inference_service"),
    [
        pytest.param(
            {"name": "mlserver-cuda-cpu-fallback"},
            get_model_storage_uri_dict(model_format_name=ModelFormat.ONNX),
            {"deployment_mode": KServeDeploymentType.STANDARD, "gpu": True},
            {
                "name": "resnet-50-onnx",
                "gpu_count": 0,
                "enable_external_route": True,
                "timeout": Timeout.TIMEOUT_10MIN,
            },
            id="test_mlserver_cuda_silent_cpu_fallback",
            marks=[pytest.mark.tier1],
        ),
    ],
    indirect=True,
)
class TestSilentCPUFallback:
    """TC-FALLBACK-001: mlserver-cuda-runtime silently falls back to CPU without GPU resources."""

    def test_silent_cpu_fallback(
        self,
        admin_client: DynamicClient,
        mlserver_inference_service: InferenceService,
    ) -> None:
        """Verify silent CPU fallback when mlserver-cuda-runtime runs without GPU resources.

        Given an ISVC using mlserver-cuda-runtime without GPU resource requests
        When the predictor pod schedules on a CPU node
        Then the pod is Running, CUDA init failure appears in logs,
        And inference succeeds via the CPU execution provider.
        """
        predictor_pods = get_pods_by_isvc_label(client=admin_client, isvc=mlserver_inference_service)
        assert predictor_pods, "No predictor pods found for CUDA ISVC deployed without GPU resources"
        predictor_pod = predictor_pods[0]

        pod_phase = predictor_pod.instance.status.phase
        assert pod_phase == "Running", f"Expected predictor pod Running (silent CPU fallback) but got {pod_phase!r}"

        for container in predictor_pod.instance.spec.containers:
            container_data: dict[str, Any] = container.to_dict()
            resources_data = container_data.get("resources") or {}
            pod_requests = resources_data.get("requests") or {}
            pod_limits = resources_data.get("limits") or {}
            gpu_key = Labels.Nvidia.NVIDIA_COM_GPU
            assert gpu_key not in pod_requests, f"Container {container_data.get('name')!r} should not have GPU requests"
            assert gpu_key not in pod_limits, f"Container {container_data.get('name')!r} should not have GPU limits"

        container_logs = predictor_pod.log(container=Containers.KSERVE_CONTAINER_NAME)
        assert _CUDA_INIT_FAILURE_PATTERN in container_logs, (
            f"Expected CUDA init failure containing {_CUDA_INIT_FAILURE_PATTERN!r} in logs"
        )

        validate_inference_request(
            isvc=mlserver_inference_service,
            input_query=_ONNX_MODEL_CONFIG["rest_query"],
            model_version=_ONNX_MODEL_CONFIG["model_version"],
            model_output_type=_ONNX_MODEL_CONFIG["output_type"],
            protocol=Protocols.REST,
        )


@pytest.mark.parametrize(
    ("model_namespace", "s3_models_storage_uri", "mlserver_serving_runtime", "mlserver_inference_service"),
    [
        pytest.param(
            {"name": "mlserver-gpu-sched-fail"},
            get_model_storage_uri_dict(model_format_name=ModelFormat.ONNX),
            {"deployment_mode": KServeDeploymentType.STANDARD, "gpu": True},
            {
                "name": "resnet-50-onnx",
                "gpu_count": 1,
                "wait": False,
                "wait_for_predictor_pods": False,
                "timeout": Timeout.TIMEOUT_2MIN,
            },
            id="test_mlserver_cuda_gpu_scheduling_failure",
            marks=[pytest.mark.tier1],
        ),
    ],
    indirect=True,
)
class TestGPUSchedulingFailure:
    """TC-FALLBACK-002: GPU resource request stays Pending when no GPU nodes are available.

    This test must run on clusters WITHOUT GPU nodes. It fails if GPU nodes are
    detected — running on a GPU cluster indicates a quality gate misconfiguration.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_gpu_nodes_available(self, gpu_count_on_cluster: int) -> None:
        """Fail if cluster has GPUs — this test is only valid on CPU-only clusters."""
        if gpu_count_on_cluster > 0:
            pytest.fail(
                f"Cluster has {gpu_count_on_cluster} GPU(s) — cannot test GPU scheduling failure. "
                "This test must run on a CPU-only cluster."
            )

    def test_gpu_scheduling_failure(
        self,
        admin_client: DynamicClient,
        mlserver_inference_service: InferenceService,
        mlserver_pending_predictor_pods: list[Pod],
    ) -> None:
        """Verify GPU scheduling failure when no GPU nodes are available.

        Given an ISVC requesting 1 GPU on a cluster with no GPU nodes
        When the predictor pod attempts to schedule
        Then the pod remains Pending with Unschedulable events,
        And the ISVC is not Ready and has no inference URL.
        """
        predictor_pod = mlserver_pending_predictor_pods[0]

        pod_phase = predictor_pod.instance.status.phase
        assert pod_phase == "Pending", (
            f"Expected predictor pod Pending due to GPU scheduling failure but got {pod_phase!r}"
        )

        scheduling_events = Event.list(
            client=admin_client,
            namespace=mlserver_inference_service.namespace,
            field_selector=f"involvedObject.name={predictor_pod.name}",
        )
        if scheduling_events:
            LOGGER.info(
                event="Pod scheduling events",
                pod_name=predictor_pod.name,
                events=[
                    {
                        "reason": getattr(evt.instance, "reason", None) or getattr(evt, "reason", None),
                        "message": getattr(evt.instance, "message", None) or getattr(evt, "message", None),
                    }
                    for evt in scheduling_events
                ],
            )

        isvc_conditions = getattr(mlserver_inference_service.instance.status, "conditions", None) or []
        ready_condition = next(
            (cond for cond in isvc_conditions if getattr(cond, "type", None) == "Ready"),
            None,
        )
        is_ready = ready_condition is not None and getattr(ready_condition, "status", None) == "True"
        assert not is_ready, "ISVC Ready condition should not be True when GPU scheduling fails"
