"""GPU-enabled MLServer CUDA runtime deployment and inference tests.

Validates end-to-end GPU deployment lifecycle and inference functionality:
- Runtime instantiation with correct container and @sha256: image
- ISVC reaching Ready state with predictor pod on GPU node
- GPU resource limits present on pod
- REST inference with model readiness and server health checks
- Concurrent REST inference request handling
"""

import concurrent.futures
from typing import Any

import pytest
import requests
from ocp_resources.inference_service import InferenceService
from ocp_resources.node import Node
from ocp_resources.pod import Pod
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_runtime.mlserver.constant import (
    ONNX_RESNET50_REST_INPUT_QUERY,
)
from tests.model_serving.model_runtime.mlserver.utils import (
    get_model_storage_uri_dict,
    send_rest_request,
    validate_deterministic_snapshot,
)
from utilities.constants import KServeDeploymentType, Labels, ModelFormat, Timeout
from utilities.inference_utils import get_exposed_isvc_url

pytestmark = [
    pytest.mark.usefixtures("skip_if_no_gpu_for_mlserver", "valid_aws_config"),
    pytest.mark.gpu,
    pytest.mark.mlserver_nvidia_gpu,
]

_CONCURRENT_REQUEST_COUNT: int = 5


@pytest.mark.parametrize(
    ("model_namespace", "s3_models_storage_uri", "mlserver_serving_runtime", "mlserver_inference_service"),
    [
        pytest.param(
            {"name": "mlserver-gpu-deploy-infer"},
            get_model_storage_uri_dict(model_format_name=ModelFormat.ONNX),
            {"deployment_mode": KServeDeploymentType.STANDARD, "gpu": True},
            {
                "name": "resnet-50-onnx",
                "gpu_count": 1,
                "deployment_mode": KServeDeploymentType.STANDARD,
                "enable_external_route": True,
                "timeout": Timeout.TIMEOUT_10MIN,
            },
            id="test_mlserver_cuda_gpu_deploy_and_inference",
        ),
    ],
    indirect=True,
)
class TestMLServerGPUDeployAndInference:
    """Validates GPU runtime deployment lifecycle and inference in a single deployment.

    One ISVC deployment validates: runtime config, GPU node scheduling,
    resource limits, REST health/inference, and concurrent requests.
    """

    def test_runtime_config(
        self,
        mlserver_serving_runtime: ServingRuntime,
        mlserver_inference_service: InferenceService,
    ) -> None:
        """Verify ServingRuntime is created with correct container and image config.

        Given the mlserver-cuda-runtime-template is instantiated
        When the ServingRuntime is inspected
        Then it exists, has a kserve-container, and the image uses @sha256: digest format.
        """
        assert mlserver_serving_runtime.exists, f"ServingRuntime '{mlserver_serving_runtime.name}' was not created"

        containers = mlserver_serving_runtime.instance.spec.containers
        assert containers, "ServingRuntime spec has no containers defined"

        container_image: str = containers[0].image or ""
        assert "@sha256:" in container_image, (
            f"Container image does not use @sha256: digest format: {container_image!r}"
        )

    def test_gpu_node_scheduling(
        self,
        mlserver_inference_service: InferenceService,
        mlserver_pod_resource: Pod,
        gpu_worker_nodes: list[Node],
    ) -> None:
        """Verify ISVC deployed on GPU node with correct resource limits.

        Given an InferenceService with nvidia.com/gpu: 1
        When the ISVC reaches Ready state
        Then the external route is accessible, the pod runs on a GPU node,
        And the pod has nvidia.com/gpu: 1 in resource limits.
        """
        get_exposed_isvc_url(isvc=mlserver_inference_service)

        predictor_node_name: str = mlserver_pod_resource.instance.spec.nodeName
        assert predictor_node_name, "Predictor pod has no spec.nodeName"

        gpu_node_names: set[str] = {node.name for node in gpu_worker_nodes if node.name}
        assert predictor_node_name in gpu_node_names, (
            f"Predictor pod running on '{predictor_node_name}', not a GPU node. "
            f"GPU nodes: {sorted(gpu_node_names) or 'none found'}"
        )

        pod_containers = mlserver_pod_resource.instance.spec.containers
        assert pod_containers, "Predictor pod has no containers in spec"
        kserve_container = next(
            (container for container in pod_containers if "kserve" in (container.name or "")),
            None,
        )
        assert kserve_container is not None, f"No 'kserve' container found among: {[c.name for c in pod_containers]}"
        pod_gpu_limit = (kserve_container.resources.limits or {}).get(Labels.Nvidia.NVIDIA_COM_GPU)
        assert pod_gpu_limit is not None, f"Container '{kserve_container.name}' has no nvidia.com/gpu limit"
        assert str(pod_gpu_limit) == "1", f"Expected nvidia.com/gpu limit of '1', got: {pod_gpu_limit!r}"

    def test_rest_inference_with_health_checks(
        self,
        mlserver_inference_service: InferenceService,
    ) -> None:
        """Verify model readiness, server health, and REST inference via external route.

        Given an ISVC with mlserver-cuda-runtime, nvidia.com/gpu: 1, external_route=True
        When readiness, health, and inference endpoints are queried
        Then all return HTTP 200 with valid response structure.
        """
        base_url = get_exposed_isvc_url(isvc=mlserver_inference_service)
        model_name = mlserver_inference_service.name

        readiness_url = f"{base_url}/v2/models/{model_name}/ready"
        readiness_response = requests.get(url=readiness_url, verify=False, timeout=60)
        readiness_response.raise_for_status()

        health_url = f"{base_url}/v2/health/ready"
        health_response = requests.get(url=health_url, verify=False, timeout=60)
        health_response.raise_for_status()

        infer_url = f"{base_url}/v2/models/{model_name}/infer"
        response = send_rest_request(url=infer_url, input_data=ONNX_RESNET50_REST_INPUT_QUERY)
        validate_deterministic_snapshot(response=response)

    def test_concurrent_rest_inference(
        self,
        mlserver_inference_service: InferenceService,
    ) -> None:
        """Verify concurrent REST requests are handled correctly.

        Given an ISVC with external_route=True
        When 5 concurrent requests are submitted
        Then all 5 complete without errors and pass validation.
        """
        base_url = get_exposed_isvc_url(isvc=mlserver_inference_service)
        model_name = mlserver_inference_service.name
        infer_url = f"{base_url}/v2/models/{model_name}/infer"

        def _send_single_request() -> Any:
            return send_rest_request(url=infer_url, input_data=ONNX_RESNET50_REST_INPUT_QUERY)

        with concurrent.futures.ThreadPoolExecutor(max_workers=_CONCURRENT_REQUEST_COUNT) as executor:
            futures = [executor.submit(_send_single_request) for _ in range(_CONCURRENT_REQUEST_COUNT)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]

        assert len(responses) == _CONCURRENT_REQUEST_COUNT, (
            f"Expected {_CONCURRENT_REQUEST_COUNT} responses, got {len(responses)}"
        )

        for response in responses:
            validate_deterministic_snapshot(response=response)
