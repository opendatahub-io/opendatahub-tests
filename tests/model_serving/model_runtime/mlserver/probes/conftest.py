from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_runtime.mlserver.constant import PREDICT_RESOURCES
from tests.model_serving.model_runtime.mlserver.probes.utils import MLSERVER_LIVENESS_PROBE, MLSERVER_READINESS_PROBE
from utilities.constants import Containers, KServeDeploymentType, RuntimeTemplates, Timeout
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def mlserver_probes_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_runtime_image: str,
) -> Generator[ServingRuntime, Any, Any]:
    """MLServer ServingRuntime with readiness and liveness httpGet probes on kserve-container."""
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="mlserver-runtime",
        namespace=model_namespace.name,
        template_name=RuntimeTemplates.MLSERVER,
        deployment_type=request.param["deployment_type"],
        runtime_image=mlserver_runtime_image,
        containers={
            Containers.KSERVE_CONTAINER_NAME: {
                "readinessProbe": MLSERVER_READINESS_PROBE,
                "livenessProbe": MLSERVER_LIVENESS_PROBE,
            }
        },
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def mlserver_probes_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_probes_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    mlserver_model_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    """MLServer InferenceService with probe-enabled runtime backed by S3 model storage."""
    supported_formats = mlserver_probes_serving_runtime.instance.spec.supportedModelFormats
    assert supported_formats, "ServingRuntime has no supportedModelFormats configured"
    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": mlserver_probes_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": supported_formats[0].name,
        "model_service_account": mlserver_model_service_account.name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.STANDARD),
        "external_route": True,
        "resources": PREDICT_RESOURCES.get("resources"),
        "volumes": PREDICT_RESOURCES.get("volumes"),
        "volumes_mounts": PREDICT_RESOURCES.get("volume_mounts"),
        "timeout": request.param.get("timeout", Timeout.TIMEOUT_20MIN),
    }

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture
def mlserver_probes_pod_resource(
    admin_client: DynamicClient, mlserver_probes_inference_service: InferenceService
) -> Pod:
    """Get the predictor pod for the MLServer probe InferenceService.

    Raises:
        AssertionError: If no pods found for the InferenceService.
    """
    pods = get_pods_by_isvc_label(client=admin_client, isvc=mlserver_probes_inference_service)
    assert pods, f"No pods found for InferenceService {mlserver_probes_inference_service.name}"
    return pods[0]
