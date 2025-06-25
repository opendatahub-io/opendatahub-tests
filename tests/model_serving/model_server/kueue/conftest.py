from typing import Generator, Any, Dict

import pytest
from kubernetes.dynamic import DynamicClient
from _pytest.fixtures import FixtureRequest
from utilities.kueue_utils import (
    create_local_queue,
    create_cluster_queue,
    create_resource_flavor,
    LocalQueue,
    ClusterQueue,
    ResourceFlavor,
)
from ocp_resources.namespace import Namespace
from utilities.constants import ModelAndFormat, KServeDeploymentType
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate
from ocp_resources.secret import Secret
from ocp_resources.inference_service import InferenceService
from ocp_resources.serving_runtime import ServingRuntime
from utilities.constants import RuntimeTemplates, ModelFormat
import logging

BASIC_LOGGER = logging.getLogger(name="basic")


def kueue_resource_groups(
    flavor_name: str,
    cpu_quota: int,
    memory_quota: str,
) -> list[Dict[str, Any]]:
    return [
        {
            "coveredResources": ["cpu", "memory"],
            "flavors": [
                {
                    "name": flavor_name,
                    "resources": [
                        {"name": "cpu", "nominalQuota": cpu_quota},
                        {"name": "memory", "nominalQuota": memory_quota},
                    ],
                }
            ],
        }
    ]


@pytest.fixture(scope="class")
def kueue_cluster_queue_from_template(
    request: FixtureRequest,
    admin_client: DynamicClient,
) -> Generator[ClusterQueue, Any, Any]:
    if request.param.get("name") is None:
        raise ValueError("name is required")
    with create_cluster_queue(
        name=request.param.get("name"),
        client=admin_client,
        resource_groups=kueue_resource_groups(
            request.param.get("resource_flavor_name"), request.param.get("cpu_quota"), request.param.get("memory_quota")
        ),
        namespace_selector=request.param.get("namespace_selector", {}),
    ) as cluster_queue:
        yield cluster_queue


@pytest.fixture(scope="class")
def kueue_resource_flavor_from_template(
    request: FixtureRequest,
    admin_client: DynamicClient,
) -> Generator[ResourceFlavor, Any, Any]:
    if request.param.get("name") is None:
        raise ValueError("name is required")
    with create_resource_flavor(
        name=request.param.get("name"),
        client=admin_client,
    ) as resource_flavor:
        yield resource_flavor


@pytest.fixture(scope="class")
def kueue_local_queue_from_template(
    request: FixtureRequest,
    unprivileged_model_namespace: Namespace,
    admin_client: DynamicClient,
) -> Generator[LocalQueue, Any, Any]:
    if request.param.get("name") is None:
        raise ValueError("name is required")
    if request.param.get("cluster_queue") is None:
        raise ValueError("cluster_queue is required")
    with create_local_queue(
        name=request.param.get("name"),
        namespace=unprivileged_model_namespace.name,
        cluster_queue=request.param.get("cluster_queue"),
        client=admin_client,
    ) as local_queue:
        yield local_queue


@pytest.fixture(scope="class")
def kueue_raw_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    kueue_kserve_serving_runtime: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=f"{request.param['name']}-raw",
        namespace=unprivileged_model_namespace.name,
        external_route=True,
        runtime=kueue_kserve_serving_runtime.name,
        storage_path=request.param["model-dir"],
        storage_key=ci_endpoint_s3_secret.name,
        model_format=ModelAndFormat.OPENVINO_IR,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_version=request.param["model-version"],
        labels=request.param.get("labels", {}),
        resources=request.param.get(
            "resources", {"requests": {"cpu": "1", "memory": "8Gi"}, "limits": {"cpu": "2", "memory": "10Gi"}}
        ),
        min_replicas=request.param.get("min-replicas", 1),
        max_replicas=request.param.get("max-replicas", 2),
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def kueue_kserve_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    kueue_kserve_serving_runtime: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    deployment_mode = request.param["deployment-mode"]
    isvc_kwargs = {
        "client": admin_client,
        "name": f"{request.param['name']}-{deployment_mode.lower()}",
        "namespace": unprivileged_model_namespace.name,
        "runtime": kueue_kserve_serving_runtime.name,
        "storage_path": request.param["model-dir"],
        "storage_key": ci_endpoint_s3_secret.name,
        "model_format": ModelAndFormat.OPENVINO_IR,
        "deployment_mode": deployment_mode,
        "model_version": request.param["model-version"],
        "labels": request.param.get("labels", {}),
        "resources": request.param.get(
            "resources", {"requests": {"cpu": "1", "memory": "8Gi"}, "limits": {"cpu": "2", "memory": "10Gi"}}
        ),
        "min_replicas": request.param.get("min-replicas", 1),
        "max_replicas": request.param.get("max-replicas", 2),
    }

    if env_vars := request.param.get("env-vars"):
        isvc_kwargs["model_env_variables"] = env_vars

    if (min_replicas := request.param.get("min-replicas")) is not None:
        isvc_kwargs["min_replicas"] = min_replicas
        if min_replicas == 0:
            isvc_kwargs["wait_for_predictor_pods"] = False

    if scale_metric := request.param.get("scale-metric"):
        isvc_kwargs["scale_metric"] = scale_metric

    if (scale_target := request.param.get("scale-target")) is not None:
        isvc_kwargs["scale_target"] = scale_target

    if (resources := request.param.get("resources")) is not None:
        isvc_kwargs["resources"] = resources

    print("isvc_kwargs before create_isvc", isvc_kwargs)
    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def kueue_kserve_serving_runtime(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    runtime_kwargs = {
        "client": unprivileged_client,
        "namespace": unprivileged_model_namespace.name,
        "name": request.param["runtime-name"],
        "template_name": RuntimeTemplates.OVMS_KSERVE,
        "multi_model": False,
        "resources": {
            ModelFormat.OVMS: {
                "requests": {"cpu": "1", "memory": "8Gi"},
                "limits": {"cpu": "2", "memory": "10Gi"},
            }
        },
    }

    if model_format_name := request.param.get("model-format"):
        runtime_kwargs["model_format_name"] = model_format_name

    if supported_model_formats := request.param.get("supported-model-formats"):
        runtime_kwargs["supported_model_formats"] = supported_model_formats

    if runtime_image := request.param.get("runtime-image"):
        runtime_kwargs["runtime_image"] = runtime_image

    with ServingRuntimeFromTemplate(**runtime_kwargs) as model_runtime:
        yield model_runtime
