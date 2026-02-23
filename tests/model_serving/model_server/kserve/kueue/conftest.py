from collections.abc import Generator
from typing import Any

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from utilities.constants import KServeDeploymentType, ModelAndFormat, ModelFormat, RuntimeTemplates
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


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
