from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime

from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelInferenceRuntime,
    Protocols,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def http_s3_openvino_second_model_mesh_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    ci_model_mesh_endpoint_s3_secret: Secret,
    model_mesh_model_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    # Dynamically select the used ServingRuntime by passing "runtime-fixture-name" request.param
    runtime = request.getfixturevalue(argname=request.param["runtime-fixture-name"])
    with create_isvc(
        client=admin_client,
        name=f"{Protocols.HTTP}-{ModelFormat.OPENVINO}-2",
        namespace=model_namespace.name,
        runtime=runtime.name,
        model_service_account=model_mesh_model_service_account.name,
        storage_key=ci_model_mesh_endpoint_s3_secret.name,
        storage_path=request.param["model-path"],
        model_format=request.param["model-format"],
        deployment_mode=KServeDeploymentType.MODEL_MESH,
        model_version=request.param["model-version"],
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def http_s3_ovms_external_route_model_mesh_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        namespace=model_namespace.name,
        name=f"{Protocols.HTTP}-{ModelInferenceRuntime.OPENVINO_RUNTIME}-exposed",
        template_name=RuntimeTemplates.OVMS_MODEL_MESH,
        multi_model=True,
        protocol="REST",
        resources={
            "ovms": {
                "requests": {"cpu": "1", "memory": "4Gi"},
                "limits": {"cpu": "2", "memory": "8Gi"},
            },
        },
        enable_external_route=True,
    ) as model_runtime:
        yield model_runtime
