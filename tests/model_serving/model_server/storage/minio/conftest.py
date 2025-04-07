from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime

from utilities.constants import KServeDeploymentType
from utilities.inference_utils import create_isvc


@pytest.fixture(scope="class")
def kserve_ovms_minio_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_data_connection: Secret,
    ovms_kserve_serving_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace.name,
        deployment_mode=request.param["deployment-mode"],
        model_format=request.param["model-format"],
        runtime=ovms_kserve_serving_runtime.name,
        storage_key=minio_data_connection.name,
        storage_path=request.param["model-dir"],
        model_version=request.param["model-version"],
        external_route=request.param.get("external-route"),
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def minio_service_account(
    admin_client: DynamicClient, minio_data_connection: Secret
) -> Generator[ServiceAccount, Any, Any]:
    with ServiceAccount(
        client=admin_client,
        namespace=minio_data_connection.namespace,
        name="ci-models-bucket-sa",
        secrets=[{"name": minio_data_connection.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def model_mesh_ovms_minio_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    minio_data_connection: Secret,
    minio_service_account: ServiceAccount,
    model_namespace: Namespace,
    http_s3_ovms_model_mesh_serving_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace.name,
        runtime=http_s3_ovms_model_mesh_serving_runtime.name,
        model_service_account=minio_service_account.name,
        storage_key=minio_data_connection.name,
        storage_path=request.param["model-dir"],
        model_format=request.param["model-format"],
        deployment_mode=KServeDeploymentType.MODEL_MESH,
        model_version=request.param["model-version"],
    ) as isvc:
        yield isvc
