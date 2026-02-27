from collections.abc import Generator
from typing import Any

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.serving_runtime import ServingRuntime

from utilities.inference_utils import create_isvc
from utilities.minio import create_minio_data_connection_secret


@pytest.fixture(scope="class")
def kserve_ovms_minio_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    unprivileged_minio_data_connection: Secret,
    ovms_kserve_serving_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name=request.param["name"],
        namespace=unprivileged_model_namespace.name,
        deployment_mode=request.param["deployment-mode"],
        model_format=request.param["model-format"],
        runtime=ovms_kserve_serving_runtime.name,
        storage_key=unprivileged_minio_data_connection.name,
        storage_path=request.param["model-dir"],
        model_version=request.param["model-version"],
        external_route=request.param.get("external-route"),
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def unprivileged_minio_data_connection(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    minio_service: Service,
) -> Generator[Secret, Any, Any]:
    with create_minio_data_connection_secret(
        minio_service=minio_service,
        model_namespace=unprivileged_model_namespace.name,
        aws_s3_bucket=request.param["bucket"],
        client=unprivileged_client,
    ) as secret:
        yield secret
