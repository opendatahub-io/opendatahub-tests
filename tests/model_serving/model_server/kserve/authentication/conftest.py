from collections.abc import Generator
from typing import Any
from urllib.parse import urlparse

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_runtime.mlserver.constant import MODEL_CONFIGS
from tests.model_serving.model_server.utils import arch_onnx_s3_path, wait_for_raw_isvc_https_infer_ready
from utilities.constants import (
    Annotations,
    KServeDeploymentType,
    ModelFormat,
    ModelName,
    Protocols,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
from utilities.infra import (
    create_inference_token,
    create_isvc_view_role,
    wait_for_inference_deployment_replicas,
)
from utilities.logger import RedactedString
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG
from utilities.serving_runtime import ServingRuntimeFromTemplate


# HTTP/REST model serving
@pytest.fixture(scope="class")
def auth_model_config(cluster_arch: str) -> dict[str, Any]:
    """Select a model and matching inference request for the cluster architecture."""
    if cluster_arch == "amd64":
        return {
            "path": arch_onnx_s3_path(cluster_arch),
            "template": RuntimeTemplates.OVMS_KSERVE,
            "inference": ONNX_INFERENCE_CONFIG,
        }

    model = MODEL_CONFIGS[ModelFormat.ONNX]
    return {
        "path": arch_onnx_s3_path(cluster_arch),
        "template": RuntimeTemplates.MLSERVER,
        "request": model["rest_query"],
    }


@pytest.fixture(scope="class")
def http_raw_view_role(
    unprivileged_client: DynamicClient,
    http_s3_ovms_raw_inference_service: InferenceService,
) -> Generator[Role, Any, Any]:
    with create_isvc_view_role(
        client=unprivileged_client,
        isvc=http_s3_ovms_raw_inference_service,
        name=f"{http_s3_ovms_raw_inference_service.name}-view",
        resource_names=[http_s3_ovms_raw_inference_service.name],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def http_raw_role_binding(
    unprivileged_client: DynamicClient,
    http_raw_view_role: Role,
    model_service_account: ServiceAccount,
    http_s3_ovms_raw_inference_service: InferenceService,
) -> Generator[RoleBinding, Any, Any]:
    with RoleBinding(
        client=unprivileged_client,
        namespace=model_service_account.namespace,
        name=f"{Protocols.HTTP}-{model_service_account.name}-view",
        role_ref_name=http_raw_view_role.name,
        role_ref_kind=http_raw_view_role.kind,
        subjects_kind=model_service_account.kind,
        subjects_name=model_service_account.name,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def http_raw_inference_token(model_service_account: ServiceAccount, http_raw_role_binding: RoleBinding) -> str:
    return RedactedString(value=create_inference_token(model_service_account=model_service_account))


@pytest.fixture()
def patched_remove_raw_authentication_isvc(
    unprivileged_client: DynamicClient,
    http_s3_ovms_raw_inference_service: InferenceService,
    http_raw_inference_token: str,
    auth_model_config: dict[str, Any],
) -> Generator[InferenceService, Any, Any]:
    with ResourceEditor(
        patches={
            http_s3_ovms_raw_inference_service: {
                "metadata": {
                    "annotations": {Annotations.KserveAuth.SECURITY: "false"},
                }
            }
        }
    ):
        http_s3_ovms_raw_inference_service.wait_for_condition(
            condition=http_s3_ovms_raw_inference_service.Condition.READY,
            status=http_s3_ovms_raw_inference_service.Condition.Status.TRUE,
            timeout=120,
        )
        wait_for_inference_deployment_replicas(
            client=unprivileged_client,
            isvc=http_s3_ovms_raw_inference_service,
        )
        wait_for_raw_isvc_https_infer_ready(
            isvc=http_s3_ovms_raw_inference_service, token=None, request_body=auth_model_config.get("request")
        )
        yield http_s3_ovms_raw_inference_service

    # ResourceEditor restores auth on exit; wait for ISVC to reconcile before next test
    http_s3_ovms_raw_inference_service.wait_for_condition(
        condition=http_s3_ovms_raw_inference_service.Condition.READY,
        status=http_s3_ovms_raw_inference_service.Condition.Status.TRUE,
        timeout=120,
    )
    wait_for_inference_deployment_replicas(
        client=unprivileged_client,
        isvc=http_s3_ovms_raw_inference_service,
    )
    wait_for_raw_isvc_https_infer_ready(
        isvc=http_s3_ovms_raw_inference_service,
        token=http_raw_inference_token,
        request_body=auth_model_config.get("request"),
    )


@pytest.fixture(scope="class")
def model_service_account_2(
    unprivileged_client: DynamicClient, models_endpoint_s3_secret: Secret
) -> Generator[ServiceAccount, Any, Any]:
    with ServiceAccount(
        client=unprivileged_client,
        namespace=models_endpoint_s3_secret.namespace,
        name="models-bucket-sa-2",
        secrets=[{"name": models_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def http_s3_ovms_raw_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    http_s3_ovms_serving_runtime: ServingRuntime,
    ci_s3_bucket_name: str,
    ci_endpoint_s3_secret: Secret,
    cluster_arch: str,
    model_service_account: ServiceAccount,
    auth_model_config: dict[str, Any],
) -> Generator[InferenceService, Any, Any]:
    # Construct storage URI from CI bucket
    model_path = auth_model_config["path"] if cluster_arch == "arm64" else request.param["model-dir"]
    storage_uri = f"s3://{ci_s3_bucket_name}/{model_path}/"
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.ONNX}",
        namespace=unprivileged_model_namespace.name,
        runtime=http_s3_ovms_serving_runtime.name,
        storage_key=request.getfixturevalue("models_endpoint_s3_secret").name
        if cluster_arch == "arm64"
        else ci_endpoint_s3_secret.name,
        storage_path=urlparse(storage_uri).path,
        model_format=ModelFormat.ONNX
        if cluster_arch == "arm64"
        else http_s3_ovms_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_service_account=model_service_account.name,
        enable_auth=True,
        external_route=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def http_s3_ovms_raw_inference_service_2(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    http_s3_ovms_serving_runtime: ServingRuntime,
    ci_s3_bucket_name: str,
    ci_endpoint_s3_secret: Secret,
    cluster_arch: str,
    model_service_account_2: ServiceAccount,
    auth_model_config: dict[str, Any],
) -> Generator[InferenceService, Any, Any]:
    # Construct storage URI from CI bucket
    model_path = auth_model_config["path"] if cluster_arch == "arm64" else request.param["model-dir"]
    storage_uri = f"s3://{ci_s3_bucket_name}/{model_path}/"
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.ONNX}-2",
        namespace=unprivileged_model_namespace.name,
        runtime=http_s3_ovms_serving_runtime.name,
        storage_key=request.getfixturevalue("models_endpoint_s3_secret").name
        if cluster_arch == "arm64"
        else ci_endpoint_s3_secret.name,
        storage_path=urlparse(storage_uri).path,
        model_format=ModelFormat.ONNX
        if cluster_arch == "arm64"
        else http_s3_ovms_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_service_account=model_service_account_2.name,
        enable_auth=True,
        external_route=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def http_s3_ovms_serving_runtime(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    auth_model_config: dict[str, Any],
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelName.MNIST}-runtime",
        namespace=unprivileged_model_namespace.name,
        template_name=auth_model_config["template"],
        multi_model=False,
        enable_http=True,
        enable_grpc=False,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def unprivileged_s3_ovms_raw_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    http_s3_ovms_serving_runtime: ServingRuntime,
    unprivileged_ci_endpoint_s3_secret: Secret,
    auth_model_config: dict[str, Any],
    cluster_arch: str,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.ONNX}-raw",
        namespace=unprivileged_model_namespace.name,
        runtime=http_s3_ovms_serving_runtime.name,
        model_format=ModelFormat.ONNX
        if cluster_arch == "arm64"
        else http_s3_ovms_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        storage_key=unprivileged_ci_endpoint_s3_secret.name,
        storage_path=auth_model_config["path"] if cluster_arch == "arm64" else request.param["model-dir"],
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def unprivileged_ci_endpoint_s3_secret(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
    cluster_arch: str,
) -> Generator[Secret, Any, Any]:
    from utilities.infra import s3_endpoint_secret

    with s3_endpoint_secret(
        client=unprivileged_client,
        name="ci-bucket-unprivileged",
        namespace=unprivileged_model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=request.getfixturevalue("models_s3_bucket_region")
        if cluster_arch == "arm64"
        else ci_s3_bucket_region,
        aws_s3_bucket=request.getfixturevalue("models_s3_bucket_name")
        if cluster_arch == "arm64"
        else ci_s3_bucket_name,
        aws_s3_endpoint=request.getfixturevalue("models_s3_bucket_endpoint")
        if cluster_arch == "arm64"
        else ci_s3_bucket_endpoint,
    ) as secret:
        yield secret
