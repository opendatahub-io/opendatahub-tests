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
    get_pods_by_isvc_label,
)
from utilities.jira import is_jira_open
from utilities.logger import RedactedString
from utilities.serving_runtime import ServingRuntimeFromTemplate


# HTTP/REST model serving
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
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    http_s3_ovms_raw_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    predictor_pod = get_pods_by_isvc_label(
        client=unprivileged_client,
        isvc=http_s3_ovms_raw_inference_service,
    )[0]

    with ResourceEditor(
        patches={
            http_s3_ovms_raw_inference_service: {
                "metadata": {
                    "annotations": {Annotations.KserveAuth.SECURITY: "false"},
                }
            }
        }
    ):
        if is_jira_open(jira_id="RHOAIENG-19275", admin_client=admin_client):
            predictor_pod.wait_deleted()

        yield http_s3_ovms_raw_inference_service


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
    model_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    # Construct storage URI from CI bucket
    storage_uri = f"s3://{ci_s3_bucket_name}/{request.param['model-dir']}/"
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.ONNX}",
        namespace=unprivileged_model_namespace.name,
        runtime=http_s3_ovms_serving_runtime.name,
        storage_key=ci_endpoint_s3_secret.name,
        storage_path=urlparse(storage_uri).path,
        model_format=http_s3_ovms_serving_runtime.instance.spec.supportedModelFormats[0].name,
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
    model_service_account_2: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    # Construct storage URI from CI bucket
    storage_uri = f"s3://{ci_s3_bucket_name}/{request.param['model-dir']}/"
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.ONNX}-2",
        namespace=unprivileged_model_namespace.name,
        runtime=http_s3_ovms_serving_runtime.name,
        storage_key=ci_endpoint_s3_secret.name,
        storage_path=urlparse(storage_uri).path,
        model_format=http_s3_ovms_serving_runtime.instance.spec.supportedModelFormats[0].name,
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
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelName.MNIST}-runtime",
        namespace=unprivileged_model_namespace.name,
        template_name=RuntimeTemplates.OVMS_KSERVE,
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
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.ONNX}-raw",
        namespace=unprivileged_model_namespace.name,
        runtime=http_s3_ovms_serving_runtime.name,
        model_format=http_s3_ovms_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        storage_key=unprivileged_ci_endpoint_s3_secret.name,
        storage_path=request.param["model-dir"],
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def unprivileged_ci_endpoint_s3_secret(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
) -> Generator[Secret, Any, Any]:
    from utilities.infra import s3_endpoint_secret

    with s3_endpoint_secret(
        client=unprivileged_client,
        name="ci-bucket-unprivileged",
        namespace=unprivileged_model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=ci_s3_bucket_region,
        aws_s3_bucket=ci_s3_bucket_name,
        aws_s3_endpoint=ci_s3_bucket_endpoint,
    ) as secret:
        yield secret
