import shlex
from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.role_binding import RoleBinding
from ocp_resources.role import Role
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.authorino import Authorino
from ocp_resources.serving_runtime import ServingRuntime
from pyhelper_utils.shell import run_command
from pytest_testconfig import config as py_config

from utilities.infra import create_isvc_view_role, create_ns, s3_endpoint_secret
from tests.model_serving.model_server.utils import create_isvc, get_pods_by_isvc_label
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    Protocols,
    ModelInferenceRuntime,
    RuntimeTemplates,
)
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="session")
def skip_if_no_authorino_operator(admin_client: DynamicClient):
    name = "authorino"
    if not Authorino(
        client=admin_client,
        name=name,
        namespace=f"{py_config['applications_namespace']}-auth-provider",
    ).exists:
        pytest.skip(f"{name} operator is missing from the cluster")


# GRPC model serving
@pytest.fixture(scope="class")
def grpc_model_service_account(admin_client: DynamicClient, models_endpoint_s3_secret: Secret) -> ServiceAccount:
    with ServiceAccount(
        client=admin_client,
        namespace=models_endpoint_s3_secret.namespace,
        name=f"{Protocols.GRPC}-models-bucket-sa",
        secrets=[{"name": models_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def grpc_s3_caikit_serving_runtime(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> ServingRuntime:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=f"{Protocols.GRPC}-{ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME}",
        namespace=model_namespace.name,
        template_name=RuntimeTemplates.CAIKIT_TGIS_SERVING,
        multi_model=False,
        enable_http=False,
        enable_grpc=True,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def grpc_s3_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    grpc_s3_caikit_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    grpc_model_service_account: ServiceAccount,
) -> InferenceService:
    with create_isvc(
        client=admin_client,
        name=f"{Protocols.GRPC}-{ModelFormat.CAIKIT}",
        namespace=model_namespace.name,
        runtime=grpc_s3_caikit_serving_runtime.name,
        storage_uri=s3_models_storage_uri,
        model_format=grpc_s3_caikit_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        model_service_account=grpc_model_service_account.name,
        enable_auth=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def http_view_role(
    admin_client: DynamicClient,
    http_s3_caikit_serverless_inference_service: InferenceService,
) -> Role:
    with create_isvc_view_role(
        client=admin_client,
        isvc=http_s3_caikit_serverless_inference_service,
        name=f"{http_s3_caikit_serverless_inference_service.name}-view",
        resource_names=[http_s3_caikit_serverless_inference_service.name],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def http_role_binding(
    admin_client: DynamicClient,
    http_view_role: Role,
    http_model_service_account: ServiceAccount,
    http_s3_caikit_serverless_inference_service: InferenceService,
) -> RoleBinding:
    with RoleBinding(
        client=admin_client,
        namespace=http_model_service_account.namespace,
        name=f"{Protocols.HTTP}-{http_model_service_account.name}-view",
        role_ref_name=http_view_role.name,
        role_ref_kind=http_view_role.kind,
        subjects_kind=http_model_service_account.kind,
        subjects_name=http_model_service_account.name,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def http_inference_token(http_model_service_account: ServiceAccount, http_role_binding: RoleBinding) -> str:
    return run_command(
        command=shlex.split(
            f"oc create token -n {http_model_service_account.namespace} {http_model_service_account.name}"
        )
    )[1].strip()


@pytest.fixture()
def patched_remove_authentication_isvc(
    admin_client: DynamicClient,
    http_s3_caikit_serverless_inference_service: InferenceService,
) -> InferenceService:
    with ResourceEditor(
        patches={
            http_s3_caikit_serverless_inference_service: {
                "metadata": {
                    "annotations": {"security.opendatahub.io/enable-auth": "false"},
                }
            }
        }
    ):
        predictor_pod = get_pods_by_isvc_label(
            client=admin_client,
            isvc=http_s3_caikit_serverless_inference_service,
        )[0]
        predictor_pod.wait_deleted()

        yield http_s3_caikit_serverless_inference_service


@pytest.fixture(scope="class")
def grpc_view_role(admin_client: DynamicClient, grpc_s3_inference_service: InferenceService) -> Role:
    with create_isvc_view_role(
        client=admin_client,
        isvc=grpc_s3_inference_service,
        name=f"{grpc_s3_inference_service.name}-view",
        resource_names=[grpc_s3_inference_service.name],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def grpc_role_binding(
    admin_client: DynamicClient,
    grpc_view_role: Role,
    grpc_model_service_account: ServiceAccount,
    grpc_s3_inference_service: InferenceService,
) -> RoleBinding:
    with RoleBinding(
        client=admin_client,
        namespace=grpc_model_service_account.namespace,
        name=f"{Protocols.GRPC}-{grpc_model_service_account.name}-view",
        role_ref_name=grpc_view_role.name,
        role_ref_kind=grpc_view_role.kind,
        subjects_kind=grpc_model_service_account.kind,
        subjects_name=grpc_model_service_account.name,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def grpc_inference_token(grpc_model_service_account: ServiceAccount, grpc_role_binding: RoleBinding) -> str:
    return run_command(
        command=shlex.split(
            f"oc create token -n {grpc_model_service_account.namespace} {grpc_model_service_account.name}"
        )
    )[1].strip()


@pytest.fixture(scope="class")
def http_s3_caikit_serverless_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    http_s3_caikit_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    http_model_service_account: ServiceAccount,
) -> InferenceService:
    with create_isvc(
        client=admin_client,
        name=f"{Protocols.HTTP}-{ModelFormat.CAIKIT}",
        namespace=model_namespace.name,
        runtime=http_s3_caikit_serving_runtime.name,
        storage_uri=s3_models_storage_uri,
        model_format=http_s3_caikit_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        model_service_account=http_model_service_account.name,
        enable_auth=True,
    ) as isvc:
        yield isvc


# Unprivileged user tests
@pytest.fixture(scope="class")
def unprivileged_model_namespace(
    request: FixtureRequest, unprivileged_client: DynamicClient
) -> Generator[Namespace, Any, Any]:
    with create_ns(unprivileged_client=unprivileged_client, name=request.param["name"]) as ns:
        yield ns


@pytest.fixture(scope="class")
def unprivileged_s3_caikit_serving_runtime(
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> ServingRuntime:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        unprivileged_client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME}",
        namespace=unprivileged_model_namespace.name,
        template_name=RuntimeTemplates.CAIKIT_TGIS_SERVING,
        multi_model=False,
        enable_http=True,
        enable_grpc=False,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def unprivileged_models_endpoint_s3_secret(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Secret:
    with s3_endpoint_secret(
        admin_client=unprivileged_client,
        name="models-bucket-secret",
        namespace=unprivileged_model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_bucket=models_s3_bucket_name,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def unprivileged_model_service_account(
    unprivileged_client: DynamicClient, unprivileged_models_endpoint_s3_secret: Secret
) -> ServiceAccount:
    with ServiceAccount(
        client=unprivileged_client,
        namespace=unprivileged_models_endpoint_s3_secret.namespace,
        name=f"{Protocols.HTTP}-models-bucket-sa",
        secrets=[{"name": unprivileged_models_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def unprivileged_s3_caikit_serverless_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    unprivileged_s3_caikit_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    unprivileged_model_service_account: ServiceAccount,
) -> InferenceService:
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.CAIKIT}",
        namespace=unprivileged_model_namespace.name,
        runtime=unprivileged_s3_caikit_serving_runtime.name,
        storage_uri=s3_models_storage_uri,
        model_format=unprivileged_s3_caikit_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        model_service_account=unprivileged_model_service_account.name,
    ) as isvc:
        yield isvc
