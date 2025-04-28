from typing import Any, Generator
from urllib.parse import urlparse

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
from ocp_resources.serving_runtime import ServingRuntime

from utilities.inference_utils import create_isvc
from utilities.infra import (
    create_isvc_view_role,
    get_pods_by_isvc_label,
    create_inference_token,
)
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    Protocols,
    ModelInferenceRuntime,
    RuntimeTemplates,
)
from utilities.jira import is_jira_open
from utilities.serving_runtime import ServingRuntimeFromTemplate
from utilities.constants import Annotations


# GRPC model serving
@pytest.fixture(scope="class")
def grpc_model_service_account(
    unprivileged_client: DynamicClient, models_endpoint_s3_secret: Secret
) -> Generator[ServiceAccount, Any, Any]:
    with ServiceAccount(
        client=unprivileged_client,
        namespace=models_endpoint_s3_secret.namespace,
        name=f"{Protocols.GRPC}-models-bucket-sa",
        secrets=[{"name": models_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def grpc_s3_caikit_serving_runtime(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=unprivileged_client,
        name=f"{Protocols.GRPC}-{ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME}",
        namespace=unprivileged_model_namespace.name,
        template_name=RuntimeTemplates.CAIKIT_TGIS_SERVING,
        multi_model=False,
        enable_http=False,
        enable_grpc=True,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def grpc_s3_inference_service(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    grpc_s3_caikit_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    models_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.GRPC}-{ModelFormat.CAIKIT}",
        namespace=unprivileged_model_namespace.name,
        runtime=grpc_s3_caikit_serving_runtime.name,
        model_format=grpc_s3_caikit_serving_runtime.instance.spec.supportedModelFormats[0].name,
        storage_key=models_endpoint_s3_secret.name,
        storage_path=urlparse(s3_models_storage_uri).path,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        enable_auth=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def http_view_role(
    unprivileged_client: DynamicClient,
    http_s3_caikit_serverless_inference_service: InferenceService,
) -> Generator[Role, Any, Any]:
    with create_isvc_view_role(
        client=unprivileged_client,
        isvc=http_s3_caikit_serverless_inference_service,
        name=f"{http_s3_caikit_serverless_inference_service.name}-view",
        resource_names=[http_s3_caikit_serverless_inference_service.name],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def http_raw_view_role(
    unprivileged_client: DynamicClient,
    http_s3_caikit_raw_inference_service: InferenceService,
) -> Generator[Role, Any, Any]:
    with create_isvc_view_role(
        client=unprivileged_client,
        isvc=http_s3_caikit_raw_inference_service,
        name=f"{http_s3_caikit_raw_inference_service.name}-view",
        resource_names=[http_s3_caikit_raw_inference_service.name],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def http_role_binding(
    unprivileged_client: DynamicClient,
    http_view_role: Role,
    model_service_account: ServiceAccount,
    http_s3_caikit_serverless_inference_service: InferenceService,
) -> Generator[RoleBinding, Any, Any]:
    with RoleBinding(
        client=unprivileged_client,
        namespace=model_service_account.namespace,
        name=f"{Protocols.HTTP}-{model_service_account.name}-view",
        role_ref_name=http_view_role.name,
        role_ref_kind=http_view_role.kind,
        subjects_kind=model_service_account.kind,
        subjects_name=model_service_account.name,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def http_raw_role_binding(
    unprivileged_client: DynamicClient,
    http_raw_view_role: Role,
    model_service_account: ServiceAccount,
    http_s3_caikit_raw_inference_service: InferenceService,
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
def http_inference_token(model_service_account: ServiceAccount, http_role_binding: RoleBinding) -> str:
    return create_inference_token(model_service_account=model_service_account)


@pytest.fixture(scope="class")
def http_raw_inference_token(model_service_account: ServiceAccount, http_raw_role_binding: RoleBinding) -> str:
    return create_inference_token(model_service_account=model_service_account)


@pytest.fixture()
def patched_remove_authentication_isvc(
    http_s3_caikit_serverless_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    with ResourceEditor(
        patches={
            http_s3_caikit_serverless_inference_service: {
                "metadata": {
                    "annotations": {Annotations.KserveAuth.SECURITY: "false"},
                }
            }
        }
    ):
        yield http_s3_caikit_serverless_inference_service


@pytest.fixture()
def patched_remove_raw_authentication_isvc(
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    http_s3_caikit_raw_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    predictor_pod = get_pods_by_isvc_label(
        client=unprivileged_client,
        isvc=http_s3_caikit_raw_inference_service,
    )[0]

    with ResourceEditor(
        patches={
            http_s3_caikit_raw_inference_service: {
                "metadata": {
                    "annotations": {Annotations.KserveAuth.SECURITY: "false"},
                }
            }
        }
    ):
        if is_jira_open(jira_id="RHOAIENG-19275", admin_client=admin_client):
            predictor_pod.wait_deleted()

        yield http_s3_caikit_raw_inference_service


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
def grpc_view_role(
    unprivileged_client: DynamicClient, grpc_s3_inference_service: InferenceService
) -> Generator[Role, Any, Any]:
    with create_isvc_view_role(
        client=unprivileged_client,
        isvc=grpc_s3_inference_service,
        name=f"{grpc_s3_inference_service.name}-view",
        resource_names=[grpc_s3_inference_service.name],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def grpc_role_binding(
    unprivileged_client: DynamicClient,
    grpc_view_role: Role,
    grpc_model_service_account: ServiceAccount,
    grpc_s3_inference_service: InferenceService,
) -> Generator[RoleBinding, Any, Any]:
    with RoleBinding(
        client=unprivileged_client,
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
    return create_inference_token(model_service_account=grpc_model_service_account)


@pytest.fixture(scope="class")
def http_s3_caikit_serverless_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    http_s3_caikit_tgis_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    models_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.CAIKIT}",
        namespace=unprivileged_model_namespace.name,
        runtime=http_s3_caikit_tgis_serving_runtime.name,
        model_format=http_s3_caikit_tgis_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        enable_auth=True,
        storage_key=models_endpoint_s3_secret.name,
        storage_path=urlparse(s3_models_storage_uri).path,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def http_s3_caikit_raw_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    http_s3_caikit_tgis_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    models_endpoint_s3_secret: Secret,
    model_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.CAIKIT}",
        namespace=unprivileged_model_namespace.name,
        runtime=http_s3_caikit_tgis_serving_runtime.name,
        storage_key=models_endpoint_s3_secret.name,
        storage_path=urlparse(s3_models_storage_uri).path,
        model_format=http_s3_caikit_tgis_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_service_account=model_service_account.name,
        enable_auth=True,
        external_route=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def http_s3_caikit_raw_inference_service_2(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    http_s3_caikit_tgis_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    model_service_account_2: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.CAIKIT}-2",
        namespace=unprivileged_model_namespace.name,
        runtime=http_s3_caikit_tgis_serving_runtime.name,
        storage_uri=s3_models_storage_uri,
        model_format=http_s3_caikit_tgis_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_service_account=model_service_account_2.name,
        enable_auth=True,
        external_route=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def http_s3_caikit_tgis_serving_runtime(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME}",
        namespace=unprivileged_model_namespace.name,
        template_name=RuntimeTemplates.CAIKIT_TGIS_SERVING,
        multi_model=False,
        enable_http=True,
        enable_grpc=False,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture()
def patched_remove_authentication_model_mesh_runtime(
    http_s3_ovms_model_mesh_serving_runtime: ServingRuntime,
) -> Generator[ServingRuntime, Any, Any]:
    with ResourceEditor(
        patches={
            http_s3_ovms_model_mesh_serving_runtime: {
                "metadata": {
                    "annotations": {"enable-auth": "false"},
                }
            }
        }
    ):
        yield http_s3_ovms_model_mesh_serving_runtime


@pytest.fixture(scope="class")
def http_model_mesh_view_role(
    unprivileged_client: DynamicClient,
    http_s3_openvino_model_mesh_inference_service: InferenceService,
    http_s3_ovms_model_mesh_serving_runtime: ServingRuntime,
) -> Generator[Role, Any, Any]:
    with Role(
        client=unprivileged_client,
        name=f"{http_s3_openvino_model_mesh_inference_service.name}-view",
        namespace=http_s3_openvino_model_mesh_inference_service.namespace,
        rules=[
            {"apiGroups": [""], "resources": ["services"], "verbs": ["get"]},
        ],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def http_model_mesh_role_binding(
    unprivileged_client: DynamicClient,
    http_model_mesh_view_role: Role,
    ci_service_account: ServiceAccount,
) -> Generator[RoleBinding, Any, Any]:
    with RoleBinding(
        client=unprivileged_client,
        namespace=ci_service_account.namespace,
        name=f"{Protocols.HTTP}-{ci_service_account.name}-view",
        role_ref_name=http_model_mesh_view_role.name,
        role_ref_kind=http_model_mesh_view_role.kind,
        subjects_kind=ci_service_account.kind,
        subjects_name=ci_service_account.name,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def http_model_mesh_inference_token(
    ci_service_account: ServiceAccount, http_model_mesh_role_binding: RoleBinding
) -> str:
    return create_inference_token(model_service_account=ci_service_account)
