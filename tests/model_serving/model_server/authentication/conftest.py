from typing import Any, Generator

import pytest
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
    RuntimeTemplates,
    RunTimeConfigs,
    ModelAndFormat,
    ModelVersion,
)
from utilities.jira import is_jira_open
from utilities.logger import RedactedString
from utilities.serving_runtime import ServingRuntimeFromTemplate
from utilities.constants import Annotations


@pytest.fixture(scope="class")
def model_service_account_2(
    unprivileged_client: DynamicClient, ci_endpoint_s3_secret: Secret
) -> Generator[ServiceAccount, Any, Any]:
    with ServiceAccount(
        client=unprivileged_client,
        namespace=ci_endpoint_s3_secret.namespace,
        name="models-bucket-sa-2",
        secrets=[{"name": ci_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


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
    return RedactedString(value=create_inference_token(model_service_account=ci_service_account))


@pytest.fixture(scope="class")
def ovms_kserve_serving_runtime_auth(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-ovms-runtime",
        namespace=unprivileged_model_namespace.name,
        template_name=RuntimeTemplates.OVMS_KSERVE,
        multi_model=False,
        model_format_name=RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG["model-format"],
        resources={
            ModelFormat.OVMS: {
                "requests": {"cpu": "1", "memory": "4Gi"},
                "limits": {"cpu": "2", "memory": "8Gi"},
            }
        },
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def http_ovms_serverless_inference_service(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ovms_kserve_serving_runtime_auth: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.ONNX}",
        namespace=unprivileged_model_namespace.name,
        runtime=ovms_kserve_serving_runtime_auth.name,
        model_format=ModelAndFormat.OPENVINO_IR,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        enable_auth=True,
        storage_key=ci_endpoint_s3_secret.name,
        storage_path="test-dir",
        model_version=ModelVersion.OPSET13,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def http_ovms_raw_inference_service(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ovms_kserve_serving_runtime_auth: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
    model_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.ONNX}",
        namespace=unprivileged_model_namespace.name,
        runtime=ovms_kserve_serving_runtime_auth.name,
        storage_key=ci_endpoint_s3_secret.name,
        storage_path="test-dir",
        model_format=ModelAndFormat.OPENVINO_IR,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_service_account=model_service_account.name,
        enable_auth=True,
        external_route=True,
        model_version=ModelVersion.OPSET13,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def http_ovms_raw_inference_service_2(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ovms_kserve_serving_runtime_auth: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
    model_service_account_2: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.ONNX}-2",
        namespace=unprivileged_model_namespace.name,
        runtime=ovms_kserve_serving_runtime_auth.name,
        storage_key=ci_endpoint_s3_secret.name,
        storage_path="test-dir",
        model_format=ModelAndFormat.OPENVINO_IR,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_service_account=model_service_account_2.name,
        enable_auth=True,
        external_route=True,
        model_version=ModelVersion.OPSET13,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def http_ovms_view_role(
    unprivileged_client: DynamicClient,
    http_ovms_serverless_inference_service: InferenceService,
) -> Generator[Role, Any, Any]:
    with create_isvc_view_role(
        client=unprivileged_client,
        isvc=http_ovms_serverless_inference_service,
        name=f"{http_ovms_serverless_inference_service.name}-view",
        resource_names=[http_ovms_serverless_inference_service.name],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def http_ovms_raw_view_role(
    unprivileged_client: DynamicClient,
    http_ovms_raw_inference_service: InferenceService,
) -> Generator[Role, Any, Any]:
    with create_isvc_view_role(
        client=unprivileged_client,
        isvc=http_ovms_raw_inference_service,
        name=f"{http_ovms_raw_inference_service.name}-view",
        resource_names=[http_ovms_raw_inference_service.name],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def http_ovms_role_binding(
    unprivileged_client: DynamicClient,
    http_ovms_view_role: Role,
    model_service_account: ServiceAccount,
    http_ovms_serverless_inference_service: InferenceService,
) -> Generator[RoleBinding, Any, Any]:
    with RoleBinding(
        client=unprivileged_client,
        namespace=model_service_account.namespace,
        name=f"{Protocols.HTTP}-{model_service_account.name}-ovms-view",
        role_ref_name=http_ovms_view_role.name,
        role_ref_kind=http_ovms_view_role.kind,
        subjects_kind=model_service_account.kind,
        subjects_name=model_service_account.name,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def http_ovms_raw_role_binding(
    unprivileged_client: DynamicClient,
    http_ovms_raw_view_role: Role,
    model_service_account: ServiceAccount,
    http_ovms_raw_inference_service: InferenceService,
) -> Generator[RoleBinding, Any, Any]:
    with RoleBinding(
        client=unprivileged_client,
        namespace=model_service_account.namespace,
        name=f"{Protocols.HTTP}-{model_service_account.name}-ovms-view",
        role_ref_name=http_ovms_raw_view_role.name,
        role_ref_kind=http_ovms_raw_view_role.kind,
        subjects_kind=model_service_account.kind,
        subjects_name=model_service_account.name,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def http_ovms_inference_token(model_service_account: ServiceAccount, http_ovms_role_binding: RoleBinding) -> str:
    return RedactedString(value=create_inference_token(model_service_account=model_service_account))


@pytest.fixture(scope="class")
def http_ovms_raw_inference_token(
    model_service_account: ServiceAccount, http_ovms_raw_role_binding: RoleBinding
) -> str:
    return RedactedString(value=create_inference_token(model_service_account=model_service_account))


@pytest.fixture()
def patched_remove_ovms_authentication_isvc(
    http_ovms_serverless_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    with ResourceEditor(
        patches={
            http_ovms_serverless_inference_service: {
                "metadata": {
                    "annotations": {Annotations.KserveAuth.SECURITY: "false"},
                }
            }
        }
    ):
        yield http_ovms_serverless_inference_service


@pytest.fixture()
def patched_remove_ovms_raw_authentication_isvc(
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    http_ovms_raw_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    predictor_pod = get_pods_by_isvc_label(
        client=unprivileged_client,
        isvc=http_ovms_raw_inference_service,
    )[0]

    with ResourceEditor(
        patches={
            http_ovms_raw_inference_service: {
                "metadata": {
                    "annotations": {Annotations.KserveAuth.SECURITY: "false"},
                }
            }
        }
    ):
        if is_jira_open(jira_id="RHOAIENG-19275", admin_client=admin_client):
            predictor_pod.wait_deleted()

        yield http_ovms_raw_inference_service
