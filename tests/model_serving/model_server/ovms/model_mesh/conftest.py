from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import ResourceEditor
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime

from utilities.constants import (
    Protocols,
)
from utilities.infra import create_inference_token, create_isvc_view_role


@pytest.fixture(scope="class")
def model_mesh_view_role(
    unprivileged_client: DynamicClient,
    http_s3_openvino_model_mesh_inference_service: ServingRuntime,
) -> Generator[Role, Any, Any]:
    with create_isvc_view_role(
        client=unprivileged_client,
        isvc=http_s3_openvino_model_mesh_inference_service,
        name=f"{http_s3_openvino_model_mesh_inference_service.name}-view",
        resource_names=[http_s3_openvino_model_mesh_inference_service.name],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def model_mesh_role_binding(
    unprivileged_client: DynamicClient,
    model_mesh_view_role: Role,
    ci_service_account: ServiceAccount,
) -> Generator[RoleBinding, Any, Any]:
    with RoleBinding(
        client=unprivileged_client,
        namespace=ci_service_account.namespace,
        name=f"{Protocols.HTTP}-{ci_service_account.name}-view",
        role_ref_name=model_mesh_view_role.name,
        role_ref_kind=model_mesh_view_role.kind,
        subjects_kind=ci_service_account.kind,
        subjects_name=ci_service_account.name,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def model_mesh_inference_token(
    ci_service_account: ServiceAccount,
    model_mesh_role_binding: RoleBinding,
) -> str:
    return create_inference_token(model_service_account=ci_service_account)


@pytest.fixture()
def patched_model_mesh_sr_with_authentication(
    http_s3_ovms_model_mesh_serving_runtime: ServingRuntime,
) -> Generator[None, None, None]:
    with ResourceEditor(
        patches={
            http_s3_ovms_model_mesh_serving_runtime: {
                "metadata": {
                    "annotations": {"enable-auth": "true"},
                }
            }
        }
    ):
        yield
