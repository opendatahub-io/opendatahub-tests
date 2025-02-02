import shlex

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import ResourceEditor
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pyhelper_utils.shell import run_command

from utilities.constants import (
    Protocols,
)
from utilities.infra import create_isvc_view_role


@pytest.fixture(scope="class")
def model_mesh_view_role(
    admin_client: DynamicClient,
    http_s3_openvino_model_mesh_inference_service: ServingRuntime,
) -> Role:
    with create_isvc_view_role(
        client=admin_client,
        isvc=http_s3_openvino_model_mesh_inference_service,
        name=f"{http_s3_openvino_model_mesh_inference_service.name}-view",
        resource_names=[http_s3_openvino_model_mesh_inference_service.name],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def model_mesh_role_binding(
    admin_client: DynamicClient,
    model_mesh_view_role: Role,
    model_mesh_model_service_account: ServiceAccount,
) -> RoleBinding:
    with RoleBinding(
        client=admin_client,
        namespace=model_mesh_model_service_account.namespace,
        name=f"{Protocols.HTTP}-{model_mesh_model_service_account.name}-view",
        role_ref_name=model_mesh_view_role.name,
        role_ref_kind=model_mesh_view_role.kind,
        subjects_kind=model_mesh_model_service_account.kind,
        subjects_name=model_mesh_model_service_account.name,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def model_mesh_inference_token(
    model_mesh_model_service_account: ServiceAccount,
    model_mesh_role_binding: RoleBinding,
) -> str:
    return run_command(
        command=shlex.split(
            f"oc create token -n {model_mesh_model_service_account.namespace} {model_mesh_model_service_account.name}"
        )
    )[1].strip()


@pytest.fixture()
def patched_model_mesh_sr_with_authentication(
    admin_client: DynamicClient,
    http_s3_ovms_model_mesh_serving_runtime: ServingRuntime,
) -> None:
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
