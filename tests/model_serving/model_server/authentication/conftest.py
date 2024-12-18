import shlex

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import ResourceEditor
from ocp_resources.role_binding import RoleBinding
from ocp_resources.role import Role
from ocp_resources.service_account import ServiceAccount
from ocp_resources.authorino import Authorino
from pyhelper_utils.shell import run_command


from tests.model_serving.model_server.authentication.utils import (
    create_isvc_view_role,
)
from tests.model_serving.model_server.utils import get_pods_by_isvc_label
from utilities.constants import Protocols


@pytest.fixture(scope="session")
def skip_if_no_authorino_operator(admin_client: DynamicClient):
    name = "authorino"
    if not Authorino(
        client=admin_client,
        name=name,
        namespace="redhat-ods-applications-auth-provider",
    ).exists:
        pytest.skip(f"{name} operator is missing from the cluster")


@pytest.fixture(scope="class")
def http_view_role(admin_client: DynamicClient, http_s3_inference_service: InferenceService) -> Role:
    with create_isvc_view_role(
        client=admin_client,
        isvc=http_s3_inference_service,
        name=f"{http_s3_inference_service.name}-view",
        resource_names=[http_s3_inference_service.name],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def http_role_binding(
    admin_client: DynamicClient,
    http_view_role: Role,
    http_model_service_account: ServiceAccount,
    http_s3_inference_service: InferenceService,
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
    admin_client: DynamicClient, http_s3_inference_service: InferenceService
) -> InferenceService:
    with ResourceEditor(
        patches={
            http_s3_inference_service: {
                "metadata": {
                    "annotations": {"security.opendatahub.io/enable-auth": "false"},
                }
            }
        }
    ):
        predictor_pod = get_pods_by_isvc_label(
            client=admin_client,
            isvc=http_s3_inference_service,
        )[0]
        predictor_pod.wait_deleted()

        yield http_s3_inference_service


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
