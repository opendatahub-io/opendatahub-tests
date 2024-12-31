import shlex
from typing import Generator

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
from pyhelper_utils.shell import run_command

from tests.model_serving.model_server.utils import create_isvc
from utilities.constants import (
    KServeDeploymentType,
    ModelAndFormat,
    ModelFormat,
    ModelVersion,
    Protocols,
    RuntimeQueryKeys,
    RuntimeTemplates,
)
from utilities.infra import create_isvc_view_role, create_storage_config_secret, s3_endpoint_secret
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def http_s3_openvino_model_mesh_serving_runtime(
    admin_client: DynamicClient,
    ns_with_modelmesh_enabled: Namespace,
) -> ServingRuntime:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=f"{Protocols.HTTP}-{RuntimeQueryKeys.OPENVINO_RUNTIME}",
        namespace=ns_with_modelmesh_enabled.name,
        template_name=RuntimeTemplates.OVMS_MODEL_MESH,
        multi_model=True,
        resources={
            "ovms": {
                "requests": {"cpu": "1", "memory": "4Gi"},
                "limits": {"cpu": "2", "memory": "8Gi"},
            }
        },
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def ci_model_mesh_endpoint_s3_secret(
    admin_client: DynamicClient,
    ns_with_modelmesh_enabled: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
) -> Secret:
    with s3_endpoint_secret(
        admin_client=admin_client,
        name="ci-bucket-secret",
        namespace=ns_with_modelmesh_enabled.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=ci_s3_bucket_region,
        aws_s3_bucket=ci_s3_bucket_name,
        aws_s3_endpoint=ci_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def storage_config_secret(
    admin_client: DynamicClient,
    ci_model_mesh_endpoint_s3_secret: Secret,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
) -> Generator[Secret, None, None]:
    with create_storage_config_secret(
        admin_client=admin_client,
        endpoint_secret_name=ci_model_mesh_endpoint_s3_secret.name,
        namespace=ci_model_mesh_endpoint_s3_secret.namespace,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_bucket=ci_s3_bucket_name,
        aws_s3_region=ci_s3_bucket_region,
        aws_s3_endpoint=ci_s3_bucket_endpoint,
    ) as storage_config:
        yield storage_config


@pytest.fixture(scope="class")
def model_mesh_model_service_account(
    admin_client: DynamicClient, ci_model_mesh_endpoint_s3_secret: Secret
) -> ServiceAccount:
    with ServiceAccount(
        client=admin_client,
        namespace=ci_model_mesh_endpoint_s3_secret.namespace,
        name=f"{Protocols.HTTP}-models-bucket-sa",
        secrets=[{"name": ci_model_mesh_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def http_s3_openvino_model_mesh_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    ns_with_modelmesh_enabled: Namespace,
    http_s3_openvino_model_mesh_serving_runtime: ServingRuntime,
    ci_model_mesh_endpoint_s3_secret: Secret,
    storage_config_secret: Secret,
    model_mesh_model_service_account: ServiceAccount,
) -> InferenceService:
    isvc_kwargs = {
        "client": admin_client,
        "name": f"{Protocols.HTTP}-{ModelFormat.OPENVINO}",
        "namespace": ns_with_modelmesh_enabled.name,
        "runtime": http_s3_openvino_model_mesh_serving_runtime.name,
        "model_service_account": model_mesh_model_service_account.name,
        "storage_key": ci_model_mesh_endpoint_s3_secret.name,
        "storage_path": request.param.get("model-path"),
        "model_format": ModelAndFormat.OPENVINO_IR,
        "deployment_mode": KServeDeploymentType.MODEL_MESH,
        "model_version": ModelVersion.OPSET1,
    }

    enable_auth = False

    if hasattr(request, "param"):
        enable_auth = request.param.get("enable-auth")

    isvc_kwargs["enable_auth"] = enable_auth

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


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
    http_s3_openvino_model_mesh_serving_runtime: ServingRuntime,
) -> None:
    with ResourceEditor(
        patches={
            http_s3_openvino_model_mesh_serving_runtime: {
                "metadata": {
                    "annotations": {"enable-auth": "true"},
                }
            }
        }
    ):
        yield
