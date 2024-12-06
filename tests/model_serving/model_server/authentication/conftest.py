import shlex

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
from ocp_resources.authorino import Authorino
from pyhelper_utils.shell import run_command


from tests.model_serving.model_server.authentication.utils import (
    create_isvc_view_role,
)
from tests.model_serving.model_server.utils import create_isvc, get_pods_by_isvc_label
from utilities.constants import KServeDeploymentType, ModelFormat, Protocols, RuntimeQueryKeys, RuntimeTemplates
from utilities.infra import s3_endpoint_secret
from utilities.serving_runtime import ServingRuntimeFromTemplate


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
def s3_models_storage_uri(request, models_s3_bucket_name) -> str:
    return f"s3://{models_s3_bucket_name}/{request.param['model-dir']}/"


@pytest.fixture(scope="class")
def endpoint_s3_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Secret:
    with s3_endpoint_secret(
        admin_client=admin_client,
        name="models-bucket-secret",
        namespace=model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_bucket=models_s3_bucket_name,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret


# HTTP model serving
@pytest.fixture(scope="class")
def http_model_service_account(admin_client: DynamicClient, endpoint_s3_secret: Secret) -> ServiceAccount:
    with ServiceAccount(
        client=admin_client,
        namespace=endpoint_s3_secret.namespace,
        name=f"{Protocols.HTTP}-models-bucket-sa",
        secrets=[{"name": endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def http_s3_serving_runtime(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> ServingRuntime:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=f"{Protocols.HTTP}-{RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME}",
        namespace=model_namespace.name,
        template_name=RuntimeTemplates.CAIKIT_TGIS_SERVING,
        multi_model=False,
        enable_http=True,
        enable_grpc=False,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def http_s3_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    http_s3_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    http_model_service_account: ServiceAccount,
) -> InferenceService:
    with create_isvc(
        client=admin_client,
        name=f"{Protocols.HTTP}-{ModelFormat.CAIKIT}",
        namespace=model_namespace.name,
        runtime=http_s3_serving_runtime.name,
        storage_uri=s3_models_storage_uri,
        model_format=http_s3_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        model_service_account=http_model_service_account.name,
        enable_auth=True,
    ) as isvc:
        yield isvc


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


# GRPC model serving
@pytest.fixture(scope="class")
def grpc_model_service_account(admin_client: DynamicClient, endpoint_s3_secret: Secret) -> ServiceAccount:
    with ServiceAccount(
        client=admin_client,
        namespace=endpoint_s3_secret.namespace,
        name=f"{Protocols.GRPC}-models-bucket-sa",
        secrets=[{"name": endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def grpc_s3_serving_runtime(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> ServingRuntime:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=f"{Protocols.GRPC}-{RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME}",
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
    grpc_s3_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    grpc_model_service_account: ServiceAccount,
) -> InferenceService:
    with create_isvc(
        client=admin_client,
        name=f"{Protocols.GRPC}-{ModelFormat.CAIKIT}",
        namespace=model_namespace.name,
        runtime=grpc_s3_serving_runtime.name,
        storage_uri=s3_models_storage_uri,
        model_format=grpc_s3_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        model_service_account=grpc_model_service_account.name,
        enable_auth=True,
    ) as isvc:
        yield isvc


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
