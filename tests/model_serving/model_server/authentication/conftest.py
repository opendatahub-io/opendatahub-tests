import shlex

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.service_mesh_member import ServiceMeshMember
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.authorino import Authorino
from pyhelper_utils.shell import run_command
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.authentication.utils import base64_encode_str
from tests.model_serving.model_server.utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="session")
def skip_if_no_authorino_operator(admin_client: DynamicClient):
    if not Authorino(
        client=admin_client,
        name="authorino",
        namespace="redhat-ods-applications-auth-provider",
    ).exists:
        pytest.skip("Authorino operator is missing from the cluster")


@pytest.fixture(scope="class")
def s3_models_storage_uri(request) -> str:
    return f"s3://{py_config['model_s3_bucket_name']}/{request.param['model-dir']}/"


@pytest.fixture(scope="class")
def endpoint_s3_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    aws_access_key: str,
    aws_secret_access_key: str,
) -> Secret:
    data = {
        "AWS_ACCESS_KEY_ID": base64_encode_str(text=aws_access_key),
        "AWS_SECRET_ACCESS_KEY": base64_encode_str(text=aws_secret_access_key),
        "AWS_S3_BUCKET": base64_encode_str(text=py_config["model_s3_bucket_name"]),
        "AWS_S3_ENDPOINT": base64_encode_str(text=py_config["model_s3_endpoint"]),
    }
    name = model_namespace.name

    with Secret(
        client=admin_client,
        namespace=name,
        name=name,
        data_dict=data,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def model_service_account(admin_client: DynamicClient, endpoint_s3_secret: Secret) -> ServiceAccount:
    with ServiceAccount(
        client=admin_client,
        namespace=endpoint_s3_secret.namespace,
        name="models-bucket-sa",
        secrets=[{"name": endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def s3_serving_runtime(
    request,
    admin_client: DynamicClient,
    service_mesh_member: ServiceMeshMember,
    model_namespace: Namespace,
) -> ServingRuntime:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace.name,
        template_name=request.param["template-name"],
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def s3_inference_service(
    request,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    s3_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    model_service_account: ServiceAccount,
) -> InferenceService:
    isvc_kwargs = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": s3_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": s3_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "deployment_mode": request.param.get("deployment-mode", "Serverless"),
        "model_service_account": model_service_account.name,
    }

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    if enable_auth := request.param.get("enable-model-auth"):
        isvc_kwargs["enable_auth"] = enable_auth

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def inference_token(model_namespace: Namespace, model_service_account: ServiceAccount) -> str:
    return run_command(command=shlex.split(f"oc create token -n {model_namespace.name} {model_service_account.name}"))[
        1
    ].strip()
