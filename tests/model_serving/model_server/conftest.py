from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.utils import create_isvc
from utilities.infra import s3_endpoint_secret
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="session")
def skip_if_no_deployed_openshift_serverless(admin_client: DynamicClient):
    csvs = list(
        ClusterServiceVersion.get(
            client=admin_client,
            namespace="openshift-serverless",
            label_selector="operators.coreos.com/serverless-operator.openshift-serverless",
        )
    )
    if not csvs:
        pytest.skip("OpenShift Serverless is not deployed")

    csv = csvs[0]

    if not (csv.exists and csv.status == csv.Status.SUCCEEDED):
        pytest.skip("OpenShift Serverless is not deployed")


@pytest.fixture(scope="class")
def models_endpoint_s3_secret(
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
def model_service_account(admin_client: DynamicClient, models_endpoint_s3_secret: Secret) -> ServiceAccount:
    with ServiceAccount(
        client=admin_client,
        namespace=models_endpoint_s3_secret.namespace,
        name="models-bucket-sa",
        secrets=[{"name": models_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def serving_runtime_from_template(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    runtime_kwargs = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "template_name": request.param["template-name"],
        "multi_model": request.param["multi-model"],
    }

    if (enable_http := request.param.get("enable-http")) is not None:
        runtime_kwargs["enable_http"] = enable_http

    if (enable_grpc := request.param.get("enable-grpc")) is not None:
        runtime_kwargs["enable_grpc"] = enable_grpc

    with ServingRuntimeFromTemplate(**runtime_kwargs) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def ci_s3_storage_uri(request: FixtureRequest, ci_s3_bucket_name: str) -> str:
    return f"s3://{ci_s3_bucket_name}/{request.param['model-dir']}/"


@pytest.fixture(scope="class")
def s3_models_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    serving_runtime_from_template: ServingRuntime,
    s3_models_storage_uri: str,
    model_service_account: ServiceAccount,
) -> InferenceService:
    isvc_kwargs = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": serving_runtime_from_template.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": serving_runtime_from_template.instance.spec.supportedModelFormats[0].name,
        "model_service_account": model_service_account.name,
        "deployment_mode": request.param["deployment-mode"],
    }

    if (external_route := request.param.get("external-route")) is not None:
        isvc_kwargs["enable_auth"] = external_route

    if (enable_auth := request.param.get("enable-auth")) is not None:
        isvc_kwargs["enable_auth"] = enable_auth

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc
