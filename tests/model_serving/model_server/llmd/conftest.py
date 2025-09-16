from typing import Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount

from utilities.constants import Timeout
from utilities.infra import s3_endpoint_secret
from utilities.llmd_utils import create_gateway, create_llmisvc
from utilities.llmd_constants import (
    DEFAULT_GATEWAY_NAMESPACE,
    VLLM_STORAGE_OCI,
    VLLM_CPU_IMAGE,
    DEFAULT_S3_STORAGE_PATH,
)


@pytest.fixture(scope="class")
def gateway_namespace(admin_client: DynamicClient) -> str:
    return DEFAULT_GATEWAY_NAMESPACE


@pytest.fixture(scope="class")
def llmd_s3_secret(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret, None, None]:
    with s3_endpoint_secret(
        client=admin_client,
        name="llmd-s3-secret",
        namespace=unprivileged_model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_bucket=models_s3_bucket_name,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def llmd_s3_service_account(
    admin_client: DynamicClient, llmd_s3_secret: Secret
) -> Generator[ServiceAccount, None, None]:
    with ServiceAccount(
        client=admin_client,
        namespace=llmd_s3_secret.namespace,
        name="llmd-s3-service-account",
        secrets=[{"name": llmd_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def llmd_gateway(
    request: FixtureRequest,
    admin_client: DynamicClient,
    gateway_namespace: str,
) -> Generator[Gateway, None, None]:
    if isinstance(request.param, str):
        gateway_class_name = request.param
        kwargs = {}
    else:
        gateway_class_name = request.param.get("gateway_class_name", "openshift-default")
        kwargs = {k: v for k, v in request.param.items() if k != "gateway_class_name"}
    with create_gateway(
        client=admin_client,
        namespace=gateway_namespace,
        gateway_class_name=gateway_class_name,
        wait_for_condition=True,
        timeout=Timeout.TIMEOUT_5MIN,
        **kwargs,
    ) as gateway:
        yield gateway


@pytest.fixture(scope="class")
def llmd_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[LLMInferenceService, None, None]:
    if isinstance(request.param, str):
        name_suffix = request.param
        kwargs = {}
    else:
        name_suffix = request.param.get("name_suffix", "basic")
        kwargs = {k: v for k, v in request.param.items() if k != "name_suffix"}

    service_name = kwargs.get("name", f"llm-{name_suffix}")

    if "llmd_gateway" in request.fixturenames:
        request.getfixturevalue(argname="llmd_gateway")
    container_resources = kwargs.get(
        "container_resources",
        {
            "limits": {"cpu": "1", "memory": "10Gi"},
            "requests": {"cpu": "100m", "memory": "8Gi"},
        },
    )

    with create_llmisvc(
        client=admin_client,
        name=service_name,
        namespace=unprivileged_model_namespace.name,
        storage_uri=kwargs.get("storage_uri", VLLM_STORAGE_OCI),
        container_image=kwargs.get("container_image", VLLM_CPU_IMAGE),
        container_resources=container_resources,
        wait=True,
        timeout=Timeout.TIMEOUT_15MIN,
        **{k: v for k, v in kwargs.items() if k != "name"},
    ) as llm_service:
        yield llm_service


@pytest.fixture(scope="class")
def llmd_inference_service_s3(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    llmd_s3_secret: Secret,
    llmd_s3_service_account: ServiceAccount,
) -> Generator[LLMInferenceService, None, None]:
    if isinstance(request.param, str):
        name_suffix = request.param
        kwargs = {"storage_path": DEFAULT_S3_STORAGE_PATH}
    else:
        name_suffix = request.param.get("name_suffix", "s3")
        kwargs = {k: v for k, v in request.param.items() if k != "name_suffix"}

    service_name = kwargs.get("name", f"llm-{name_suffix}")

    if "storage_key" not in kwargs:
        kwargs["storage_key"] = llmd_s3_secret.name

    if "storage_path" not in kwargs:
        kwargs["storage_path"] = DEFAULT_S3_STORAGE_PATH

    container_resources = kwargs.get(
        "container_resources",
        {
            "limits": {"cpu": "1", "memory": "10Gi"},
            "requests": {"cpu": "100m", "memory": "8Gi"},
        },
    )

    with create_llmisvc(
        client=admin_client,
        name=service_name,
        namespace=unprivileged_model_namespace.name,
        storage_key=kwargs.get("storage_key"),
        storage_path=kwargs.get("storage_path"),
        container_image=kwargs.get("container_image", VLLM_CPU_IMAGE),
        container_resources=container_resources,
        service_account=llmd_s3_service_account.name,
        wait=True,
        timeout=Timeout.TIMEOUT_15MIN,
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ["name", "storage_key", "storage_path", "container_image", "container_resources"]
        },
    ) as llm_service:
        yield llm_service
