from collections.abc import Generator
from copy import deepcopy
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_runtime.vllm.constant import ACCELERATOR_IDENTIFIER, PREDICT_RESOURCES, TEMPLATE_MAP
from tests.model_serving.model_runtime.vllm.utils import (
    add_image_pull_secrets_if_configured,
    dedupe_vllm_cli_args,
    kserve_s3_endpoint_secret,
    validate_supported_quantization_schema,
)
from utilities.constants import KServeDeploymentType, Labels, RuntimeTemplates
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def negative_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    supported_accelerator_type: str,
    vllm_runtime_image: str,
) -> Generator[ServingRuntime]:
    accelerator_type = supported_accelerator_type.lower()
    template_name = TEMPLATE_MAP.get(accelerator_type, RuntimeTemplates.VLLM_CUDA)
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="vllm-neg-runtime",
        namespace=model_namespace.name,
        template_name=template_name,
        deployment_type=request.param.get("deployment_mode", KServeDeploymentType.RAW_DEPLOYMENT),
        runtime_image=vllm_runtime_image,
        support_tgis_open_ai_endpoints=True,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def negative_isvc_no_wait(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    negative_serving_runtime: ServingRuntime,
    supported_accelerator_type: str,
    s3_models_storage_uri: str,
    vllm_model_service_account: ServiceAccount,
    kserve_registry_pull_secret: Secret | None,
) -> Generator[InferenceService, Any, Any]:
    """ISVC that does not wait for readiness — used to inspect failure conditions."""
    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": negative_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": negative_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "model_service_account": vllm_model_service_account.name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.RAW_DEPLOYMENT),
        "external_route": True,
        "wait": False,
        "wait_for_predictor_pods": False,
        "timeout": request.param.get("timeout", 120),
    }

    accelerator_type = supported_accelerator_type.lower()
    gpu_count = request.param.get("gpu_count", 1)
    identifier = ACCELERATOR_IDENTIFIER.get(accelerator_type, Labels.Nvidia.NVIDIA_COM_GPU)
    resources = deepcopy(PREDICT_RESOURCES["resources"])
    resources["requests"][identifier] = gpu_count
    resources["limits"][identifier] = gpu_count
    isvc_kwargs["resources"] = resources

    if arguments := request.param.get("runtime_argument"):
        isvc_kwargs["argument"] = dedupe_vllm_cli_args(arguments=arguments)

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    add_image_pull_secrets_if_configured(
        isvc_kwargs=isvc_kwargs,
        kserve_registry_pull_secret=kserve_registry_pull_secret,
    )

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def bad_s3_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret]:
    """S3 secret with deliberately wrong credentials."""
    with kserve_s3_endpoint_secret(
        admin_client=admin_client,
        name="bad-s3-secret",
        namespace=model_namespace.name,
        aws_access_key="INVALID_ACCESS_KEY",
        aws_secret_access_key="INVALID_SECRET_KEY",
        aws_s3_region=models_s3_bucket_region,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def bad_s3_service_account(
    admin_client: DynamicClient,
    bad_s3_secret: Secret,
) -> Generator[ServiceAccount]:
    with ServiceAccount(
        client=admin_client,
        namespace=bad_s3_secret.namespace,
        name="bad-s3-sa",
        secrets=[{"name": bad_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def negative_isvc_bad_creds(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    negative_serving_runtime: ServingRuntime,
    supported_accelerator_type: str,
    s3_models_storage_uri: str,
    bad_s3_service_account: ServiceAccount,
    kserve_registry_pull_secret: Secret | None,
) -> Generator[InferenceService, Any, Any]:
    """ISVC using invalid S3 credentials — expects storage download failure."""
    accelerator_type = supported_accelerator_type.lower()
    gpu_count = request.param.get("gpu_count", 1)
    identifier = ACCELERATOR_IDENTIFIER.get(accelerator_type, Labels.Nvidia.NVIDIA_COM_GPU)
    resources = deepcopy(PREDICT_RESOURCES["resources"])
    resources["requests"][identifier] = gpu_count
    resources["limits"][identifier] = gpu_count

    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": negative_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": negative_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "model_service_account": bad_s3_service_account.name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.RAW_DEPLOYMENT),
        "external_route": True,
        "wait": False,
        "wait_for_predictor_pods": False,
        "timeout": 120,
        "resources": resources,
    }

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    add_image_pull_secrets_if_configured(
        isvc_kwargs=isvc_kwargs,
        kserve_registry_pull_secret=kserve_registry_pull_secret,
    )

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc
