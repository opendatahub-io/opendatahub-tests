"""Shared fixtures for vLLM-Omni serving runtime tests."""

import json
from collections.abc import Generator
from copy import deepcopy
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_runtime.utils import deselect_tests_for_missing_templates
from tests.model_serving.model_runtime.vllm.constant import ACCELERATOR_IDENTIFIER
from tests.model_serving.model_runtime.vllm.modelcar.constant import (
    PULL_SECRET_ACCESS_TYPE,
    PULL_SECRET_NAME,
    SUPPORTED_MODELCAR_REGISTRY_HOSTS,
)
from tests.model_serving.model_runtime.vllm.modelcar.utils import (
    normalize_registry_pull_auth,
    validate_registry_pull_auth,
)
from tests.model_serving.model_runtime.vllm.utils import (
    add_image_pull_secrets_if_configured,
    dedupe_vllm_cli_args,
    kserve_s3_endpoint_secret,
)
from tests.model_serving.model_runtime.vllm_omni.constant import (
    OMNI_MULTI_GPU_RESOURCES,
    OMNI_SINGLE_GPU_RESOURCES,
    OMNI_TEMPLATE_MAP,
    OMNI_VOLUME_MOUNTS,
    OMNI_VOLUMES,
)
from utilities.constants import AcceleratorType, Containers, KServeDeploymentType, Labels
from utilities.inference_utils import create_isvc, get_exposed_isvc_url
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = structlog.get_logger(name=__name__)


def pytest_collection_modifyitems(items: list[pytest.Item], config: pytest.Config) -> None:
    """Deselect fast template tests whose template does not exist on the cluster."""
    from pytest_testconfig import config as py_config

    deselect_tests_for_missing_templates(
        items=items,
        config=config,
        applications_namespace=py_config.get("applications_namespace", "redhat-ods-applications"),
    )


@pytest.fixture(scope="session")
def vllm_omni_runtime_image(pytestconfig: pytest.Config) -> str | None:
    """Resolve the vLLM-Omni runtime image from CLI option (--vllm-omni-runtime-image).

    Returns None when not set, causing the ServingRuntime template's default image to be used.
    """
    runtime_image = pytestconfig.option.vllm_omni_runtime_image
    if not runtime_image:
        return None
    return runtime_image


@pytest.fixture(scope="session")
def skip_if_no_vllm_omni_multi_gpu(
    supported_accelerator_type: str | None,
    gpu_count_on_cluster: int,
) -> None:
    """Skip test unless the cluster provides at least 2 NVIDIA GPUs."""
    if not supported_accelerator_type or supported_accelerator_type.lower() != AcceleratorType.NVIDIA:
        pytest.skip(f"vLLM-Omni requires NVIDIA GPUs. Found accelerator: '{supported_accelerator_type or 'None'}'.")
    if gpu_count_on_cluster < 2:
        pytest.skip(f"vLLM-Omni multi-GPU tests require at least 2 GPUs. Found: {gpu_count_on_cluster}.")


@pytest.fixture(scope="class")
def vllm_omni_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    supported_accelerator_type: str,
    vllm_omni_runtime_image: str | None,
) -> Generator[ServingRuntime, Any, Any]:
    """Class-scoped vLLM-Omni ServingRuntime from the platform template."""
    accelerator_type = supported_accelerator_type.lower()
    template_name = OMNI_TEMPLATE_MAP.get(accelerator_type)
    if not template_name:
        pytest.fail(
            f"No vLLM-Omni template available for accelerator '{accelerator_type}'. "
            f"Supported accelerators: {list(OMNI_TEMPLATE_MAP.keys())}. "
            "vLLM-Omni Tech Preview only supports NVIDIA CUDA."
        )
    runtime_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": "vllm-omni-runtime",
        "namespace": model_namespace.name,
        "template_name": template_name,
        "deployment_type": request.param.get("deployment_mode", KServeDeploymentType.STANDARD),
    }
    if vllm_omni_runtime_image:
        runtime_kwargs["runtime_image"] = vllm_omni_runtime_image

    with ServingRuntimeFromTemplate(**runtime_kwargs) as runtime:
        yield runtime


@pytest.fixture(scope="class")
def kserve_registry_pull_secret(  # noqa: UFN001 — sibling vllm/ conftest has same fixture; not inheritable
    admin_client: DynamicClient,
    model_namespace: Namespace,
    registry_pull_secret: list[str],
    registry_host: list[str],
) -> Generator[Secret | None, Any, Any]:
    """Create a dockerconfigjson pull secret when OCI registry credentials are configured."""
    if not registry_host:
        yield None
        return

    if len(registry_host) != len(registry_pull_secret):
        raise ValueError(
            f"registry_host count ({len(registry_host)}) must match "
            f"registry_pull_secret count ({len(registry_pull_secret)})"
        )

    unsupported_hosts = set(registry_host) - SUPPORTED_MODELCAR_REGISTRY_HOSTS
    if unsupported_hosts:
        raise ValueError(f"Unsupported OCI registry hosts: {sorted(unsupported_hosts)}")

    auths: dict[str, dict[str, str]] = {}
    for host, raw_auth in zip(registry_host, registry_pull_secret):
        auth = normalize_registry_pull_auth(raw_value=raw_auth, expected_host=host)
        validate_registry_pull_auth(auth=auth)
        auths[host] = {"auth": auth}

    docker_config_json = json.dumps({"auths": auths})
    with Secret(
        client=admin_client,
        name=PULL_SECRET_NAME,
        namespace=model_namespace.name,
        string_data={
            ".dockerconfigjson": docker_config_json,
            "ACCESS_TYPE": PULL_SECRET_ACCESS_TYPE,
            "OCI_HOST": ",".join(registry_host),
        },
        type="kubernetes.io/dockerconfigjson",
        wait_for_resource=True,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def vllm_omni_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    vllm_omni_serving_runtime: ServingRuntime,
    supported_accelerator_type: str,
    s3_models_storage_uri: str,
    vllm_omni_model_service_account: ServiceAccount,
    kserve_registry_pull_secret: Secret | None,
) -> Generator[InferenceService, Any, Any]:
    """Class-scoped vLLM-Omni InferenceService with auto-scaled resources.

    Resources are selected based on gpu_count:
      - gpu_count > 1: heavy spec (32Gi/64Gi) for multi-stage pipeline models
      - gpu_count == 1: light spec (16Gi/32Gi) for single-GPU TTS models
    Can be overridden via "resources" in request.param.
    """
    accelerator_type = supported_accelerator_type.lower()
    identifier = ACCELERATOR_IDENTIFIER.get(accelerator_type, Labels.Nvidia.NVIDIA_COM_GPU)
    gpu_count = request.param.get("gpu_count")

    base_resources = OMNI_MULTI_GPU_RESOURCES if gpu_count > 1 else OMNI_SINGLE_GPU_RESOURCES
    resources = deepcopy(x=request.param.get("resources", base_resources["resources"]))
    resources["requests"][identifier] = gpu_count
    resources["limits"][identifier] = gpu_count

    timeout = request.param.get("timeout")

    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": request.param.get("name", "vllm-omni-isvc"),
        "namespace": model_namespace.name,
        "runtime": vllm_omni_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": vllm_omni_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "model_service_account": vllm_omni_model_service_account.name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.STANDARD),
        "external_route": True,
        "resources": resources,
    }
    if timeout:
        isvc_kwargs["timeout"] = timeout
    if gpu_count > 1:
        isvc_kwargs["volumes"] = OMNI_VOLUMES
        isvc_kwargs["volumes_mounts"] = OMNI_VOLUME_MOUNTS
    if arguments := request.param.get("runtime_argument"):
        arguments = [arg for arg in arguments if not arg.startswith("--tensor-parallel-size")]
        arguments.append(f"--tensor-parallel-size={gpu_count}")
        isvc_kwargs["argument"] = dedupe_vllm_cli_args(arguments=arguments)

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    if model_env_variables := request.param.get("model_env_variables"):
        isvc_kwargs["model_env_variables"] = model_env_variables

    add_image_pull_secrets_if_configured(
        isvc_kwargs=isvc_kwargs,
        kserve_registry_pull_secret=kserve_registry_pull_secret,
    )

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def vllm_omni_isvc_url(vllm_omni_inference_service: InferenceService) -> str:
    """Cached external route URL — one Route lookup per class."""
    return get_exposed_isvc_url(isvc=vllm_omni_inference_service)


@pytest.fixture(scope="class")
def vllm_omni_pod_resource(
    admin_client: DynamicClient,
    vllm_omni_inference_service: InferenceService,
) -> Pod:
    """Class-scoped predictor pod for the vLLM-Omni InferenceService."""
    from utilities.infra import get_pods_by_isvc_label

    pods = list(get_pods_by_isvc_label(client=admin_client, isvc=vllm_omni_inference_service))
    assert pods, f"No predictor pods found for ISVC {vllm_omni_inference_service.name}"
    return pods[0]


@pytest.fixture(scope="class")
def vllm_omni_pod_logs(vllm_omni_pod_resource: Pod) -> str:
    """Class-scoped cached pod logs to avoid multiple large log stream fetches."""
    return vllm_omni_pod_resource.log(container=Containers.KSERVE_CONTAINER_NAME)


@pytest.fixture(scope="class")
def vllm_omni_model_service_account(
    admin_client: DynamicClient,
    vllm_omni_kserve_endpoint_s3_secret: Secret,
) -> Generator[ServiceAccount, Any, Any]:
    """ServiceAccount with S3 credentials for vLLM-Omni ISVCs."""
    with ServiceAccount(
        client=admin_client,
        namespace=vllm_omni_kserve_endpoint_s3_secret.namespace,
        name="vllm-omni-models-bucket-sa",
        secrets=[{"name": vllm_omni_kserve_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def vllm_omni_kserve_endpoint_s3_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret, Any, Any]:
    """KServe S3 endpoint secret for vLLM-Omni model storage access."""
    with kserve_s3_endpoint_secret(
        admin_client=admin_client,
        name="vllm-omni-models-bucket-secret",
        namespace=model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret
