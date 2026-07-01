import json
from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_runtime.vllm.constant import ACCELERATOR_IDENTIFIER, PREDICT_RESOURCES, TEMPLATE_MAP
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
    skip_if_not_deployment_mode,
    validate_supported_quantization_schema,
)
from utilities.constants import AcceleratorType, KServeDeploymentType, Labels, RuntimeTemplates
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = structlog.get_logger(name=__name__)

SUPPORTED_CPU_X86_ACCELERATORS: set[str] = {AcceleratorType.CPU_x86}


@pytest.fixture(scope="session")
def skip_if_no_supported_cpu_x86_accelerator_type(supported_accelerator_type: str | None) -> None:
    """Skip test unless the cluster provides the x86 CPU accelerator."""
    if not supported_accelerator_type or supported_accelerator_type.lower() not in SUPPORTED_CPU_X86_ACCELERATORS:
        pytest.skip(
            f"Test requires a supported vLLM x86 CPU accelerator. "
            f"Found: '{supported_accelerator_type or 'None'}'. "
            f"Expected one of: {SUPPORTED_CPU_X86_ACCELERATORS}."
        )


@pytest.fixture(scope="class")
def serving_runtime(
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
        name="vllm-runtime",
        namespace=model_namespace.name,
        template_name=template_name,
        deployment_type=request.param["deployment_mode"],
        runtime_image=vllm_runtime_image,
        support_tgis_open_ai_endpoints=True,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def kserve_registry_pull_secret(
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
def vllm_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    serving_runtime: ServingRuntime,
    supported_accelerator_type: str,
    s3_models_storage_uri: str,
    vllm_model_service_account: ServiceAccount,
    kserve_registry_pull_secret: Secret | None,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": serving_runtime.instance.spec.supportedModelFormats[0].name,
        "model_service_account": vllm_model_service_account.name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.STANDARD),
        "external_route": True,
    }
    accelerator_type = supported_accelerator_type.lower()
    gpu_count = request.param.get("gpu_count")
    timeout = request.param.get("timeout")
    identifier = ACCELERATOR_IDENTIFIER.get(accelerator_type, Labels.Nvidia.NVIDIA_COM_GPU)
    resources: Any = PREDICT_RESOURCES["resources"]
    resources["requests"][identifier] = gpu_count
    resources["limits"][identifier] = gpu_count
    isvc_kwargs["resources"] = resources
    if timeout:
        isvc_kwargs["timeout"] = timeout
    if gpu_count > 1:
        isvc_kwargs["volumes"] = PREDICT_RESOURCES["volumes"]
        isvc_kwargs["volumes_mounts"] = PREDICT_RESOURCES["volume_mounts"]
    if arguments := request.param.get("runtime_argument"):
        arguments = [arg for arg in arguments if not arg.startswith(("--tensor-parallel-size", "--quantization"))]
        arguments.append(f"--tensor-parallel-size={gpu_count}")
        if quantization := request.param.get("quantization"):
            validate_supported_quantization_schema(q_type=quantization)
            arguments.append(f"--quantization={quantization}")
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
def vllm_model_service_account(admin_client: DynamicClient, kserve_endpoint_s3_secret: Secret) -> ServiceAccount:
    with ServiceAccount(
        client=admin_client,
        namespace=kserve_endpoint_s3_secret.namespace,
        name="models-bucket-sa",
        secrets=[{"name": kserve_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def kserve_endpoint_s3_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret]:
    with kserve_s3_endpoint_secret(
        admin_client=admin_client,
        name="models-bucket-secret",
        namespace=model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture
def skip_if_not_raw_deployment(vllm_inference_service: InferenceService) -> None:
    skip_if_not_deployment_mode(
        isvc=vllm_inference_service,
        deployment_types=KServeDeploymentType.RAW_DEPLOYMENT_MODES,
    )
