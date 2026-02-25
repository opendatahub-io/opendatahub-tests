"""
Pytest fixtures for OpenVINO model serving runtime tests.

This module provides fixtures for:
- Setting up OpenVINO serving runtimes and templates
- Creating inference services and related Kubernetes resources
- Managing S3 secrets and service accounts
- Providing test utilities like snapshots and pod resources
- OVMS smoke test Pod and ConfigMap for in-cluster script execution
"""

from pathlib import Path
from typing import cast, Any, Generator
import copy

import pytest
from syrupy.extensions.json import JSONSnapshotExtension

from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount

from tests.model_serving.model_runtime.openvino.constant import PREDICT_RESOURCES

from utilities.constants import (
    KServeDeploymentType,
    Labels,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label
from utilities.serving_runtime import ServingRuntimeFromTemplate

from simple_logger.logger import get_logger


LOGGER = get_logger(name=__name__)

OVMS_SMOKE_SCRIPTS_DIR = Path(__file__).parent / "smoke"
OVMS_SMOKE_SCRIPT_NAMES = ("ovms_smoketest.py", "smoke.py")
OVMS_SMOKE_CONFIGMAP_NAME = "ovms-smoke-scripts"
OVMS_SMOKE_POD_NAME = "ovms-smoke-pod"
OVMS_SMOKE_SCRIPTS_MOUNT_PATH = "/scripts"


@pytest.fixture(scope="class")
def openvino_serving_runtime(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime, None, None]:
    """
    Provides a ServingRuntime resource for OpenVINO with the specified protocol and deployment type.

    Args:
        request (pytest.FixtureRequest): Pytest fixture request containing parameters.
        admin_client (DynamicClient): Kubernetes dynamic client.
        model_namespace (Namespace): Kubernetes namespace for model deployment.
        openvino_runtime_image (str): The container image for the OpenVINO runtime.
        protocol (str): The protocol to use (e.g., REST or GRPC).

    Yields:
        ServingRuntime: An instance of the OpenVINO ServingRuntime configured as per parameters.
    """
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="openvino-runtime",
        namespace=model_namespace.name,
        template_name=RuntimeTemplates.OVMS_KSERVE,
        deployment_type=request.param["deployment_type"],
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def openvino_inference_service(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    openvino_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    openvino_model_service_account: ServiceAccount,
    gpu_count_on_cluster: int,
) -> Generator[InferenceService, Any, Any]:
    """
    Creates and yields a configured InferenceService instance for OpenVINO testing.

    Args:
        request (pytest.FixtureRequest): Pytest fixture request containing test parameters.
        admin_client (DynamicClient): Kubernetes dynamic client.
        model_namespace (Namespace): Kubernetes namespace for model deployment.
        openvino_serving_runtime (ServingRuntime): The OpenVINO ServingRuntime instance.
        s3_models_storage_uri (str): URI for the S3 storage location of models.
        openvino_model_service_account (ServiceAccount): Service account for the model.

    Yields:
        InferenceService: A configured InferenceService resource.
    """
    params = getattr(request, "param", {})
    service_config = {
        "client": admin_client,
        "name": params.get("name"),
        "namespace": model_namespace.name,
        "runtime": openvino_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": openvino_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "model_service_account": openvino_model_service_account.name,
        "deployment_mode": params.get("deployment_type", KServeDeploymentType.RAW_DEPLOYMENT),
        "external_route": params.get("enable_external_route", False),
    }

    gpu_count = params.get("gpu_count", 0)
    timeout = params.get("timeout")
    min_replicas = params.get("min-replicas")

    resources = copy.deepcopy(cast(dict[str, dict[str, str]], PREDICT_RESOURCES["resources"]))

    if gpu_count > 0:
        if gpu_count_on_cluster < gpu_count:
            raise ResourceNotFoundError(
                f"Not enough GPU available for test execution, required:{gpu_count}, available:{gpu_count_on_cluster}"
            )
        identifier = Labels.Nvidia.NVIDIA_COM_GPU
        resources["requests"][identifier] = gpu_count
        resources["limits"][identifier] = gpu_count
        service_config["volumes"] = copy.deepcopy(PREDICT_RESOURCES["volumes"])
        service_config["volumes_mounts"] = copy.deepcopy(PREDICT_RESOURCES["volume_mounts"])

    service_config["resources"] = resources

    if timeout:
        service_config["timeout"] = timeout

    if min_replicas:
        service_config["min_replicas"] = min_replicas

    with create_isvc(**service_config) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def openvino_model_service_account(admin_client: DynamicClient, kserve_s3_secret: Secret) -> ServiceAccount:
    """
    Creates and yields a ServiceAccount linked to the provided S3 secret for OpenVINO models.

    Args:
        admin_client (DynamicClient): Kubernetes dynamic client.
        kserve_s3_secret (Secret): The Kubernetes secret containing S3 credentials.

    Yields:
        ServiceAccount: A ServiceAccount configured with access to the S3 secret.
    """
    with ServiceAccount(
        client=admin_client,
        namespace=kserve_s3_secret.namespace,
        name="openvino-models-bucket-sa",
        secrets=[{"name": kserve_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture
def openvino_response_snapshot(snapshot: Any) -> Any:
    """
    Provides a snapshot fixture configured to use JSONSnapshotExtension for OpenVINO responses.

    Args:
        snapshot (Any): The base snapshot fixture.

    Returns:
        Any: Snapshot fixture extended with JSONSnapshotExtension.
    """
    return snapshot.use_extension(extension_class=JSONSnapshotExtension)


@pytest.fixture
def openvino_pod_resource(
    admin_client: DynamicClient,
    openvino_inference_service: InferenceService,
) -> Pod:
    """
    Retrieves the first Kubernetes Pod associated with the given OpenVINO InferenceService.

    Args:
        admin_client (DynamicClient): Kubernetes dynamic client.
        openvino_inference_service (InferenceService): The OpenVINO InferenceService resource.

    Returns:
        Pod: The first Pod found for the InferenceService.

    Raises:
        RuntimeError: If no pods are found for the specified InferenceService.
    """
    pods = get_pods_by_isvc_label(client=admin_client, isvc=openvino_inference_service)
    if not pods:
        raise ResourceNotFoundError(f"No pods found for InferenceService {openvino_inference_service.name}")
    return pods[0]

def _load_ovms_smoke_scripts_data() -> dict[str, str]:
    """Load smoke script file contents for ConfigMap data."""
    data: dict[str, str] = {}
    for name in OVMS_SMOKE_SCRIPT_NAMES:
        path = OVMS_SMOKE_SCRIPTS_DIR / name
        data[name] = path.read_text()
    return data


@pytest.fixture(scope="class")
def ovms_smoke_scripts_configmap(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ConfigMap, None, None]:
    """
    ConfigMap containing OVMS smoke test scripts to run inside the container.

    Args:
        admin_client: Kubernetes dynamic client.
        model_namespace: Namespace for the ConfigMap.

    Yields:
        ConfigMap: ConfigMap with ovms_smoketest.py and smoke.py data.
    """
    data = _load_ovms_smoke_scripts_data()
    with ConfigMap(
        client=admin_client,
        name=OVMS_SMOKE_CONFIGMAP_NAME,
        namespace=model_namespace.name,
        data=data,
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def ovms_smoke_pod(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    ovms_runtime_image: str,
    ovms_smoke_scripts_configmap: ConfigMap,
) -> Generator[Pod, None, None]:
    """
    Pod that runs OVMS smoke scripts inside OpenShift using the OVMS runtime image.

    The smoke scripts are mounted read-only via ConfigMap (not copied).
    The container runs both scripts in sequence; the Pod succeeds only if both exit 0.

    Args:
        admin_client: Kubernetes dynamic client.
        model_namespace: Namespace for the Pod.
        ovms_runtime_image: Container image for OVMS runtime (from CLI or template).
        ovms_smoke_scripts_configmap: ConfigMap with smoke script contents.

    Yields:
        Pod: The completed Pod resource (phase Succeeded when both scripts exit 0).
    """
    run_cmd = (
        f"python {OVMS_SMOKE_SCRIPTS_MOUNT_PATH}/ovms_smoketest.py && "
        f"python {OVMS_SMOKE_SCRIPTS_MOUNT_PATH}/smoke.py"
    )
    # Use writable dirs under /tmp so non-root container can cache models and configs.
    # HF_HOME is the preferred cache for Hugging Face (TRANSFORMERS_CACHE is deprecated in v5).
    env_vars = [
        {"name": "HOME", "value": "/tmp"},
        {"name": "HF_HOME", "value": "/tmp/hf_cache"},
        {"name": "MPLCONFIGDIR", "value": "/tmp/matplotlib"},
    ]
    with Pod(
        client=admin_client,
        name=OVMS_SMOKE_POD_NAME,
        namespace=model_namespace.name,
        restart_policy="Never",
        containers=[
            {
                "name": "ovms-smoke",
                "image": ovms_runtime_image,
                "command": ["/bin/sh", "-c"],
                "args": [run_cmd],
                "env": env_vars,
                "volumeMounts": [
                    {
                        "name": "smoke-scripts",
                        "mountPath": OVMS_SMOKE_SCRIPTS_MOUNT_PATH,
                        "readOnly": True,
                    }
                ],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "runAsNonRoot": True,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
            }
        ],
        volumes=[
            {
                "name": "smoke-scripts",
                "configMap": {"name": ovms_smoke_scripts_configmap.name},
            }
        ],
    ) as pod:
        LOGGER.info("Waiting for OVMS smoke Pod to complete")
        pod.wait_for_status(status=Pod.Status.SUCCEEDED, timeout=300)
        yield pod
