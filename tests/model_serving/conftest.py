from __future__ import annotations

import copy
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pytest
import structlog
from _pytest.fixtures import FixtureRequest
from kubernetes.client.rest import ApiException
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from timeout_sampler import TimeoutExpiredError

from tests.model_serving.model_runtime.mlserver.constant import PREDICT_RESOURCES
from utilities.constants import (
    KServeDeploymentType,
    Labels,
    ModelFormat,
    ModelInferenceRuntime,
    RuntimeTemplates,
)
from utilities.general import download_model_data
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = structlog.get_logger(name=__name__)

# ---------------------------------------------------------------------------
# Multi-model serving constants
# ---------------------------------------------------------------------------
MLSERVER_CUDA_TEMPLATE: str = "mlserver-cuda-runtime-template"
MLSERVER_CUDA_RUNTIME: str = "mlserver-cuda-runtime"

KSERVE_CONTAINER: str = "kserve-container"
MULTI_MODEL_PVC_SIZE: str = "10Gi"
STORAGE_EXHAUSTION_PVC_SIZE: str = "1Mi"

# S3 model-path prefixes — each maps to a distinct S3 directory and PVC subdirectory.
# The 10-model happy-path repository and every negative-scenario variant are staged
# as separate subdirectories on the shared PVC so fixtures are isolated.
S3_PATH_10_MODEL: str = "mlserver/model_repository"
S3_PATH_MULTI_VERSION: str = "mlserver/multi_version"
S3_PATH_ONE_FAILED: str = "mlserver/one_failed_model"
S3_PATH_MALFORMED: str = "mlserver/malformed_settings"
S3_PATH_UNSUPPORTED_IMPL: str = "mlserver/unsupported_impl"
S3_PATH_MISSING_ARTIFACT: str = "mlserver/missing_artifact"

# Static-profile environment defaults recorded per RHAI-49122
STATIC_PROFILE_ENV_VARS: list[str] = [
    "CHECK_REPOSITORY_POPULATED",
    "MLSERVER_STRICT_READINESS",
    "MLSERVER_EMPTY_REGISTRY_READINESS",
    "MLSERVER_LOAD_MODELS_AT_STARTUP",
    "MLSERVER_MODELS_DIR",
    "MLSERVER_PARALLEL_WORKERS",
]


@pytest.fixture(scope="session")
def root_dir(pytestconfig: pytest.Config) -> Any:
    """
    Provides the root directory path of the pytest project for the entire test session.

    Args:
        pytestconfig (pytest.Config): The pytest configuration object.

    Returns:
        Any: The root path of the pytest project.
    """
    return pytestconfig.rootpath


@pytest.fixture(scope="class")
def protocol(request: FixtureRequest) -> str:
    """
    Provides the protocol type parameter for the test class.

    Args:
        request (pytest.FixtureRequest): The pytest fixture request object.

    Returns:
        str: The protocol type specified in the test parameter.
    """
    return request.param["protocol_type"]


@pytest.fixture(scope="session")
def s3_models_storage_uri(request: FixtureRequest, models_s3_bucket_name: str) -> str:
    """
    Build S3 model URI from fixture params.

    Supports:
    - relative model paths via ``model-dir`` + default/global bucket
    - optional per-test bucket override via ``bucket-name``
    - fully-qualified S3 URIs in ``model-dir`` (e.g. ``s3://bucket/prefix``)
    """
    model_dir = request.param["model-dir"].strip()

    if model_dir.startswith("s3://"):
        return f"{model_dir.rstrip('/')}/"

    bucket_name = request.param.get("bucket-name", models_s3_bucket_name).strip()
    normalized_model_dir = model_dir.strip("/")
    return f"s3://{bucket_name}/{normalized_model_dir}/"


@pytest.fixture(scope="class")
def kserve_s3_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret]:
    """
    Creates and yields a Kubernetes Secret configured for S3 access in KServe.

    Args:
        admin_client (DynamicClient): Kubernetes dynamic client.
        model_namespace (Namespace): Namespace where the secret will be created.
        aws_access_key_id (str): AWS access key ID.
        aws_secret_access_key (str): AWS secret access key.
        models_s3_bucket_region (str): AWS S3 bucket region.
        models_s3_bucket_endpoint (str): AWS S3 bucket endpoint URL.

    Yields:
        Secret: A Kubernetes Secret configured with the provided AWS credentials and S3 endpoint.
    """
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


@contextmanager
def kserve_s3_endpoint_secret(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_endpoint: str,
    aws_s3_region: str,
) -> Generator[Secret, Any, Any]:
    """
    Context manager that creates a temporary Kubernetes Secret for KServe
    to access an S3-compatible storage endpoint.

    Args:
        admin_client (DynamicClient): Kubernetes dynamic client for resource operations.
        name (str): Name of the Secret resource.
        namespace (str): Kubernetes namespace in which to create the Secret.
        aws_access_key (str): AWS access key ID for authentication.
        aws_secret_access_key (str): AWS secret access key for authentication.
        aws_s3_endpoint (str): S3 endpoint URL (e.g., https://s3.example.com).
        aws_s3_region (str): AWS region for the S3 service.

    Yields:
        Secret: The created Kubernetes Secret object within the context.
    """
    with Secret(
        client=admin_client,
        name=name,
        namespace=namespace,
        annotations={
            "serving.kserve.io/s3-endpoint": (aws_s3_endpoint.replace("https://", "").replace("http://", "")),
            "serving.kserve.io/s3-region": aws_s3_region,
            "serving.kserve.io/s3-useanoncredential": "false",
            "serving.kserve.io/s3-verifyssl": "0",
            "serving.kserve.io/s3-usehttps": "1",
        },
        string_data={
            "AWS_ACCESS_KEY_ID": aws_access_key,
            "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
        },
        wait_for_resource=True,
    ) as secret:
        yield secret


# ---------------------------------------------------------------------------
# Multi-model serving fixtures
# ---------------------------------------------------------------------------


def _append_env_to_serving_runtime(
    runtime: ServingRuntime,
    env_vars: list[dict[str, str]],
) -> None:
    """Append environment variables to the first container of a live ServingRuntime.

    Reads the current env list, appends any vars not already present, then
    sends a merge-patch with the full container array.  This avoids
    overwriting template-provided env vars (which the ``containers`` kwarg
    of ``ServingRuntimeFromTemplate`` would do).

    Args:
        runtime: An already-created ServingRuntime resource.
        env_vars: List of ``{"name": ..., "value": ...}`` dicts to append.
    """
    runtime.reload()
    containers = runtime.instance.spec.get("containers", [])
    if not containers:
        return

    existing_env = list(containers[0].get("env", []))
    existing_names = {e.get("name") for e in existing_env}
    for var in env_vars:
        if var["name"] not in existing_names:
            existing_env.append(var)

    containers[0]["env"] = existing_env
    runtime.update(resource_dict={"spec": {"containers": [dict(c) for c in containers]}})


def _record_pod_image(pod: Pod) -> dict[str, str]:
    """Extract container image reference (repository, tag, digest) from a pod.

    Satisfies RHAI-49122 image traceability requirement.

    Args:
        pod: Running Kubernetes pod.

    Returns:
        Dictionary with repository, tag, digest, and full image string.
    """
    containers = pod.instance.spec.get("containers", [])
    if not containers:
        return {"image": "", "repository": "", "tag": "", "digest": ""}

    image: str = containers[0].get("image", "")
    repository = image
    tag = ""
    digest = ""

    if "@" in image:
        repository, digest = image.rsplit("@", 1)
    if ":" in repository:
        repository, tag = repository.rsplit(":", 1)

    # Also capture imageID from status for the digest that was actually pulled
    container_statuses = pod.instance.status.get("containerStatuses", [])
    if container_statuses:
        image_id = container_statuses[0].get("imageID", "")
        if "@" in image_id and not digest:
            digest = image_id.rsplit("@", 1)[1]

    return {"image": image, "repository": repository, "tag": tag, "digest": digest}


def _record_static_profile_env(pod: Pod) -> dict[str, str | None]:
    """Record static-profile environment variable defaults from a deployed pod.

    Args:
        pod: Running Kubernetes pod.

    Returns:
        Dictionary mapping env var names to their values (None if unset).
    """
    env_values: dict[str, str | None] = {}
    containers = pod.instance.spec.get("containers", [])
    if not containers:
        return env_values

    env_list = containers[0].get("env", [])
    env_map = {e.get("name"): e.get("value") for e in env_list}
    for var_name in STATIC_PROFILE_ENV_VARS:
        env_values[var_name] = env_map.get(var_name)

    return env_values


@pytest.fixture(scope="class")
def multi_model_service_account(
    admin_client: DynamicClient,
    kserve_s3_secret: Secret,
) -> Generator[ServiceAccount]:
    """ServiceAccount for multi-model InferenceServices, linked to the S3 secret."""
    with ServiceAccount(
        client=admin_client,
        namespace=kserve_s3_secret.namespace,
        name="multi-model-sa",
        secrets=[{"name": kserve_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def mlserver_serving_runtime(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_runtime_image: str,
) -> Generator[ServingRuntime]:
    """MLServer ServingRuntime from the CPU template.

    multiModel is left at the template default (false) — that flag is the
    ModelMesh compatibility switch and must remain false for the SR to be
    selectable by KServe-native InferenceServices.  Multi-model behavior
    comes from MLServer's directory-scanning model repository, not from
    the spec field.

    Patches MLSERVER_PARALLEL_WORKERS=1 onto the live resource because the
    template change has not landed yet.
    """
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=ModelInferenceRuntime.MLSERVER_RUNTIME,
        namespace=model_namespace.name,
        template_name=RuntimeTemplates.MLSERVER,
        deployment_type=KServeDeploymentType.RAW_DEPLOYMENT,
        runtime_image=mlserver_runtime_image,
    ) as runtime:
        _append_env_to_serving_runtime(
            runtime=runtime,
            env_vars=[{"name": "MLSERVER_PARALLEL_WORKERS", "value": "1"}],
        )
        yield runtime


@pytest.fixture(scope="class")
def multi_model_pvc(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[PersistentVolumeClaim]:
    """PVC for the 10-model repository layout."""
    with PersistentVolumeClaim(
        name="multi-model-pvc",
        namespace=model_namespace.name,
        client=admin_client,
        size=MULTI_MODEL_PVC_SIZE,
        accessmodes="ReadWriteOnce",
    ) as pvc:
        yield pvc


def _stage_s3_to_pvc(
    admin_client: DynamicClient,
    model_namespace: str,
    pvc_name: str,
    model_path: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    bucket_name: str,
    endpoint: str,
    region: str,
) -> str:
    """Download an S3 prefix into a PVC subdirectory.

    Thin wrapper around ``download_model_data`` so every staging fixture
    uses the same call convention.

    Args:
        admin_client: Kubernetes dynamic client.
        model_namespace: Namespace name.
        pvc_name: Target PVC name.
        model_path: S3 prefix **and** PVC subdirectory (they are the same
            by convention — ``download_model_data`` uses ``model_path``
            for both the source key-prefix and the ``subPath`` mount).
        aws_access_key_id: AWS access key.
        aws_secret_access_key: AWS secret key.
        bucket_name: S3 bucket name.
        endpoint: S3 endpoint URL.
        region: S3 region.

    Returns:
        ``model_path`` unchanged — callers build ``pvc://{pvc}/{path}``.
    """
    return download_model_data(
        client=admin_client,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        model_namespace=model_namespace,
        model_pvc_name=pvc_name,
        bucket_name=bucket_name,
        aws_endpoint_url=endpoint,
        aws_default_region=region,
        model_path=model_path,
        use_sub_path=True,
        restricted_scc_init=True,
    )


@pytest.fixture(scope="class")
def multi_model_staged_data(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    multi_model_pvc: PersistentVolumeClaim,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_endpoint: str,
    models_s3_bucket_region: str,
) -> str:
    """Stage the 10-model repository onto the PVC via S3 download.

    Downloads ``s3://{bucket}/mlserver/model_repository/`` into PVC
    subdirectory ``mlserver/model_repository``.  The ISVC then references
    ``pvc://{pvc_name}/mlserver/model_repository``.

    Returns:
        Model path prefix within the PVC.
    """
    return _stage_s3_to_pvc(
        admin_client=admin_client,
        model_namespace=model_namespace.name,
        pvc_name=multi_model_pvc.name,
        model_path=S3_PATH_10_MODEL,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        bucket_name=models_s3_bucket_name,
        endpoint=models_s3_bucket_endpoint,
        region=models_s3_bucket_region,
    )


@pytest.fixture(scope="class")
def multi_model_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    multi_model_pvc: PersistentVolumeClaim,
    multi_model_staged_data: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """10-model InferenceService using PVC storage.

    After deployment, records the pod image reference (RHAI-49122) and
    static-profile environment defaults.
    """
    storage_uri = f"pvc://{multi_model_pvc.name}/{multi_model_staged_data}"

    with create_isvc(
        client=admin_client,
        name="multi-model-isvc",
        namespace=model_namespace.name,
        runtime=mlserver_serving_runtime.name,
        storage_uri=storage_uri,
        model_format=mlserver_serving_runtime.instance.spec.supportedModelFormats[0].name,
        model_service_account=multi_model_service_account.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
        external_route=True,
        wait_for_predictor_pods=True,
    ) as isvc:
        # Record image and env vars for traceability (RHAI-49122)
        pods = get_pods_by_isvc_label(client=admin_client, isvc=isvc)
        if pods:
            image_info = _record_pod_image(pods[0])
            LOGGER.info("multi_model_inference_service image", **image_info)
            env_info = _record_static_profile_env(pods[0])
            LOGGER.info("multi_model_inference_service env defaults", **env_info)
        yield isvc


@pytest.fixture(scope="class")
def multi_version_staged_data(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    multi_model_pvc: PersistentVolumeClaim,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_endpoint: str,
    models_s3_bucket_region: str,
) -> str:
    """Stage the multi-version model directory (shared-model v1.0 + v2.0) onto PVC."""
    return _stage_s3_to_pvc(
        admin_client=admin_client,
        model_namespace=model_namespace.name,
        pvc_name=multi_model_pvc.name,
        model_path=S3_PATH_MULTI_VERSION,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        bucket_name=models_s3_bucket_name,
        endpoint=models_s3_bucket_endpoint,
        region=models_s3_bucket_region,
    )


@pytest.fixture(scope="class")
def multi_version_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    multi_model_pvc: PersistentVolumeClaim,
    multi_version_staged_data: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """InferenceService with shared-model v1.0 and v2.0 for version disambiguation (TC-E2E-005)."""
    storage_uri = f"pvc://{multi_model_pvc.name}/{multi_version_staged_data}"

    with create_isvc(
        client=admin_client,
        name="multi-version-isvc",
        namespace=model_namespace.name,
        runtime=mlserver_serving_runtime.name,
        storage_uri=storage_uri,
        model_format=mlserver_serving_runtime.instance.spec.supportedModelFormats[0].name,
        model_service_account=multi_model_service_account.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
        wait_for_predictor_pods=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def single_model_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """Single-model ISVC for regression test (TC-E2E-009)."""
    with create_isvc(
        client=admin_client,
        name="single-model-isvc",
        namespace=model_namespace.name,
        runtime=mlserver_serving_runtime.name,
        storage_uri=s3_models_storage_uri,
        model_format=ModelFormat.SKLEARN,
        model_service_account=multi_model_service_account.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
        wait_for_predictor_pods=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def startup_measurement_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    multi_model_pvc: PersistentVolumeClaim,
    multi_model_staged_data: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """Fresh 10-model ISVC for startup time measurement (TC-E2E-007).

    Uses wait_for_predictor_pods=False so the test can measure startup itself.
    """
    storage_uri = f"pvc://{multi_model_pvc.name}/{multi_model_staged_data}"

    with create_isvc(
        client=admin_client,
        name="startup-measure-isvc",
        namespace=model_namespace.name,
        runtime=mlserver_serving_runtime.name,
        storage_uri=storage_uri,
        model_format=mlserver_serving_runtime.instance.spec.supportedModelFormats[0].name,
        model_service_account=multi_model_service_account.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def cuda_serving_runtime(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_runtime_image: str,
) -> Generator[ServingRuntime]:
    """CUDA-variant MLServer ServingRuntime for GPU tests (TC-E2E-008).

    Same parallel-workers pin as the CPU variant.
    """
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=MLSERVER_CUDA_RUNTIME,
        namespace=model_namespace.name,
        template_name=MLSERVER_CUDA_TEMPLATE,
        deployment_type=KServeDeploymentType.RAW_DEPLOYMENT,
        runtime_image=mlserver_runtime_image,
    ) as runtime:
        _append_env_to_serving_runtime(
            runtime=runtime,
            env_vars=[{"name": "MLSERVER_PARALLEL_WORKERS", "value": "1"}],
        )
        yield runtime


@pytest.fixture(scope="class")
def cuda_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    cuda_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """GPU-enabled ONNX ISVC using the CUDA template (TC-E2E-008)."""
    resources: dict[str, Any] = copy.deepcopy(PREDICT_RESOURCES["resources"])
    resources["requests"][Labels.Nvidia.NVIDIA_COM_GPU] = 1
    resources["limits"][Labels.Nvidia.NVIDIA_COM_GPU] = 1

    with create_isvc(
        client=admin_client,
        name="cuda-multi-model-isvc",
        namespace=model_namespace.name,
        runtime=cuda_serving_runtime.name,
        storage_uri=s3_models_storage_uri,
        model_format=ModelFormat.ONNX,
        model_service_account=multi_model_service_account.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        resources=resources,
        volumes=copy.deepcopy(PREDICT_RESOURCES["volumes"]),
        volumes_mounts=copy.deepcopy(PREDICT_RESOURCES["volume_mounts"]),
        wait_for_predictor_pods=True,
    ) as isvc:
        yield isvc


# ---------------------------------------------------------------------------
# Negative-scenario fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def empty_storage_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """ISVC with an empty storage path — zero models (TC-NEG-002).

    Creates a dedicated PVC with no model data staged.
    """
    with (
        PersistentVolumeClaim(
            name="empty-storage-pvc",
            namespace=model_namespace.name,
            client=admin_client,
            size="1Gi",
            accessmodes="ReadWriteOnce",
        ) as pvc,
        create_isvc(
            client=admin_client,
            name="empty-storage-isvc",
            namespace=model_namespace.name,
            runtime=mlserver_serving_runtime.name,
            storage_uri=f"pvc://{pvc.name}/",
            model_format=mlserver_serving_runtime.instance.spec.supportedModelFormats[0].name,
            model_service_account=multi_model_service_account.name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
            wait_for_predictor_pods=False,
        ) as isvc,
    ):
        yield isvc


@pytest.fixture(scope="class")
def one_failed_model_staged_data(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    multi_model_pvc: PersistentVolumeClaim,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_endpoint: str,
    models_s3_bucket_region: str,
) -> str:
    """Stage 10-model set with one bad artifact reference onto its own PVC subdir."""
    return _stage_s3_to_pvc(
        admin_client=admin_client,
        model_namespace=model_namespace.name,
        pvc_name=multi_model_pvc.name,
        model_path=S3_PATH_ONE_FAILED,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        bucket_name=models_s3_bucket_name,
        endpoint=models_s3_bucket_endpoint,
        region=models_s3_bucket_region,
    )


@pytest.fixture(scope="class")
def one_failed_model_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    multi_model_pvc: PersistentVolumeClaim,
    one_failed_model_staged_data: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """ISVC where one model references a non-existent artifact (TC-NEG-003).

    Staged from ``s3://{bucket}/mlserver/one_failed_model/`` — a 10-model
    directory where one model-settings.json points to a missing file,
    resulting in 9 loaded + 1 failed.
    """
    storage_uri = f"pvc://{multi_model_pvc.name}/{one_failed_model_staged_data}"

    with create_isvc(
        client=admin_client,
        name="one-failed-model-isvc",
        namespace=model_namespace.name,
        runtime=mlserver_serving_runtime.name,
        storage_uri=storage_uri,
        model_format=mlserver_serving_runtime.instance.spec.supportedModelFormats[0].name,
        model_service_account=multi_model_service_account.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def malformed_settings_staged_data(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    multi_model_pvc: PersistentVolumeClaim,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_endpoint: str,
    models_s3_bucket_region: str,
) -> str:
    """Stage model set containing one invalid-JSON model-settings.json onto PVC."""
    return _stage_s3_to_pvc(
        admin_client=admin_client,
        model_namespace=model_namespace.name,
        pvc_name=multi_model_pvc.name,
        model_path=S3_PATH_MALFORMED,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        bucket_name=models_s3_bucket_name,
        endpoint=models_s3_bucket_endpoint,
        region=models_s3_bucket_region,
    )


@pytest.fixture(scope="class")
def malformed_settings_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    multi_model_pvc: PersistentVolumeClaim,
    malformed_settings_staged_data: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """ISVC with one subdirectory containing invalid JSON model-settings (TC-NEG-004).

    Staged from ``s3://{bucket}/mlserver/malformed_settings/`` — includes
    sklearn-iris-1, xgb-classifier-1, and a ``bad-model`` subdirectory
    whose model-settings.json is syntactically broken.
    """
    storage_uri = f"pvc://{multi_model_pvc.name}/{malformed_settings_staged_data}"

    with create_isvc(
        client=admin_client,
        name="malformed-settings-isvc",
        namespace=model_namespace.name,
        runtime=mlserver_serving_runtime.name,
        storage_uri=storage_uri,
        model_format=mlserver_serving_runtime.instance.spec.supportedModelFormats[0].name,
        model_service_account=multi_model_service_account.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def unsupported_impl_staged_data(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    multi_model_pvc: PersistentVolumeClaim,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_endpoint: str,
    models_s3_bucket_region: str,
) -> str:
    """Stage model set containing one custom.MyModel implementation onto PVC."""
    return _stage_s3_to_pvc(
        admin_client=admin_client,
        model_namespace=model_namespace.name,
        pvc_name=multi_model_pvc.name,
        model_path=S3_PATH_UNSUPPORTED_IMPL,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        bucket_name=models_s3_bucket_name,
        endpoint=models_s3_bucket_endpoint,
        region=models_s3_bucket_region,
    )


@pytest.fixture(scope="class")
def unsupported_impl_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    multi_model_pvc: PersistentVolumeClaim,
    unsupported_impl_staged_data: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """ISVC with one subdirectory using custom.MyModel implementation (TC-NEG-005).

    Staged from ``s3://{bucket}/mlserver/unsupported_impl/`` — includes
    sklearn-iris-1, xgb-classifier-1, and a ``custom-model`` subdirectory
    whose model-settings.json specifies ``"implementation": "custom.MyModel"``.
    """
    storage_uri = f"pvc://{multi_model_pvc.name}/{unsupported_impl_staged_data}"

    with create_isvc(
        client=admin_client,
        name="unsupported-impl-isvc",
        namespace=model_namespace.name,
        runtime=mlserver_serving_runtime.name,
        storage_uri=storage_uri,
        model_format=mlserver_serving_runtime.instance.spec.supportedModelFormats[0].name,
        model_service_account=multi_model_service_account.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def resource_exhaustion_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    multi_model_pvc: PersistentVolumeClaim,
    multi_model_staged_data: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """ISVC with intentionally low resource limits to trigger OOM (TC-NEG-006).

    Shares the 10-model happy-path staging data — the scenario exercises
    resource limits, not a different model layout.
    """
    storage_uri = f"pvc://{multi_model_pvc.name}/{multi_model_staged_data}"
    low_resources: dict[str, dict[str, str]] = {
        "requests": {"cpu": "100m", "memory": "64Mi"},
        "limits": {"cpu": "200m", "memory": "128Mi"},
    }

    with create_isvc(
        client=admin_client,
        name="resource-exhaustion-isvc",
        namespace=model_namespace.name,
        runtime=mlserver_serving_runtime.name,
        storage_uri=storage_uri,
        model_format=mlserver_serving_runtime.instance.spec.supportedModelFormats[0].name,
        model_service_account=multi_model_service_account.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        resources=low_resources,
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def missing_artifact_staged_data(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    multi_model_pvc: PersistentVolumeClaim,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_endpoint: str,
    models_s3_bucket_region: str,
) -> str:
    """Stage model set containing one missing-artifact reference onto PVC."""
    return _stage_s3_to_pvc(
        admin_client=admin_client,
        model_namespace=model_namespace.name,
        pvc_name=multi_model_pvc.name,
        model_path=S3_PATH_MISSING_ARTIFACT,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        bucket_name=models_s3_bucket_name,
        endpoint=models_s3_bucket_endpoint,
        region=models_s3_bucket_region,
    )


@pytest.fixture(scope="class")
def missing_artifact_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    multi_model_pvc: PersistentVolumeClaim,
    missing_artifact_staged_data: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """ISVC where one model references ./nonexistent.joblib (TC-NEG-007).

    Staged from ``s3://{bucket}/mlserver/missing_artifact/`` — includes
    sklearn-iris-1, xgb-classifier-1, and a ``missing-model`` subdirectory
    whose model-settings.json ``uri`` points to ``./nonexistent.joblib``.
    """
    storage_uri = f"pvc://{multi_model_pvc.name}/{missing_artifact_staged_data}"

    with create_isvc(
        client=admin_client,
        name="missing-artifact-isvc",
        namespace=model_namespace.name,
        runtime=mlserver_serving_runtime.name,
        storage_uri=storage_uri,
        model_format=mlserver_serving_runtime.instance.spec.supportedModelFormats[0].name,
        model_service_account=multi_model_service_account.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def invalid_credentials_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[InferenceService]:
    """ISVC with intentionally wrong S3 credentials (TC-NEG-008).

    Creates a bad secret and SA in-line, then deploys the ISVC expecting
    storage initialization failure.
    """
    with (
        kserve_s3_endpoint_secret(
            admin_client=admin_client,
            name="bad-credentials-secret",
            namespace=model_namespace.name,
            aws_access_key="INVALID_KEY_ID",
            aws_secret_access_key="INVALID_SECRET_KEY",
            aws_s3_region=models_s3_bucket_region,
            aws_s3_endpoint=models_s3_bucket_endpoint,
        ) as bad_secret,
        ServiceAccount(
            client=admin_client,
            namespace=model_namespace.name,
            name="bad-credentials-sa",
            secrets=[{"name": bad_secret.name}],
        ) as bad_sa,
        create_isvc(
            client=admin_client,
            name="invalid-creds-isvc",
            namespace=model_namespace.name,
            runtime=mlserver_serving_runtime.name,
            storage_uri="s3://nonexistent-bucket/models/",
            model_format=ModelFormat.SKLEARN,
            model_service_account=bad_sa.name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
            wait_for_predictor_pods=False,
        ) as isvc,
    ):
        yield isvc


@pytest.fixture(scope="class")
def storage_exhaustion_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    multi_model_service_account: ServiceAccount,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_endpoint: str,
    models_s3_bucket_region: str,
) -> Generator[InferenceService]:
    """ISVC with an undersized PVC to trigger ENOSPC (TC-NEG-009)."""
    with PersistentVolumeClaim(
        name="undersized-pvc",
        namespace=model_namespace.name,
        client=admin_client,
        size=STORAGE_EXHAUSTION_PVC_SIZE,
        accessmodes="ReadWriteOnce",
    ) as pvc:
        # Attempt to stage data — expected to fail or partially succeed
        try:
            model_path = download_model_data(
                client=admin_client,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                model_namespace=model_namespace.name,
                model_pvc_name=pvc.name,
                bucket_name=models_s3_bucket_name,
                aws_endpoint_url=models_s3_bucket_endpoint,
                aws_default_region=models_s3_bucket_region,
                model_path="mlserver/model_repository",
                use_sub_path=True,
                restricted_scc_init=True,
            )
            storage_uri = f"pvc://{pvc.name}/{model_path}"
        except (TimeoutExpiredError, ApiException, OSError) as exc:
            LOGGER.info("Storage exhaustion staging failed as expected", error=str(exc))
            storage_uri = f"pvc://{pvc.name}/"

        with create_isvc(
            client=admin_client,
            name="storage-exhaustion-isvc",
            namespace=model_namespace.name,
            runtime=mlserver_serving_runtime.name,
            storage_uri=storage_uri,
            model_format=mlserver_serving_runtime.instance.spec.supportedModelFormats[0].name,
            model_service_account=multi_model_service_account.name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
            wait_for_predictor_pods=False,
        ) as isvc:
            yield isvc


# ---------------------------------------------------------------------------
# NFR fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def baseline_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """Single-model baseline for PSAP benchmark comparison (TC-NFR-001)."""
    with create_isvc(
        client=admin_client,
        name="baseline-isvc",
        namespace=model_namespace.name,
        runtime=mlserver_serving_runtime.name,
        storage_uri=s3_models_storage_uri,
        model_format=ModelFormat.SKLEARN,
        model_service_account=multi_model_service_account.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
        wait_for_predictor_pods=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def single_model_fleet(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[list[InferenceService]]:
    """Fleet of 50 single-model ISVCs for scalability comparison (TC-NFR-002).

    Creates 50 ISVCs sequentially with app=single-model-bench label.
    """
    fleet: list[InferenceService] = []
    contexts: list[Any] = []

    for i in range(50):
        ctx = create_isvc(
            client=admin_client,
            name=f"single-bench-{i:03d}",
            namespace=model_namespace.name,
            runtime=mlserver_serving_runtime.name,
            storage_uri=s3_models_storage_uri,
            model_format=ModelFormat.SKLEARN,
            model_service_account=multi_model_service_account.name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
            labels={"app": "single-model-bench"},
            wait_for_predictor_pods=True,
        )
        isvc = ctx.__enter__()
        contexts.append(ctx)
        fleet.append(isvc)

    yield fleet

    for ctx in reversed(contexts):
        try:
            ctx.__exit__(None, None, None)
        except ApiException as exc:
            LOGGER.warning("Single-model fleet teardown error", error=str(exc))


@pytest.fixture(scope="class")
def multi_model_fleet(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    multi_model_pvc: PersistentVolumeClaim,
    multi_model_staged_data: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[list[InferenceService]]:
    """Fleet of 5 multi-model ISVCs for scalability comparison (TC-NFR-002).

    Each ISVC points to the same 10-model PVC repository.
    """
    storage_uri = f"pvc://{multi_model_pvc.name}/{multi_model_staged_data}"
    fleet: list[InferenceService] = []
    contexts: list[Any] = []

    for i in range(5):
        ctx = create_isvc(
            client=admin_client,
            name=f"multi-bench-{i:03d}",
            namespace=model_namespace.name,
            runtime=mlserver_serving_runtime.name,
            storage_uri=storage_uri,
            model_format=mlserver_serving_runtime.instance.spec.supportedModelFormats[0].name,
            model_service_account=multi_model_service_account.name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
            labels={"app": "multi-model-bench"},
            wait_for_predictor_pods=True,
        )
        isvc = ctx.__enter__()
        contexts.append(ctx)
        fleet.append(isvc)

    yield fleet

    for ctx in reversed(contexts):
        try:
            ctx.__exit__(None, None, None)
        except ApiException as exc:
            LOGGER.warning("Multi-model fleet teardown error", error=str(exc))
