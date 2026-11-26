from __future__ import annotations

import contextlib
import copy
from collections.abc import Generator
from typing import Any

import pytest
import structlog
from _pytest.fixtures import FixtureRequest
from kubernetes.client.rest import ApiException
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from timeout_sampler import TimeoutExpiredError

from tests.model_serving.model_runtime.mlserver.constant import PREDICT_RESOURCES
from tests.model_serving.model_runtime.vllm.utils import kserve_s3_endpoint_secret
from tests.model_serving.multi_model_helpers import (
    append_env_to_serving_runtime,
    record_pod_image,
    record_static_profile_env,
    stage_s3_to_pvc,
)
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
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
    teardown_resources: bool,
) -> Generator[Secret]:
    """S3 access Secret for KServe InferenceServices.

    Post-upgrade: looks up the existing Secret by name without re-creating it.
    """
    if getattr(pytestconfig.option, "post_upgrade", False):
        secret = Secret(
            client=admin_client,
            namespace=model_namespace.name,
            name="models-bucket-secret",
        )
        yield secret
    else:
        with kserve_s3_endpoint_secret(
            admin_client=admin_client,
            name="models-bucket-secret",
            namespace=model_namespace.name,
            aws_access_key=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_s3_region=models_s3_bucket_region,
            aws_s3_endpoint=models_s3_bucket_endpoint,
        ) as secret:
            secret.teardown = teardown_resources
            yield secret


# ---------------------------------------------------------------------------
# Multi-model serving fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def multi_model_service_account(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    kserve_s3_secret: Secret,
    model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[ServiceAccount]:
    """ServiceAccount for multi-model InferenceServices, linked to the S3 secret.

    Post-upgrade: looks up the existing ServiceAccount by name without
    re-creating it.
    """
    if getattr(pytestconfig.option, "post_upgrade", False):
        sa = ServiceAccount(
            client=admin_client,
            namespace=kserve_s3_secret.namespace,
            name="multi-model-sa",
        )
        yield sa
    else:
        with ServiceAccount(
            client=admin_client,
            namespace=kserve_s3_secret.namespace,
            name="multi-model-sa",
            secrets=[{"name": kserve_s3_secret.name}],
            teardown=teardown_resources,
        ) as sa:
            yield sa


@pytest.fixture(scope="class")
def mlserver_serving_runtime(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_runtime_image: str,
    teardown_resources: bool,
) -> Generator[ServingRuntime]:
    """MLServer ServingRuntime from the CPU template.

    multiModel is left at the template default (false) — that flag is the
    ModelMesh compatibility switch and must remain false for the SR to be
    selectable by KServe-native InferenceServices.  Multi-model behavior
    comes from MLServer's directory-scanning model repository, not from
    the spec field.

    Patches MLSERVER_PARALLEL_WORKERS=1 onto the live resource because the
    template change has not landed yet.

    Post-upgrade: looks up the existing ServingRuntime by name without
    re-creating it.
    """
    if getattr(pytestconfig.option, "post_upgrade", False):
        runtime = ServingRuntime(
            client=admin_client,
            name=ModelInferenceRuntime.MLSERVER_RUNTIME,
            namespace=model_namespace.name,
        )
        yield runtime
        runtime.clean_up()
    else:
        with ServingRuntimeFromTemplate(
            client=admin_client,
            name=ModelInferenceRuntime.MLSERVER_RUNTIME,
            namespace=model_namespace.name,
            template_name=RuntimeTemplates.MLSERVER,
            deployment_type=KServeDeploymentType.RAW_DEPLOYMENT,
            runtime_image=mlserver_runtime_image,
            teardown=teardown_resources,
        ) as runtime:
            append_env_to_serving_runtime(
                runtime=runtime,
                env_vars=[{"name": "MLSERVER_PARALLEL_WORKERS", "value": "1"}],
            )
            yield runtime


@pytest.fixture(scope="class")
def multi_model_pvc(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[PersistentVolumeClaim]:
    """PVC for the 10-model repository layout.

    Post-upgrade: looks up the existing PVC by name without re-creating it.
    """
    if getattr(pytestconfig.option, "post_upgrade", False):
        pvc = PersistentVolumeClaim(
            name="multi-model-pvc",
            namespace=model_namespace.name,
            client=admin_client,
        )
        yield pvc
    else:
        with PersistentVolumeClaim(
            name="multi-model-pvc",
            namespace=model_namespace.name,
            client=admin_client,
            size=MULTI_MODEL_PVC_SIZE,
            accessmodes="ReadWriteOnce",
            teardown=teardown_resources,
        ) as pvc:
            yield pvc


@pytest.fixture(scope="class")
def multi_model_staged_data(
    pytestconfig: pytest.Config,
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

    Post-upgrade: returns the model path without re-downloading.

    Returns:
        Model path prefix within the PVC.
    """
    if getattr(pytestconfig.option, "post_upgrade", False):
        return S3_PATH_10_MODEL

    return stage_s3_to_pvc(
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
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    multi_model_pvc: PersistentVolumeClaim,
    multi_model_staged_data: str,
    multi_model_service_account: ServiceAccount,
    teardown_resources: bool,
) -> Generator[InferenceService]:
    """10-model InferenceService using PVC storage.

    After deployment, records the pod image reference (RHAI-49122) and
    static-profile environment defaults.

    Post-upgrade: looks up the existing InferenceService by name without
    re-creating it.
    """
    if getattr(pytestconfig.option, "post_upgrade", False):
        isvc = InferenceService(
            client=admin_client,
            name="multi-model-isvc",
            namespace=model_namespace.name,
        )
        yield isvc
        isvc.clean_up()
    else:
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
            teardown=teardown_resources,
        ) as isvc:
            # Record image and env vars for traceability (RHAI-49122)
            pods = get_pods_by_isvc_label(client=admin_client, isvc=isvc)
            if pods:
                image_info = record_pod_image(pods[0])
                LOGGER.info("multi_model_inference_service image", **image_info)
                env_info = record_static_profile_env(pods[0])
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
    return stage_s3_to_pvc(
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
        wait=False,
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def cuda_serving_runtime(
    admin_client: DynamicClient,
    model_namespace: Namespace,
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
        runtime_image=None,
    ) as runtime:
        append_env_to_serving_runtime(
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
            wait=False,
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
    return stage_s3_to_pvc(
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
        wait=False,
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
    return stage_s3_to_pvc(
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
        wait=False,
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
    return stage_s3_to_pvc(
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
        wait=False,
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
        wait=False,
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
    return stage_s3_to_pvc(
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
        wait=False,
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def invalid_credentials_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[InferenceService]:
    """ISVC with intentionally wrong S3 credentials (TC-NEG-008).

    Uses bad credentials against the real S3 bucket so the error is
    specifically an authentication rejection, not a missing-bucket 404.
    """
    with (
        kserve_s3_endpoint_secret(
            admin_client=admin_client,
            name="bad-credentials-secret",
            namespace=model_namespace.name,
            aws_access_key="INVALID_KEY_ID",
            aws_secret_access_key="INVALID_SECRET_KEY",  # pragma: allowlist secret
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
            storage_uri=f"s3://{models_s3_bucket_name}/{S3_PATH_10_MODEL}/",
            model_format=ModelFormat.SKLEARN,
            model_service_account=bad_sa.name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
            wait=False,
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
        staging_failed = False
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
            staging_failed = True

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
            wait=False,
            wait_for_predictor_pods=False,
        ) as isvc:
            isvc._staging_failed = staging_failed
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
    Uses ExitStack so already-created ISVCs are cleaned up if setup fails mid-loop.
    """
    with contextlib.ExitStack() as stack:
        fleet: list[InferenceService] = []
        for i in range(50):
            isvc = stack.enter_context(
                cm=create_isvc(
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
            )
            fleet.append(isvc)
        yield fleet


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
    Uses ExitStack so already-created ISVCs are cleaned up if setup fails mid-loop.

    Note: all 5 ISVCs mount the same ReadWriteOnce PVC. This assumes either a
    single-node cluster or that the scheduler co-locates all pods on the same
    node. On multi-node clusters with no node-affinity constraint, pods may
    land on different nodes and fail to attach the RWO volume. If fleet tests
    need to run on multi-node clusters, change ``multi_model_pvc`` to use
    ReadWriteMany (requires an RWX-capable StorageClass).
    """
    storage_uri = f"pvc://{multi_model_pvc.name}/{multi_model_staged_data}"
    with contextlib.ExitStack() as stack:
        fleet: list[InferenceService] = []
        for i in range(5):
            isvc = stack.enter_context(
                cm=create_isvc(
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
            )
            fleet.append(isvc)
        yield fleet
