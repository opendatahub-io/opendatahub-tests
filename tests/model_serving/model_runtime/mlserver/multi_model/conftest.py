"""
Pytest fixtures for MLServer multi-model serving tests.

Provides fixtures for multi-model deployment scenarios including E2E,
negative, NFR, and upgrade test categories.

Source: RHAISTRAT-3033
"""

from __future__ import annotations

import contextlib
import copy
from collections.abc import Generator
from typing import Any

import pytest
import structlog
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
from tests.model_serving.model_runtime.mlserver.multi_model.constant import (
    MULTI_MODEL_PVC_SIZE,
    S3_PATH_10_MODEL,
    S3_PATH_MALFORMED,
    S3_PATH_MISSING_ARTIFACT,
    S3_PATH_MULTI_VERSION,
    S3_PATH_ONE_FAILED,
    S3_PATH_UNSUPPORTED_IMPL,
    STORAGE_EXHAUSTION_PVC_SIZE,
)
from tests.model_serving.model_runtime.mlserver.multi_model.utils import (
    append_env_to_serving_runtime,
    record_pod_image,
    record_static_profile_env,
    stage_s3_to_pvc,
)
from tests.model_serving.model_runtime.vllm.utils import kserve_s3_endpoint_secret
from tests.model_serving.model_server.upgrade.utils import (
    load_baseline_from_configmap,
    save_baseline_to_configmap,
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
# S3 / ServiceAccount fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def multi_model_s3_secret(
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


@pytest.fixture(scope="class")
def multi_model_service_account(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    multi_model_s3_secret: Secret,
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
            namespace=multi_model_s3_secret.namespace,
            name="multi-model-sa",
        )
        yield sa
    else:
        with ServiceAccount(
            client=admin_client,
            namespace=multi_model_s3_secret.namespace,
            name="multi-model-sa",
            secrets=[{"name": multi_model_s3_secret.name}],
            teardown=teardown_resources,
        ) as sa:
            yield sa


# ---------------------------------------------------------------------------
# ServingRuntime fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def multi_model_serving_runtime(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_runtime_image: str,
    teardown_resources: bool,
) -> Generator[ServingRuntime]:
    """MLServer ServingRuntime from the CPU template for multi-model tests.

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
def cuda_serving_runtime(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime]:
    """CUDA-variant MLServer ServingRuntime for GPU tests (TC-E2E-008).

    Same parallel-workers pin as the CPU variant.
    """
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=ModelInferenceRuntime.MLSERVER_CUDA_RUNTIME,
        namespace=model_namespace.name,
        template_name=RuntimeTemplates.MLSERVER_CUDA,
        deployment_type=KServeDeploymentType.RAW_DEPLOYMENT,
        runtime_image=None,
    ) as runtime:
        append_env_to_serving_runtime(
            runtime=runtime,
            env_vars=[{"name": "MLSERVER_PARALLEL_WORKERS", "value": "1"}],
        )
        yield runtime


# ---------------------------------------------------------------------------
# PVC and data-staging fixtures
# ---------------------------------------------------------------------------


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

    Post-upgrade: returns the model path without re-downloading.
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


# ---------------------------------------------------------------------------
# InferenceService fixtures — E2E
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def multi_model_inference_service(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    multi_model_serving_runtime: ServingRuntime,
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
            runtime=multi_model_serving_runtime.name,
            storage_uri=storage_uri,
            model_format=multi_model_serving_runtime.instance.spec.supportedModelFormats[0].name,
            model_service_account=multi_model_service_account.name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
            external_route=True,
            wait_for_predictor_pods=True,
            teardown=teardown_resources,
        ) as isvc:
            pods = get_pods_by_isvc_label(client=admin_client, isvc=isvc)
            if pods:
                image_info = record_pod_image(pods[0])
                LOGGER.info("multi_model_inference_service image", **image_info)
                env_info = record_static_profile_env(pods[0])
                LOGGER.info("multi_model_inference_service env defaults", **env_info)
            yield isvc


@pytest.fixture(scope="class")
def multi_version_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    multi_model_serving_runtime: ServingRuntime,
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
        runtime=multi_model_serving_runtime.name,
        storage_uri=storage_uri,
        model_format=multi_model_serving_runtime.instance.spec.supportedModelFormats[0].name,
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
    multi_model_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """Single-model ISVC for regression test (TC-E2E-009)."""
    with create_isvc(
        client=admin_client,
        name="single-model-isvc",
        namespace=model_namespace.name,
        runtime=multi_model_serving_runtime.name,
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
    multi_model_serving_runtime: ServingRuntime,
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
        runtime=multi_model_serving_runtime.name,
        storage_uri=storage_uri,
        model_format=multi_model_serving_runtime.instance.spec.supportedModelFormats[0].name,
        model_service_account=multi_model_service_account.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
        wait=False,
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


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
# InferenceService fixtures — Negative
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def empty_storage_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    multi_model_serving_runtime: ServingRuntime,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """ISVC with an empty storage path — zero models (TC-NEG-002)."""
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
            runtime=multi_model_serving_runtime.name,
            storage_uri=f"pvc://{pvc.name}/",
            model_format=multi_model_serving_runtime.instance.spec.supportedModelFormats[0].name,
            model_service_account=multi_model_service_account.name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
            wait=False,
            wait_for_predictor_pods=False,
        ) as isvc,
    ):
        yield isvc


@pytest.fixture(scope="class")
def one_failed_model_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    multi_model_serving_runtime: ServingRuntime,
    multi_model_pvc: PersistentVolumeClaim,
    one_failed_model_staged_data: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """ISVC where one model references a non-existent artifact (TC-NEG-003)."""
    storage_uri = f"pvc://{multi_model_pvc.name}/{one_failed_model_staged_data}"

    with create_isvc(
        client=admin_client,
        name="one-failed-model-isvc",
        namespace=model_namespace.name,
        runtime=multi_model_serving_runtime.name,
        storage_uri=storage_uri,
        model_format=multi_model_serving_runtime.instance.spec.supportedModelFormats[0].name,
        model_service_account=multi_model_service_account.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
        wait=False,
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def malformed_settings_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    multi_model_serving_runtime: ServingRuntime,
    multi_model_pvc: PersistentVolumeClaim,
    malformed_settings_staged_data: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """ISVC with one subdirectory containing invalid JSON model-settings (TC-NEG-004)."""
    storage_uri = f"pvc://{multi_model_pvc.name}/{malformed_settings_staged_data}"

    with create_isvc(
        client=admin_client,
        name="malformed-settings-isvc",
        namespace=model_namespace.name,
        runtime=multi_model_serving_runtime.name,
        storage_uri=storage_uri,
        model_format=multi_model_serving_runtime.instance.spec.supportedModelFormats[0].name,
        model_service_account=multi_model_service_account.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
        wait=False,
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def unsupported_impl_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    multi_model_serving_runtime: ServingRuntime,
    multi_model_pvc: PersistentVolumeClaim,
    unsupported_impl_staged_data: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """ISVC with one subdirectory using custom.MyModel implementation (TC-NEG-005)."""
    storage_uri = f"pvc://{multi_model_pvc.name}/{unsupported_impl_staged_data}"

    with create_isvc(
        client=admin_client,
        name="unsupported-impl-isvc",
        namespace=model_namespace.name,
        runtime=multi_model_serving_runtime.name,
        storage_uri=storage_uri,
        model_format=multi_model_serving_runtime.instance.spec.supportedModelFormats[0].name,
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
    multi_model_serving_runtime: ServingRuntime,
    multi_model_pvc: PersistentVolumeClaim,
    multi_model_staged_data: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """ISVC with intentionally low resource limits to trigger OOM (TC-NEG-006)."""
    storage_uri = f"pvc://{multi_model_pvc.name}/{multi_model_staged_data}"
    low_resources: dict[str, dict[str, str]] = {
        "requests": {"cpu": "100m", "memory": "64Mi"},
        "limits": {"cpu": "200m", "memory": "128Mi"},
    }

    with create_isvc(
        client=admin_client,
        name="resource-exhaustion-isvc",
        namespace=model_namespace.name,
        runtime=multi_model_serving_runtime.name,
        storage_uri=storage_uri,
        model_format=multi_model_serving_runtime.instance.spec.supportedModelFormats[0].name,
        model_service_account=multi_model_service_account.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        resources=low_resources,
        wait=False,
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def missing_artifact_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    multi_model_serving_runtime: ServingRuntime,
    multi_model_pvc: PersistentVolumeClaim,
    missing_artifact_staged_data: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """ISVC where one model references ./nonexistent.joblib (TC-NEG-007)."""
    storage_uri = f"pvc://{multi_model_pvc.name}/{missing_artifact_staged_data}"

    with create_isvc(
        client=admin_client,
        name="missing-artifact-isvc",
        namespace=model_namespace.name,
        runtime=multi_model_serving_runtime.name,
        storage_uri=storage_uri,
        model_format=multi_model_serving_runtime.instance.spec.supportedModelFormats[0].name,
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
    multi_model_serving_runtime: ServingRuntime,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[InferenceService]:
    """ISVC with intentionally wrong S3 credentials (TC-NEG-008)."""
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
            runtime=multi_model_serving_runtime.name,
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
    multi_model_serving_runtime: ServingRuntime,
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
            runtime=multi_model_serving_runtime.name,
            storage_uri=storage_uri,
            model_format=multi_model_serving_runtime.instance.spec.supportedModelFormats[0].name,
            model_service_account=multi_model_service_account.name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
            wait=False,
            wait_for_predictor_pods=False,
        ) as isvc:
            isvc._staging_failed = staging_failed
            yield isvc


# ---------------------------------------------------------------------------
# InferenceService fixtures — NFR
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def baseline_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    multi_model_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """Single-model baseline for PSAP benchmark comparison (TC-NFR-001)."""
    with create_isvc(
        client=admin_client,
        name="baseline-isvc",
        namespace=model_namespace.name,
        runtime=multi_model_serving_runtime.name,
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
    multi_model_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[list[InferenceService]]:
    """Fleet of 50 single-model ISVCs for scalability comparison (TC-NFR-002)."""
    with contextlib.ExitStack() as stack:
        fleet: list[InferenceService] = []
        for i in range(50):
            isvc = stack.enter_context(
                cm=create_isvc(
                    client=admin_client,
                    name=f"single-bench-{i:03d}",
                    namespace=model_namespace.name,
                    runtime=multi_model_serving_runtime.name,
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
    multi_model_serving_runtime: ServingRuntime,
    multi_model_pvc: PersistentVolumeClaim,
    multi_model_staged_data: str,
    multi_model_service_account: ServiceAccount,
) -> Generator[list[InferenceService]]:
    """Fleet of 5 multi-model ISVCs for scalability comparison (TC-NFR-002)."""
    storage_uri = f"pvc://{multi_model_pvc.name}/{multi_model_staged_data}"
    with contextlib.ExitStack() as stack:
        fleet: list[InferenceService] = []
        for i in range(5):
            isvc = stack.enter_context(
                cm=create_isvc(
                    client=admin_client,
                    name=f"multi-bench-{i:03d}",
                    namespace=model_namespace.name,
                    runtime=multi_model_serving_runtime.name,
                    storage_uri=storage_uri,
                    model_format=multi_model_serving_runtime.instance.spec.supportedModelFormats[0].name,
                    model_service_account=multi_model_service_account.name,
                    deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
                    resources=copy.deepcopy(PREDICT_RESOURCES["resources"]),
                    labels={"app": "multi-model-bench"},
                    wait_for_predictor_pods=True,
                )
            )
            fleet.append(isvc)
        yield fleet


# ---------------------------------------------------------------------------
# Upgrade fixtures
# ---------------------------------------------------------------------------


def _capture_serving_runtime_snapshot(runtime: ServingRuntime) -> dict[str, Any]:
    """Capture ServingRuntime spec and annotations for baseline comparison."""
    instance = runtime.instance.to_dict()
    spec = instance.get("spec", {})
    annotations = instance.get("metadata", {}).get("annotations", {})
    return {
        "name": runtime.name,
        "namespace": runtime.namespace,
        "spec": spec,
        "annotations": annotations,
    }


@pytest.fixture(scope="class")
def pre_upgrade_serving_runtime_snapshot(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    multi_model_serving_runtime: ServingRuntime,
    model_namespace: Namespace,
) -> dict[str, Any]:
    """ServingRuntime spec snapshot for pre/post upgrade comparison.

    Pre-upgrade: captures the live ServingRuntime state and saves it to a ConfigMap.
    Post-upgrade: loads the previously saved snapshot from the ConfigMap.
    """
    if getattr(pytestconfig.option, "post_upgrade", False):
        baselines = load_baseline_from_configmap(
            client=admin_client,
            namespace=model_namespace.name,
        )
        snapshot = baselines.get("serving_runtime_snapshot")
        assert snapshot is not None, (
            f"Missing 'serving_runtime_snapshot' key in baseline ConfigMap for namespace "
            f"'{model_namespace.name}'. Ensure pre-upgrade tests ran successfully."
        )
        return snapshot

    runtime = multi_model_serving_runtime
    runtime.reload()
    snapshot = _capture_serving_runtime_snapshot(runtime=runtime)
    save_baseline_to_configmap(
        client=admin_client,
        namespace=model_namespace.name,
        baselines={"serving_runtime_snapshot": snapshot},
    )
    return snapshot
