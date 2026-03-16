from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
import yaml
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.storage_class import StorageClass
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger

from utilities.constants import (
    DscComponents,
    KServeDeploymentType,
    Labels,
    ModelAndFormat,
    ModelFormat,
    RuntimeTemplates,
    StorageClassName,
)
from utilities.data_science_cluster_utils import (
    get_dsc_ready_condition,
    wait_for_dsc_reconciliation,
)
from utilities.inference_utils import create_isvc
from utilities.infra import (
    s3_endpoint_secret,
    update_configmap_data,
)
from utilities.kueue_utils import (
    ClusterQueue,
    LocalQueue,
    ResourceFlavor,
    create_cluster_queue,
    create_local_queue,
    create_resource_flavor,
    wait_for_kueue_crds_available,
)
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def models_endpoint_s3_secret(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret, Any, Any]:
    with s3_endpoint_secret(
        client=unprivileged_client,
        name="models-bucket-secret",
        namespace=unprivileged_model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_bucket=models_s3_bucket_name,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret


# HTTP model serving
@pytest.fixture(scope="class")
def model_service_account(
    unprivileged_client: DynamicClient, models_endpoint_s3_secret: Secret
) -> Generator[ServiceAccount, Any, Any]:
    with ServiceAccount(
        client=unprivileged_client,
        namespace=models_endpoint_s3_secret.namespace,
        name="models-bucket-sa",
        secrets=[{"name": models_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def serving_runtime_from_template(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    runtime_kwargs = {
        "client": unprivileged_client,
        "name": request.param["name"],
        "namespace": unprivileged_model_namespace.name,
        "template_name": request.param["template-name"],
        "multi_model": request.param["multi-model"],
        "models_priorities": request.param.get("models-priorities"),
        "supported_model_formats": request.param.get("supported-model-formats"),
    }

    if (enable_http := request.param.get("enable-http")) is not None:
        runtime_kwargs["enable_http"] = enable_http

    if (enable_grpc := request.param.get("enable-grpc")) is not None:
        runtime_kwargs["enable_grpc"] = enable_grpc

    with ServingRuntimeFromTemplate(**runtime_kwargs) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def s3_models_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    serving_runtime_from_template: ServingRuntime,
    models_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": unprivileged_client,
        "name": request.param["name"],
        "namespace": unprivileged_model_namespace.name,
        "runtime": serving_runtime_from_template.name,
        "model_format": serving_runtime_from_template.instance.spec.supportedModelFormats[0].name,
        "deployment_mode": request.param["deployment-mode"],
        "storage_key": models_endpoint_s3_secret.name,
        "storage_path": request.param["model-dir"],
    }

    if (external_route := request.param.get("external-route")) is not None:
        isvc_kwargs["external_route"] = external_route

    if (enable_auth := request.param.get("enable-auth")) is not None:
        isvc_kwargs["enable_auth"] = enable_auth

    if (scale_metric := request.param.get("scale-metric")) is not None:
        isvc_kwargs["scale_metric"] = scale_metric

    if (scale_target := request.param.get("scale-target")) is not None:
        isvc_kwargs["scale_target"] = scale_target

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture(scope="function")
def s3_models_inference_service_patched_annotations(
    request: FixtureRequest, s3_models_inference_service: InferenceService
) -> InferenceService:
    if hasattr(request, "param"):
        with ResourceEditor(
            patches={
                s3_models_inference_service: {
                    "metadata": {
                        "annotations": request.param["annotations"],
                    }
                }
            }
        ):
            yield s3_models_inference_service


@pytest.fixture(scope="class")
def model_pvc(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    access_mode = "ReadWriteOnce"
    pvc_kwargs = {
        "name": "model-pvc",
        "namespace": unprivileged_model_namespace.name,
        "client": unprivileged_client,
        "size": request.param["pvc-size"],
    }
    if hasattr(request, "param"):
        access_mode = request.param.get("access-modes")

        if storage_class_name := request.param.get("storage-class-name"):
            pvc_kwargs["storage_class"] = storage_class_name

    pvc_kwargs["accessmodes"] = access_mode

    with PersistentVolumeClaim(**pvc_kwargs) as pvc:
        pvc.wait_for_status(status=pvc.Status.BOUND, timeout=120)
        yield pvc


@pytest.fixture(scope="session")
def skip_if_no_nfs_storage_class(admin_client: DynamicClient) -> None:
    if not StorageClass(client=admin_client, name=StorageClassName.NFS).exists:
        pytest.skip(f"StorageClass {StorageClassName.NFS} is missing from the cluster")


@pytest.fixture(scope="class")
def ovms_kserve_serving_runtime(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    runtime_kwargs = {
        "client": unprivileged_client,
        "namespace": unprivileged_model_namespace.name,
        "name": request.param["runtime-name"],
        "template_name": RuntimeTemplates.OVMS_KSERVE,
        "multi_model": False,
        "resources": {
            ModelFormat.OVMS: {
                "requests": {"cpu": "1", "memory": "4Gi"},
                "limits": {"cpu": "2", "memory": "8Gi"},
            }
        },
    }

    if model_format_name := request.param.get("model-format"):
        runtime_kwargs["model_format_name"] = model_format_name

    if supported_model_formats := request.param.get("supported-model-formats"):
        runtime_kwargs["supported_model_formats"] = supported_model_formats

    if runtime_image := request.param.get("runtime-image"):
        runtime_kwargs["runtime_image"] = runtime_image

    with ServingRuntimeFromTemplate(**runtime_kwargs) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def ci_endpoint_s3_secret(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
) -> Generator[Secret, Any, Any]:
    with s3_endpoint_secret(
        client=unprivileged_client,
        name="ci-bucket-secret",
        namespace=unprivileged_model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=ci_s3_bucket_region,
        aws_s3_bucket=ci_s3_bucket_name,
        aws_s3_endpoint=ci_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def ovms_kserve_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ovms_kserve_serving_runtime: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    deployment_mode = request.param["deployment-mode"]
    isvc_kwargs = {
        "client": unprivileged_client,
        "name": f"{request.param['name']}-{deployment_mode.lower()}",
        "namespace": unprivileged_model_namespace.name,
        "runtime": ovms_kserve_serving_runtime.name,
        "storage_path": request.param["model-dir"],
        "storage_key": ci_endpoint_s3_secret.name,
        "model_format": ModelAndFormat.OPENVINO_IR,
        "deployment_mode": deployment_mode,
        "model_version": request.param["model-version"],
    }

    if env_vars := request.param.get("env-vars"):
        isvc_kwargs["model_env_variables"] = env_vars

    if (min_replicas := request.param.get("min-replicas")) is not None:
        isvc_kwargs["min_replicas"] = min_replicas
        if min_replicas == 0:
            isvc_kwargs["wait_for_predictor_pods"] = False

    if max_replicas := request.param.get("max-replicas"):
        isvc_kwargs["max_replicas"] = max_replicas

    if scale_metric := request.param.get("scale-metric"):
        isvc_kwargs["scale_metric"] = scale_metric

    if (scale_target := request.param.get("scale-target")) is not None:
        isvc_kwargs["scale_target"] = scale_target

    isvc_kwargs["stop_resume"] = request.param.get("stop", False)

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def ovms_raw_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ovms_kserve_serving_runtime: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name=f"{request.param['name']}-raw",
        namespace=unprivileged_model_namespace.name,
        external_route=True,
        runtime=ovms_kserve_serving_runtime.name,
        storage_path=request.param["model-dir"],
        storage_key=ci_endpoint_s3_secret.name,
        model_format=ModelAndFormat.OPENVINO_IR,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_version=request.param["model-version"],
        stop_resume=request.param.get("stop", False),
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def user_workload_monitoring_config_map(
    admin_client: DynamicClient, cluster_monitoring_config: ConfigMap
) -> Generator[ConfigMap]:
    uwm_namespace = "openshift-user-workload-monitoring"

    data = {
        "config.yaml": yaml.dump({
            "prometheus": {
                "logLevel": "debug",
                "retention": "15d",
                "volumeClaimTemplate": {"spec": {"resources": {"requests": {"storage": "40Gi"}}}},
            }
        })
    }

    with update_configmap_data(
        client=admin_client,
        name="user-workload-monitoring-config",
        namespace=uwm_namespace,
        data=data,
    ) as cm:
        yield cm

    # UWM PVCs are not deleted once the configmap is deleted; forcefully deleting the PVCs to avoid having left-overs
    for pvc in PersistentVolumeClaim.get(client=admin_client, namespace=uwm_namespace):
        pvc.clean_up()


@pytest.fixture(scope="class")
def model_car_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    serving_runtime_from_template: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    deployment_mode = request.param.get("deployment-mode", KServeDeploymentType.RAW_DEPLOYMENT)
    with create_isvc(
        client=unprivileged_client,
        name=f"model-car-{deployment_mode.lower()}",
        namespace=unprivileged_model_namespace.name,
        runtime=serving_runtime_from_template.name,
        storage_uri=request.param["storage-uri"],
        model_format=serving_runtime_from_template.instance.spec.supportedModelFormats[0].name,
        deployment_mode=deployment_mode,
        external_route=request.param.get("external-route", True),
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="session")
def skip_if_no_gpu_available(gpu_count_on_cluster: int) -> None:
    """Skip test if no GPUs are available on the cluster."""
    if gpu_count_on_cluster < 1:
        pytest.skip("No GPUs available on cluster, skipping GPU test")


@pytest.fixture(scope="class")
def gpu_model_car_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    serving_runtime_from_template: ServingRuntime,
    gpu_count_on_cluster: int,
) -> Generator[InferenceService, Any, Any]:
    """Create a GPU-accelerated model car inference service."""
    from copy import deepcopy

    from tests.model_serving.model_runtime.openvino.constant import PREDICT_RESOURCES

    deployment_mode = request.param.get("deployment-mode", KServeDeploymentType.RAW_DEPLOYMENT)
    gpu_count = request.param.get("gpu-count", 1)

    if gpu_count_on_cluster < gpu_count:
        pytest.skip(f"Not enough GPUs available. Required: {gpu_count}, Available: {gpu_count_on_cluster}")

    resources = deepcopy(x=PREDICT_RESOURCES["resources"])
    resources["requests"][Labels.Nvidia.NVIDIA_COM_GPU] = str(gpu_count)
    resources["limits"][Labels.Nvidia.NVIDIA_COM_GPU] = str(gpu_count)

    with create_isvc(
        client=unprivileged_client,
        name=f"gpu-model-car-{deployment_mode.lower()}",
        namespace=unprivileged_model_namespace.name,
        runtime=serving_runtime_from_template.name,
        storage_uri=request.param["storage-uri"],
        model_format=serving_runtime_from_template.instance.spec.supportedModelFormats[0].name,
        deployment_mode=deployment_mode,
        external_route=request.param.get("external-route", True),
        wait_for_predictor_pods=False,
        resources=resources,
        volumes=deepcopy(x=PREDICT_RESOURCES["volumes"]),
        volumes_mounts=deepcopy(x=PREDICT_RESOURCES["volume_mounts"]),
    ) as isvc:
        yield isvc


# Kueue Fixtures
def _is_kueue_operator_installed(admin_client: DynamicClient) -> bool:
    """Check if the Kueue operator is installed and ready."""
    try:
        csvs = list(
            ClusterServiceVersion.get(
                client=admin_client,
                namespace=py_config.get("applications_namespace", "openshift-operators"),
            )
        )
        for csv in csvs:
            if csv.name.startswith("kueue") and csv.status == csv.Status.SUCCEEDED:
                LOGGER.info(f"Found Kueue operator CSV: {csv.name}")
                return True
        return False
    except ResourceNotFoundError:
        return False


@pytest.fixture(scope="session")
def ensure_kueue_unmanaged_in_dsc(
    admin_client: DynamicClient, dsc_resource: DataScienceCluster
) -> Generator[None, Any]:
    """Set DSC Kueue to Unmanaged and wait for CRDs to be available."""
    try:
        if not _is_kueue_operator_installed(admin_client):
            pytest.skip("Kueue operator is not installed, skipping Kueue tests")

        # Check current Kueue state
        kueue_management_state = dsc_resource.instance.spec.components[DscComponents.KUEUE].managementState

        with ExitStack() as stack:
            # Only patch if Kueue is not already Unmanaged
            if kueue_management_state != DscComponents.ManagementState.UNMANAGED:
                LOGGER.info(f"Patching Kueue from {kueue_management_state} to Unmanaged")
                # Read timestamp BEFORE applying patch
                ready_condition = get_dsc_ready_condition(dsc=dsc_resource)
                pre_patch_time = ready_condition.get("lastTransitionTime") if ready_condition else None

                dsc_dict = {
                    "spec": {
                        "components": {
                            DscComponents.KUEUE: {"managementState": DscComponents.ManagementState.UNMANAGED}
                        }
                    }
                }
                stack.enter_context(cm=ResourceEditor(patches={dsc_resource: dsc_dict}))

                # Wait for DSC to reconcile the patch
                wait_for_dsc_reconciliation(dsc=dsc_resource, baseline_time=pre_patch_time)
            else:
                LOGGER.info("Kueue already Unmanaged, no patch needed")

            # Always wait for Kueue CRDs and controller pods (regardless of patch)
            wait_for_kueue_crds_available(client=admin_client)
            yield

    except (AttributeError, KeyError) as e:
        pytest.skip(f"Kueue component not found in DSC: {e}")


def kueue_resource_groups(
    flavor_name: str,
    cpu_quota: int,
    memory_quota: str,
) -> list[dict[str, Any]]:
    return [
        {
            "coveredResources": ["cpu", "memory"],
            "flavors": [
                {
                    "name": flavor_name,
                    "resources": [
                        {"name": "cpu", "nominalQuota": cpu_quota},
                        {"name": "memory", "nominalQuota": memory_quota},
                    ],
                }
            ],
        }
    ]


@pytest.fixture(scope="class")
def kueue_cluster_queue_from_template(
    request: FixtureRequest,
    admin_client: DynamicClient,
    ensure_kueue_unmanaged_in_dsc,
) -> Generator[ClusterQueue, Any]:
    if request.param.get("name") is None:
        raise ValueError("name is required")
    with create_cluster_queue(
        name=request.param.get("name"),
        client=admin_client,
        resource_groups=kueue_resource_groups(
            request.param.get("resource_flavor_name"), request.param.get("cpu_quota"), request.param.get("memory_quota")
        ),
        namespace_selector=request.param.get("namespace_selector", {}),
    ) as cluster_queue:
        yield cluster_queue


@pytest.fixture(scope="class")
def kueue_resource_flavor_from_template(
    request: FixtureRequest,
    admin_client: DynamicClient,
    ensure_kueue_unmanaged_in_dsc,
) -> Generator[ResourceFlavor, Any]:
    if request.param.get("name") is None:
        raise ValueError("name is required")
    with create_resource_flavor(
        name=request.param.get("name"),
        client=admin_client,
    ) as resource_flavor:
        yield resource_flavor


@pytest.fixture(scope="class")
def kueue_local_queue_from_template(
    request: FixtureRequest,
    unprivileged_model_namespace: Namespace,
    admin_client: DynamicClient,
    ensure_kueue_unmanaged_in_dsc,
) -> Generator[LocalQueue, Any]:
    if request.param.get("name") is None:
        raise ValueError("name is required")
    if request.param.get("cluster_queue") is None:
        raise ValueError("cluster_queue is required")
    with create_local_queue(
        name=request.param.get("name"),
        namespace=unprivileged_model_namespace.name,
        cluster_queue=request.param.get("cluster_queue"),
        client=admin_client,
    ) as local_queue:
        yield local_queue
