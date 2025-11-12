from typing import Generator, Any, Dict

import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from _pytest.fixtures import FixtureRequest
from utilities.kueue_utils import (
    create_local_queue,
    create_cluster_queue,
    create_resource_flavor,
    LocalQueue,
    ClusterQueue,
    ResourceFlavor,
)
from ocp_resources.namespace import Namespace
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.resource import ResourceEditor
from utilities.constants import ModelAndFormat, KServeDeploymentType, DscComponents
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate
from ocp_resources.secret import Secret
from ocp_resources.inference_service import InferenceService
from ocp_resources.serving_runtime import ServingRuntime
from utilities.constants import RuntimeTemplates, ModelFormat
from pytest_testconfig import config as py_config
import logging

BASIC_LOGGER = logging.getLogger(name="basic")


def _is_kueue_operator_installed(admin_client: DynamicClient) -> bool:
    try:
        csvs = list(
            ClusterServiceVersion.get(
                dyn_client=admin_client,
                namespace=py_config.get("applications_namespace", "openshift-operators"),
            )
        )
        for csv in csvs:
            if csv.name.startswith("kueue") and csv.status == csv.Status.SUCCEEDED:
                BASIC_LOGGER.info(f"Found Kueue operator CSV: {csv.name}")
                return True
        return False
    except ResourceNotFoundError:
        return False


@pytest.fixture(scope="session", autouse=True)
def ensure_kueue_unmanaged_in_dsc(
    admin_client: DynamicClient, dsc_resource: DataScienceCluster
) -> Generator[None, Any, None]:
    try:
        if not _is_kueue_operator_installed(admin_client):
            pytest.skip("Kueue operator is not installed, skipping Kueue tests")

        dsc_resource.get()
        kueue_management_state = dsc_resource.instance.spec.components[DscComponents.KUEUE].managementState

        if kueue_management_state == DscComponents.ManagementState.UNMANAGED:
            BASIC_LOGGER.info("Kueue is already Unmanaged in DSC, proceeding with tests")
            yield
        else:
            BASIC_LOGGER.info(f"Kueue management state is {kueue_management_state}, updating to Unmanaged")
            dsc_dict = {
                "spec": {
                    "components": {DscComponents.KUEUE: {"managementState": DscComponents.ManagementState.UNMANAGED}}
                }
            }

            with ResourceEditor(patches={dsc_resource: dsc_dict}):
                BASIC_LOGGER.info("Updated Kueue to Unmanaged, waiting for DSC to be ready")
                dsc_resource.wait_for_condition(condition="Ready", status="True", timeout=300)
                BASIC_LOGGER.info("DSC is ready, proceeding with tests")
                yield

            BASIC_LOGGER.info(f"Restoring Kueue management state to {kueue_management_state}")
            restore_dict = {"spec": {"components": {DscComponents.KUEUE: {"managementState": kueue_management_state}}}}
            with ResourceEditor(patches={dsc_resource: restore_dict}):
                dsc_resource.wait_for_condition(condition="Ready", status="True", timeout=300)
                BASIC_LOGGER.info("Restored Kueue management state")

    except (AttributeError, KeyError) as e:
        pytest.skip(f"Kueue component not found in DSC: {e}")


def kueue_resource_groups(
    flavor_name: str,
    cpu_quota: int,
    memory_quota: str,
) -> list[Dict[str, Any]]:
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
) -> Generator[ClusterQueue, Any, Any]:
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
) -> Generator[ResourceFlavor, Any, Any]:
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
) -> Generator[LocalQueue, Any, Any]:
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


@pytest.fixture(scope="class")
def kueue_raw_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    kueue_kserve_serving_runtime: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=f"{request.param['name']}-raw",
        namespace=unprivileged_model_namespace.name,
        external_route=True,
        runtime=kueue_kserve_serving_runtime.name,
        storage_path=request.param["model-dir"],
        storage_key=ci_endpoint_s3_secret.name,
        model_format=ModelAndFormat.OPENVINO_IR,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_version=request.param["model-version"],
        labels=request.param.get("labels", {}),
        resources=request.param.get(
            "resources", {"requests": {"cpu": "1", "memory": "8Gi"}, "limits": {"cpu": "2", "memory": "10Gi"}}
        ),
        min_replicas=request.param.get("min-replicas", 1),
        max_replicas=request.param.get("max-replicas", 2),
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def kueue_kserve_serving_runtime(
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
                "requests": {"cpu": "1", "memory": "8Gi"},
                "limits": {"cpu": "2", "memory": "10Gi"},
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
