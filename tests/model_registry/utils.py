from typing import Any
from simple_logger.logger import get_logger

from kubernetes.dynamic import DynamicClient
from timeout_sampler import TimeoutSampler, TimeoutExpiredError
from model_registry import ModelRegistry as ModelRegistryClient
from model_registry.types import RegisteredModel
from ocp_resources.pod import Pod
from ocp_resources.service import Service
from ocp_resources.model_registry import ModelRegistry
from kubernetes.dynamic.exceptions import ResourceNotFoundError, NotFoundError

from tests.model_registry.constants import MODEL_NAME, MODEL_DESCRIPTION
from utilities.exceptions import ProtocolNotSupportedError, TooManyServicesError
from utilities.constants import Protocols, ModelFormat

ADDRESS_ANNOTATION_PREFIX: str = "routing.opendatahub.io/external-address-"

LOGGER = get_logger(name=__name__)


def get_mr_service_by_label(client: DynamicClient, ns: str, mr_instance: ModelRegistry) -> Service:
    """
    Args:
        client (DynamicClient): OCP Client to use.
        ns (str): Namespace name where to find the Service
        mr_instance (ModelRegistry): Model Registry instance

    Returns:
        Service: The matching Service

    Raises:
        ResourceNotFoundError: if no service is found.
    """
    if svc := [
        svcs
        for svcs in Service.get(
            dyn_client=client,
            namespace=ns,
            label_selector=f"app={mr_instance.name},component=model-registry",
        )
    ]:
        if len(svc) == 1:
            return svc[0]
        raise TooManyServicesError(svc)
    raise ResourceNotFoundError(f"{mr_instance.name} has no Service")


def get_endpoint_from_mr_service(svc: Service, protocol: str) -> str:
    if protocol in (Protocols.REST, Protocols.GRPC):
        return svc.instance.metadata.annotations[f"{ADDRESS_ANNOTATION_PREFIX}{protocol}"]
    else:
        raise ProtocolNotSupportedError(protocol)


def get_pod_container_error_status(pod: Pod) -> str | None:
    """
    Check container error status for a given pod and if any containers is in waiting state, return that information
    """
    pod_instance_status = pod.instance.status
    for container_status in pod_instance_status.get("containerStatuses", []):
        if waiting_container := container_status.get("state", {}).get("waiting"):
            return waiting_container["reason"] if waiting_container.get("reason") else waiting_container
    return ""


def get_not_running_pods(pods: list[Pod]) -> list[dict[str, Any]]:
    # Gets all the non-running pods from a given namespace.
    # Note: We need to keep track of pods marked for deletion as not running. This would ensure any
    # pod that was spun up in place of pod marked for deletion, are not ignored
    pods_not_running = []
    try:
        for pod in pods:
            pod_instance = pod.instance
            if container_status_error := get_pod_container_error_status(pod=pod):
                pods_not_running.append({pod.name: container_status_error})

            if pod_instance.metadata.get("deletionTimestamp") or pod_instance.status.phase not in (
                pod.Status.RUNNING,
                pod.Status.SUCCEEDED,
            ):
                pods_not_running.append({pod.name: pod.status})
    except (ResourceNotFoundError, NotFoundError) as exc:
        LOGGER.warning("Ignoring pod that disappeared during cluster sanity check: %s", exc)
    return pods_not_running


def wait_for_pods_running(
    admin_client: DynamicClient,
    namespace_name: str,
    number_of_consecutive_checks: int = 1,
) -> bool | None:
    """
    Waits for all pods in a given namespace to reach Running/Completed state. To avoid catching all pods in running
    state too soon, use number_of_consecutive_checks with appropriate values.
    """
    samples = TimeoutSampler(
        wait_timeout=180,
        sleep=5,
        func=get_not_running_pods,
        pods=list(Pod.get(dyn_client=admin_client, namespace=namespace_name)),
        exceptions_dict={NotFoundError: [], ResourceNotFoundError: []},
    )
    sample = None
    try:
        current_check = 0
        for sample in samples:
            if not sample:
                current_check += 1
                if current_check >= number_of_consecutive_checks:
                    return True
            else:
                current_check = 0
    except TimeoutExpiredError:
        if sample:
            LOGGER.error(
                f"timeout waiting for all pods in namespace {namespace_name} to reach "
                f"running state, following pods are in not running state: {sample}"
            )
            raise
    return None


def register_model(model_registry_client: ModelRegistryClient) -> RegisteredModel:
    return model_registry_client.register_model(
        name=MODEL_NAME,
        uri="https://storage-place.my-company.com",
        version="2.0.0",
        description=MODEL_DESCRIPTION,
        model_format_name=ModelFormat.ONNX,
        model_format_version="1",
        storage_key="my-data-connection",
        storage_path="path/to/model",
        metadata={
            "int_key": 1,
            "bool_key": False,
            "float_key": 3.14,
            "str_key": "str_value",
        },
    )


def get_and_validate_registered_model(
    model_registry_client: ModelRegistryClient,
    model_name: str,
    registered_model: RegisteredModel,
) -> None:
    """
    Get and validate a registered model.
    """
    model = model_registry_client.get_registered_model(name=model_name)
    expected_attrs = {
        "id": registered_model.id,
        "name": registered_model.name,
        "description": registered_model.description,
        "owner": registered_model.owner,
        "state": registered_model.state,
    }
    LOGGER.info(f"Expected: {expected_attrs}")
    errors = [
        f"Unexpected {attr} expected: {expected}, received {getattr(model, attr)}"
        for attr, expected in expected_attrs.items()
        if getattr(model, attr) != expected
    ]
    assert not errors, f"Model Registry validation failed with error: {errors}"


class ModelRegistryV1Alpha1(ModelRegistry):
    api_version = f"{ModelRegistry.ApiGroup.MODELREGISTRY_OPENDATAHUB_IO}/{ModelRegistry.ApiVersion.V1ALPHA1}"