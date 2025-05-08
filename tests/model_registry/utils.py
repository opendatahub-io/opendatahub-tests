import uuid
from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.service import Service
from ocp_resources.model_registry import ModelRegistry
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError, TimeoutSampler
from kubernetes.dynamic.exceptions import NotFoundError
from tests.model_registry.constants import MR_DB_IMAGE_DIGEST
from utilities.exceptions import ProtocolNotSupportedError, TooManyServicesError
from utilities.constants import Protocols, Annotations

ADDRESS_ANNOTATION_PREFIX: str = "routing.opendatahub.io/external-address-"

LOGGER = get_logger(name=__name__)


def get_mr_service_by_label(client: DynamicClient, ns: Namespace, mr_instance: ModelRegistry) -> Service:
    """
    Args:
        client (DynamicClient): OCP Client to use.
        ns (Namespace): Namespace object where to find the Service
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
            namespace=ns.name,
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


def get_model_registry_deployment_template_dict(secret_name: str, resource_name: str) -> dict[str, Any]:
    return {
        "metadata": {
            "labels": {
                "name": resource_name,
                "sidecar.istio.io/inject": "false",
            }
        },
        "spec": {
            "containers": [
                {
                    "env": [
                        {
                            "name": "MYSQL_USER",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "key": "database-user",
                                    "name": secret_name,
                                }
                            },
                        },
                        {
                            "name": "MYSQL_PASSWORD",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "key": "database-password",
                                    "name": secret_name,
                                }
                            },
                        },
                        {
                            "name": "MYSQL_ROOT_PASSWORD",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "key": "database-password",
                                    "name": secret_name,
                                }
                            },
                        },
                        {
                            "name": "MYSQL_DATABASE",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "key": "database-name",
                                    "name": secret_name,
                                }
                            },
                        },
                    ],
                    "args": [
                        "--datadir",
                        "/var/lib/mysql/datadir",
                        "--default-authentication-plugin=mysql_native_password",
                    ],
                    "image": MR_DB_IMAGE_DIGEST,
                    "imagePullPolicy": "IfNotPresent",
                    "livenessProbe": {
                        "exec": {
                            "command": [
                                "/bin/bash",
                                "-c",
                                "mysqladmin -u${MYSQL_USER} -p${MYSQL_ROOT_PASSWORD} ping",
                            ]
                        },
                        "initialDelaySeconds": 15,
                        "periodSeconds": 10,
                        "timeoutSeconds": 5,
                    },
                    "name": "mysql",
                    "ports": [{"containerPort": 3306, "protocol": "TCP"}],
                    "readinessProbe": {
                        "exec": {
                            "command": [
                                "/bin/bash",
                                "-c",
                                'mysql -D ${MYSQL_DATABASE} -u${MYSQL_USER} -p${MYSQL_ROOT_PASSWORD} -e "SELECT 1"',
                            ]
                        },
                        "initialDelaySeconds": 10,
                        "timeoutSeconds": 5,
                    },
                    "securityContext": {"capabilities": {}, "privileged": False},
                    "terminationMessagePath": "/dev/termination-log",
                    "volumeMounts": [
                        {
                            "mountPath": "/var/lib/mysql",
                            "name": f"{resource_name}-data",
                        }
                    ],
                }
            ],
            "dnsPolicy": "ClusterFirst",
            "restartPolicy": "Always",
            "volumes": [
                {
                    "name": f"{resource_name}-data",
                    "persistentVolumeClaim": {"claimName": resource_name},
                }
            ],
        },
    }


def get_model_registry_db_label_dict(db_resource_name: str) -> dict[str, str]:
    return {
        Annotations.KubernetesIo.NAME: db_resource_name,
        Annotations.KubernetesIo.INSTANCE: db_resource_name,
        Annotations.KubernetesIo.PART_OF: db_resource_name,
    }


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


def generate_random_name(prefix: str, length: int = 8) -> str:
    """
    Generates a name with a required prefix and a random suffix derived from a UUID.

    The length of the random suffix can be controlled, defaulting to 8 characters.
    The suffix is taken from the beginning of a V4 UUID's hex representation.

    Args:
        prefix (str): The required prefix for the generated name.
        ength (int, optional): The desired length for the UUID-derived suffix.
                               Defaults to 8. Must be between 1 and 32.

    Returns:
        str: A string in the format "prefix-uuid_suffix".

    Raises:
        ValueError: If prefix is empty, or if length is not between 1 and 32.
    """
    if not prefix:
        raise ValueError("Prefix cannot be empty or None.")
    if not isinstance(length, int) or not (1 <= length <= 32):
        raise ValueError("suffix_length must be an integer between 1 and 32.")
    # Generate a new random UUID (version 4)
    random_uuid = uuid.uuid4()
    # Use the first 'length' characters of the hexadecimal representation of the UUID as the suffix.
    # random_uuid.hex is 32 characters long.
    suffix = random_uuid.hex[:length]
    return f"{prefix}-{suffix}"


def generate_namespace_name(file_path: str) -> str:
    return (file_path.removesuffix(".py").replace("/", "-").replace("_", "-"))[-63:].split("-", 1)[-1]
