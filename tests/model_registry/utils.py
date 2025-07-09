import json
from typing import Any, List

import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from ocp_resources.service import Service
from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError, TimeoutSampler
from kubernetes.dynamic.exceptions import NotFoundError
from tests.model_registry.constants import MR_DB_IMAGE_DIGEST
from tests.model_registry.exceptions import ModelRegistryResourceNotFoundError
from utilities.exceptions import ProtocolNotSupportedError, TooManyServicesError
from utilities.constants import Protocols, Annotations
from model_registry import ModelRegistry as ModelRegistryClient
from model_registry.types import RegisteredModel

ADDRESS_ANNOTATION_PREFIX: str = "routing.opendatahub.io/external-address-"

LOGGER = get_logger(name=__name__)


def get_mr_service_by_label(client: DynamicClient, namespace_name: str, mr_instance: ModelRegistry) -> Service:
    """
    Args:
        client (DynamicClient): OCP Client to use.
        namespace_name (str): Namespace name associated with the service
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
            namespace=namespace_name,
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


def generate_namespace_name(file_path: str) -> str:
    return (file_path.removesuffix(".py").replace("/", "-").replace("_", "-"))[-63:].split("-", 1)[-1]


def add_mysql_certs_volumes_to_deployment(
    spec: dict[str, Any],
    ca_configmap_name: str,
) -> list[dict[str, Any]]:
    """
    Adds the MySQL certs volumes to the deployment.

    Args:
        spec: The spec of the deployment
        ca_configmap_name: The name of the CA configmap

    Returns:
        The volumes with the MySQL certs volumes added
    """

    volumes = list(spec["volumes"])
    volumes.extend([
        {"name": ca_configmap_name, "configMap": {"name": ca_configmap_name}},
        {"name": "mysql-server-cert", "secret": {"secretName": "mysql-server-cert"}},  # pragma: allowlist secret
        {"name": "mysql-server-key", "secret": {"secretName": "mysql-server-key"}},  # pragma: allowlist secret
    ])

    return volumes


def apply_mysql_args_and_volume_mounts(
    my_sql_container: dict[str, Any],
    ca_configmap_name: str,
    ca_mount_path: str,
) -> dict[str, Any]:
    """
    Applies the MySQL args and volume mounts to the MySQL container.

    Args:
        my_sql_container: The MySQL container
        ca_configmap_name: The name of the CA configmap
        ca_mount_path: The mount path of the CA

    Returns:
        The MySQL container with the MySQL args and volume mounts applied
    """

    mysql_args = list(my_sql_container.get("args", []))
    mysql_args.extend([
        f"--ssl-ca={ca_mount_path}/ca/ca-bundle.crt",
        f"--ssl-cert={ca_mount_path}/server_cert/tls.crt",
        f"--ssl-key={ca_mount_path}/server_key/tls.key",
    ])

    volumes_mounts = list(my_sql_container.get("volumeMounts", []))
    volumes_mounts.extend([
        {"name": ca_configmap_name, "mountPath": f"{ca_mount_path}/ca", "readOnly": True},
        {
            "name": "mysql-server-cert",
            "mountPath": f"{ca_mount_path}/server_cert",
            "readOnly": True,
        },
        {
            "name": "mysql-server-key",
            "mountPath": f"{ca_mount_path}/server_key",
            "readOnly": True,
        },
    ])

    my_sql_container["args"] = mysql_args
    my_sql_container["volumeMounts"] = volumes_mounts
    return my_sql_container


def get_and_validate_registered_model(
    model_registry_client: ModelRegistryClient,
    model_name: str,
    registered_model: RegisteredModel = None,
) -> List[str]:
    """
    Get and validate a registered model.
    """
    model = model_registry_client.get_registered_model(name=model_name)
    if registered_model is not None:
        expected_attrs = {
            "id": registered_model.id,
            "name": registered_model.name,
            "description": registered_model.description,
            "owner": registered_model.owner,
            "state": registered_model.state,
        }
    else:
        expected_attrs = {
            "name": model_name,
        }
    errors = [
        f"Unexpected {attr} expected: {expected}, received {getattr(model, attr)}"
        for attr, expected in expected_attrs.items()
        if getattr(model, attr) != expected
    ]
    return errors


def execute_model_registry_get_command(url: str, headers: dict[str, str], json_output: bool = True) -> dict[Any, Any]:
    """
    Executes model registry get commands against model registry rest end point

    Args:
        url (str): Model registry endpoint for rest calls
        headers (dict[str, str]): HTTP headers for get calls
        json_output(bool): Whether to output JSON response

    Returns: json output or dict of raw output.
    """
    resp = requests.get(url=url, headers=headers, verify=False)
    LOGGER.info(f"url: {url}, status code: {resp.status_code}, rep: {resp.text}")
    if resp.status_code not in [200, 201]:
        raise ModelRegistryResourceNotFoundError(
            f"Failed to get ModelRegistry resource: {url}, {resp.status_code}: {resp.text}"
        )
    if json_output:
        try:
            return json.loads(resp.text)
        except json.JSONDecodeError:
            LOGGER.error(f"Unable to parse {resp.text}")
            raise
    else:
        return {"raw_output": resp.text}
