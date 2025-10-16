import json
from typing import Any, List

import requests
from kubernetes.dynamic import DynamicClient

from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from simple_logger.logger import get_logger
from timeout_sampler import retry
from tests.model_registry.constants import (
    MR_DB_IMAGE_DIGEST,
    MODEL_REGISTRY_DB_SECRET_STR_DATA,
    MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
    DB_BASE_RESOURCES_NAME,
    OAUTH_PROXY_CONFIG_DICT,
    MARIADB_MY_CNF,
    PORT_MAP,
    MODEL_REGISTRY_POD_FILTER,
)
from tests.model_registry.exceptions import ModelRegistryResourceNotFoundError
from utilities.exceptions import ProtocolNotSupportedError, TooManyServicesError
from utilities.constants import Protocols, Annotations, Timeout
from model_registry import ModelRegistry as ModelRegistryClient
from model_registry.types import RegisteredModel


ADDRESS_ANNOTATION_PREFIX: str = "routing.opendatahub.io/external-address-"
MARIA_DB_IMAGE = (
    "registry.redhat.io/rhel9/mariadb-1011@sha256:5608cce9ca8fed81027c97336d526b80320b0f4517ca5d3d141c0bbd7d563f8a"
)
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


def get_model_registry_deployment_template_dict(
    secret_name: str, resource_name: str, db_backend: str
) -> dict[str, Any]:
    base_dict = {
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
                    "name": db_backend,
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
    if db_backend == "mariadb":
        base_dict["metadata"]["labels"]["app"] = db_backend
        base_dict["metadata"]["labels"]["component"] = "database"
        base_dict["spec"]["containers"][0]["image"] = MARIA_DB_IMAGE
        base_dict["spec"]["containers"][0]["env"].append({
            "name": "MARIADB_ROOT_PASSWORD",
            "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "database-password"}},
        })
        base_dict["spec"]["containers"][0]["volumeMounts"] = [
            {"mountPath": "/var/lib/mysql", "name": f"{db_backend}-data"},
            {
                "mountPath": "/etc/mysql/conf.d",
                "name": f"{db_backend}-config",
            },
        ]
        del base_dict["spec"]["containers"][0]["args"]
        base_dict["spec"]["volumes"] = [
            {
                "name": f"{db_backend}-data",
                "persistentVolumeClaim": {"claimName": resource_name},
            },
            {
                "name": f"{db_backend}-config",
                "configMap": {"name": resource_name},
            },
        ]
    return base_dict


def get_model_registry_db_label_dict(db_resource_name: str) -> dict[str, str]:
    return {
        Annotations.KubernetesIo.NAME: db_resource_name,
        Annotations.KubernetesIo.INSTANCE: db_resource_name,
        Annotations.KubernetesIo.PART_OF: db_resource_name,
    }


@retry(exceptions_dict={TimeoutError: []}, wait_timeout=Timeout.TIMEOUT_2MIN, sleep=5)
def wait_for_new_running_mr_pod(
    admin_client: DynamicClient,
    orig_pod_name: str,
    namespace: str,
    instance_name: str,
) -> Pod:
    """
    Wait for the model registry pod to be replaced.

    Args:
        admin_client (DynamicClient): The admin client.
        orig_pod_name (str): The name of the original pod.
        namespace (str): The namespace of the pod.
        instance_name (str): The name of the instance.
    Returns:
        Pod object.

    Raises:
        TimeoutError: If the pods are not replaced.

    """
    LOGGER.info("Waiting for pod to be replaced")
    pods = list(
        Pod.get(
            dyn_client=admin_client,
            namespace=namespace,
            label_selector=MODEL_REGISTRY_POD_FILTER,
        )
    )
    if pods and len(pods) == 1:
        if pods[0].name != orig_pod_name and pods[0].status == Pod.Status.RUNNING:
            return pods[0]
    raise TimeoutError(f"Timeout waiting for pod {orig_pod_name} to be replaced")


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


def get_mr_service_objects(
    base_name: str,
    namespace: str,
    client: DynamicClient,
    teardown_resources: bool,
    num: int,
) -> list[Service]:
    services = []
    annotation = {
        "template.openshift.io/expose-uri": r"mysql://{.spec.clusterIP}:{.spec.ports[?(.name==\mysql\)].port}"
    }
    for num_service in range(0, num):
        name = f"{base_name}{num_service}"
        services.append(
            Service(
                client=client,
                name=name,
                namespace=namespace,
                ports=[
                    {
                        "name": "mysql",
                        "nodePort": 0,
                        "port": 3306,
                        "protocol": "TCP",
                        "appProtocol": "tcp",
                        "targetPort": 3306,
                    }
                ],
                selector={
                    "name": name,
                },
                label=get_model_registry_db_label_dict(db_resource_name=name),
                annotations=annotation,
                teardown=teardown_resources,
            )
        )
    return services


def get_mr_configmap_objects(
    base_name: str,
    namespace: str,
    client: DynamicClient,
    teardown_resources: bool,
    num: int,
    db_backend: str,
) -> list[Service]:
    config_maps = []
    if db_backend == "mariadb":
        for num_config_map in range(0, num):
            name = f"{base_name}{num_config_map}"
            config_maps.append(
                ConfigMap(
                    client=client,
                    name=name,
                    namespace=namespace,
                    data={"my.cnf": MARIADB_MY_CNF},
                    label=get_model_registry_db_label_dict(db_resource_name=name),
                    teardown=teardown_resources,
                )
            )
    return config_maps


def get_mr_pvc_objects(
    base_name: str, namespace: str, client: DynamicClient, teardown_resources: bool, num: int
) -> list[PersistentVolumeClaim]:
    pvcs = []
    for num_pvc in range(0, num):
        name = f"{base_name}{num_pvc}"
        pvcs.append(
            PersistentVolumeClaim(
                accessmodes="ReadWriteOnce",
                name=name,
                namespace=namespace,
                client=client,
                size="5Gi",
                label=get_model_registry_db_label_dict(db_resource_name=name),
                teardown=teardown_resources,
            )
        )
    return pvcs


def get_mr_secret_objects(
    base_name: str, namespace: str, client: DynamicClient, teardown_resources: bool, num: int
) -> list[Secret]:
    secrets = []
    for num_secret in range(0, num):
        name = f"{base_name}{num_secret}"
        secrets.append(
            Secret(
                client=client,
                name=name,
                namespace=namespace,
                string_data=MODEL_REGISTRY_DB_SECRET_STR_DATA,
                label=get_model_registry_db_label_dict(db_resource_name=name),
                annotations=MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
                teardown=teardown_resources,
            )
        )
    return secrets


def get_mr_deployment_objects(
    base_name: str,
    namespace: str,
    client: DynamicClient,
    teardown_resources: bool,
    db_backend: str,
    num: int,
) -> list[Deployment]:
    deployments = []

    for num_deployment in range(0, num):
        name = f"{base_name}{num_deployment}"
        selectors = {"matchLabels": {"name": name}}
        if db_backend == "mariadb":
            selectors["matchLabels"]["app"] = db_backend
            selectors["matchLabels"]["component"] = "database"
        secret_name = f"{DB_BASE_RESOURCES_NAME}{num_deployment}"
        deployments.append(
            Deployment(
                name=name,
                client=client,
                namespace=namespace,
                annotations={
                    "template.alpha.openshift.io/wait-for-ready": "true",
                },
                label=get_model_registry_db_label_dict(db_resource_name=name),
                replicas=1,
                revision_history_limit=0,
                selector=selectors,
                strategy={"type": "Recreate"},
                template=get_model_registry_deployment_template_dict(
                    secret_name=secret_name, resource_name=name, db_backend=db_backend
                ),
                wait_for_resource=True,
                teardown=teardown_resources,
            )
        )
    return deployments


def get_mr_standard_labels(resource_name: str) -> dict[str, str]:
    return {
        Annotations.KubernetesIo.NAME: resource_name,
        Annotations.KubernetesIo.INSTANCE: resource_name,
        Annotations.KubernetesIo.PART_OF: resource_name,
        Annotations.KubernetesIo.CREATED_BY: resource_name,
    }


def get_model_registry_objects(
    base_name: str,
    namespace: str,
    client: DynamicClient,
    teardown_resources: bool,
    params: dict[str, Any],
    num: int,
    db_backend: str,
) -> list[Any]:
    model_registry_objects = []
    for num_mr in range(0, num):
        name = f"{base_name}{num_mr}"
        mysql = get_mysql_config(
            base_name=f"{DB_BASE_RESOURCES_NAME}{num_mr}", namespace=namespace, db_backend=db_backend
        )
        if "sslRootCertificateConfigMap" in params:
            mysql["sslRootCertificateConfigMap"] = params["sslRootCertificateConfigMap"]
        model_registry_objects.append(
            ModelRegistry(
                client=client,
                name=name,
                namespace=namespace,
                label=get_mr_standard_labels(resource_name=name),
                grpc={},
                rest={},
                oauth_proxy=OAUTH_PROXY_CONFIG_DICT,
                mysql=mysql,
                wait_for_resource=True,
                teardown=teardown_resources,
            )
        )
    return model_registry_objects


def get_model_registry_metadata_resources(
    base_name: str,
    namespace: str,
    client: DynamicClient,
    teardown_resources: bool,
    num_resources: int,
    db_backend: str,
) -> dict[Any, Any]:
    return {
        Secret: get_mr_secret_objects(
            client=client,
            namespace=namespace,
            base_name=base_name,
            num=num_resources,
            teardown_resources=teardown_resources,
        ),
        PersistentVolumeClaim: get_mr_pvc_objects(
            client=client,
            namespace=namespace,
            base_name=base_name,
            num=num_resources,
            teardown_resources=teardown_resources,
        ),
        Service: get_mr_service_objects(
            client=client,
            namespace=namespace,
            base_name=base_name,
            num=num_resources,
            teardown_resources=teardown_resources,
        ),
        ConfigMap: get_mr_configmap_objects(
            client=client,
            namespace=namespace,
            base_name=base_name,
            num=num_resources,
            teardown_resources=teardown_resources,
            db_backend=db_backend,
        ),
        Deployment: get_mr_deployment_objects(
            client=client,
            namespace=namespace,
            base_name=base_name,
            num=num_resources,
            teardown_resources=teardown_resources,
            db_backend=db_backend,
        ),
    }


def get_mysql_config(base_name: str, namespace: str, db_backend: str) -> dict[str, Any]:
    return {
        "host": f"{base_name}.{namespace}.svc.cluster.local",
        "database": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
        "passwordSecret": {"key": "database-password", "name": base_name},
        "port": PORT_MAP[db_backend],
        "skipDBCreation": False,
        "username": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
    }


def validate_no_grpc_container(deployment_containers: list[dict[str, Any]]) -> None:
    grpc_container = None
    for container in deployment_containers:
        if "grpc" in container["name"]:
            grpc_container = container
    assert not grpc_container, f"GRPC container found: {grpc_container}"


def validate_mlmd_removal_in_model_registry_pod_log(
    deployment_containers: list[dict[str, Any]], pod_object: Pod
) -> None:
    errors = []
    embedmd_message = "EmbedMD service connected"
    for container in deployment_containers:
        container_name = container["name"]
        LOGGER.info(f"Checking {container_name}")
        log = pod_object.log(container=container_name)
        if "rest" in container_name:
            if embedmd_message not in log:
                errors.append(f"Missing {embedmd_message} in {container_name} log")
        if "MLMD" in log:
            errors.append(f"MLMD reference found in {container_name} log")
    assert not errors, f"Log validation failed with error(s): {errors}"


def get_model_catalog_pod(
    client: DynamicClient, model_registry_namespace: str, label_selector: str = "component=model-catalog"
) -> list[Pod]:
    return list(Pod.get(namespace=model_registry_namespace, label_selector=label_selector, dyn_client=client))


def get_rest_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "accept": "application/json",
        "Content-Type": "application/json",
    }
