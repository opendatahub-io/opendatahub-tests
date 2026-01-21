import base64
import json
import time
from typing import Any, List, Dict

import requests
from kubernetes.dynamic import DynamicClient

from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from utilities.resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from simple_logger.logger import get_logger
from timeout_sampler import retry
from tests.model_registry.constants import (
    MR_DB_IMAGE_DIGEST,
    MODEL_REGISTRY_DB_SECRET_STR_DATA,
    MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
    DB_BASE_RESOURCES_NAME,
    MARIADB_MY_CNF,
    PORT_MAP,
    MODEL_REGISTRY_POD_FILTER,
    MR_POSTGRES_DB_OBJECT,
)
from tests.model_registry.exceptions import ModelRegistryResourceNotFoundError
from utilities.exceptions import ProtocolNotSupportedError, TooManyServicesError
from utilities.constants import Protocols, Annotations, Timeout, PodNotFound
from model_registry import ModelRegistry as ModelRegistryClient
from model_registry.types import RegisteredModel

from utilities.general import wait_for_pods_running
from utilities.user_utils import get_byoidc_issuer_url

ADDRESS_ANNOTATION_PREFIX: str = "routing.opendatahub.io/external-address-"
MARIA_DB_IMAGE = (
    "registry.redhat.io/rhel9/mariadb-1011@sha256:092407d87f8017bb444a462fb3d38ad5070429e94df7cf6b91d82697f36d0fa9"
)
LOGGER = get_logger(name=__name__)


class TransientUnauthorizedError(Exception):
    """Exception for transient 401 Unauthorized errors that should be retried."""

    pass


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
            client=client,
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
            client=admin_client,
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
                size="3Gi",
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
        mysql = None
        if db_backend != "default":
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
                rest={},
                kube_rbac_proxy={},
                mysql=mysql if mysql else None,
                postgres={"generateDeployment": True} if db_backend == "default" else None,
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
    client: DynamicClient, model_registry_namespace: str, label_selector: str = "app.kubernetes.io/name=model-catalog"
) -> list[Pod]:
    return list(Pod.get(namespace=model_registry_namespace, label_selector=label_selector, client=client))


def get_rest_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "accept": "application/json",
        "Content-Type": "application/json",
    }


def wait_for_model_catalog_pod_ready_after_deletion(
    client: DynamicClient, model_registry_namespace: str, consecutive_try: int = 6
) -> bool:
    model_catalog_pods = get_model_catalog_pod(
        client=client,
        model_registry_namespace=model_registry_namespace,
    )
    # We can wait for the pods to reflect updated catalog, however, deleting them ensures the updated config is
    # applied immediately.
    for pod in model_catalog_pods:
        pod.delete()
    # After the deletion, we need to wait for the pod to be spinned up and get to ready state.
    assert wait_for_model_catalog_pod_created(client=client, model_registry_namespace=model_registry_namespace)
    wait_for_pods_running(
        admin_client=client, namespace_name=model_registry_namespace, number_of_consecutive_checks=consecutive_try
    )
    return True


@retry(wait_timeout=30, sleep=5, exceptions_dict={PodNotFound: []})
def wait_for_model_catalog_pod_created(client: DynamicClient, model_registry_namespace: str) -> bool:
    pods = get_model_catalog_pod(client=client, model_registry_namespace=model_registry_namespace)
    if pods:
        return True
    raise PodNotFound("Model catalog pod not found")


def execute_get_call(
    url: str, headers: dict[str, str], verify: bool | str = False, params: dict[str, Any] | None = None
) -> requests.Response:
    LOGGER.info(f"Executing get call: {url}")
    if params:
        LOGGER.info(f"params: {params}")
    resp = requests.get(url=url, headers=headers, verify=verify, timeout=60, params=params)
    LOGGER.info(f"Encoded url from requests library: {resp.url}")
    if resp.status_code not in [200, 201]:
        # Raise custom exception for 401 errors that can be retried (OAuth/kube-rbac-proxy initialization)
        if resp.status_code == 401:
            raise TransientUnauthorizedError(f"Get call failed for resource: {url}, 401: {resp.text}")
        # Raise regular exception for other errors (400, 403, 404, etc.) that should fail immediately
        raise ResourceNotFoundError(f"Get call failed for resource: {url}, {resp.status_code}: {resp.text}")
    return resp


@retry(wait_timeout=90, sleep=5, exceptions_dict={ResourceNotFoundError: [], TransientUnauthorizedError: []})
def wait_for_model_catalog_api(url: str, headers: dict[str, str], verify: bool | str = False) -> requests.Response:
    """
    Wait for model catalog API to be ready and fully initialized checks both /sources and /models endpoints
    to ensure OAuth/kube-rbac-proxy is fully initialized.
    """
    LOGGER.info(f"Waiting for model catalog API at {url}sources")
    execute_get_call(url=f"{url}sources", headers=headers, verify=verify)
    LOGGER.info(f"Verifying model catalog API readiness at {url}models")

    return execute_get_call(url=f"{url}models", headers=headers, verify=verify)


def execute_get_command(
    url: str, headers: dict[str, str], verify: bool | str = False, params: dict[str, Any] | None = None
) -> dict[Any, Any]:
    resp = execute_get_call(url=url, headers=headers, verify=verify, params=params)
    try:
        return json.loads(resp.text)
    except json.JSONDecodeError:
        LOGGER.error(f"Unable to parse {resp.text}")
        raise


def get_sample_yaml_str(models: list[str]) -> str:
    model_str: str = ""
    for model in models:
        model_str += f"""
{get_model_str(model=model)}
"""
    return f"""source: Hugging Face
models:
{model_str}
"""


def validate_model_catalog_sources(
    model_catalog_sources_url: str, rest_headers: dict[str, str], expected_catalog_values: dict[str, str]
) -> None:
    results = execute_get_command(
        url=model_catalog_sources_url,
        headers=rest_headers,
    )["items"]
    LOGGER.info(f"Model catalog sources: {results}")
    ids_from_query = [result_entry["id"] for result_entry in results]
    ids_expected = [expected_entry["id"] for expected_entry in expected_catalog_values]
    LOGGER.info(f"IDs expected: {ids_expected}, IDs found: {ids_from_query}")
    assert set(ids_expected).issubset(set(ids_from_query)), f"Expected: {expected_catalog_values}. Actual: {results}"


def get_catalog_str(ids: list[str]) -> str:
    catalog_str: str = ""
    for index, id in enumerate(ids):
        catalog_str += f"""
- name: Sample Catalog {index}
  id: {id}
  type: yaml
  enabled: true
  properties:
    yamlCatalogPath: {id.replace("_", "-")}.yaml
"""
    return f"""catalogs:
{catalog_str}
"""


def get_model_str(model: str) -> str:
    current_time = int(time.time() * 1000)
    return f"""
- name: {model}
  description: test description.
  readme: |-
    # test read me information {model}
  provider: Mistral AI
  logo: temp placeholder logo
  license: apache-2.0
  licenseLink: https://www.apache.org/licenses/LICENSE-2.0.txt
  libraryName: transformers
  artifacts:
    - uri: https://huggingface.co/{model}/resolve/main/consolidated.safetensors
  createTimeSinceEpoch: \"{str(current_time - 10000)}\"
  lastUpdateTimeSinceEpoch: \"{str(current_time)}\"
"""


class ResourceNotDeleted(Exception):
    pass


@retry(wait_timeout=360, sleep=5, exceptions_dict={ResourceNotDeleted: []})
def wait_for_default_resource_cleanedup(admin_client: DynamicClient, namespace_name: str) -> bool:
    objects_not_deleted = []
    for kind in [Service, PersistentVolumeClaim, Deployment, Secret]:
        LOGGER.info(f"Checking if {kind} {MR_POSTGRES_DB_OBJECT[kind]} is deleted")
        kind_obj = kind(client=admin_client, namespace=namespace_name, name=MR_POSTGRES_DB_OBJECT[kind])
        if kind_obj.exists:
            objects_not_deleted.append(f"{kind_obj.kind} - {kind_obj.name}")
    if not objects_not_deleted:
        return True
    raise ResourceNotDeleted(f"Following objects are not deleted: {objects_not_deleted}")


def get_mr_user_token(admin_client: DynamicClient, user_credentials_rbac: dict[str, str]) -> str:
    url = f"{get_byoidc_issuer_url(admin_client=admin_client)}/protocol/openid-connect/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "python-requests"}

    data = {
        "username": user_credentials_rbac["username"],
        "password": user_credentials_rbac["password"],
        "grant_type": "password",
        "client_id": "oc-cli",
        "scope": "openid",
    }

    try:
        LOGGER.info(f"Requesting token for user {user_credentials_rbac['username']} in byoidc environment")
        response = requests.post(
            url=url,
            headers=headers,
            data=data,
            allow_redirects=True,
            timeout=30,
            verify=True,  # Set to False if you need to skip SSL verification
        )
        response.raise_for_status()
        json_response = response.json()

        # Validate that we got an access token
        if "id_token" not in json_response:
            LOGGER.error("Warning: No id_token in response")
            raise AssertionError(f"No id_token in response: {json_response}")
        return json_response["id_token"]
    except Exception as e:
        raise e


def get_byoidc_user_credentials(username: str = None) -> Dict[str, str]:
    """
    Get user credentials from byoidc-credentials secret.

    Args:
        username: Specific username to look up. If None, returns first user.

    Returns:
        Dictionary with username and password for the specified user.

    Raises:
        ValueError: If username not found or no users/passwords in secret.
        AssertionError: If users or passwords lists are empty.
    """
    credentials_secret = Secret(name="byoidc-credentials", namespace="oidc", ensure_exists=True)
    credential_data = credentials_secret.instance.data
    user_names = base64.b64decode(credential_data.users).decode().split(",")
    passwords = base64.b64decode(credential_data.passwords).decode().split(",")

    # Assert that both lists are not empty
    assert user_names and user_names != [""], "No usernames found in byoidc-credentials secret"
    assert passwords and passwords != [""], "No passwords found in byoidc-credentials secret"

    # Use specified username or default to first user
    requested_username = username if username else user_names[0]

    if requested_username and requested_username in user_names:
        # Find the index of the requested username
        user_index = user_names.index(requested_username)
        if user_index < len(passwords):
            selected_username = user_names[user_index]
            selected_password = passwords[user_index]
        else:
            raise ValueError(f"Password not found for user '{requested_username}' at index {user_index}")
    elif requested_username:
        raise ValueError(f"Username '{requested_username}' not found in byoidc credentials")
    else:
        raise ValueError("No users found in byoidc credentials")

    LOGGER.info(f"Using byoidc-credentials username='{selected_username}'")
    return {
        "username": selected_username,
        "password": selected_password,
    }
