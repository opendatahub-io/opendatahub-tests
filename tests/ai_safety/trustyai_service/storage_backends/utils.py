"""Helpers for deploying a TrustyAI service against a SQL storage backend.

The service is deployed directly rather than through a `TrustyAIService` CR
because the operator cannot select the PostgreSQL or SQLite backends yet; see
this package's README.md.
"""

import json
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from timeout_sampler import TimeoutSampler

from tests.ai_safety.trustyai_service.storage_backends.constants import (
    ENDPOINT_DATA_UPLOAD,
    ENDPOINT_HEALTH,
    ENDPOINT_HEALTH_READY,
    ENDPOINT_INFERENCE_IDS,
    ENDPOINT_INFO,
    ENDPOINT_INFO_NAMES,
    ENDPOINT_INFO_TAGS,
    POSTGRES_CERTS_MOUNT_PATH,
    POSTGRES_CONFIG_MOUNT_PATH,
    POSTGRES_PORT,
    SERVICE_CA_CONFIGMAP,
    SERVICE_CA_KEY,
    SERVICE_HEALTH_PORT,
    SERVICE_TLS_MOUNT_PATH,
    SERVICE_TLS_PORT,
    SERVING_CERT_ANNOTATION,
)
from tests.ai_safety.trustyai_service.utils import generate_db_tls_certs
from utilities.certificates_utils import create_ca_bundle_file
from utilities.image_constants import SharedImages

LOGGER = structlog.get_logger(name=__name__)

# PostgreSQL refuses to start when the key file is group/other readable, unless the
# file is owned by root. Secret volumes are root-owned, so 0640 is accepted.
_KEY_FILE_MODE: int = 0o640

_POSTGRES_SSL_CONF: str = f"""ssl = on
ssl_cert_file = '{POSTGRES_CERTS_MOUNT_PATH}/tls.crt'
ssl_key_file = '{POSTGRES_CERTS_MOUNT_PATH}/tls.key'
ssl_ca_file = '{POSTGRES_CERTS_MOUNT_PATH}/ca.crt'
"""


def wait_for_deployment_pods_ready(
    client: DynamicClient,
    namespace: str,
    label_selector: str,
    timeout: int = 600,
) -> list[Pod]:
    """Wait until at least one pod matching `label_selector` reports Ready.

    Args:
        client: Kubernetes dynamic client.
        namespace: Namespace to look in.
        label_selector: Label selector identifying the pods.
        timeout: Seconds to wait for the pods to appear and become ready.

    Returns:
        list[Pod]: The ready pods.
    """

    def _pods() -> list[Pod]:
        return list(Pod.get(client=client, namespace=namespace, label_selector=label_selector))

    for sample in TimeoutSampler(wait_timeout=timeout, sleep=5, func=lambda: bool(_pods())):
        if sample:
            break

    pods = _pods()
    for pod in pods:
        pod.wait_for_condition(condition=Pod.Condition.READY, status="True", timeout=timeout)
    return pods


@contextmanager
def create_standalone_postgres(
    client: DynamicClient,
    namespace_name: str,
    name: str,
    db_credentials_secret_name: str,
    teardown: bool = True,
) -> Generator[Deployment, Any, Any]:
    """Deploy a single-pod PostgreSQL with TLS enabled.

    The server certificate carries the Service DNS names, so the TrustyAI service
    can connect with `sslmode=verify-full`.

    Args:
        client: Kubernetes dynamic client.
        namespace_name: Namespace to deploy into.
        name: Name shared by the Deployment, Service and certificate SANs.
        db_credentials_secret_name: Secret holding databaseUsername/databasePassword/databaseName.
        teardown: Whether to delete the resources on exit.

    Yields:
        Deployment: The running PostgreSQL deployment.
    """
    ca_cert, server_cert, server_key = generate_db_tls_certs(namespace_name=namespace_name, service_name=name)

    with (
        Secret(
            client=client,
            name=f"{name}-ca",
            namespace=namespace_name,
            string_data={"ca.crt": ca_cert},
            teardown=teardown,
        ),
        Secret(
            client=client,
            name=f"{name}-server-tls",
            namespace=namespace_name,
            string_data={"tls.crt": server_cert, "tls.key": server_key, "ca.crt": ca_cert},
            teardown=teardown,
        ),
        ConfigMap(
            client=client,
            name=f"{name}-config",
            namespace=namespace_name,
            data={"ssl.conf": _POSTGRES_SSL_CONF},
            teardown=teardown,
        ),
        Service(
            client=client,
            kind_dict={
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": name, "namespace": namespace_name},
                "spec": {
                    "selector": {"name": name},
                    "ports": [
                        {"port": POSTGRES_PORT, "targetPort": POSTGRES_PORT, "name": "postgres", "protocol": "TCP"}
                    ],
                },
            },
            teardown=teardown,
        ),
    ):
        deployment_template = {
            "metadata": {"labels": {"name": name, "app": "postgres", "component": "database"}},
            "spec": {
                "containers": [
                    {
                        "name": "postgres",
                        "image": SharedImages.POSTGRESQL_15,
                        "imagePullPolicy": "IfNotPresent",
                        "env": [
                            {
                                "name": "POSTGRESQL_USER",
                                "valueFrom": {
                                    "secretKeyRef": {"name": db_credentials_secret_name, "key": "databaseUsername"}
                                },
                            },
                            {
                                "name": "POSTGRESQL_PASSWORD",
                                "valueFrom": {
                                    "secretKeyRef": {"name": db_credentials_secret_name, "key": "databasePassword"}
                                },
                            },
                            {
                                "name": "POSTGRESQL_DATABASE",
                                "valueFrom": {
                                    "secretKeyRef": {"name": db_credentials_secret_name, "key": "databaseName"}
                                },
                            },
                        ],
                        "ports": [{"containerPort": POSTGRES_PORT, "protocol": "TCP"}],
                        "volumeMounts": [
                            {"name": "certs", "mountPath": POSTGRES_CERTS_MOUNT_PATH, "readOnly": True},
                            {"name": "config", "mountPath": POSTGRES_CONFIG_MOUNT_PATH, "readOnly": True},
                            {"name": "data", "mountPath": "/var/lib/pgsql/data"},
                        ],
                        "readinessProbe": {
                            "exec": {"command": ["/usr/libexec/check-container"]},
                            "initialDelaySeconds": 10,
                            "periodSeconds": 10,
                            "timeoutSeconds": 10,
                        },
                        "livenessProbe": {
                            "exec": {"command": ["/usr/libexec/check-container", "--live"]},
                            "initialDelaySeconds": 30,
                            "periodSeconds": 15,
                            "timeoutSeconds": 10,
                        },
                    }
                ],
                "volumes": [
                    {
                        "name": "certs",
                        "secret": {"secretName": f"{name}-server-tls", "defaultMode": _KEY_FILE_MODE},
                    },
                    {"name": "config", "configMap": {"name": f"{name}-config"}},
                    {"name": "data", "emptyDir": {}},
                ],
            },
        }

        with Deployment(
            client=client,
            name=name,
            namespace=namespace_name,
            replicas=1,
            selector={"matchLabels": {"name": name}},
            template=deployment_template,
            label={"name": name, "app": "postgres"},
            teardown=teardown,
        ) as deployment:
            deployment.wait_for_replicas(deployed=True, timeout=600)
            wait_for_deployment_pods_ready(client=client, namespace=namespace_name, label_selector=f"name={name}")
            yield deployment


def _service_container_env(env: dict[str, str], db_credentials_secret_name: str | None) -> list[dict[str, Any]]:
    """Build the service container env, sourcing DB credentials from the secret."""
    container_env: list[dict[str, Any]] = [{"name": key, "value": value} for key, value in env.items()]

    if db_credentials_secret_name:
        container_env += [
            {
                "name": "DATABASE_USERNAME",
                "valueFrom": {"secretKeyRef": {"name": db_credentials_secret_name, "key": "databaseUsername"}},
            },
            {
                "name": "DATABASE_PASSWORD",
                "valueFrom": {"secretKeyRef": {"name": db_credentials_secret_name, "key": "databasePassword"}},
            },
            {
                "name": "DATABASE_NAME",
                "valueFrom": {"secretKeyRef": {"name": db_credentials_secret_name, "key": "databaseName"}},
            },
            {
                "name": "DATABASE_SERVICE",
                "valueFrom": {"secretKeyRef": {"name": db_credentials_secret_name, "key": "databaseService"}},
            },
            {
                "name": "DATABASE_PORT",
                "valueFrom": {"secretKeyRef": {"name": db_credentials_secret_name, "key": "databasePort"}},
            },
        ]
    return container_env


def service_ca_certificate(client: DynamicClient, namespace: str) -> str:
    """Return the cluster's service-CA bundle, injected into every namespace.

    The router needs it to validate the serving certificate that terminates the
    reencrypt Route at the service container.
    """
    config_map = ConfigMap(
        client=client,
        name=SERVICE_CA_CONFIGMAP,
        namespace=namespace,
        ensure_exists=True,
    )
    return str(config_map.instance.data[SERVICE_CA_KEY])


@contextmanager
def create_storage_backend_service(
    client: DynamicClient,
    namespace_name: str,
    name: str,
    image: str,
    env: dict[str, str],
    db_credentials_secret_name: str | None = None,
    ca_secret_name: str | None = None,
    ca_mount_path: str | None = None,
    data_volume_mount_path: str | None = None,
    wait_for_ready: bool = True,
    teardown: bool = True,
) -> Generator[Deployment, Any, Any]:
    """Deploy the TrustyAI service image directly against a storage backend.

    Args:
        client: Kubernetes dynamic client.
        namespace_name: Namespace to deploy into.
        name: Name shared by the Deployment, Service and Route.
        image: TrustyAI service image to run.
        env: Literal env vars for the container, e.g. SERVICE_STORAGE_FORMAT.
        db_credentials_secret_name: Secret to source DATABASE_* connection env from.
        ca_secret_name: Secret holding `ca.crt` for the database CA.
        ca_mount_path: Directory to mount `ca_secret_name` at.
        data_volume_mount_path: Directory backed by an emptyDir, for file-backed SQLite.
        wait_for_ready: Wait for the pod to report Ready. Set False when the test
            expects the service to fail its startup checks.
        teardown: Whether to delete the resources on exit.

    Yields:
        Deployment: The service deployment, ready unless `wait_for_ready` is False.
    """
    serving_cert_secret = f"{name}-serving-cert"

    # The service only listens on a routable address when it finds serving
    # certificates; without them the API stays on loopback (main.py::run_server).
    volumes: list[dict[str, Any]] = [
        {"name": "serving-cert", "secret": {"secretName": serving_cert_secret, "defaultMode": 0o440}}
    ]
    volume_mounts: list[dict[str, Any]] = [
        {"name": "serving-cert", "mountPath": SERVICE_TLS_MOUNT_PATH, "readOnly": True}
    ]

    if ca_secret_name and ca_mount_path:
        volumes.append({"name": "db-ca", "secret": {"secretName": ca_secret_name}})
        volume_mounts.append({"name": "db-ca", "mountPath": ca_mount_path, "readOnly": True})

    if data_volume_mount_path:
        volumes.append({"name": "storage-data", "emptyDir": {}})
        volume_mounts.append({"name": "storage-data", "mountPath": data_volume_mount_path})

    deployment_template = {
        "metadata": {"labels": {"name": name, "app": name}},
        "spec": {
            "containers": [
                {
                    "name": "trustyai-service",
                    "image": image,
                    "imagePullPolicy": "IfNotPresent",
                    "env": _service_container_env(env=env, db_credentials_secret_name=db_credentials_secret_name),
                    "ports": [
                        {"containerPort": SERVICE_TLS_PORT, "protocol": "TCP", "name": "https"},
                        {"containerPort": SERVICE_HEALTH_PORT, "protocol": "TCP", "name": "health"},
                    ],
                    "readinessProbe": {
                        "httpGet": {"path": f"/{ENDPOINT_HEALTH_READY}", "port": SERVICE_HEALTH_PORT},
                        "initialDelaySeconds": 10,
                        "periodSeconds": 10,
                    },
                    "volumeMounts": volume_mounts,
                }
            ],
            "volumes": volumes,
        },
    }

    with (
        Service(
            client=client,
            kind_dict={
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": name,
                    "namespace": namespace_name,
                    # service-ca issues the serving certificate the container mounts.
                    "annotations": {SERVING_CERT_ANNOTATION: serving_cert_secret},
                },
                "spec": {
                    "selector": {"name": name},
                    "ports": [
                        {"port": SERVICE_TLS_PORT, "targetPort": SERVICE_TLS_PORT, "name": "https", "protocol": "TCP"}
                    ],
                },
            },
            teardown=teardown,
        ),
        Route(
            client=client,
            kind_dict={
                "apiVersion": "route.openshift.io/v1",
                "kind": "Route",
                "metadata": {"name": name, "namespace": namespace_name},
                "spec": {
                    "to": {"kind": "Service", "name": name},
                    "port": {"targetPort": SERVICE_TLS_PORT},
                    "tls": {
                        "termination": "reencrypt",
                        "destinationCACertificate": service_ca_certificate(client=client, namespace=namespace_name),
                        "insecureEdgeTerminationPolicy": "Redirect",
                    },
                },
            },
            teardown=teardown,
        ),
        Deployment(
            client=client,
            name=name,
            namespace=namespace_name,
            replicas=1,
            selector={"matchLabels": {"name": name}},
            template=deployment_template,
            label={"name": name, "app": name},
            teardown=teardown,
        ) as deployment,
    ):
        if wait_for_ready:
            deployment.wait_for_replicas(deployed=True, timeout=600)
            wait_for_deployment_pods_ready(client=client, namespace=namespace_name, label_selector=f"name={name}")
        yield deployment


def get_pod_logs(client: DynamicClient, namespace: str, label_selector: str) -> str:
    """Return the concatenated logs of every pod matching `label_selector`."""
    logs = []
    for pod in Pod.get(client=client, namespace=namespace, label_selector=label_selector):
        try:
            logs.append(pod.log(container="trustyai-service"))
        except Exception as exc:  # noqa: BLE001 - a crash-looping pod may have no readable log yet
            LOGGER.warning(f"Could not read logs from pod {pod.name}: {exc}")
    return "\n".join(logs)


def wait_for_pod_log_message(
    client: DynamicClient,
    namespace: str,
    label_selector: str,
    message: str,
    timeout: int = 300,
) -> str:
    """Wait for `message` to appear in the logs of a pod matching `label_selector`.

    Args:
        client: Kubernetes dynamic client.
        namespace: Namespace to look in.
        label_selector: Label selector identifying the pods.
        message: Substring to wait for.
        timeout: Seconds to wait.

    Returns:
        str: The logs containing the message.

    Raises:
        TimeoutExpiredError: If the message never appears.
    """

    def _matching_logs() -> str | None:
        logs = get_pod_logs(client=client, namespace=namespace, label_selector=label_selector)
        return logs if message in logs else None

    matched = ""
    for sample in TimeoutSampler(wait_timeout=timeout, sleep=10, func=_matching_logs):
        if sample:
            matched = sample
            break
    return matched


def build_upload_payload(
    model_name: str,
    n_rows: int = 5,
    data_tag: str | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Build a KServe-shaped payload for `/data/upload`.

    Args:
        model_name: Model the rows belong to.
        n_rows: Number of rows to upload.
        data_tag: Optional tag applied to every row.
        offset: Value offset, so repeated uploads carry distinguishable data.

    Returns:
        dict: Payload accepted by the data upload endpoint.
    """
    payload: dict[str, Any] = {
        "model_name": model_name,
        "is_ground_truth": False,
        "request": {
            "inputs": [
                {
                    "name": "input",
                    "shape": [n_rows, 2],
                    "datatype": "INT64",
                    "data": [[offset + i, offset + i + 1] for i in range(n_rows)],
                }
            ]
        },
        "response": {
            "model_name": model_name,
            "outputs": [
                {
                    "name": "output",
                    "shape": [n_rows, 1],
                    "datatype": "INT64",
                    "data": [[(offset + i) * 2] for i in range(n_rows)],
                }
            ],
        },
    }
    if data_tag:
        payload["data_tag"] = data_tag
    return payload


def wait_for_model_observations(
    storage_client: StorageBackendClient,
    model_name: str,
    expected: int,
    timeout: int = 120,
) -> dict[str, Any]:
    """Wait until `/info` reports `expected` observations for `model_name`.

    The upload endpoint returns once the write is committed, but the service
    metadata is assembled from a data source that may still be refreshing, so the
    read is polled rather than asserted once.

    Args:
        storage_client: Client for the service under test.
        model_name: Model to look for.
        expected: Observation count to wait for.
        timeout: Seconds to wait.

    Returns:
        dict: The `/info` entry for `model_name`.

    Raises:
        TimeoutExpiredError: If the count is not reached in time.
    """

    def _observations() -> dict[str, Any] | None:
        response = storage_client.info()
        if response.status_code != requests.codes.ok:
            LOGGER.warning(f"/info returned {response.status_code}: {response.text}")
            return None
        model_info = response.json().get(model_name)
        if model_info and model_info.get("data", {}).get("observations") == expected:
            return model_info
        return None

    sampler = TimeoutSampler(wait_timeout=timeout, sleep=5, func=_observations)
    model_info: dict[str, Any] = {}
    for sample in sampler:
        if sample:
            model_info = sample
            break
    return model_info


class StorageBackendClient:
    """HTTP client for a directly deployed TrustyAI service.

    Unlike `TrustyAIServiceClient`, this talks to a Route in front of the service
    container itself: the deployment under test has no kube-rbac-proxy, because it
    is not created by the operator.
    """

    def __init__(self, client: DynamicClient, namespace: str, name: str, timeout: int = 60) -> None:
        self.route = Route(client=client, namespace=namespace, name=name, ensure_exists=True)
        self.cert_path = create_ca_bundle_file(client=client)
        self.timeout = timeout

    def _url(self, endpoint: str) -> str:
        return f"https://{self.route.host}/{endpoint.lstrip('/')}"

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> requests.Response:
        """GET `endpoint` on the service route."""
        return requests.get(url=self._url(endpoint), params=params, verify=self.cert_path, timeout=self.timeout)

    def post(self, endpoint: str, json: dict[str, Any]) -> requests.Response:
        """POST `json` to `endpoint` on the service route."""
        return requests.post(url=self._url(endpoint), json=json, verify=self.cert_path, timeout=self.timeout)

    def delete(self, endpoint: str, json: dict[str, Any]) -> requests.Response:
        """DELETE `endpoint` on the service route with a JSON body."""
        return requests.delete(url=self._url(endpoint), json=json, verify=self.cert_path, timeout=self.timeout)

    def upload(self, payload: dict[str, Any]) -> requests.Response:
        """Upload inference data."""
        return self.post(endpoint=ENDPOINT_DATA_UPLOAD, json=payload)

    def info(self) -> requests.Response:
        """Get the service metadata for every known model."""
        return self.get(endpoint=ENDPOINT_INFO)

    def inference_ids(self, model_name: str, params: dict[str, Any] | None = None) -> requests.Response:
        """List the inference ids stored for `model_name`."""
        return self.get(endpoint=f"{ENDPOINT_INFERENCE_IDS}/{model_name}", params=params)

    def apply_name_mapping(self, payload: dict[str, Any]) -> requests.Response:
        """Apply a name mapping."""
        return self.post(endpoint=ENDPOINT_INFO_NAMES, json=payload)

    def clear_name_mapping(self, model_name: str) -> requests.Response:
        """Clear a name mapping.

        The API expects the request body to be a JSON-encoded string (the model id),
        not an object with a modelId field.
        """
        return requests.delete(
            url=self._url(ENDPOINT_INFO_NAMES),
            data=json.dumps(model_name),
            headers={"Content-Type": "application/json"},
            verify=self.cert_path,
            timeout=self.timeout,
        )

    def tags(self, model_name: str) -> requests.Response:
        """Get per-tag row counts for `model_name`."""
        return self.get(endpoint=ENDPOINT_INFO_TAGS, params={"modelId": model_name})

    def health(self) -> requests.Response:
        """Get the combined health payload."""
        return self.get(endpoint=ENDPOINT_HEALTH)

    def readiness(self) -> requests.Response:
        """Get the readiness payload."""
        return self.get(endpoint=ENDPOINT_HEALTH_READY)
