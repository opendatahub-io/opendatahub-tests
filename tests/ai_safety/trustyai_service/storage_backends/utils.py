"""Helpers for exercising the operator-managed TrustyAI SQL database service."""

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

from tests.ai_safety.trustyai_service.constants import TAI_METRICS_CONFIG
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
)
from tests.ai_safety.trustyai_service.utils import create_trustyai_service, generate_db_tls_certs
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import TRUSTYAI_SERVICE_NAME
from utilities.image_constants import SharedImages

LOGGER = structlog.get_logger(name=__name__)

_KEY_FILE_MODE: int = 0o640
_POSTGRES_SSL_CONF: str = f"""ssl = on
ssl_cert_file = '{POSTGRES_CERTS_MOUNT_PATH}/tls.crt'
ssl_key_file = '{POSTGRES_CERTS_MOUNT_PATH}/tls.key'
ssl_ca_file = '{POSTGRES_CERTS_MOUNT_PATH}/ca.crt'
"""


@contextmanager
def create_standalone_postgres(
    client: DynamicClient,
    namespace_name: str,
    name: str,
    db_credentials_secret_name: str,
    teardown: bool = True,
) -> Generator[Deployment, Any, Any]:
    """Create only the supporting PostgreSQL database resources with TLS enabled."""
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
                    "ports": [{"port": POSTGRES_PORT, "targetPort": POSTGRES_PORT, "name": "postgres"}],
                },
            },
            teardown=teardown,
        ),
    ):
        template = {
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
                    {"name": "certs", "secret": {"secretName": f"{name}-server-tls", "defaultMode": _KEY_FILE_MODE}},
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
            template=template,
            label={"name": name, "app": "postgres"},
            teardown=teardown,
        ) as deployment:
            deployment.wait_for_replicas(deployed=True, timeout=600)
            yield deployment


@contextmanager
def create_database_backed_trustyai_service(
    client: DynamicClient,
    namespace: str,
    name: str,
    credentials_secret_name: str,
    wait_for_replicas: bool = True,
    teardown: bool = True,
    database_tuning_config_map_name: str | None = None,
) -> Generator[tuple[Any, Deployment], Any, Any]:
    """Create the service through its CR and yield the operator-managed Deployment."""
    storage = {"format": "DATABASE", "size": "1Gi", "databaseConfigurations": credentials_secret_name}
    if database_tuning_config_map_name:
        storage["databaseTuning"] = database_tuning_config_map_name
    with create_trustyai_service(
        client=client,
        namespace=namespace,
        name=name,
        storage=storage,
        metrics=TAI_METRICS_CONFIG,
        wait_for_replicas=wait_for_replicas,
        teardown=teardown,
    ) as service:
        deployment = Deployment(client=client, namespace=namespace, name=name)
        deployment.wait(timeout=300)
        if wait_for_replicas:
            deployment.wait_for_replicas(deployed=True, timeout=600)
        yield service, deployment


def update_database_tuning_config_map(config_map: ConfigMap, data: dict[str, str]) -> None:
    """Replace tuning values after the CR is running; env changes need a pod restart."""
    resource_dict = config_map.instance.to_dict()
    resource_dict["data"] = data
    config_map.update(resource_dict=resource_dict)


def restart_trustyai_service_pod(
    client: DynamicClient,
    namespace: str,
    deployment: Deployment,
    wait_for_replicas: bool = True,
    timeout: int = 600,
) -> None:
    """Replace the CR-managed pod so it reads the latest ConfigMap-backed environment."""
    selector = deployment_pod_selector(deployment=deployment)
    pods = list(Pod.get(client=client, namespace=namespace, label_selector=selector))
    assert pods, f"No service pods found for Deployment {deployment.name}"
    for pod in pods:
        pod.delete(wait=True)
    if wait_for_replicas:
        deployment.wait_for_replicas(deployed=True, timeout=timeout)


def wait_for_deployment_unready(
    client: DynamicClient,
    namespace: str,
    name: str,
    timeout: int = 300,
) -> bool:
    """Wait until the service Deployment has no ready replicas after a rejected setting."""

    def _no_ready_replicas() -> bool:
        deployment = Deployment(client=client, namespace=namespace, name=name, ensure_exists=True)
        return not deployment.instance.status.get("readyReplicas")

    return any(TimeoutSampler(wait_timeout=timeout, sleep=5, func=_no_ready_replicas))


def wait_for_trustyai_condition(
    service: Any,
    condition_type: str,
    status: str,
    timeout: int = 300,
) -> dict[str, Any]:
    """Wait for a TrustyAIService status condition and return it.

    Args:
        service: The TrustyAIService custom resource.
        condition_type: Condition type to look for, such as `DatabaseTuningAvailable`.
        status: Wanted condition status, `"True"` or `"False"`.
        timeout: Seconds to wait.

    Returns:
        The matching condition as a dict.
    """

    def _condition() -> dict[str, Any] | None:
        conditions = service.instance.to_dict().get("status", {}).get("conditions") or []
        return next((item for item in conditions if item["type"] == condition_type and item["status"] == status), None)

    for condition in TimeoutSampler(wait_timeout=timeout, sleep=5, func=_condition):
        if condition:
            return condition
    raise TimeoutError(f"{condition_type}={status} not reported on {service.name}")


def patch_service_deployment(
    deployment: Deployment,
    env: dict[str, str] | None = None,
    config_map_name: str | None = None,
    wait_for_replicas: bool = True,
) -> None:
    """Apply test-only environment inputs to an operator-created Deployment."""
    container_patch: dict[str, Any] = {"name": TRUSTYAI_SERVICE_NAME}
    if env:
        container_patch["env"] = [{"name": key, "value": value} for key, value in env.items()]
    if config_map_name:
        container_patch["envFrom"] = [{"configMapRef": {"name": config_map_name}}]
    patch = {"spec": {"template": {"spec": {"containers": [container_patch]}}}}
    deployment.api.patch(
        body=patch,
        name=deployment.name,
        namespace=deployment.namespace,
        content_type="application/strategic-merge-patch+json",
    )
    if wait_for_replicas:
        deployment.wait_for_replicas(deployed=True, timeout=600)


def deployment_pod_selector(deployment: Deployment) -> str:
    """Build the selector for pods controlled by a CR-generated Deployment."""
    labels = deployment.instance.spec.selector.matchLabels
    return ",".join(f"{key}={value}" for key, value in labels.items())


def get_pod_logs(client: DynamicClient, namespace: str, label_selector: str) -> str:
    """Return logs of pods matching a selector, tolerating pods without a current log."""
    logs: list[str] = []
    for pod in Pod.get(client=client, namespace=namespace, label_selector=label_selector):
        try:
            logs.append(pod.log(container=TRUSTYAI_SERVICE_NAME))
        except Exception as exc:  # noqa: BLE001 - a crash-looping pod may not have a readable log yet
            LOGGER.warning(f"Could not read logs from pod {pod.name}: {exc}")
    return "\n".join(logs)


def wait_for_pod_log_message(
    client: DynamicClient,
    namespace: str,
    label_selector: str,
    message: str,
    timeout: int = 300,
) -> str:
    """Wait until a matching TrustyAI pod logs a message and return those logs."""

    def _logs() -> str | None:
        logs = get_pod_logs(client=client, namespace=namespace, label_selector=label_selector)
        return logs if message in logs else None

    for sample in TimeoutSampler(wait_timeout=timeout, sleep=10, func=_logs):
        if sample:
            return sample
    return ""


def build_upload_payload(
    model_name: str,
    n_rows: int = 5,
    data_tag: str | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Build a KServe-shaped payload for `/data/upload`."""
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
    """Wait until `/info` reports the expected observation count."""

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
    for sample in sampler:
        if sample:
            return sample
    return {}


def run_python_in_service_pod(
    client: DynamicClient,
    namespace: str,
    code: str,
    deployment: Deployment | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Run a Python probe in a pod selected by the operator-managed Deployment."""
    if deployment is None:
        if name is None:
            raise ValueError("Pass either deployment or name")
        deployment = Deployment(client=client, namespace=namespace, name=name, ensure_exists=True)
    selector = deployment_pod_selector(deployment=deployment)
    pods = list(Pod.get(client=client, namespace=namespace, label_selector=selector))
    assert pods, f"No pod found for Deployment {deployment.name}"
    output = pods[0].execute(command=["python", "-c", code], container=TRUSTYAI_SERVICE_NAME).strip()
    return json.loads(output.splitlines()[-1])


class StorageBackendClient:
    """Authenticated HTTP client for the TrustyAIService CR Route."""

    def __init__(
        self,
        client: DynamicClient,
        namespace: str,
        name: str,
        token: str,
        timeout: int = 60,
    ) -> None:
        self.route = Route(client=client, namespace=namespace, name=name, ensure_exists=True)
        self.cert_path = create_ca_bundle_file(client=client)
        self.headers = {"Authorization": f"Bearer {token}"}
        self.timeout = timeout

    def _url(self, endpoint: str) -> str:
        return f"https://{self.route.host}/{endpoint.lstrip('/')}"

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> requests.Response:
        return requests.get(
            url=self._url(endpoint),
            params=params,
            headers=self.headers,
            verify=self.cert_path,
            timeout=self.timeout,
        )

    def post(self, endpoint: str, json: dict[str, Any]) -> requests.Response:
        return requests.post(
            url=self._url(endpoint),
            json=json,
            headers=self.headers,
            verify=self.cert_path,
            timeout=self.timeout,
        )

    def upload(self, payload: dict[str, Any]) -> requests.Response:
        return self.post(endpoint=ENDPOINT_DATA_UPLOAD, json=payload)

    def info(self) -> requests.Response:
        return self.get(endpoint=ENDPOINT_INFO)

    def inference_ids(self, model_name: str, params: dict[str, Any] | None = None) -> requests.Response:
        return self.get(endpoint=f"{ENDPOINT_INFERENCE_IDS}/{model_name}", params=params)

    def apply_name_mapping(self, payload: dict[str, Any]) -> requests.Response:
        return self.post(endpoint=ENDPOINT_INFO_NAMES, json=payload)

    def clear_name_mapping(self, model_name: str) -> requests.Response:
        return requests.delete(
            url=self._url(ENDPOINT_INFO_NAMES),
            data=json.dumps(model_name),
            headers={**self.headers, "Content-Type": "application/json"},
            verify=self.cert_path,
            timeout=self.timeout,
        )

    def tags(self, model_name: str) -> requests.Response:
        return self.get(endpoint=ENDPOINT_INFO_TAGS, params={"modelId": model_name})

    def health(self) -> requests.Response:
        return self.get(endpoint=ENDPOINT_HEALTH)

    def readiness(self) -> requests.Response:
        return self.get(endpoint=ENDPOINT_HEALTH_READY)
