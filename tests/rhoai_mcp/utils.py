import copy
import json
from typing import Any

import requests
import structlog
import yaml
from fastmcp import Client
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.service import Service
from pytest_testconfig import config as py_config
from tenacity import retry as tenacity_retry
from tenacity import retry_if_not_result, stop_after_delay, wait_exponential
from timeout_sampler import retry

from tests.rhoai_mcp.constants import (
    RHOAI_MCP_APP_NAME,
    RHOAI_MCP_HEALTH_PATH,
    RHOAI_MCP_ODH_STABLE,
    RHOAI_MCP_PORT,
    RHOAI_MCP_RHOAI_VERSION,
)
from tests.rhoai_mcp.image_constants import RhoaiMcpImages
from utilities.infra import is_disconnected_cluster

_logger = structlog.get_logger(name=__name__)

_RETRY_EXCEPTIONS: dict[type, list] = {
    requests.exceptions.ConnectTimeout: [],
    requests.exceptions.ReadTimeout: [],
    requests.exceptions.ConnectionError: [lambda exc: not isinstance(exc, requests.exceptions.SSLError)],
}

_MODEL_CATALOG_NAMESPACES = ("rhoai-model-registries", "odh-model-registries", "model-registries")
_MODEL_CATALOG_SERVICE_PATTERN = "model-catalog"
_MODEL_CATALOG_PORT = 8443


def discover_model_catalog_url(client: DynamicClient) -> str | None:
    """Probe well-known namespaces for a Model Catalog service.

    Returns the in-cluster HTTPS URL if found, None otherwise.
    """
    from kubernetes.dynamic.exceptions import ForbiddenError, NotFoundError

    for namespace in _MODEL_CATALOG_NAMESPACES:
        try:
            services = Service.get(client=client, namespace=namespace)
            for service in services:
                if _MODEL_CATALOG_SERVICE_PATTERN in service.name:
                    url = f"https://{service.name}.{namespace}.svc:{_MODEL_CATALOG_PORT}"
                    _logger.info(msg=f"Discovered Model Catalog service: {url}")
                    return url
        except NotFoundError:
            _logger.debug(msg=f"Namespace {namespace} not found, skipping")
            continue
        except ForbiddenError:
            _logger.debug(msg=f"No permission to list services in {namespace}, skipping")
            continue
    return None


def get_rhoai_mcp_image(client: DynamicClient) -> str:
    """Return the rhoai-mcp container image appropriate for the target cluster."""
    if is_disconnected_cluster(client=client):
        return RhoaiMcpImages.RHOAI_MCP_RHOAI_DIGEST

    if py_config["distribution"] == "upstream":
        return RHOAI_MCP_ODH_STABLE

    # py_config["distribution"] == "downstream"
    return RHOAI_MCP_RHOAI_VERSION


def deployment_template_with_image(image: str) -> dict[str, Any]:
    """Return a deep copy of the pod template with *image* set on the main container."""
    template = copy.deepcopy(_DEPLOYMENT_TEMPLATE)
    template["spec"]["containers"][0]["image"] = image
    return template


_CORE_K8S_KINDS = frozenset({
    "ConfigMap",
    "Namespace",
    "Secret",
    "Service",
    "ServiceAccount",
})


def dry_run_validate_manifests(
    dyn_api: DynamicClient,
    configs: dict[str, str],
    default_namespace: str,
    required_kinds: frozenset[str] = _CORE_K8S_KINDS,
) -> int:
    """Submit each K8s manifest in *configs* as a server-side dry-run create.

    Returns the number of manifests that were successfully validated.
    Raises for *required_kinds* that are not registered on the cluster;
    unknown CRDs are skipped with a warning.
    """
    validated = 0
    for config_name, content in configs.items():
        for manifest in yaml.safe_load_all(content):
            if not isinstance(manifest, dict):
                continue
            api_version = manifest.get("apiVersion")
            kind = manifest.get("kind")
            if not api_version or not kind:
                continue

            name = manifest.get("metadata", {}).get("name", "<unnamed>")
            try:
                resource = dyn_api.resources.get(api_version=api_version, kind=kind)
            except ResourceNotFoundError:
                if kind in required_kinds:
                    raise
                _logger.warning(
                    "Skipping dry-run for %s/%s '%s' in config '%s': CRD not registered on cluster",
                    api_version,
                    kind,
                    name,
                    config_name,
                )
                continue

            namespace = manifest.get("metadata", {}).get("namespace", default_namespace)
            resource.create(body=manifest, namespace=namespace, dry_run="All")
            validated += 1
    return validated


_DEPLOYMENT_TEMPLATE: dict[str, Any] = {
    "metadata": {
        "labels": {
            "app.kubernetes.io/component": "server",
            "app.kubernetes.io/name": RHOAI_MCP_APP_NAME,
        },
    },
    "spec": {
        "containers": [
            {
                "name": RHOAI_MCP_APP_NAME,
                "image": "",
                "imagePullPolicy": "Always",
                "args": ["--transport", "$(RHOAI_MCP_TRANSPORT)"],
                "envFrom": [{"configMapRef": {"name": f"{RHOAI_MCP_APP_NAME}-config"}}],
                "ports": [
                    {
                        "containerPort": RHOAI_MCP_PORT,
                        "name": "http",
                        "protocol": "TCP",
                    }
                ],
                "livenessProbe": {
                    "httpGet": {"path": RHOAI_MCP_HEALTH_PATH, "port": "http"},
                    "initialDelaySeconds": 10,
                    "periodSeconds": 30,
                    "timeoutSeconds": 5,
                    "failureThreshold": 3,
                },
                "readinessProbe": {
                    "httpGet": {"path": RHOAI_MCP_HEALTH_PATH, "port": "http"},
                    "initialDelaySeconds": 5,
                    "periodSeconds": 10,
                    "timeoutSeconds": 5,
                    "failureThreshold": 12,
                },
                # Model Catalog sync loads benchmark data into in-memory SQLite;
                # config generation can block the event loop for tens of seconds.
                # 1Gi accommodates the larger dataset vs the base kustomize 512Mi.
                "resources": {
                    "requests": {"cpu": "100m", "memory": "256Mi"},
                    "limits": {"cpu": "500m", "memory": "1Gi"},
                },
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "readOnlyRootFilesystem": True,
                },
                "volumeMounts": [{"name": "tmp", "mountPath": "/tmp"}],
            }
        ],
        "securityContext": {
            "runAsNonRoot": True,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "serviceAccountName": RHOAI_MCP_APP_NAME,
        "volumes": [{"name": "tmp", "emptyDir": {}}],
    },
}


@retry(wait_timeout=120, sleep=5, exceptions_dict=_RETRY_EXCEPTIONS)
def probe_health(url: str, ca_bundle_file: str) -> requests.Response:
    """GET the health endpoint, retrying on transient network failures."""
    return requests.get(url, verify=ca_bundle_file, timeout=10)


def parse_tool_result(result: object) -> dict:
    """Parse the JSON payload from a call_tool response."""
    return json.loads(result.content[0].text)


@tenacity_retry(
    stop=stop_after_delay(300),
    wait=wait_exponential(min=5, max=30),
    retry=retry_if_not_result(lambda data: data.get("status") == "Ready"),
)
async def wait_for_model_ready(client: Client, name: str, namespace: str) -> dict:
    """Poll get_inference_service until the model reports Ready or timeout."""
    result = await client.call_tool(
        name="get_inference_service",
        arguments={"name": name, "namespace": namespace},
    )
    return parse_tool_result(result=result)
