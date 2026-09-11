import copy
import json
import logging
from typing import Any

import requests
from fastmcp import Client
from kubernetes.dynamic import DynamicClient
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

_logger = logging.getLogger(__name__)

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
    from kubernetes.dynamic.exceptions import NotFoundError

    svc_api = client.resources.get(api_version="v1", kind="Service")
    for namespace in _MODEL_CATALOG_NAMESPACES:
        try:
            services = svc_api.get(namespace=namespace)
        except NotFoundError:
            _logger.debug("Namespace %s not found, skipping", namespace)
            continue
        for svc in services.items:
            if _MODEL_CATALOG_SERVICE_PATTERN in svc.metadata.name:
                url = f"https://{svc.metadata.name}.{namespace}.svc:{_MODEL_CATALOG_PORT}"
                _logger.info("Discovered Model Catalog service: %s", url)
                return url
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
