from typing import Any

import requests
import structlog

from tests.rhoai_mcp.constants import (
    RHOAI_MCP_APP_NAME,
    RHOAI_MCP_HEALTH_PATH,
    RHOAI_MCP_PORT,
)
from tests.rhoai_mcp.image_constants import RhoaiMcpImages

LOGGER = structlog.get_logger(name=__name__)


class TransientHealthError(Exception):
    """Recoverable failure while polling the rhoai-mcp health endpoint."""


TRANSIENT_HEALTH_EXCEPTIONS: dict[type, list[Any]] = {TransientHealthError: []}


def get_deployment_template() -> dict[str, Any]:
    """Return the Kubernetes pod template for the rhoai-mcp Deployment."""
    labels = {
        "app.kubernetes.io/component": "server",
        "app.kubernetes.io/name": RHOAI_MCP_APP_NAME,
    }
    return {
        "metadata": {"labels": labels},
        "spec": {
            "containers": [
                {
                    "name": RHOAI_MCP_APP_NAME,
                    "image": RhoaiMcpImages.RHOAI_MCP,
                    "imagePullPolicy": "Always",
                    "args": ["--transport", "sse"],
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
                        "failureThreshold": 3,
                    },
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "128Mi"},
                        "limits": {"cpu": "500m", "memory": "512Mi"},
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


def probe_health(url: str, ca_bundle_file: str) -> requests.Response:
    """GET the health endpoint, retrying only on transient network failures."""
    try:
        return requests.get(url, verify=ca_bundle_file, timeout=10)
    except (
        requests.exceptions.ConnectTimeout,
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectionError,
    ) as err:
        if isinstance(err, requests.exceptions.SSLError):
            raise
        LOGGER.warning(f"Transient error checking rhoai-mcp health: {err}")
        raise TransientHealthError(str(err)) from err
