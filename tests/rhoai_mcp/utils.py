from typing import Any

import requests
from timeout_sampler import retry

from tests.rhoai_mcp.constants import (
    RHOAI_MCP_APP_NAME,
    RHOAI_MCP_HEALTH_PATH,
    RHOAI_MCP_PORT,
)
from tests.rhoai_mcp.image_constants import RhoaiMcpImages

_RETRY_EXCEPTIONS: dict[type, list] = {
    requests.exceptions.ConnectTimeout: [],
    requests.exceptions.ReadTimeout: [],
    requests.exceptions.ConnectionError: [lambda exc: not isinstance(exc, requests.exceptions.SSLError)],
}


DEPLOYMENT_TEMPLATE: dict[str, Any] = {
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
                "image": RhoaiMcpImages.RHOAI_MCP,
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


@retry(wait_timeout=120, sleep=5, exceptions_dict=_RETRY_EXCEPTIONS)
def probe_health(url: str, ca_bundle_file: str) -> requests.Response:
    """GET the health endpoint, retrying on transient network failures."""
    return requests.get(url, verify=ca_bundle_file, timeout=10)
