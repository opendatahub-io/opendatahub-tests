from typing import Any
from urllib.parse import urlparse

import yaml

from tests.ogx.constants import OGX_CORE_VLLM_API_TOKEN, OGX_CORE_VLLM_URL
from tests.ogx.praxis.constants import (
    MIGRATION_FALLBACK_TENANT,
    PRAXIS_GATEWAY_APP_LABEL,
    PRAXIS_GATEWAY_PORT,
    TARGET_CONVERSATION_ITEMS_TABLE,
    TARGET_CONVERSATIONS_TABLE,
    TARGET_RESPONSES_TABLE,
)
from tests.ogx.praxis.utils import praxis_connection_string

# Names internal to the Praxis configuration rendered by `praxis_gateway_config`.
_PRAXIS_FILTER_CHAIN_NAME: str = "store-pipeline"
_PRAXIS_VLLM_UPSTREAM_NAME: str = "vllm-upstream"
_PRAXIS_OGX_UPSTREAM_NAME: str = "ogx-upstream"
_OGX_API_PORT: int = 8321


def build_praxis_gateway_config(namespace: str, ogx_service_name: str) -> str:
    """Render the Praxis configuration that puts it in front of the target database.

    The pipeline is the one the OGX repository's `migration-e2e-praxis` workflow
    uses. The store filters own the three tables the migration writes to, while
    Responses create requests are translated to Chat Completions and sent
    directly to the configured vLLM inference endpoint.

    The rendered document embeds the target DSN and vLLM API token, so it must
    only ever be written into a Secret.

    Args:
        namespace: Namespace holding the target PostgreSQL service.
        ogx_service_name: Name of the Service exposing the OGX API.

    Returns:
        The Praxis configuration as a YAML document.

    Raises:
        ValueError: If ``OGX_CORE_VLLM_URL`` is not a valid HTTP(S) URL.
    """
    vllm_url = urlparse(url=OGX_CORE_VLLM_URL)
    if vllm_url.scheme not in {"http", "https"} or not vllm_url.hostname:
        raise ValueError("OGX_CORE_VLLM_URL must be a valid HTTP(S) URL")

    vllm_port = vllm_url.port or (443 if vllm_url.scheme == "https" else 80)
    vllm_host = vllm_url.hostname
    vllm_endpoint = f"{vllm_host}:{vllm_port}"
    vllm_host_header = vllm_url.netloc
    vllm_cluster: dict[str, Any] = {
        "name": _PRAXIS_VLLM_UPSTREAM_NAME,
        "endpoints": [vllm_endpoint],
        "read_timeout_ms": 300000,
    }
    if vllm_url.scheme == "https":
        vllm_cluster["tls"] = {"sni": vllm_host}

    ogx_host = f"{ogx_service_name}.{namespace}.svc.cluster.local"
    ogx_cluster: dict[str, Any] = {
        "name": _PRAXIS_OGX_UPSTREAM_NAME,
        "endpoints": [f"{ogx_host}:{_OGX_API_PORT}"],
        "http": {"authority": ogx_host},
    }

    postgres_backend: dict[str, Any] = {
        "backend": "postgres",
        "database_url": praxis_connection_string(namespace=namespace),
        # The DSN addresses a Service by name, which Praxis rejects as a private
        # target unless it is opted into, and the test PostgreSQL serves no TLS.
        "allow_private_database_url": True,
        "ssl_mode": "disable",
    }
    config: dict[str, Any] = {
        "insecure_options": {
            "allow_private_endpoints": True,
            "allow_private_upstreams": True,
        },
        "listeners": [
            {
                "name": PRAXIS_GATEWAY_APP_LABEL,
                "address": f"0.0.0.0:{PRAXIS_GATEWAY_PORT}",
                "filter_chains": [_PRAXIS_FILTER_CHAIN_NAME],
            }
        ],
        "filter_chains": [
            {
                "name": _PRAXIS_FILTER_CHAIN_NAME,
                "filters": [
                    # Classifies the operation from the request head. `openai_conversations`
                    # requires it earlier in the same chain.
                    {"filter": "ai_operation"},
                    {
                        "filter": "openai_responses_request",
                        "initialize_state": False,
                        "on_invalid": "reject",
                    },
                    {"filter": "openai_responses_request", "on_invalid": "reject"},
                    {"filter": "state_owner", "mode": "single_tenant", "tenant_id": MIGRATION_FALLBACK_TENANT},
                    {
                        "filter": "openai_conversations",
                        "conversations_table": TARGET_CONVERSATIONS_TABLE,
                        "items_table": TARGET_CONVERSATION_ITEMS_TABLE,
                        **postgres_backend,
                    },
                    {
                        "filter": "openai_responses_store",
                        "responses_table": TARGET_RESPONSES_TABLE,
                        "conversations_table": TARGET_CONVERSATIONS_TABLE,
                        **postgres_backend,
                    },
                    {
                        "filter": "iterative_request_router",
                        "initial_step": "inference",
                        "max_iterations": 1,
                        "timeout_ms": 300000,
                        "steps": [
                            {
                                "name": "inference",
                                "filters": [
                                    {"filter": "openai_responses_to_chat_completions"},
                                    {
                                        "filter": "path_rewrite",
                                        "replace": {
                                            "pattern": "^/v1/responses/?$",
                                            "replacement": "/v1/chat/completions",
                                        },
                                        "conditions": [{"when": {"path_prefix": "/v1/responses", "methods": ["POST"]}}],
                                    },
                                    {
                                        "filter": "router",
                                        "routes": [
                                            {"path_prefix": "/v1/files", "cluster": _PRAXIS_OGX_UPSTREAM_NAME},
                                            {
                                                "path_prefix": "/v1/vector_stores",
                                                "cluster": _PRAXIS_OGX_UPSTREAM_NAME,
                                            },
                                            {"path_prefix": "/", "cluster": _PRAXIS_VLLM_UPSTREAM_NAME},
                                        ],
                                    },
                                    {
                                        "filter": "credential_injection",
                                        "clusters": [
                                            {
                                                "name": _PRAXIS_VLLM_UPSTREAM_NAME,
                                                "header": "Authorization",
                                                "value": OGX_CORE_VLLM_API_TOKEN,
                                                "header_prefix": "Bearer ",
                                                "strip_client_credential": True,
                                            }
                                        ],
                                    },
                                    {
                                        "filter": "headers",
                                        "request_set": [{"name": "Host", "value": vllm_host_header}],
                                    },
                                    {"filter": "load_balancer", "clusters": [vllm_cluster, ogx_cluster]},
                                ],
                                "on_result": [{"default": True, "done": True}],
                            }
                        ],
                    },
                ],
            }
        ],
    }
    return yaml.safe_dump(data=config, sort_keys=False)
