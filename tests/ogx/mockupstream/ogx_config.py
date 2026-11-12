"""OGXServer spec + run-config helpers for mock-upstream-backed test servers.

All three feature suites (MaaS token passthrough, MCP tool runtime, OTel
metrics) deploy OGX the same way: a ConfigMap holding a full ``run.yaml``
(referenced via ``spec.overrideConfig``) plus an OGXServer CR with the workload
resources and route-ingress network policy the shared test fixtures need.
This module centralizes both: ``build_mock_run_config`` assembles the run.yaml
(parameterized the way ``build_ogx_server_config`` parameterizes server specs)
and ``build_override_config_server_spec`` assembles the OGXServer spec.
"""

from typing import Any

from tests.ogx.constants import OGX_METRICS_PORT
from tests.ogx.mockupstream.constants import (
    MOCK_MAAS_MODEL_ID,
    MOCK_MCP_CONNECTOR_ID,
    MOCK_MCP_SERVER_LABEL,
    PASSTHROUGH_FORWARD_HEADERS,
    PASSTHROUGH_PROVIDER_ID,
    PASSTHROUGH_PROVIDER_TYPE,
)

# Route-ingress network policy matching the shared ogx_test_route/ogx_client
# fixtures (ingress to the OGX API port from the router and host network only).
# A custom ``network`` spec replaces the operator's default ingress rules
# entirely, so the default rule letting monitoring namespaces reach the metrics
# port must be re-added here — otherwise user-workload Prometheus cannot scrape
# 9464 and the UWM telemetry test cannot pass.
_INGRESS_NETWORK_POLICY = {
    "policy": {
        "ingress": [
            {
                "ports": [{"port": 8321, "protocol": "TCP"}],
                "from": [
                    {"namespaceSelector": {"matchLabels": {"network.openshift.io/policy-group": "ingress"}}},
                    {"namespaceSelector": {"matchLabels": {"network.openshift.io/policy-group": "host-network"}}},
                ],
            },
            {
                "ports": [{"port": OGX_METRICS_PORT, "protocol": "TCP"}],
                "from": [
                    {"namespaceSelector": {"matchLabels": {"network.openshift.io/policy-group": "monitoring"}}},
                ],
            },
        ]
    }
}


def build_mock_run_config(
    distro_name: str,
    mock_service_url: str,
    api_key: str,
    mcp_endpoint: str | None = None,
) -> dict[str, Any]:
    """Build the run.yaml for an OGX server whose inference path is the mock MaaS.

    Mirrors the provider entry the Gen AI BFF registers (a single
    ``genai-bff-proxy`` ``remote::passthrough`` inference provider pointing at
    the mock upstream, plus one registered LLM model), on SQLite state stores.

    Args:
        distro_name: ``distro_name`` embedded in the config.
        mock_service_url: In-cluster URL of the mock MaaS upstream (OGX appends
            ``/v1`` automatically).
        api_key: The passthrough provider's ``api_key``. Empty when the
            per-request credential flow is under test (the user token then
            arrives via ``X-OGX-Provider-Data``); a dummy value when a
            non-empty key is required for OGX's internal OpenAI-compatible
            client (e.g. the responses API's model calls).
        mcp_endpoint: When set, register the mock MCP server as a connector
            and persist connectors (used by the MCP tool-runtime tests).
    """
    apis = ["inference", "responses", "file_processors", "files", "tool_runtime", "vector_io"]
    providers: dict[str, list[dict[str, Any]]] = {
        "inference": [
            {
                "provider_id": PASSTHROUGH_PROVIDER_ID,
                "provider_type": PASSTHROUGH_PROVIDER_TYPE,
                "config": {
                    "base_url": mock_service_url,
                    "api_key": api_key,
                    "forward_headers": dict(PASSTHROUGH_FORWARD_HEADERS),
                },
            },
        ],
    }
    stores: dict[str, Any] = {"metadata": {"backend": "kv_default", "namespace": "registry"}}
    config: dict[str, Any] = {
        "version": 2,
        "distro_name": distro_name,
        "apis": apis,
        "providers": providers,
        "storage": {
            "backends": {
                "kv_default": {"type": "kv_sqlite", "db_path": f"/tmp/{distro_name}-kvstore.db"},
                "sql_default": {"type": "sql_sqlite", "db_path": f"/tmp/{distro_name}-sqlstore.db"},
            },
            "stores": stores,
        },
        "registered_resources": {
            "models": [
                {
                    "provider_id": PASSTHROUGH_PROVIDER_ID,
                    "model_id": MOCK_MAAS_MODEL_ID,
                    "provider_model_id": MOCK_MAAS_MODEL_ID,
                    "model_type": "llm",
                    "metadata": {"provider_resource_id": MOCK_MAAS_MODEL_ID},
                }
            ]
        },
    }

    # The API set the inline::builtin responses provider requires, plus
    # tool_runtime and vector_io. The shared ogx_client fixture's readiness
    # check (and the dashboard's Playground configuration) expect a full
    # server, so every mock-backed suite deploys this set.
    providers.update({
        "responses": [
            {
                "provider_id": "builtin",
                "provider_type": "inline::builtin",
                "config": {
                    "persistence": {
                        "responses": {
                            "table_name": "responses",
                            "backend": "sql_default",
                            "max_write_queue_size": 10000,
                            "num_writers": 4,
                        }
                    }
                },
            }
        ],
        "file_processors": [{"provider_id": "auto", "provider_type": "inline::auto", "config": {}}],
        "files": [
            {
                "provider_id": "localfs-files",
                "provider_type": "inline::localfs",
                "config": {
                    "storage_dir": f"/tmp/{distro_name}-files",
                    "metadata_store": {"table_name": "files_metadata", "backend": "sql_default"},
                },
            }
        ],
        "tool_runtime": [
            {
                "provider_id": "model-context-protocol",
                "provider_type": "remote::model-context-protocol",
                "config": {},
            }
        ],
        "vector_io": [
            {
                "provider_id": "faiss",
                "provider_type": "inline::faiss",
                "config": {"persistence": {"namespace": "faiss", "backend": "kv_default"}},
            }
        ],
    })

    if mcp_endpoint:
        stores["connectors"] = {"table_name": "connectors", "backend": "sql_default"}
        config["connectors"] = [
            {
                "connector_id": MOCK_MCP_CONNECTOR_ID,
                "url": mcp_endpoint,
                "connector_type": "mcp",
                "server_label": MOCK_MCP_SERVER_LABEL,
            }
        ]

    return config


def build_override_config_server_spec(
    configmap_name: str,
    configmap_key: str,
    workload_overrides: dict[str, Any] | None = None,
    monitoring: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the OGXServer spec for a server configured via ``spec.overrideConfig``.

    The referenced ConfigMap (with label ``ogx.io/watch: "true"``) must exist in
    the same namespace before the OGXServer is created.

    Args:
        configmap_name: ConfigMap holding the full run.yaml config.
        configmap_key: ConfigMap key holding the config content.
        workload_overrides: Extra ``workload`` fields (e.g. ``overrides.env``
            for OTEL_* variables), merged over the defaults.
        monitoring: ``spec.monitoring`` (e.g. Prometheus metrics endpoint).

    Returns:
        OGXServerSpec configuration dict for ``create_ogx_server``.
    """
    workload: dict[str, Any] = {
        "resources": {
            "requests": {"cpu": "1", "memory": "1Gi"},
            "limits": {"cpu": "2", "memory": "2Gi"},
        },
    }
    if workload_overrides:
        workload.update(workload_overrides)

    config: dict[str, Any] = {
        "distribution": {"name": "rh"},
        "overrideConfig": {"name": configmap_name, "key": configmap_key},
        "workload": workload,
        "network": _INGRESS_NETWORK_POLICY,
    }
    if monitoring:
        config["monitoring"] = monitoring
    return config
