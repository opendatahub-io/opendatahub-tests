"""OGXServer spec + run-config helpers for mock-upstream-backed test servers.

All three feature suites (MaaS token passthrough, MCP tool runtime, OTel
metrics) deploy OGX the same way: a ConfigMap holding a full ``run.yaml``
(referenced via ``spec.overrideConfig``) plus an OGXServer CR with the workload
resources and route-ingress network policy the shared test fixtures need.
This module centralizes that spec; the suites only build their ``run.yaml``.
"""

from typing import Any

# Route-ingress network policy matching the shared ogx_test_route/ogx_client
# fixtures (ingress to the OGX API port from the router and host network only).
_INGRESS_NETWORK_POLICY = {
    "policy": {
        "ingress": [
            {
                "ports": [{"port": 8321, "protocol": "TCP"}],
                "from": [
                    {"namespaceSelector": {"matchLabels": {"network.openshift.io/policy-group": "ingress"}}},
                    {"namespaceSelector": {"matchLabels": {"network.openshift.io/policy-group": "host-network"}}},
                ],
            }
        ]
    }
}


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
