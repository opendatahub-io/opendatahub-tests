"""Fixtures for the OGX telemetry (Prometheus metrics / OTLP export) tests.

The ``ogx_server`` fixture defined here shadows the one in
``tests/ogx/conftest.py`` for tests under ``tests/ogx/otel/``. The server is
deployed with:

- ``spec.monitoring`` enabled (metrics port 9464) — the operator then creates
  the ServiceMonitor and restricts the metrics port to monitoring namespaces,
  matching the shipped telemetry design (RHAIENG-5155/5156): an auth-exempt
  pull endpoint for cluster-monitoring Prometheus, never exposed via Route.
- OTEL env pointing the existing OTLP push export at the mock MaaS upstream,
  which doubles as a passive OTLP receiver so the export path is observable.

The model is the same mock MaaS upstream used by the MaaS passthrough suite
(echo server), since only the telemetry it triggers is under test.
"""

from collections.abc import Generator
from typing import Any

import pytest
import structlog
import yaml
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.namespace import Namespace

from tests.ogx.mockupstream.constants import (
    MOCK_MAAS_MODEL_ID,
    PASSTHROUGH_FORWARD_HEADERS,
    PASSTHROUGH_PROVIDER_ID,
    PASSTHROUGH_PROVIDER_TYPE,
)
from tests.ogx.mockupstream.deployments import MockUpstream, deploy_mock_maas_server
from tests.ogx.mockupstream.ogx_config import build_override_config_server_spec
from tests.ogx.utils import create_ogx_server
from utilities.resources.ogx_server import OgxServer

LOGGER = structlog.get_logger(name=__name__)

OTEL_SERVER_NAME = "ogx-otel-test"
OTEL_CONFIGMAP_NAME = "ogx-otel-config"
OTEL_CONFIGMAP_KEY = "run.yaml"

# Metrics endpoint exposed by the OGX operator when spec.monitoring is enabled
# (auth-exempt, separate from the API port, not routed — per the telemetry
# design, only network-isolated scrapers may reach it).
OGX_METRICS_PORT = 9464

# The passthrough provider needs a non-empty api_key for OGX's internal
# OpenAI-compatible client; the credential flow is not under test here.
_MOCK_INFERENCE_API_KEY = "mock-inference-key"


def build_otel_run_config(mock_maas_url: str) -> dict[str, Any]:
    """Build the run.yaml for the telemetry test server.

    Same minimal inference-only shape as the MaaS passthrough suite; the
    telemetry-specific parts (OTEL env, metrics port) live on the OGXServer
    spec, not in the run config.
    """
    return {
        "version": 2,
        "distro_name": OTEL_SERVER_NAME,
        "apis": ["inference"],
        "providers": {
            "inference": [
                {
                    "provider_id": PASSTHROUGH_PROVIDER_ID,
                    "provider_type": PASSTHROUGH_PROVIDER_TYPE,
                    "config": {
                        "base_url": mock_maas_url,
                        "api_key": _MOCK_INFERENCE_API_KEY,
                        "forward_headers": dict(PASSTHROUGH_FORWARD_HEADERS),
                    },
                },
            ],
        },
        "storage": {
            "backends": {
                "kv_default": {"type": "kv_sqlite", "db_path": "/tmp/otel-test-kvstore.db"},
                "sql_default": {"type": "sql_sqlite", "db_path": "/tmp/otel-test-sqlstore.db"},
            },
            "stores": {"metadata": {"backend": "kv_default", "namespace": "registry"}},
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


@pytest.fixture(scope="class")
def mock_maas_upstream(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[MockUpstream, Any, Any]:
    """Deploy the mock MaaS upstream; it also receives OGX's OTLP exports."""
    with deploy_mock_maas_server(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace,
        teardown=teardown_resources,
    ) as upstream:
        yield upstream


@pytest.fixture(scope="class")
def ogx_server(  # noqa: UFN001 — intentionally shadows the parent conftest fixture (see module docstring)
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    enabled_ogx_operator: DataScienceCluster,
    mock_maas_upstream: MockUpstream,
    teardown_resources: bool,
    request: FixtureRequest,
) -> Generator[OgxServer, Any, Any]:
    """OGXServer with the Prometheus metrics endpoint and OTLP export enabled."""
    namespace = unprivileged_model_namespace.name
    config_yaml = yaml.dump(build_otel_run_config(mock_maas_url=mock_maas_upstream.service_url))

    if pytestconfig.option.post_upgrade:
        configmap = ConfigMap(
            client=unprivileged_client,
            name=OTEL_CONFIGMAP_NAME,
            namespace=namespace,
            ensure_exists=True,
        )
        configmap.wait(timeout=60)
        ogx_srv = OgxServer(
            client=unprivileged_client,
            name=OTEL_SERVER_NAME,
            namespace=namespace,
            ensure_exists=True,
        )
        ogx_srv.wait_for_status(status=OgxServer.Status.READY, timeout=600)
        yield ogx_srv
        if teardown_resources:
            ogx_srv.clean_up()
        return

    with (
        ConfigMap(
            client=unprivileged_client,
            name=OTEL_CONFIGMAP_NAME,
            namespace=namespace,
            data={OTEL_CONFIGMAP_KEY: config_yaml},
            label={"ogx.io/watch": "true"},
            wait_for_resource=True,
            teardown=teardown_resources,
        ),
        create_ogx_server(
            client=unprivileged_client,
            name=OTEL_SERVER_NAME,
            namespace=namespace,
            config=build_override_config_server_spec(
                configmap_name=OTEL_CONFIGMAP_NAME,
                configmap_key=OTEL_CONFIGMAP_KEY,
                workload_overrides={
                    "overrides": {
                        "env": [
                            {"name": "OTEL_SERVICE_NAME", "value": OTEL_SERVER_NAME},
                            {
                                "name": "OTEL_EXPORTER_OTLP_ENDPOINT",
                                "value": mock_maas_upstream.service_url,
                            },
                            {"name": "OTEL_EXPORTER_OTLP_PROTOCOL", "value": "http/protobuf"},
                            # Speed up the periodic OTLP metric export so tests
                            # observe exports within seconds instead of the
                            # 60s SDK default.
                            {"name": "OTEL_METRIC_EXPORT_INTERVAL", "value": "5000"},
                        ]
                    }
                },
                monitoring={"enabled": True, "metricsPort": OGX_METRICS_PORT},
            ),
            teardown=teardown_resources,
        ) as ogx_srv,
    ):
        ogx_srv.wait_for_status(status=OgxServer.Status.READY, timeout=600)
        yield ogx_srv
