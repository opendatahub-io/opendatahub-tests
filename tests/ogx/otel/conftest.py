"""Fixtures for the OGX telemetry (Prometheus metrics / OTLP export) tests.

The ``ogx_server`` fixture defined here shadows the one in
``tests/ogx/conftest.py`` for tests under ``tests/ogx/otel/``. The server is
deployed with:

- ``spec.monitoring`` enabled (metrics port from ``OGX_METRICS_PORT``) — the
  operator then creates the ServiceMonitor and restricts the metrics port to
  monitoring namespaces, matching the shipped telemetry design: an auth-exempt
  pull endpoint for cluster-monitoring Prometheus, never exposed via Route.
- OTEL env pointing the existing OTLP push export at the mock MaaS upstream,
  which doubles as a passive OTLP receiver so the export path is observable.

The model is the same mock MaaS upstream used by the MaaS passthrough suite
(echo server), since only the telemetry it triggers is under test.
"""

from collections.abc import Generator
from typing import Any

import pytest
import yaml
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.namespace import Namespace

from tests.ogx.constants import OGX_METRICS_PORT
from tests.ogx.mockupstream.deployments import MockUpstream
from tests.ogx.mockupstream.ogx_config import (
    build_mock_run_config,
    build_override_config_server_spec,
)
from tests.ogx.utils import create_ogx_server
from utilities.resources.ogx_server import OgxServer

OTEL_SERVER_NAME = "ogx-otel-test"
OTEL_CONFIGMAP_NAME = "ogx-otel-config"
OTEL_CONFIGMAP_KEY = "run.yaml"

# The passthrough provider needs a non-empty api_key for OGX's internal
# OpenAI-compatible client; the credential flow is not under test here.
_MOCK_INFERENCE_API_KEY = "mock-inference-key"  # pragma: allowlist secret


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
    config_yaml = yaml.dump(
        build_mock_run_config(
            distro_name=OTEL_SERVER_NAME,
            mock_service_url=mock_maas_upstream.service_url,
            api_key=_MOCK_INFERENCE_API_KEY,
        )
    )

    if pytestconfig.option.post_upgrade:
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
