"""Fixtures for the MaaS token passthrough tests.

The ``ogx_server`` fixture defined here shadows the one in
``tests/ogx/conftest.py`` for tests under ``tests/ogx/maaspassthrough/``: the
server is deployed with a full run.yaml override whose only inference path is
the ``genai-bff-proxy`` ``remote::passthrough`` provider pointing at an
in-cluster mock MaaS upstream, mirroring the configuration class the Gen AI
BFF registers for Playgrounds in RHOAI 3.6. The shared
``ogx_server_deployment`` / ``ogx_test_route`` / ``ogx_client`` fixtures
resolve against this shadow.
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
from tests.ogx.mockupstream.deployments import MockUpstream
from tests.ogx.mockupstream.ogx_config import build_override_config_server_spec
from tests.ogx.utils import create_ogx_server
from utilities.resources.ogx_server import OgxServer

LOGGER = structlog.get_logger(name=__name__)

MAAS_PASSTHROUGH_SERVER_NAME = "ogx-maas-passthrough-test"
MAAS_PASSTHROUGH_CONFIGMAP_NAME = "ogx-maas-passthrough-config"
MAAS_PASSTHROUGH_CONFIGMAP_KEY = "run.yaml"


def build_maas_passthrough_run_config(mock_service_url: str) -> dict[str, Any]:
    """Build the run.yaml for the MaaS token passthrough server.

    Mirrors the provider entry the Gen AI BFF registers in RHOAI 3.6: a single
    ``genai-bff-proxy`` ``remote::passthrough`` inference provider with an
    empty ``api_key`` (credentials flow per-request via ``X-OGX-Provider-Data``
    and the ``forward_headers`` mapping) plus one registered LLM model served
    by the mock upstream.
    """
    return {
        "version": 2,
        "distro_name": MAAS_PASSTHROUGH_SERVER_NAME,
        "apis": ["inference"],
        "providers": {
            "inference": [
                {
                    "provider_id": PASSTHROUGH_PROVIDER_ID,
                    "provider_type": PASSTHROUGH_PROVIDER_TYPE,
                    "config": {
                        # OGX appends /v1 automatically (-> /v1/chat/completions).
                        "base_url": mock_service_url,
                        # Empty on purpose: the user token must arrive per-request
                        # via X-OGX-Provider-Data, never baked into the provider.
                        "api_key": "",
                        "forward_headers": dict(PASSTHROUGH_FORWARD_HEADERS),
                    },
                },
            ],
        },
        "storage": {
            "backends": {
                "kv_default": {"type": "kv_sqlite", "db_path": "/tmp/maas-passthrough-kvstore.db"},
                "sql_default": {"type": "sql_sqlite", "db_path": "/tmp/maas-passthrough-sqlstore.db"},
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
def ogx_server(  # noqa: UFN001 — intentionally shadows the parent conftest fixture (see module docstring)
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    enabled_ogx_operator: DataScienceCluster,
    mock_maas_upstream: MockUpstream,
    teardown_resources: bool,
    request: FixtureRequest,
) -> Generator[OgxServer, Any, Any]:
    """OGXServer started with the genai-bff-proxy passthrough provider.

    The ConfigMap (with the ``ogx.io/watch: "true"`` label required by the
    operator for ``spec.overrideConfig`` references) is created before the
    OGXServer to avoid a reconcile failure on a missing reference.
    """
    namespace = unprivileged_model_namespace.name
    config_yaml = yaml.dump(build_maas_passthrough_run_config(mock_service_url=mock_maas_upstream.service_url))

    if pytestconfig.option.post_upgrade:
        configmap = ConfigMap(
            client=unprivileged_client,
            name=MAAS_PASSTHROUGH_CONFIGMAP_NAME,
            namespace=namespace,
            ensure_exists=True,
        )
        configmap.wait(timeout=60)
        ogx_srv = OgxServer(
            client=unprivileged_client,
            name=MAAS_PASSTHROUGH_SERVER_NAME,
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
            name=MAAS_PASSTHROUGH_CONFIGMAP_NAME,
            namespace=namespace,
            data={MAAS_PASSTHROUGH_CONFIGMAP_KEY: config_yaml},
            label={"ogx.io/watch": "true"},
            wait_for_resource=True,
            teardown=teardown_resources,
        ),
        create_ogx_server(
            client=unprivileged_client,
            name=MAAS_PASSTHROUGH_SERVER_NAME,
            namespace=namespace,
            config=build_override_config_server_spec(
                configmap_name=MAAS_PASSTHROUGH_CONFIGMAP_NAME,
                configmap_key=MAAS_PASSTHROUGH_CONFIGMAP_KEY,
            ),
            teardown=teardown_resources,
        ) as ogx_srv,
    ):
        ogx_srv.wait_for_status(status=OgxServer.Status.READY, timeout=600)
        yield ogx_srv


@pytest.fixture(scope="class")
def mock_maas_debug_url(mock_maas_upstream: MockUpstream) -> str:
    """External URL of the mock MaaS debug endpoints (last-headers/requests)."""
    return mock_maas_upstream.route_url
