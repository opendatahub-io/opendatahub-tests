"""Fixtures for the MaaS token passthrough tests.

The ``ogx_server`` fixture defined here shadows the one in
``tests/ogx/conftest.py`` for tests under ``tests/ogx/maaspassthrough/``: the
server is deployed with a full run.yaml override whose only inference path is
the ``genai-bff-proxy`` ``remote::passthrough`` provider pointing at an
in-cluster mock MaaS upstream, mirroring the configuration class the Gen AI
BFF registers for Playgrounds. The shared
``ogx_server_deployment`` / ``ogx_test_route`` / ``ogx_client`` fixtures
resolve against this shadow.
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

from tests.ogx.maaspassthrough.constants import (
    MAAS_PASSTHROUGH_CONFIGMAP_KEY,
    MAAS_PASSTHROUGH_CONFIGMAP_NAME,
    MAAS_PASSTHROUGH_SERVER_NAME,
)
from tests.ogx.mockupstream.deployments import MockUpstream
from tests.ogx.mockupstream.ogx_config import (
    build_mock_run_config,
    build_override_config_server_spec,
)
from tests.ogx.utils import create_ogx_server
from utilities.resources.ogx_server import OgxServer


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
    OGXServer to avoid a reconcile failure on a missing reference. The
    passthrough provider's api_key is empty on purpose: the user token must
    arrive per-request via ``X-OGX-Provider-Data``, never baked into the
    provider.
    """
    namespace = unprivileged_model_namespace.name
    config_yaml = yaml.dump(
        build_mock_run_config(
            distro_name=MAAS_PASSTHROUGH_SERVER_NAME,
            mock_service_url=mock_maas_upstream.service_url,
            api_key="",
        )
    )

    if pytestconfig.option.post_upgrade:
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
