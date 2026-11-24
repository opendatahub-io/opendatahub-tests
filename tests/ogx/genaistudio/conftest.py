"""Fixtures for Gen AI Studio Playground configuration tests.

The ``ogx_server`` fixture defined here *shadows* the one in
``tests/ogx/conftest.py`` for tests under ``tests/ogx/genaistudio/``. The shared
``ogx_server_deployment`` / ``ogx_test_route`` / ``ogx_client`` fixtures resolve
against this shadow, so the Playground deployment gets the same route, client
and readiness handling as the regular OGX tests — but the server is deployed
the way the odh-dashboard deploys it for a Playground instead of via the
env-var driven provider configuration.
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

from tests.ogx.constants import (
    GENAI_PLAYGROUND_CONFIGMAP_KEY,
    GENAI_PLAYGROUND_CONFIGMAP_LABELS,
    GENAI_PLAYGROUND_CONFIGMAP_NAME,
    GENAI_PLAYGROUND_SERVER_NAME,
)
from tests.ogx.genaistudio.config import (
    build_genai_playground_config,
    build_genai_playground_server_config,
)
from tests.ogx.utils import create_ogx_server
from utilities.resources.ogx_server import OgxServer

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def ogx_server(  # noqa: UFN001 — intentionally shadows the parent conftest fixture (see module docstring)
    request: FixtureRequest,
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    enabled_ogx_operator: DataScienceCluster,
    teardown_resources: bool,
) -> Generator[OgxServer, Any, Any]:
    """OGXServer started with the Gen AI Studio Playground configuration.

    Shadows ``tests/ogx/conftest.py::ogx_server`` and reproduces what the
    odh-dashboard BFF's ``InstallOGXServer`` (tag v3.6.0) creates for a
    Playground:

    1. ConfigMap (``GENAI_PLAYGROUND_CONFIGMAP_NAME``, key ``config.yaml``)
       holding the generated OGX config, labeled ``ogx.io/watch: "true"``
       (required by the operator for ConfigMaps referenced via
       ``spec.overrideConfig``).
    2. OGXServer ``lsd-genai-playground`` with ``spec.distribution.name: rh``
       and ``spec.overrideConfig`` pointing at the ConfigMap. The ConfigMap is
       created first to avoid a reconcile failure on a missing reference.

    Accepts optional indirect parametrization with ``passthrough_base_url`` to
    point the ``genai-bff-proxy`` provider at an in-cluster mock proxy instead
    of the default non-routable stub (OGX connects lazily, so the stub is
    enough to verify start and provider registration).

    In post-upgrade runs, reuses the resources left by the pre-upgrade leg.
    """
    params = getattr(request, "param", {})
    namespace = unprivileged_model_namespace.name
    config_yaml = yaml.dump(
        build_genai_playground_config(
            namespace=namespace,
            passthrough_base_url=params.get("passthrough_base_url"),
        )
    )

    if pytestconfig.option.post_upgrade:
        configmap = ConfigMap(
            client=unprivileged_client,
            name=GENAI_PLAYGROUND_CONFIGMAP_NAME,
            namespace=namespace,
            ensure_exists=True,
        )
        configmap.wait(timeout=60)
        ogx_srv = OgxServer(
            client=unprivileged_client,
            name=GENAI_PLAYGROUND_SERVER_NAME,
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
            name=GENAI_PLAYGROUND_CONFIGMAP_NAME,
            namespace=namespace,
            data={GENAI_PLAYGROUND_CONFIGMAP_KEY: config_yaml},
            label=dict(GENAI_PLAYGROUND_CONFIGMAP_LABELS),
            wait_for_resource=True,
            teardown=teardown_resources,
        ),
        create_ogx_server(
            client=unprivileged_client,
            name=GENAI_PLAYGROUND_SERVER_NAME,
            namespace=namespace,
            config=build_genai_playground_server_config(),
            teardown=teardown_resources,
        ) as ogx_srv,
    ):
        ogx_srv.wait_for_status(status=OgxServer.Status.READY, timeout=600)
        yield ogx_srv
