"""Fixtures for the MCP tool runtime tests.

The ``ogx_server`` fixture defined here shadows the one in
``tests/ogx/conftest.py`` for tests under ``tests/ogx/mcp/``. The server is
deployed with a run.yaml override that:

- registers an in-cluster mock MCP server as a connector (startup
  registration; exposed read-only via the admin connectors API), and
- enables the responses API with the ``inline::builtin`` provider, which is
  what executes MCP tools server-side on behalf of a model.

The model itself is the mock MaaS upstream through the
``genai-bff-proxy`` ``remote::passthrough`` provider: it deterministically
requests the mock MCP server's ``get_weather`` tool, so the whole loop
(model tool-call -> OGX MCP invocation -> tool result in the response) is
assertable without any external dependency.
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
    MOCK_MCP_CONNECTOR_ID,
    MOCK_MCP_SERVER_LABEL,
    PASSTHROUGH_FORWARD_HEADERS,
    PASSTHROUGH_PROVIDER_ID,
    PASSTHROUGH_PROVIDER_TYPE,
)
from tests.ogx.mockupstream.deployments import MockUpstream, deploy_mock_mcp_server
from tests.ogx.mockupstream.ogx_config import build_override_config_server_spec
from tests.ogx.utils import create_ogx_server
from utilities.resources.ogx_server import OgxServer

LOGGER = structlog.get_logger(name=__name__)

MCP_SERVER_NAME = "ogx-mcp-test"
MCP_CONFIGMAP_NAME = "ogx-mcp-config"
MCP_CONFIGMAP_KEY = "run.yaml"

# The passthrough provider needs a non-empty api_key for OGX's internal
# OpenAI-compatible client (the builtin responses provider calls the inference
# provider through it). Unlike the MaaS passthrough suite, the credential flow
# is not under test here.
_MOCK_INFERENCE_API_KEY = "mock-inference-key"  # pragma: allowlist secret


def build_mcp_run_config(mock_maas_url: str, mock_mcp_url: str) -> dict[str, Any]:
    """Build the run.yaml for the MCP tool runtime server.

    Enables the API set the ``inline::builtin`` responses provider requires
    (responses, vector_io, files, file_processors) plus inference and
    tool_runtime, registers the mock MCP server as a connector, and registers
    the mock MaaS model the responses API drives the tool loop with.
    """
    return {
        "version": 2,
        "distro_name": MCP_SERVER_NAME,
        "apis": ["responses", "file_processors", "files", "inference", "tool_runtime", "vector_io"],
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
                        "storage_dir": "/tmp/mcp-test-files",
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
        },
        "storage": {
            "backends": {
                "kv_default": {"type": "kv_sqlite", "db_path": "/tmp/mcp-test-kvstore.db"},
                "sql_default": {"type": "sql_sqlite", "db_path": "/tmp/mcp-test-sqlstore.db"},
            },
            "stores": {
                "metadata": {"backend": "kv_default", "namespace": "registry"},
                "connectors": {"table_name": "connectors", "backend": "sql_default"},
            },
        },
        "connectors": [
            {
                "connector_id": MOCK_MCP_CONNECTOR_ID,
                "url": mock_mcp_url,
                "connector_type": "mcp",
                "server_label": MOCK_MCP_SERVER_LABEL,
            }
        ],
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
def mock_mcp_upstream(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[MockUpstream, Any, Any]:
    """Deploy the mock MCP server in the test namespace (in-cluster only)."""
    with deploy_mock_mcp_server(
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
    mock_mcp_upstream: MockUpstream,
    mock_maas_upstream: MockUpstream,
    teardown_resources: bool,
    request: FixtureRequest,
) -> Generator[OgxServer, Any, Any]:
    """OGXServer with the MCP connector and the responses API enabled."""
    namespace = unprivileged_model_namespace.name
    config_yaml = yaml.dump(
        build_mcp_run_config(
            mock_maas_url=mock_maas_upstream.service_url,
            mock_mcp_url=mock_mcp_upstream.service_url,
        )
    )

    if pytestconfig.option.post_upgrade:
        configmap = ConfigMap(
            client=unprivileged_client,
            name=MCP_CONFIGMAP_NAME,
            namespace=namespace,
            ensure_exists=True,
        )
        configmap.wait(timeout=60)
        ogx_srv = OgxServer(
            client=unprivileged_client,
            name=MCP_SERVER_NAME,
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
            name=MCP_CONFIGMAP_NAME,
            namespace=namespace,
            data={MCP_CONFIGMAP_KEY: config_yaml},
            label={"ogx.io/watch": "true"},
            wait_for_resource=True,
            teardown=teardown_resources,
        ),
        create_ogx_server(
            client=unprivileged_client,
            name=MCP_SERVER_NAME,
            namespace=namespace,
            config=build_override_config_server_spec(
                configmap_name=MCP_CONFIGMAP_NAME,
                configmap_key=MCP_CONFIGMAP_KEY,
            ),
            teardown=teardown_resources,
        ) as ogx_srv,
    ):
        ogx_srv.wait_for_status(status=OgxServer.Status.READY, timeout=600)
        yield ogx_srv


@pytest.fixture(scope="class")
def ogx_admin_api_url(ogx_test_route) -> str:
    """Base URL of the OGX server, for APIs not covered by ogx_client 1.1.3."""
    return f"https://{ogx_test_route.host}"
