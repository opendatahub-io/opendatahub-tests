from __future__ import annotations

from collections.abc import Generator
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import portforward
import pytest
import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from requests import RequestException

from tests.ai_hub.image_constants import AiHubImages
from tests.ai_hub.mcp_servers.config.utils import get_mcp_catalog_sources
from tests.ai_hub.mcp_servers.integration.constants import (
    DASHBOARD_LOCAL_PORT,
    DASHBOARD_REMOTE_PORT,
    DEPLOYABLE_MCP_EXPECTED_TOOL_NAMES,
    DEPLOYABLE_MCP_LICENSE,
    DEPLOYABLE_MCP_PATH,
    DEPLOYABLE_MCP_PORT,
    DEPLOYABLE_MCP_PROVIDER,
    DEPLOYABLE_MCP_REGISTRY_NAMESPACE,
    DEPLOYABLE_MCP_SERVER_DESCRIPTION,
    DEPLOYABLE_MCP_SERVER_TITLE,
    DEPLOYABLE_MCP_SOURCE_NAME_PREFIX,
    DEPLOYABLE_MCP_TAGS,
    DEPLOYABLE_MCP_TIMESTAMP_MS,
    DEPLOYABLE_MCP_VERSION,
    MLFLOW_BFF_PATH,
    MLFLOW_BFF_REGISTER_PATH,
    MODEL_REGISTRY_BFF_PATH,
)
from tests.ai_hub.mcp_servers.integration.utils import (
    deploy_yaml_from_spec,
    extract_data_envelope,
    extract_deploy_spec,
    mcp_server_name_path_segment,
    normalize_registry_server_json,
    normalize_registry_tools,
    request_json,
    wait_for_ready_deployment,
)
from tests.ai_hub.utils import execute_get_command_with_retry, wait_for_catalog_api
from utilities.resources.pod import Pod as UtilPod

LOGGER = structlog.get_logger(name=__name__)


def _run_cleanup_request(step: str, **request_kwargs: Any) -> None:
    """Run a cleanup HTTP request, log failures, and continue teardown."""
    try:
        request_json(expect_json=False, **request_kwargs)
    except (AssertionError, RequestException) as expt:
        LOGGER.warning("Cleanup request failed", step=step, error=str(expt))


def _delete_mcp_access_endpoint(
    dashboard_api_base_url: str,
    encoded_name: str,
    endpoint_id: str | None,
    endpoint_url: str,
    headers: dict[str, str],
    workspace_name: str,
) -> None:
    """Delete a created MLflow access endpoint by id or best-effort lookup."""
    if endpoint_id:
        _run_cleanup_request(
            step="delete mlflow access endpoint by id",
            method="DELETE",
            url=(
                f"{dashboard_api_base_url}{MLFLOW_BFF_PATH}/"
                f"{encoded_name}/endpoints/{quote(endpoint_id, safe='')}"
                f"?workspace={quote(workspace_name, safe='')}"
            ),
            headers=headers,
            verify=False,
            expected_status_codes={200, 204, 404},
        )
        return

    endpoint_list_payload = request_json(
        method="GET",
        url=(
            f"{dashboard_api_base_url}{MLFLOW_BFF_PATH}/{encoded_name}/endpoints"
            f"?workspace={quote(workspace_name, safe='')}"
        ),
        headers=headers,
        verify=False,
        expected_status_codes={200, 404},
        expect_json=False,
    )
    endpoint_list_data = endpoint_list_payload.get("data", endpoint_list_payload)
    endpoint_candidates: list[dict[str, Any]] = []
    if isinstance(endpoint_list_data, list):
        endpoint_candidates = [item for item in endpoint_list_data if isinstance(item, dict)]
    elif isinstance(endpoint_list_data, dict):
        if isinstance(endpoint_list_data.get("items"), list):
            endpoint_candidates = [item for item in endpoint_list_data["items"] if isinstance(item, dict)]
        elif isinstance(endpoint_list_data.get("endpoints"), list):
            endpoint_candidates = [item for item in endpoint_list_data["endpoints"] if isinstance(item, dict)]

    matched_endpoint = next(
        (
            endpoint
            for endpoint in endpoint_candidates
            if endpoint.get("endpoint_url") == endpoint_url and endpoint.get("server_version") == DEPLOYABLE_MCP_VERSION
        ),
        None,
    )
    if matched_endpoint and matched_endpoint.get("id"):
        _run_cleanup_request(
            step="delete mlflow access endpoint by lookup",
            method="DELETE",
            url=(
                f"{dashboard_api_base_url}{MLFLOW_BFF_PATH}/"
                f"{encoded_name}/endpoints/{quote(str(matched_endpoint['id']), safe='')}"
                f"?workspace={quote(workspace_name, safe='')}"
            ),
            headers=headers,
            verify=False,
            expected_status_codes={200, 204, 404},
        )


@pytest.fixture(scope="class")
def mcp_registry_auth_headers(current_client_token: str) -> dict[str, str]:
    """Build auth headers for dashboard BFF requests in the live cluster."""
    return {
        "Authorization": f"Bearer {current_client_token}",
        "X-Forwarded-Access-Token": current_client_token,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


@pytest.fixture(scope="class")
def mcp_registry_test_metadata(model_namespace: Namespace) -> dict[str, Any]:
    """Build a unique catalog and registry payload for the happy-path test."""
    suffix = f"{model_namespace.name.removeprefix('test-')[-15:]}-{uuid4().hex[:8]}"
    deployment_name = f"kubernetes-mcp-{suffix}"
    server_name = f"{DEPLOYABLE_MCP_REGISTRY_NAMESPACE}/{deployment_name}"
    source_id = f"mcp_registry_{suffix.replace('-', '_')}"
    source_name = f"{DEPLOYABLE_MCP_SOURCE_NAME_PREFIX} {suffix}"
    yaml_catalog_path = f"mcp-registry-{suffix}.yaml"

    deploy_spec = {
        "source": {
            "type": "ContainerImage",
            "containerImage": {"ref": AiHubImages.KUBERNETES_MCP_SERVER_V0_0_65},
        },
        "config": {
            "port": DEPLOYABLE_MCP_PORT,
            "path": DEPLOYABLE_MCP_PATH,
        },
        "runtime": {"replicas": 1},
    }

    registry_server_json = {
        "name": server_name,
        "version": DEPLOYABLE_MCP_VERSION,
        "title": DEPLOYABLE_MCP_SERVER_TITLE,
        "description": DEPLOYABLE_MCP_SERVER_DESCRIPTION,
        "packages": [
            {
                "registryType": "oci",
                "identifier": AiHubImages.KUBERNETES_MCP_SERVER_V0_0_65,
                "transport": {"type": "streamable-http"},
            }
        ],
        "remotes": [
            {
                "type": "streamable-http",
                "url": f"https://{deployment_name}.invalid{DEPLOYABLE_MCP_PATH}",
            }
        ],
        "_meta": {
            "com.redhat/deploy-spec": deploy_spec,
        },
    }

    catalog_tools = [
        {"name": "resources_list", "description": "List Kubernetes resources."},
        {"name": "pods_list", "description": "List pods across the cluster."},
        {"name": "namespaces_list", "description": "List Kubernetes namespaces."},
    ]
    assert {tool["name"] for tool in catalog_tools} == set(DEPLOYABLE_MCP_EXPECTED_TOOL_NAMES)

    catalog_yaml = yaml.safe_dump(
        {
            "mcp_servers": [
                {
                    "name": server_name,
                    "description": DEPLOYABLE_MCP_SERVER_DESCRIPTION,
                    "provider": DEPLOYABLE_MCP_PROVIDER,
                    "version": DEPLOYABLE_MCP_VERSION,
                    "license": DEPLOYABLE_MCP_LICENSE,
                    "tags": list(DEPLOYABLE_MCP_TAGS),
                    "tools": catalog_tools,
                    "server_json": registry_server_json,
                    "createTimeSinceEpoch": DEPLOYABLE_MCP_TIMESTAMP_MS,
                    "lastUpdateTimeSinceEpoch": DEPLOYABLE_MCP_TIMESTAMP_MS,
                }
            ]
        },
        sort_keys=False,
    )

    return {
        "catalog_tools": catalog_tools,
        "deployment_name": deployment_name,
        "server_name": server_name,
        "source_id": source_id,
        "source_name": source_name,
        "yaml_catalog_path": yaml_catalog_path,
        "catalog_yaml": catalog_yaml,
    }


@pytest.fixture(scope="class")
def dashboard_api_base_url(dashboard_pod: UtilPod) -> Generator[str, Any]:
    """Port-forward the dashboard pod to exercise the same BFF routes the dashboard uses."""
    dashboard_url = f"https://127.0.0.1:{DASHBOARD_LOCAL_PORT}"

    try:
        with portforward.forward(
            pod_or_service=dashboard_pod.name,
            namespace=dashboard_pod.namespace,
            from_port=DASHBOARD_LOCAL_PORT,
            to_port=DASHBOARD_REMOTE_PORT,
            waiting=20,
        ):
            LOGGER.info(
                "Dashboard API port-forward established",
                dashboard_pod=dashboard_pod.name,
                dashboard_url=dashboard_url,
            )
            yield dashboard_url
    except Exception as expt:
        LOGGER.error(f"Failed to set up port forwarding for pod {dashboard_pod.name}: {expt}")
        raise


@pytest.fixture(scope="class")
def deployable_mcp_catalog_source(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mcp_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
    mcp_registry_test_metadata: dict[str, Any],
) -> Generator[None, Any]:
    """Patch the MCP catalog with one deterministic deployable server entry."""
    catalog_config_map, current_data = get_mcp_catalog_sources(
        admin_client=admin_client,
        model_registry_namespace=model_registry_namespace,
    )
    current_data.setdefault("mcp_catalogs", [])
    current_data["mcp_catalogs"].append({
        "name": mcp_registry_test_metadata["source_name"],
        "id": mcp_registry_test_metadata["source_id"],
        "type": "yaml",
        "enabled": True,
        "properties": {"yamlCatalogPath": mcp_registry_test_metadata["yaml_catalog_path"]},
        "labels": [mcp_registry_test_metadata["source_name"]],
    })

    patches = {
        "data": {
            "sources.yaml": yaml.safe_dump(current_data, sort_keys=False),
            mcp_registry_test_metadata["yaml_catalog_path"]: mcp_registry_test_metadata["catalog_yaml"],
        }
    }

    pre_patch_size = execute_get_command_with_retry(
        url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
        headers=model_registry_rest_headers,
        params={"pageSize": 1000},
    ).get("size", 0)

    with ResourceEditor(patches={catalog_config_map: patches}):
        wait_for_catalog_api(
            endpoint="mcp_servers",
            item_name="MCP servers",
            url=mcp_catalog_rest_urls[0],
            headers=model_registry_rest_headers,
            previous_size=pre_patch_size,
            expected_size=pre_patch_size + 1,
        )
        yield

    wait_for_catalog_api(
        endpoint="mcp_servers",
        item_name="MCP servers",
        url=mcp_catalog_rest_urls[0],
        headers=model_registry_rest_headers,
        previous_size=pre_patch_size + 1,
        expected_size=pre_patch_size,
    )


@pytest.fixture(scope="class")
def catalog_server_details(
    deployable_mcp_catalog_source: None,
    mcp_catalog_rest_urls: list[str],
    mcp_registry_test_metadata: dict[str, Any],
    model_registry_rest_headers: dict[str, str],
) -> dict[str, Any]:
    """Resolve the exact injected catalog entry and return its registry-ready payloads."""
    server_name = mcp_registry_test_metadata["server_name"]
    catalog_response = execute_get_command_with_retry(
        url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
        headers=model_registry_rest_headers,
        params={"pageSize": 1000},
    )
    catalog_server = next(
        (
            server
            for server in catalog_response.get("items", [])
            if server.get("name") == server_name and server.get("source_id") == mcp_registry_test_metadata["source_id"]
        ),
        None,
    )
    assert catalog_server, f"Custom MCP server {server_name} was not found in the live catalog response"
    catalog_server_id = catalog_server.get("id")
    assert catalog_server_id, f"Catalog server {server_name} did not expose an id: {catalog_server}"

    catalog_server_detail = execute_get_command_with_retry(
        url=f"{mcp_catalog_rest_urls[0]}mcp_servers/{quote(str(catalog_server_id), safe='')}",
        headers=model_registry_rest_headers,
    )
    assert catalog_server_detail.get("name") == server_name, (
        f"Catalog detail for MCP server id {catalog_server_id} did not match expected name {server_name}: "
        f"{catalog_server_detail}"
    )
    catalog_server_json = catalog_server_detail.get("serverJson")
    assert isinstance(catalog_server_json, dict), (
        f"Catalog detail for MCP server {server_name} did not include serverJson: {catalog_server_detail}"
    )
    normalized_server_json = normalize_registry_server_json(server_json=catalog_server_json)

    catalog_server_tools_response = execute_get_command_with_retry(
        url=f"{mcp_catalog_rest_urls[0]}mcp_servers/{quote(str(catalog_server_id), safe='')}/tools",
        headers=model_registry_rest_headers,
    )
    catalog_server_tools = catalog_server_tools_response.get("items") or []
    assert isinstance(catalog_server_tools, list), (
        f"Catalog tools for server {server_name} were not returned as a list: {catalog_server_tools_response}"
    )
    normalized_tools = normalize_registry_tools(tools=catalog_server_tools)
    assert {tool["name"] for tool in normalized_tools if "name" in tool} >= set(DEPLOYABLE_MCP_EXPECTED_TOOL_NAMES), (
        f"Catalog tools for server {server_name} did not include expected tools: {normalized_tools}"
    )

    return {
        "deploy_spec": extract_deploy_spec(server_json=normalized_server_json),
        "encoded_name": mcp_server_name_path_segment(name=server_name),
        "server_json": normalized_server_json,
        "server_name": server_name,
        "tools": normalized_tools,
    }


@pytest.fixture(scope="class")
def registered_mcp_server_version(
    catalog_server_details: dict[str, Any],
    dashboard_api_base_url: str,
    mcp_registry_auth_headers: dict[str, str],
    mcp_registry_test_metadata: dict[str, Any],
    model_namespace: Namespace,
) -> Generator[dict[str, Any], Any]:
    """Register the catalog-originated MCP server in MLflow and clean it up afterward."""
    workspace_name = model_namespace.name
    encoded_name = catalog_server_details["encoded_name"]
    registry_version_url = (
        f"{dashboard_api_base_url}{MLFLOW_BFF_REGISTER_PATH}?workspace={quote(workspace_name, safe='')}"
    )

    created_registry_version = False

    try:
        version_payload = request_json(
            method="POST",
            url=registry_version_url,
            headers=mcp_registry_auth_headers,
            verify=False,
            expected_status_codes={201},
            json_body={
                "name": catalog_server_details["server_name"],
                "display_name": DEPLOYABLE_MCP_SERVER_TITLE,
                "server_json": catalog_server_details["server_json"],
                "status": "active",
                "source": mcp_registry_test_metadata["source_id"],
                "tools": catalog_server_details["tools"],
            },
        )
        created_registry_version = True
        extract_data_envelope(payload=version_payload, url=registry_version_url)

        yield {
            "encoded_name": encoded_name,
            "server_name": catalog_server_details["server_name"],
            "workspace_name": workspace_name,
        }
    finally:
        if created_registry_version:
            _run_cleanup_request(
                step="deprecate mlflow registry version",
                method="PATCH",
                url=(
                    f"{dashboard_api_base_url}{MLFLOW_BFF_PATH}/{encoded_name}/versions/"
                    f"{quote(DEPLOYABLE_MCP_VERSION, safe='')}?workspace={quote(workspace_name, safe='')}"
                ),
                headers=mcp_registry_auth_headers,
                verify=False,
                expected_status_codes={200, 404},
                json_body={"status": "deprecated"},
            )
            _run_cleanup_request(
                step="delete mlflow registry version",
                method="DELETE",
                url=(
                    f"{dashboard_api_base_url}{MLFLOW_BFF_PATH}/{encoded_name}/versions/"
                    f"{quote(DEPLOYABLE_MCP_VERSION, safe='')}?workspace={quote(workspace_name, safe='')}"
                ),
                headers=mcp_registry_auth_headers,
                verify=False,
                expected_status_codes={200, 204, 404},
            )
            _run_cleanup_request(
                step="delete mlflow registry server",
                method="DELETE",
                url=(
                    f"{dashboard_api_base_url}{MLFLOW_BFF_PATH}/{encoded_name}"
                    f"?workspace={quote(workspace_name, safe='')}"
                ),
                headers=mcp_registry_auth_headers,
                verify=False,
                expected_status_codes={200, 204, 404},
            )


@pytest.fixture(scope="class")
def mcp_deployment(
    catalog_server_details: dict[str, Any],
    dashboard_api_base_url: str,
    mcp_registry_auth_headers: dict[str, str],
    mcp_registry_test_metadata: dict[str, Any],
    model_namespace: Namespace,
    registered_mcp_server_version: dict[str, Any],
) -> Generator[dict[str, Any], Any]:
    """Deploy the registered MCP server through the dashboard BFF and clean it up afterward."""
    workspace_name = model_namespace.name
    requested_deployment_name = mcp_registry_test_metadata["deployment_name"]
    deployment_url = f"{dashboard_api_base_url}{MODEL_REGISTRY_BFF_PATH}?namespace={quote(workspace_name, safe='')}"

    created_deployment = False
    deployment_name = requested_deployment_name

    try:
        deployment_payload = request_json(
            method="POST",
            url=deployment_url,
            headers=mcp_registry_auth_headers,
            verify=False,
            expected_status_codes={201},
            json_body={
                "data": {
                    "name": requested_deployment_name,
                    "displayName": "QE MCP registry deployment",
                    "registryServer": registered_mcp_server_version["server_name"],
                    "registryVersion": DEPLOYABLE_MCP_VERSION,
                    "image": catalog_server_details["deploy_spec"]["source"]["containerImage"]["ref"],
                    "yaml": deploy_yaml_from_spec(deploy_spec=catalog_server_details["deploy_spec"]),
                }
            },
        )
        created_deployment = True
        deployment_data = extract_data_envelope(payload=deployment_payload, url=deployment_url)
        assert deployment_data["name"] == requested_deployment_name

        ready_deployment = wait_for_ready_deployment(
            base_url=dashboard_api_base_url,
            deployment_name=deployment_name,
            namespace_name=workspace_name,
            headers=mcp_registry_auth_headers,
            verify=False,
        )

        yield {
            "deployment_name": deployment_name,
            "encoded_name": registered_mcp_server_version["encoded_name"],
            "ready_deployment": ready_deployment,
            "server_name": registered_mcp_server_version["server_name"],
            "workspace_name": workspace_name,
        }
    finally:
        if created_deployment:
            _run_cleanup_request(
                step="delete model registry deployment",
                method="DELETE",
                url=(
                    f"{dashboard_api_base_url}{MODEL_REGISTRY_BFF_PATH}/{deployment_name}"
                    f"?namespace={quote(workspace_name, safe='')}"
                ),
                headers=mcp_registry_auth_headers,
                verify=False,
                expected_status_codes={200, 204, 404},
            )


@pytest.fixture(scope="class")
def mcp_access_endpoint(
    dashboard_api_base_url: str,
    mcp_deployment: dict[str, Any],
    mcp_registry_auth_headers: dict[str, str],
) -> Generator[dict[str, Any], Any]:
    """Create the MLflow access endpoint record for the deployed MCP server and clean it up afterward."""
    ready_deployment = mcp_deployment["ready_deployment"]
    deployment_name = mcp_deployment["deployment_name"]
    workspace_name = mcp_deployment["workspace_name"]
    endpoint_url = ready_deployment.get("address", {}).get("url") or (
        f"http://{deployment_name}.{workspace_name}.svc.cluster.local:"
        f"{ready_deployment['port']}{ready_deployment.get('path') or DEPLOYABLE_MCP_PATH}"
    )

    endpoint_id: str | None = None

    try:
        endpoint_resource_url = (
            f"{dashboard_api_base_url}{MLFLOW_BFF_PATH}/{mcp_deployment['encoded_name']}/endpoints"
            f"?workspace={quote(workspace_name, safe='')}"
        )
        endpoint_payload = request_json(
            method="POST",
            url=endpoint_resource_url,
            headers=mcp_registry_auth_headers,
            verify=False,
            expected_status_codes={201},
            json_body={
                "endpoint_url": endpoint_url,
                "transport_type": "streamable-http",
                "server_version": DEPLOYABLE_MCP_VERSION,
            },
        )
        endpoint_data = extract_data_envelope(payload=endpoint_payload, url=endpoint_resource_url)
        endpoint_id = endpoint_data["id"]
        stored_endpoint_payload = request_json(
            method="GET",
            url=(
                f"{dashboard_api_base_url}{MLFLOW_BFF_PATH}/{mcp_deployment['encoded_name']}/endpoints/"
                f"{quote(endpoint_id, safe='')}?workspace={quote(workspace_name, safe='')}"
            ),
            headers=mcp_registry_auth_headers,
            verify=False,
            expected_status_codes={200},
        )
        stored_endpoint_data = extract_data_envelope(payload=stored_endpoint_payload, url=endpoint_resource_url)
        assert stored_endpoint_data["server_name"] == mcp_deployment["server_name"]
        assert stored_endpoint_data["server_version"] == DEPLOYABLE_MCP_VERSION
        assert stored_endpoint_data["endpoint_url"] == endpoint_url

        yield {
            **mcp_deployment,
            "endpoint_id": endpoint_id,
            "endpoint_url": endpoint_url,
        }
    finally:
        _delete_mcp_access_endpoint(
            dashboard_api_base_url=dashboard_api_base_url,
            encoded_name=str(mcp_deployment["encoded_name"]),
            endpoint_id=endpoint_id,
            endpoint_url=endpoint_url,
            headers=mcp_registry_auth_headers,
            workspace_name=workspace_name,
        )
