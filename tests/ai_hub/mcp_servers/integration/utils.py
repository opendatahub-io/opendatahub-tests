from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

import pytest
import requests
import yaml
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_hub.mcp_servers.integration.constants import MODEL_REGISTRY_BFF_PATH


def mcp_server_name_path_segment(name: str) -> str:
    segments = name.split("/")
    invalid_segments = {"", ".", ".."}
    if any(segment in invalid_segments for segment in segments):
        raise ValueError(f"Invalid MCP server name for path usage: {name}")
    return "/".join(quote(segment, safe="") for segment in segments)


def request_json(
    method: str,
    url: str,
    headers: dict[str, str],
    verify: str | bool,
    expected_status_codes: set[int],
    json_body: dict[str, Any] | None = None,
    expect_json: bool = True,
) -> dict[str, Any]:
    response = requests.request(
        method=method,
        url=url,
        headers=headers,
        json=json_body,
        verify=verify,
        timeout=60,
    )
    assert response.status_code in expected_status_codes, (
        f"{method} {url} returned {response.status_code}, expected {sorted(expected_status_codes)}: {response.text}"
    )
    if response.status_code == 204 or not response.content.strip():
        return {}
    try:
        return response.json()
    except ValueError as expt:
        if not expect_json:
            return {}
        raise AssertionError(
            f"Expected JSON response from {method} {url} with status {response.status_code}, got: {response.text!r}"
        ) from expt


def extract_data_envelope(payload: dict[str, Any], url: str) -> dict[str, Any]:
    data = payload.get("data")
    assert isinstance(data, dict), f"Expected response envelope with object data from {url}, got: {payload}"
    return data


def normalize_registry_server_json(server_json: dict[str, Any]) -> dict[str, Any]:
    """Strip catalog wrapper fields that the MLflow registry create API rejects."""
    normalized_server_json = server_json.get("server", server_json)
    assert isinstance(normalized_server_json, dict), f"Expected registry server JSON object, got: {server_json}"

    catalog_only_keys = {
        "createTimeSinceEpoch",
        "lastUpdateTimeSinceEpoch",
        "source",
        "source_id",
        "sourceId",
        "provider",
        "license",
        "tags",
        "tools",
        "id",
    }
    return {key: value for key, value in normalized_server_json.items() if key not in catalog_only_keys}


def normalize_registry_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Keep only MLflow-compatible tool fields from catalog tool objects."""
    allowed_keys = {
        "name",
        "description",
        "title",
        "inputSchema",
        "outputSchema",
        "annotations",
        "icons",
        "_meta",
    }
    normalized_tools: list[dict[str, Any]] = []
    for tool in tools:
        assert isinstance(tool, dict), f"Expected tool entry object, got: {tool}"
        normalized_tools.append({key: value for key, value in tool.items() if key in allowed_keys})
    return normalized_tools


def extract_deploy_spec(server_json: dict[str, Any]) -> dict[str, Any]:
    """Return the deploy spec embedded in catalog-originated server JSON."""
    deploy_spec = server_json.get("_meta", {}).get("com.redhat/deploy-spec")
    assert isinstance(deploy_spec, dict), f"Expected deploy spec in server JSON metadata, got: {server_json}"
    return deploy_spec


def deploy_yaml_from_spec(deploy_spec: dict[str, Any]) -> str:
    """Build deployment YAML from the catalog/registry deploy spec."""
    return yaml.safe_dump(
        {
            "config": deploy_spec["config"],
            "runtime": deploy_spec["runtime"],
        },
        sort_keys=False,
    )


async def _list_mcp_tools(endpoint_url: str) -> list[Any]:
    """Return tools advertised by the deployed MCP server over streamable HTTP."""
    async with Client(StreamableHttpTransport(url=endpoint_url)) as client:
        return await client.list_tools()


def probe_mcp_tools(endpoint_url: str) -> list[Any]:
    """Query the deployed MCP endpoint and return its advertised tools."""
    return asyncio.run(main=_list_mcp_tools(endpoint_url=endpoint_url))


def deployment_ready(deployment: dict[str, Any]) -> bool:
    for condition in deployment.get("conditions", []):
        if condition.get("type") == "Ready" and condition.get("status") == "True":
            return True
    return False


def wait_for_ready_deployment(
    base_url: str,
    deployment_name: str,
    namespace_name: str,
    headers: dict[str, str],
    verify: str | bool,
) -> dict[str, Any]:
    deployment_url = f"{base_url}{MODEL_REGISTRY_BFF_PATH}/{deployment_name}?namespace={namespace_name}"
    sampler = TimeoutSampler(
        wait_timeout=300,
        sleep=5,
        func=request_json,
        method="GET",
        url=deployment_url,
        headers=headers,
        verify=verify,
        expected_status_codes={200},
    )

    last_seen: dict[str, Any] = {}
    try:
        for sample in sampler:
            last_seen = extract_data_envelope(payload=sample, url=deployment_url)
            if deployment_ready(deployment=last_seen):
                return last_seen
    except TimeoutExpiredError as expt:
        raise AssertionError(f"MCP deployment {deployment_name} did not become Ready: {last_seen}") from expt

    pytest.fail(f"MCP deployment {deployment_name} did not become Ready: {last_seen}")
