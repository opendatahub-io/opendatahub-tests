"""Utilities for EvalHub MCP Streamable HTTP transport tests."""

from __future__ import annotations

import json
from typing import Any

import requests
import structlog
from timeout_sampler import TimeoutSampler

from tests.ai_safety.evalhub.constants import EVALHUB_VLLM_EMULATOR_PORT
from tests.ai_safety.evalhub.mcp.constants import (
    EVALHUB_MCP_CLIENT_NAME,
    EVALHUB_MCP_CLIENT_VERSION,
    EVALHUB_MCP_DEFAULT_BENCHMARK_ID,
    EVALHUB_MCP_DEFAULT_PROVIDER_ID,
    EVALHUB_MCP_PROTOCOL_VERSION,
    EVALHUB_MCP_PROXY_RESOURCE,
)
from tests.ai_safety.evalhub.utils import (
    EVALHUB_JOB_TERMINAL_STATES,
    build_headers,
    build_vllm_arc_easy_benchmark,
)

LOGGER = structlog.get_logger(name=__name__)


class McpProtocolError(Exception):
    """Raised when an MCP JSON-RPC response contains an error."""


class EvalHubMcpClient:
    """Minimal Streamable HTTP client for evalhub-mcp JSON-RPC."""

    def __init__(
        self,
        host: str,
        token: str,
        ca_bundle_file: str,
        tenant: str,
        timeout: int = 30,
    ) -> None:
        self.base_url = f"https://{host}"
        self.headers = build_headers(token=token, tenant=tenant)
        self.ca_bundle_file = ca_bundle_file
        self.timeout = timeout
        self.session_id: str | None = None
        self._request_id = 0
        self._initialized = False
        self.initialize_result: dict[str, Any] = {}

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _post(
        self,
        payload: dict[str, Any],
        notification: bool = False,
    ) -> requests.Response:
        headers = {
            **self.headers,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        response = requests.post(
            url=self.base_url,
            headers=headers,
            json=payload,
            verify=self.ca_bundle_file,
            timeout=self.timeout,
        )
        if not notification and response.headers.get("Mcp-Session-Id"):
            self.session_id = response.headers["Mcp-Session-Id"]
        return response

    @staticmethod
    def _parse_response_body(response: requests.Response) -> dict[str, Any]:
        """Parse a JSON or SSE-framed JSON-RPC response."""
        text = response.text.strip()
        if not text:
            return {}

        if text.startswith("{"):
            return json.loads(text)

        event_data: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                if event_data:
                    return json.loads("\n".join(event_data))
                continue
            if line.startswith("data:"):
                event_data.append(line.removeprefix("data:").lstrip())

        if event_data:
            return json.loads("\n".join(event_data))

        return json.loads(text)

    def initialize(self) -> dict[str, Any]:
        """Run MCP initialize + notifications/initialized handshake."""
        init_id = self._next_id()
        init_response = self._post(
            payload={
                "jsonrpc": "2.0",
                "id": init_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": EVALHUB_MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": EVALHUB_MCP_CLIENT_NAME,
                        "version": EVALHUB_MCP_CLIENT_VERSION,
                    },
                },
            }
        )
        init_response.raise_for_status()
        init_body = self._parse_response_body(response=init_response)
        if init_body.get("error"):
            raise McpProtocolError(f"initialize failed: {init_body['error']}")

        self._post(
            payload={
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            },
            notification=True,
        )
        self._initialized = True
        self.initialize_result = init_body.get("result", {})
        return self.initialize_result

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a JSON-RPC request and return the result object."""
        if not self._initialized:
            self.initialize()

        request_id = self._next_id()
        response = self._post(
            payload={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
        )
        response.raise_for_status()
        body = self._parse_response_body(response=response)

        if body.get("id") != request_id and "result" not in body and "error" not in body:
            LOGGER.warning(f"Unexpected MCP response for {method}: {body}")

        if body.get("error"):
            raise McpProtocolError(f"{method} failed: {body['error']}")

        return body.get("result", {})

    def get_health(self, path: str) -> requests.Response:
        """GET the unauthenticated MCP health endpoint."""
        return requests.get(
            url=f"{self.base_url}{path}",
            verify=self.ca_bundle_file,
            timeout=10,
        )

    def post_without_auth(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> requests.Response:
        """POST JSON-RPC without Authorization (for negative auth tests)."""
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        return requests.post(
            url=self.base_url,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params or {},
            },
            verify=self.ca_bundle_file,
            timeout=self.timeout,
        )


def validate_evalhub_mcp_initialize(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant_namespace: str,
) -> dict[str, Any]:
    """Validate MCP initialize succeeds with bearer token, tenant header, and proxy RBAC.

    Args:
        host: MCP route hostname.
        token: Bearer token for the test service account.
        ca_bundle_file: Path to the cluster CA bundle for TLS verification.
        tenant_namespace: Tenant namespace sent via X-Tenant header.

    Returns:
        The MCP initialize result object.

    Raises:
        requests.HTTPError: If the HTTP request fails.
        McpProtocolError: If the JSON-RPC response contains an error.
    """
    assert token, "Expected non-empty bearer token"
    client = EvalHubMcpClient(
        host=host,
        token=token,
        ca_bundle_file=ca_bundle_file,
        tenant=tenant_namespace,
    )
    return client.initialize()


def mcp_resource_names(result: dict[str, Any]) -> list[str]:
    """Extract resource names from a resources/list result."""
    resources = result.get("resources", [])
    return [item.get("name", "") for item in resources if isinstance(item, dict)]


def mcp_tool_names(result: dict[str, Any]) -> list[str]:
    """Extract tool names from a tools/list result."""
    tools = result.get("tools", [])
    return [item.get("name", "") for item in tools if isinstance(item, dict)]


def mcp_prompt_names(result: dict[str, Any]) -> list[str]:
    """Extract prompt names from a prompts/list result."""
    prompts = result.get("prompts", [])
    return [item.get("name", "") for item in prompts if isinstance(item, dict)]


def call_mcp_tool(
    client: EvalHubMcpClient,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Invoke an MCP tool and return the full tools/call result."""
    return client.call(
        method="tools/call",
        params={"name": name, "arguments": arguments or {}},
    )


def read_mcp_resource(client: EvalHubMcpClient, uri: str) -> dict[str, Any]:
    """Read an MCP resource by URI."""
    return client.call(method="resources/read", params={"uri": uri})


def get_mcp_prompt(
    client: EvalHubMcpClient,
    name: str,
    arguments: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fetch an MCP prompt by name."""
    return client.call(
        method="prompts/get",
        params={"name": name, "arguments": arguments or {}},
    )


def mcp_tool_is_error(result: dict[str, Any]) -> bool:
    """Return True when tools/call returned a tool-level error."""
    return result.get("isError") is True


def mcp_tool_error_text(result: dict[str, Any]) -> str:
    """Extract error text from a tools/call error result."""
    content = result.get("content", [])
    if content and isinstance(content[0], dict):
        return str(content[0].get("text", ""))
    return ""


def mcp_tool_structured(result: dict[str, Any] | None) -> dict[str, Any]:
    """Return structuredContent from a tools/call result."""
    if result is None:
        return {}
    structured = result.get("structuredContent", {})
    return structured if isinstance(structured, dict) else {}


def build_mcp_model_url(service_name: str, tenant_namespace: str) -> str:
    """Build the vLLM emulator model URL used by MCP submit_evaluation."""
    return f"http://{service_name}.{tenant_namespace}.svc.cluster.local:{EVALHUB_VLLM_EMULATOR_PORT}/v1"


def build_mcp_evaluation_arguments(
    model_url: str,
    job_name: str = "mcp-test-job",
    benchmark_id: str = EVALHUB_MCP_DEFAULT_BENCHMARK_ID,
    provider_id: str = EVALHUB_MCP_DEFAULT_PROVIDER_ID,
    collection_id: str | None = None,
) -> dict[str, Any]:
    """Build submit_evaluation tool arguments."""
    arguments: dict[str, Any] = {
        "name": job_name,
        "model": {"url": model_url, "name": "emulatedModel"},
    }
    if collection_id:
        arguments["collection"] = {"id": collection_id}
    elif benchmark_id == EVALHUB_MCP_DEFAULT_BENCHMARK_ID and provider_id == EVALHUB_MCP_DEFAULT_PROVIDER_ID:
        arguments["benchmarks"] = [build_vllm_arc_easy_benchmark()]
    else:
        arguments["benchmarks"] = [{"id": benchmark_id, "provider_id": provider_id}]
    return arguments


def submit_evaluation_via_mcp(
    client: EvalHubMcpClient,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Submit an evaluation job via MCP and return structured tool output."""
    result = call_mcp_tool(client=client, name="submit_evaluation", arguments=arguments)
    if mcp_tool_is_error(result=result):
        raise RuntimeError(f"submit_evaluation failed: {mcp_tool_error_text(result=result)}")
    structured = mcp_tool_structured(result=result)
    if not structured.get("job_id"):
        raise ValueError(f"Expected job_id in submit response: {result}")
    return structured


def format_mcp_job_status_failure(status: dict[str, Any]) -> str:
    """Format MCP get_job_status structured output for assertion messages."""
    parts = [f"state='{status.get('state', '')}'"]
    message = status.get("message")
    if message:
        parts.append(f"message={message!r}")
    benchmarks = status.get("benchmarks")
    if benchmarks:
        parts.append(f"benchmarks={benchmarks!r}")
    return "; ".join(parts)


def wait_for_mcp_job_state(
    client: EvalHubMcpClient,
    job_id: str,
    timeout: int = 600,
    sleep: int = 15,
    terminal_states: set[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Poll get_job_status until the job reaches a terminal state.

    Returns:
        A tuple of the terminal state and the last structured status payload.
    """
    states = terminal_states or EVALHUB_JOB_TERMINAL_STATES
    terminal_state = ""
    last_status: dict[str, Any] = {}
    for status_result in TimeoutSampler(
        wait_timeout=timeout,
        sleep=sleep,
        func=lambda: call_mcp_tool(
            client=client,
            name="get_job_status",
            arguments={"job_id": job_id},
        ),
    ):
        last_status = mcp_tool_structured(result=status_result)
        terminal_state = last_status.get("state", "")
        LOGGER.info(f"MCP job {job_id} state: {terminal_state}")
        if terminal_state in states:
            break
    if terminal_state in {"failed", "partially_failed"}:
        LOGGER.error(f"MCP job {job_id} failed: {format_mcp_job_status_failure(status=last_status)}")
    return terminal_state, last_status


def mcp_read_resource_text(result: dict[str, Any]) -> str:
    """Return text from the first resources/read content block."""
    contents = result.get("contents", [])
    if not contents:
        return ""
    first = contents[0]
    if isinstance(first, dict):
        return str(first.get("text", ""))
    return ""


def build_mcp_proxy_role_rules(evalhub_instance_name: str) -> list[dict[str, list[str]]]:
    """RBAC rules granting evalhubs/proxy access for one EvalHub instance."""
    return [
        {
            "apiGroups": ["trustyai.opendatahub.io"],
            "resources": [EVALHUB_MCP_PROXY_RESOURCE],
            "resourceNames": [evalhub_instance_name],
            "verbs": ["get", "create"],
        }
    ]
