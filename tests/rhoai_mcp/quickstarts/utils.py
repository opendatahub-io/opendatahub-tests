import asyncio
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from tenacity import retry as tenacity_retry
from tenacity import retry_if_not_result, stop_after_delay, wait_exponential

from tests.rhoai_mcp.utils import parse_tool_result

_FINISHED_PHASES = frozenset({"Complete", "Failed"})


def call_mcp_tool(transport: StreamableHttpTransport, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call an MCP tool from synchronous code and return its parsed JSON payload.

    Args:
        transport: Authenticated transport for the rhoai-mcp server.
        name: Name of the MCP tool to call.
        arguments: Arguments to pass to the tool.

    Returns:
        The tool's JSON response as a dict.
    """

    async def _call() -> dict[str, Any]:
        async with Client(transport) as client:
            result = await client.call_tool(name=name, arguments=arguments)
            return parse_tool_result(result=result)

    return asyncio.run(main=_call())


@tenacity_retry(
    stop=stop_after_delay(300),
    wait=wait_exponential(min=5, max=30),
    retry=retry_if_not_result(lambda data: data.get("phase") in _FINISHED_PHASES),
)
async def wait_for_quickstart_action_finished(client: Client, job_name: str) -> dict[str, Any]:
    """Poll get_quickstart_action_status until the installer Job completes or fails.

    Args:
        client: Connected MCP client.
        job_name: Name of the installer Job returned by run_quickstart_action.

    Returns:
        The final get_quickstart_action_status response.
    """
    result = await client.call_tool(name="get_quickstart_action_status", arguments={"job_name": job_name})
    return parse_tool_result(result=result)
