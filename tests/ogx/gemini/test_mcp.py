"""MCP server integration test for the remote::gemini provider.

Covers test case TC-MCP-001 from the remote_gemini_provider test plan
(RHAISTRAT-1245).
"""

import pytest
import structlog
from ogx_client import OgxClient

LOGGER = structlog.get_logger(name=__name__)

# These tests require live Gemini API access and must not run on disconnected clusters.
pytestmark = [pytest.mark.skip_on_disconnected]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [
        pytest.param(
            {"name": "test-gemini-mcp", "randomize_name": True},
            {"enable_gemini": True},
            id="gemini",
        ),
    ],
    indirect=True,
)
class TestGeminiMCP:
    """MCP server connectivity with Gemini models."""

    @pytest.mark.skip(reason="MCP harness for TC-MCP-001 is not implemented yet (RHOAIENG-101299)")
    @pytest.mark.tier2
    def test_mcp_server_connectivity_with_gemini(self, ogx_client: OgxClient) -> None:
        """Verify MCP tool invocation works with Gemini models (TC-MCP-001).

        Given: an active remote::gemini provider and an MCP server exposing tools.
        When: a message that should trigger an MCP tool is sent to a Gemini model.
        Then: the model invokes the MCP tool and incorporates its output.

        Note: TC-MCP-001's preconditions state the MCP harness is "TBD — specific
        harness to be selected by QE team". The test is skipped by marker, before
        any fixture is set up, until the harness exists.
        """
