"""OpenShell end-to-end tests.

Tier guide:
    smoke  — basic proof-of-life (can the sandbox run OpenCode at all?)
    tier1  — broader provider/model matrix, negative cases
"""

import pytest
import structlog
from openshell.sandbox import SandboxSession

from tests.openshell.constants import OPENSHELL_VLLM_MODEL

LOGGER = structlog.get_logger(name=__name__)

MIN_RESPONSE_WORDS = 3


@pytest.mark.smoke
@pytest.mark.open_shell
class TestOpenShell:
    """OpenShell smoke tests."""

    def test_opencode_proof_of_life(self, sandbox: SandboxSession) -> None:
        """Verify that a non-interactive OpenCode prompt inside the sandbox
        returns a coherent, multi-word response via the privacy router.
        """
        LOGGER.info("Executing OpenCode proof-of-life prompt inside sandbox")
        result = sandbox.exec(
            ["opencode", "run", "--model", f"rhoai/{OPENSHELL_VLLM_MODEL}", "explain openshift in one sentence"],
            timeout_seconds=300,
        )

        stderr = (result.stderr or "").strip()
        assert result.exit_code == 0, f"opencode failed (exit {result.exit_code}): {stderr}"
        if stderr:
            LOGGER.warning("opencode wrote to stderr", stderr=stderr)

        response = (result.stdout or "").strip()
        assert response, "OpenCode returned an empty response"
        word_count = len(response.split())
        assert word_count >= MIN_RESPONSE_WORDS, (
            f"Response too short ({word_count} words, need >={MIN_RESPONSE_WORDS}): {response!r}"
        )

        LOGGER.info("OpenCode proof-of-life response received", content=response, word_count=word_count)
