"""OpenShell end-to-end tests.

Each ``pytest.param`` entry wires an inference provider + model pair through
the full sandbox lifecycle.  Add new ``pytest.param`` rows (with appropriate
``marks``) to cover additional providers or tiers without duplicating test
logic.

Tier guide:
    smoke  — basic proof-of-life (can the sandbox run OpenCode at all?)
    tier1  — broader provider/model matrix, negative cases
"""

import pytest
import structlog
from openshell.sandbox import SandboxSession

from tests.openshell.constants import (
    OPENSHELL_VLLM_MODEL,
    OPENSHELL_VLLM_PROVIDER,
)

LOGGER = structlog.get_logger(name=__name__)

MIN_RESPONSE_WORDS = 3

_VLLM_PARAMS = {"provider": OPENSHELL_VLLM_PROVIDER, "model": OPENSHELL_VLLM_MODEL}


@pytest.mark.parametrize(
    "inference_route, sandbox",
    [
        pytest.param(
            _VLLM_PARAMS,
            _VLLM_PARAMS,
            id=f"provider:{OPENSHELL_VLLM_PROVIDER or 'vllm'}, model:{OPENSHELL_VLLM_MODEL}",
            marks=(pytest.mark.smoke),
        ),
    ],
    indirect=True,
)
@pytest.mark.open_shell
class TestOpenShell:
    """OpenShell tests — parametrized by provider/model, filtered by tier marks."""

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
