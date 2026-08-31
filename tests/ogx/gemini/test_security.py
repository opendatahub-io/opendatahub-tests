"""Security tests for the remote::gemini provider.

Covers test cases TC-SEC-001 and TC-SEC-002 from the remote_gemini_provider
test plan (RHAISTRAT-1245).
"""

import pytest
import structlog
from ocp_resources.pod import Pod
from ogx_client import OgxClient

from tests.ogx.constants import GEMINI_API_KEY
from tests.ogx.gemini.utils import is_gemini_provider_active

LOGGER = structlog.get_logger(name=__name__)

# These tests require live Gemini API access and must not run on disconnected clusters.
pytestmark = [pytest.mark.skip_on_disconnected]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [
        pytest.param(
            {"name": "test-gemini-security", "randomize_name": True},
            {"enable_gemini": True},
            id="gemini",
        ),
    ],
    indirect=True,
)
class TestGeminiSecurity:
    """The Gemini API key must never leak through specs, logs, or responses."""

    @pytest.mark.tier2
    def test_api_key_not_exposed(
        self,
        ogx_client: OgxClient,
        gemini_model_id: str,
        ogx_gemini_pod: Pod,
    ) -> None:
        """Verify the Gemini API key is not exposed anywhere observable (TC-SEC-001).

        Given: a deployed distribution with GEMINI_API_KEY injected from a Secret.
        When: the pod spec, container logs, the /v1/providers response, and a chat
            completion response are inspected.
        Then: the pod spec references the Secret (not a plaintext key), and the raw
            key value appears in none of the logs or API responses.
        """
        # The pod spec must reference the key via secretKeyRef, never inline.
        containers = ogx_gemini_pod.instance.spec.containers
        gemini_env_entries = [
            env for container in containers for env in (container.env or []) if env.name == "GEMINI_API_KEY"
        ]
        assert gemini_env_entries, "GEMINI_API_KEY env var not found in the pod spec"
        for env in gemini_env_entries:
            assert env.value in (None, ""), "GEMINI_API_KEY is set as a plaintext value in the pod spec"
            assert env.valueFrom and env.valueFrom.secretKeyRef, (
                "GEMINI_API_KEY is not sourced from a Secret via secretKeyRef"
            )

        # The remaining leak checks require knowing the real key value to search for.
        if not GEMINI_API_KEY:
            pytest.fail(
                reason="Gemini API key value is not available to the test; "
                "cannot verify absence in logs/responses for TC-SEC-001"
            )

        # Inspect logs from every container in the pod (including any sidecars), not
        # just the first — a leak in any container's logs is a CWE-532 exposure.
        for container in containers:
            logs = ogx_gemini_pod.log(container=container.name)
            assert GEMINI_API_KEY not in logs, f"Gemini API key value leaked into logs of container {container.name!r}"

        providers_dump = str(list(ogx_client.providers.list()))
        assert GEMINI_API_KEY not in providers_dump, "Gemini API key value leaked into /v1/providers response"

        chat = ogx_client.chat.completions.create(
            model=gemini_model_id,
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert GEMINI_API_KEY not in str(chat), "Gemini API key value leaked into a chat completion response"

    @pytest.mark.tier3
    def test_tls_enforced_for_gemini_egress(self, ogx_client: OgxClient) -> None:
        """Verify outbound Gemini connections negotiate TLS 1.2+ (TC-SEC-002).

        Given: an active remote::gemini provider making outbound calls to Gemini.
        When: the outbound TLS handshake from the OGX pod is captured and inspected.
        Then: the negotiated TLS version is 1.2 or 1.3 with a valid certificate chain.

        Note: this requires pod-level packet capture / proxy inspection tooling that
        is not part of the pytest harness (TC-SEC-002 preconditions call it out as
        environment-provided). The Gemini-side prerequisite is asserted; the capture
        step is not yet wired in, so the test fails to flag the outstanding work
        rather than silently skipping.
        """
        assert is_gemini_provider_active(ogx_client=ogx_client), "remote::gemini provider is not active"
        pytest.fail(
            reason="TLS handshake capture for TC-SEC-002 requires network inspection tooling "
            "(tcpdump/proxy) not available in the pytest harness; wire in packet capture to complete"
        )
