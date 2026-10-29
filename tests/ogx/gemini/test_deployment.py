"""Deployment / packaging tests for the remote::gemini provider.

Covers test cases TC-DEP-001, TC-DEP-002 and TC-DEP-003 from the
remote_gemini_provider test plan (RHAISTRAT-1245).

These verify, from the running distribution, that the Gemini provider is built
into the image (google-genai dependency), configured with the conditional
activation pattern, and that the operator injects GEMINI_API_KEY into the pod.
"""

import pytest
import structlog
from ocp_resources.pod import Pod
from ogx_client import OgxClient

from tests.ogx.gemini.utils import is_gemini_provider_active, pod_env_var_is_set

LOGGER = structlog.get_logger(name=__name__)

# These tests require live Gemini API access and must not run on disconnected clusters.
pytestmark = [pytest.mark.skip_on_disconnected]

# Candidate locations for the distribution run/config file inside the pod.
CONFIG_SEARCH_PATHS = ("/app", "/opt/app-root", "/etc/ogx", "/root/.ogx")


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [
        pytest.param(
            {"name": "test-gemini-deploy", "randomize_name": True},
            {"enable_gemini": True},
            id="gemini",
        ),
    ],
    indirect=True,
)
@pytest.mark.downstream_only
class TestGeminiDeployment:
    """Build/config/operator-injection checks for the Gemini provider."""

    @pytest.mark.tier1
    def test_build_includes_gemini_and_google_genai(
        self,
        ogx_client: OgxClient,
        ogx_gemini_pod: Pod,
    ) -> None:
        """Verify the image ships the Gemini provider and google-genai (TC-DEP-001).

        Given: a deployed OGX distribution image.
        When: the running pod is inspected for the google-genai package and the
            provider list is queried.
        Then: google-genai is importable and remote::gemini is available, which
            together demonstrate build.yaml included the provider and its dependency.
        """
        output = ogx_gemini_pod.execute(command=["sh", "-c", "python -c 'import google.genai' && echo IMPORT_OK"])
        assert "IMPORT_OK" in output, f"google-genai is not importable in the distribution image: {output!r}"
        assert is_gemini_provider_active(ogx_client=ogx_client), (
            "remote::gemini provider is not available despite google-genai being installed"
        )

    @pytest.mark.tier1
    def test_config_conditional_activation_pattern(self, ogx_gemini_pod: Pod) -> None:
        """Verify config.yaml uses the ENABLE_GEMINI conditional pattern (TC-DEP-002).

        Given: a deployed OGX distribution.
        When: the distribution config file inside the pod is located and inspected.
        Then: it references the conditional activation pattern
            ${env.ENABLE_GEMINI:+gemini}.
        """
        pattern = "${env.ENABLE_GEMINI:+gemini}"
        search_roots = " ".join(CONFIG_SEARCH_PATHS)
        matched_files = ogx_gemini_pod.execute(
            command=["sh", "-c", f"grep -rlsF '{pattern}' {search_roots} 2>/dev/null || true"]
        ).strip()
        assert matched_files, (
            f"Conditional activation pattern {pattern!r} not found in any file under {CONFIG_SEARCH_PATHS!r}"
        )

    @pytest.mark.tier1
    def test_operator_injects_gemini_api_key(
        self,
        ogx_client: OgxClient,
        ogx_gemini_pod: Pod,
    ) -> None:
        """Verify the operator injects GEMINI_API_KEY into the pod (TC-DEP-003).

        Given: an OgxServer configured to source GEMINI_API_KEY from a secret.
        When: the running pod's environment is inspected (without printing the value).
        Then: GEMINI_API_KEY is set and remote::gemini is active.
        """
        assert pod_env_var_is_set(pod=ogx_gemini_pod, name="GEMINI_API_KEY"), (
            "GEMINI_API_KEY is not set in the OgxServer pod environment"
        )
        assert is_gemini_provider_active(ogx_client=ogx_client), (
            "remote::gemini provider is not active despite GEMINI_API_KEY being injected"
        )
