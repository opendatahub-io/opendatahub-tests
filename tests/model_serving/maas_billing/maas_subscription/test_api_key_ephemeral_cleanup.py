from typing import Any

import portforward
import pytest
import requests
import structlog
from ocp_resources.cron_job import CronJob
from ocp_resources.network_policy import NetworkPolicy
from pytest_testconfig import config as py_config

from tests.model_serving.maas_billing.maas_subscription.utils import search_active_api_keys
from tests.model_serving.maas_billing.utils import build_maas_headers

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
)
class TestEphemeralKeyCleanup:
    """Tests for ephemeral API key cleanup (CronJob + internal endpoint)."""

    @pytest.mark.tier1
    def test_cronjob_exists_and_configured(self, maas_cleanup_cronjob: CronJob) -> None:
        """Verify the maas-api-key-cleanup CronJob exists with expected configuration."""
        spec = maas_cleanup_cronjob.instance.spec

        assert spec.schedule == "*/15 * * * *", f"Expected schedule '*/15 * * * *', got '{spec.schedule}'"
        assert spec.concurrencyPolicy == "Forbid", (
            "CronJob should use Forbid concurrency policy to prevent overlapping runs"
        )

        containers = spec.jobTemplate.spec.template.spec.containers
        assert len(containers) >= 1, "CronJob should have at least one container"
        container_spec = containers[0]
        cmd_str = " ".join(container_spec.command or [])
        assert "/internal/v1/api-keys/cleanup" in cmd_str, (
            f"CronJob command should target the internal cleanup endpoint, got: {cmd_str}"
        )

        sec_ctx = getattr(container_spec, "securityContext", None)
        assert sec_ctx is not None, "Cleanup container should have securityContext configured"
        assert sec_ctx.runAsNonRoot is True, "Cleanup container should run as non-root"
        assert sec_ctx.readOnlyRootFilesystem is True, "Cleanup container should have read-only root filesystem"

        LOGGER.info(f"[ephemeral] CronJob validated: schedule={spec.schedule}, concurrency={spec.concurrencyPolicy}")

    @pytest.mark.tier1
    def test_cleanup_networkpolicy_exists(self, maas_cleanup_networkpolicy: NetworkPolicy) -> None:
        """Verify the cleanup NetworkPolicy restricts cleanup pod egress to maas-api only."""
        spec = maas_cleanup_networkpolicy.instance.spec

        assert spec.podSelector.matchLabels.get("app") == "maas-api-cleanup", (
            f"NetworkPolicy should target app=maas-api-cleanup pods, got: {spec.podSelector.matchLabels}"
        )
        for policy_type in ("Egress", "Ingress"):
            assert policy_type in spec.policyTypes, f"NetworkPolicy should control {policy_type} traffic"

        ingress_rules = getattr(spec, "ingress", None)
        assert ingress_rules in ([], None), "Cleanup pods should have no inbound traffic allowed"

        egress_rules = getattr(spec, "egress", None)
        assert egress_rules, "NetworkPolicy should define at least one egress rule"

        LOGGER.info("[ephemeral] NetworkPolicy validated: cleanup pods restricted to maas-api egress only")

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_ephemeral_key_visible_with_include_filter(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        ephemeral_api_key: dict[str, Any],
    ) -> None:
        """Verify ephemeral key is marked as ephemeral and visible when includeEphemeral=True."""
        key_id = ephemeral_api_key["id"]

        assert ephemeral_api_key.get("ephemeral") is True, "Key should be marked as ephemeral"

        items = search_active_api_keys(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            include_ephemeral=True,
        )
        assert key_id in [item["id"] for item in items], (
            f"Ephemeral key {key_id} should appear in search with includeEphemeral=True"
        )
        LOGGER.info(f"[ephemeral] Ephemeral key {key_id} visible with includeEphemeral=True")

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_ephemeral_key_hidden_from_default_search(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        ephemeral_api_key: dict[str, Any],
    ) -> None:
        """Verify ephemeral key is hidden from default search when includeEphemeral is not set."""
        key_id = ephemeral_api_key["id"]

        default_items = search_active_api_keys(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            include_ephemeral=False,
        )
        assert key_id not in [item["id"] for item in default_items], (
            "Ephemeral key should be excluded from default search (includeEphemeral defaults to False)"
        )
        LOGGER.info(f"[ephemeral] Ephemeral key {key_id} correctly hidden from default search")

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_trigger_cleanup_preserves_active_keys(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        ephemeral_api_key: dict[str, Any],
        maas_api_pod_name: str,
    ) -> None:
        """Verify the cleanup endpoint does not delete active (non-expired) ephemeral keys."""
        applications_namespace = py_config["applications_namespace"]
        key_id = ephemeral_api_key["id"]
        api_keys_endpoint = f"{base_url}/v1/api-keys"
        auth_header = build_maas_headers(token=ocp_token_for_actor)

        LOGGER.info(f"[ephemeral] Triggering cleanup via port-forward into pod={maas_api_pod_name}")

        with portforward.forward(
            pod_or_service=maas_api_pod_name,
            namespace=applications_namespace,
            from_port=8080,
            to_port=8080,
            waiting=20,
        ):
            cleanup_response = requests.post(
                url="http://localhost:8080/internal/v1/api-keys/cleanup",
                timeout=30,
            )

        assert cleanup_response.status_code == 200, (
            f"Cleanup endpoint returned unexpected status: {cleanup_response.status_code}: "
            f"{(cleanup_response.text or '')[:200]}"
        )
        cleanup_resp = cleanup_response.json()
        deleted_count = cleanup_resp.get("deletedCount", -1)
        assert deleted_count >= 0, f"Cleanup response should have non-negative deletedCount, got: {cleanup_resp}"
        LOGGER.info(f"[ephemeral] Cleanup completed: deletedCount={deleted_count}")

        r_get = request_session_http.get(
            url=f"{api_keys_endpoint}/{key_id}",
            headers=auth_header,
            timeout=30,
        )
        assert r_get.status_code == 200, (
            f"Active ephemeral key {key_id} should survive cleanup, got {r_get.status_code}: {(r_get.text or '')[:200]}"
        )
        get_body = r_get.json()
        assert get_body.get("status") == "active", (
            f"Key should still be active after cleanup, got: {get_body.get('status')}"
        )
        LOGGER.info(f"[ephemeral] Active key {key_id} survived cleanup correctly")
