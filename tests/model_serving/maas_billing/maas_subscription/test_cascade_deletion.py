from __future__ import annotations

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger

from tests.model_serving.maas_billing.maas_subscription.utils import (
    chat_payload_for_url,
    create_maas_subscription,
    poll_expected_status,
)
from utilities.resources.maa_s_subscription import MaaSSubscription

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_inference_service_tinyllama_free",
    "maas_model_tinyllama_free",
    "maas_auth_policy_tinyllama_free",
    "maas_subscription_tinyllama_free",
)
class TestCascadeDeletion:
    """
    Tests that deleting MaaSSubscription CRs triggers proper cleanup/rebuild behavior.
    """

    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_delete_subscription_rebuilds_trlp(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        maas_model_tinyllama_free,
        model_url_tinyllama_free: str,
        maas_headers_for_actor_api_key: dict[str, str],
        maas_subscription_tinyllama_free,
    ) -> None:
        """
        Add a second subscription for the same model, then delete it.
        Verify the original subscription still allows access (HTTP 200).
        """
        # _ = maas_subscription_tinyllama_free

        with create_maas_subscription(
            admin_client=admin_client,
            subscription_namespace=maas_subscription_tinyllama_free.namespace,
            subscription_name="e2e-temp-sub",
            owner_group_name="system:authenticated",
            model_name=maas_model_tinyllama_free.name,
            model_namespace=maas_model_tinyllama_free.namespace,
            tokens_per_minute=50,
            window="1m",
            priority=0,
            teardown=False,
            wait_for_resource=True,
        ) as temp_subscription:
            temp_subscription.wait_for_condition(condition="Ready", status="True", timeout=300)

            LOGGER.info(
                "Created temporary subscription %s for model %s",
                temp_subscription.name,
                maas_model_tinyllama_free.name,
            )

            temp_subscription.clean_up(wait=True)
            LOGGER.info("Deleted temporary subscription %s", temp_subscription.name)

            payload = chat_payload_for_url(model_url=model_url_tinyllama_free)

            response = poll_expected_status(
                request_session_http=request_session_http,
                model_url=model_url_tinyllama_free,
                headers=maas_headers_for_actor_api_key,
                payload=payload,
                expected_statuses={200},
                wait_timeout=240,
                sleep=5,
                request_timeout=60,
            )

            assert response.status_code == 200, (
                "Expected 200 after deleting temporary subscription and rebuilding policies, "
                f"got {response.status_code}: {(response.text or '')[:200]}"
            )

    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_delete_last_subscription_denies_access(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        maas_free_group: str,
        maas_model_tinyllama_free,
        # maas_auth_policy_tinyllama_free,
        model_url_tinyllama_free: str,
        maas_headers_for_actor_api_key: dict[str, str],
        maas_subscription_tinyllama_free: MaaSSubscription,
    ) -> None:
        """
        Delete the only/original subscription for the model.
        Verify access is denied with 403 or 429.

        Expected behavior:
        - 403 if auth policy denies because no valid subscription exists
        - 429 if default deny token rate limit policy applies first

        Both mean: no subscription => no access.
        """

        original_name = maas_subscription_tinyllama_free.name
        original_priority = getattr(maas_subscription_tinyllama_free, "priority", 0)

        LOGGER.info("Deleting original subscription %s", original_name)
        maas_subscription_tinyllama_free.clean_up(wait=True)

        payload = chat_payload_for_url(model_url=model_url_tinyllama_free)

        try:
            response = poll_expected_status(
                request_session_http=request_session_http,
                model_url=model_url_tinyllama_free,
                headers=maas_headers_for_actor_api_key,
                payload=payload,
                expected_statuses={403, 429},
                wait_timeout=120,
                sleep=5,
                request_timeout=60,
            )

            LOGGER.info(
                "No subscription present for model %s -> received %s as expected",
                maas_model_tinyllama_free.name,
                response.status_code,
            )

            assert response.status_code in {403, 429}, (
                "Expected 403 or 429 after deleting the last subscription, "
                f"got {response.status_code}: {(response.text or '')[:200]}"
            )
        finally:
            with MaaSSubscription(
                client=admin_client,
                name=original_name,
                namespace=maas_subscription_tinyllama_free.namespace,
                owner={
                    "groups": [{"name": maas_free_group}],
                },
                model_refs=[
                    {
                        "name": maas_model_tinyllama_free.name,
                        "namespace": maas_model_tinyllama_free.namespace,
                        "tokenRateLimits": [{"limit": 100, "window": "1m"}],
                    }
                ],
                priority=original_priority,
                teardown=False,
                wait_for_resource=True,
            ) as restored_subscription:
                restored_subscription.wait_for_condition(condition="Ready", status="True", timeout=300)
                LOGGER.info("Restored original subscription %s", restored_subscription.name)
