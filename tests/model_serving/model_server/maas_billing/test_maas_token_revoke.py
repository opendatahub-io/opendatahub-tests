import pytest
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_server.maas_billing.utils import (
    verify_chat_completions,
    revoke_token,
)

LOGGER = get_logger(name=__name__)

ACTORS = [
    pytest.param({"type": "free"}, "free", id="free"),
    pytest.param({"type": "premium"}, "premium", id="premium"),
]


@pytest.mark.usefixtures(
    "maas_free_group",
    "maas_premium_group",
)
@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [
        pytest.param(
            {"name": "llm", "modelmesh-enabled": False},
            id="maas-billing-namespace",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "ocp_token_for_actor, actor_label",
    ACTORS,
    indirect=["ocp_token_for_actor"],
    scope="class",
)
class TestMaasTokenRevokeFreePremium:
    """
    For FREE and PREMIUM actors:
    - MaaS token works for /v1/models and /v1/chat/completions
    - revoke succeeds (DELETE /v1/tokens revokes ALL tokens for that user)
    - token becomes invalid after revoke (401/403), allowing propagation time
    """

    def test_token_happy_then_revoked_fails(
        self,
        request_session_http,
        base_url: str,
        model_url: str,
        ocp_token_for_actor: str,
        actor_label: str,
        maas_token_for_actor: str,
        maas_headers_for_actor: dict,
        maas_models_response_for_actor,
    ) -> None:

        models_list = maas_models_response_for_actor.json().get("data", [])

        verify_chat_completions(
            request_session_http=request_session_http,
            model_url=model_url,
            headers=maas_headers_for_actor,
            models_list=models_list,
            prompt_text="hi",
            max_tokens=16,
            request_timeout_seconds=60,
            log_prefix=f"MaaS revoke pre-check [{actor_label}]",
            expected_status_codes=(200,),
        )

        revoke_url = f"{base_url}/v1/tokens"
        LOGGER.info(f"[{actor_label}] revoke request: DELETE {revoke_url}")

        r_del = revoke_token(
            base_url=base_url,
            oc_user_token=ocp_token_for_actor,
            http_session=request_session_http,
        )

        LOGGER.info(f"[{actor_label}] revoke response: status={r_del.status_code} body={(r_del.text or '')[:200]}")

        assert r_del.status_code in (200, 202, 204), (
            f"{actor_label}: revoke failed: {r_del.status_code} {(r_del.text or '')[:200]}"
        )

        last_status = None
        last_text = None

        for resp in TimeoutSampler(
            wait_timeout=60,
            sleep=3,
            func=verify_chat_completions,
            request_session_http=request_session_http,
            model_url=model_url,
            headers=maas_headers_for_actor,
            models_list=models_list,
            prompt_text="hi",
            max_tokens=16,
            request_timeout_seconds=60,
            log_prefix=f"MaaS revoke post-check [{actor_label}]",
            expected_status_codes=(200, 401, 403),
        ):
            last_status = resp.status_code
            last_text = resp.text
            LOGGER.info(f"[{actor_label}] post-revoke status={last_status}")

            if last_status in (401, 403):
                LOGGER.info(f"{actor_label}: got expected {last_status} after revoke")
                return

        assert last_status in (401, 403), (
            f"{actor_label}: expected 401/403 after revoke, got {last_status}. body={(last_text or '')[:200]}"
        )
