import pytest
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_server.maas_billing.utils import (
    verify_chat_completions,
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
    "unprivileged_model_namespace, ocp_token_for_actor, actor_label",
    [
        pytest.param(
            {"name": "llm", "modelmesh-enabled": False},
            {"type": "free"},
            "free",
            id="maas-billing-namespace-free",
        ),
        pytest.param(
            {"name": "llm", "modelmesh-enabled": False},
            {"type": "premium"},
            "premium",
            id="maas-billing-namespace-premium",
        ),
    ],
    indirect=["unprivileged_model_namespace", "ocp_token_for_actor"],
    scope="class",
)
class TestMaasTokenRevokeFreePremium:
    """
    For FREE and PREMIUM actors:
    - MaaS token works for /v1/models and /v1/chat/completions (precondition fixture)
    - revoke succeeds (DELETE /v1/tokens revokes ALL tokens for that user) (action fixture)
    - token becomes invalid after revoke (401/403), allowing propagation time
    """

    def test_token_happy_then_revoked_fails(
        self,
        request_session_http,
        base_url: str,
        model_url: str,
        actor_label: str,
        maas_headers_for_actor: dict,
        ensure_working_maas_token_pre_revoke,
        revoke_maas_tokens_for_actor,
    ) -> None:

        last_status = None

        for resp in TimeoutSampler(
            wait_timeout=60,
            sleep=3,
            func=verify_chat_completions,
            request_session_http=request_session_http,
            model_url=model_url,
            headers=maas_headers_for_actor,
            models_list=ensure_working_maas_token_pre_revoke,
            prompt_text="hi",
            max_tokens=16,
            request_timeout_seconds=60,
            log_prefix=f"MaaS revoke post-check [{actor_label}]",
            expected_status_codes=(200, 401, 403),
        ):
            last_status = resp.status_code
            LOGGER.info(f"[{actor_label}] post-revoke status={last_status}")

            if last_status in (401, 403):
                LOGGER.info(f"{actor_label}: got expected {last_status} after revoke")
                break
