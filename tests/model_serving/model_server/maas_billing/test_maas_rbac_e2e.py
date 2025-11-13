import pytest

from simple_logger.logger import get_logger
from utilities.plugins.constant import RestHeader, OpenAIEnpoints

from tests.model_serving.model_server.maas_billing.utils import mint_token

LOGGER = get_logger(name=__name__)

MODELS_INFO = OpenAIEnpoints.MODELS_INFO
CHAT_COMPLETIONS = OpenAIEnpoints.CHAT_COMPLETIONS


@pytest.mark.usefixtures(
    "maas_free_group",
    "maas_premium_group",
)
class TestMaasRBACE2E:
    """
    Simple smoke flow:
    - mint MaaS token for a user
    - call /v1/models
    - call /v1/chat/completions

    Actors:
    - admin
    - free user (in maas-free-users group)
    - premium user (in maas-premium-users group)
    """

    @pytest.mark.parametrize(
        "ocp_token_for_actor",
        ["admin", "free", "premium"],
        indirect=True,
    )
    def test_maas_flow_for_actors(
        self,
        request_session_http,
        base_url: str,
        model_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        # 1) Mint MaaS token as this actor
        mint_resp, mint_body = mint_token(
            base_url=base_url,
            oc_user_token=ocp_token_for_actor,
            http_session=request_session_http,
            minutes=10,
        )
        LOGGER.info(f"MaaS RBAC: mint token status={mint_resp.status_code}")
        assert mint_resp.status_code in (200, 201), f"mint failed: {mint_resp.status_code} {mint_resp.text[:200]}"

        maas_token = mint_body.get("token", "")
        assert isinstance(maas_token, str) and len(maas_token) > 10, "no usable MaaS token in response"

        headers = {"Authorization": f"Bearer {maas_token}", **RestHeader.HEADERS}

        # 2) Call /v1/models
        models_url = f"{base_url}{MODELS_INFO}"
        models_resp = request_session_http.get(url=models_url, headers=headers, timeout=60)
        LOGGER.info(f"MaaS RBAC: /v1/models -> {models_resp.status_code}")
        assert models_resp.status_code == 200, f"/v1/models failed: {models_resp.status_code} {models_resp.text[:200]}"

        models = models_resp.json().get("data", [])
        assert isinstance(models, list) and models, "no models returned from /v1/models"

        model_id = models[0].get("id", "")
        assert model_id, "first model from /v1/models has no 'id'"

        # 3) Call /llm/<deployment>/v1/chat/completions
        payload = {"model": model_id, "prompt": "Hello", "max_tokens": 16}
        LOGGER.info(f"MaaS RBAC: POST {model_url} with keys={list(payload.keys())}")

        chat_resp = request_session_http.post(
            url=model_url,
            headers=headers,
            json=payload,
            timeout=60,
        )
        LOGGER.info(f"MaaS RBAC: POST {model_url} -> {chat_resp.status_code}")
        assert chat_resp.status_code == 200, (
            f"/v1/chat/completions failed: {chat_resp.status_code} {chat_resp.text[:200]} (url={model_url})"
        )

        chat_body = chat_resp.json()
        choices = chat_body.get("choices", [])
        assert isinstance(choices, list) and choices, "'choices' missing or empty"

        first_message = choices[0].get("message", {}) or {}
        first_text = first_message.get("content") or choices[0].get("text", "")
        assert isinstance(first_text, str) and first_text.strip(), "first choice has no text content"
