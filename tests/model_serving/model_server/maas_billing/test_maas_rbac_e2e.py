import pytest

from simple_logger.logger import get_logger
from utilities.plugins.constant import RestHeader, OpenAIEnpoints

from tests.model_serving.model_server.maas_billing.utils import mint_token

LOGGER = get_logger(name=__name__)

MODELS_INFO = OpenAIEnpoints.MODELS_INFO
CHAT_COMPLETIONS = OpenAIEnpoints.CHAT_COMPLETIONS


@pytest.mark.usefixtures("maas_free_group", "maas_premium_group")
class TestMaasRBACE2E:
    """
    For each actor (admin / free / premium) verify:
    - can mint a MaaS token
    - can list models
    - can call /v1/chat/completions
    """

    def _mint_maas_token(
        self,
        request_session_http,
        base_url: str,
        ocp_token_for_actor: str,
        minutes: int = 10,
    ) -> str:
        resp, body = mint_token(
            base_url=base_url,
            oc_user_token=ocp_token_for_actor,
            http_session=request_session_http,
            minutes=minutes,
        )
        LOGGER.info("MaaS RBAC: mint token status=%s", resp.status_code)
        assert resp.status_code in (200, 201), f"mint failed: {resp.status_code} {resp.text[:200]}"

        token = body.get("token", "")
        assert isinstance(token, str) and len(token) > 10, "no usable MaaS token in response"
        return token

    @pytest.mark.parametrize("ocp_token_for_actor", ["admin", "free", "premium"], indirect=True)
    def test_mint_token_for_actors(
        self,
        request_session_http,
        base_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        """Each actor can mint a MaaS token."""
        token = self._mint_maas_token(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_token_for_actor=ocp_token_for_actor,
        )
        LOGGER.info("MaaS RBAC: minted token len=%s", len(token))

    @pytest.mark.parametrize("ocp_token_for_actor", ["admin", "free", "premium"], indirect=True)
    def test_models_visible_for_actors(
        self,
        request_session_http,
        base_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        """Each actor can list models via /v1/models."""
        maas_token = self._mint_maas_token(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_token_for_actor=ocp_token_for_actor,
        )
        headers = {"Authorization": f"Bearer {maas_token}", **RestHeader.HEADERS}

        models_url = f"{base_url}{MODELS_INFO}"
        resp = request_session_http.get(url=models_url, headers=headers, timeout=60)
        LOGGER.info("MaaS RBAC: /v1/models -> %s", resp.status_code)
        assert resp.status_code == 200, f"/v1/models failed: {resp.status_code} {resp.text[:200]}"

        models = resp.json().get("data", [])
        assert isinstance(models, list) and models, "no models returned from /v1/models"

    @pytest.mark.parametrize("ocp_token_for_actor", ["admin", "free", "premium"], indirect=True)
    def test_chat_completions_for_actors(
        self,
        request_session_http,
        base_url: str,
        model_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        """Each actor can call /v1/chat/completions on a model."""
        maas_token = self._mint_maas_token(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_token_for_actor=ocp_token_for_actor,
        )
        headers = {"Authorization": f"Bearer {maas_token}", **RestHeader.HEADERS}

        # Get a model ID
        models_url = f"{base_url}{MODELS_INFO}"
        models_resp = request_session_http.get(url=models_url, headers=headers, timeout=60)
        assert models_resp.status_code == 200, f"/v1/models failed: {models_resp.status_code} {models_resp.text[:200]}"
        models = models_resp.json().get("data", [])
        assert models, "no models returned from /v1/models"
        model_id = models[0].get("id", "")
        assert model_id, "first model from /v1/models has no 'id'"

        # Call chat
        payload = {"model": model_id, "prompt": "Hello", "max_tokens": 16}
        LOGGER.info("MaaS RBAC: POST %s with keys=%s", model_url, list(payload.keys()))

        chat_resp = request_session_http.post(
            url=model_url,
            headers=headers,
            json=payload,
            timeout=60,
        )
        LOGGER.info("MaaS RBAC: POST %s -> %s", model_url, chat_resp.status_code)
        assert chat_resp.status_code == 200, (
            f"/v1/chat/completions failed: {chat_resp.status_code} {chat_resp.text[:200]} (url={model_url})"
        )

        chat_body = chat_resp.json()
        choices = chat_body.get("choices", [])
        assert isinstance(choices, list) and choices, "'choices' missing or empty"
