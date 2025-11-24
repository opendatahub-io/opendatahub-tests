import pytest
from simple_logger.logger import get_logger
from utilities.plugins.constant import RestHeader, OpenAIEnpoints

LOGGER = get_logger(name=__name__)

MODELS_INFO = OpenAIEnpoints.MODELS_INFO
CHAT_COMPLETIONS = OpenAIEnpoints.CHAT_COMPLETIONS

ACTORS = [
    {"kind": "admin"},
    {"kind": "free"},
    {"kind": "premium"},
]


@pytest.mark.usefixtures("maas_free_group", "maas_premium_group")
@pytest.mark.parametrize(
    "ocp_token_for_actor",
    ACTORS,
    indirect=True,
)
class TestMaasRBACE2E:
    """
    For each actor (admin / free / premium) verify:
    - can mint a MaaS token
    - can list models
    - can call /v1/chat/completions
    """

    def test_mint_token_for_actors(
        self,
        ocp_token_for_actor,
        maas_token_for_actor: str,
    ) -> None:
        LOGGER.info(f"MaaS RBAC: using already minted MaaS token length={len(maas_token_for_actor)}")

    def test_models_visible_for_actors(
        self,
        request_session_http,
        base_url: str,
        ocp_token_for_actor,
        maas_token_for_actor: str,
    ) -> None:
        headers = {
            "Authorization": f"Bearer {maas_token_for_actor}",
            **RestHeader.HEADERS,
        }
        models_url = f"{base_url}{MODELS_INFO}"
        response = request_session_http.get(
            url=models_url,
            headers=headers,
            timeout=60,
        )

        LOGGER.info(f"MaaS RBAC: /v1/models -> {response.status_code}")

        assert response.status_code == 200, f"/v1/models failed: {response.status_code} {response.text[:200]}"
        models = response.json().get("data", [])
        assert isinstance(models, list) and models, "no models returned from /v1/models"

    def test_chat_completions_for_actors(
        self,
        request_session_http,
        base_url: str,
        model_url: str,
        ocp_token_for_actor,
        maas_token_for_actor: str,
    ) -> None:
        headers = {
            "Authorization": f"Bearer {maas_token_for_actor}",
            **RestHeader.HEADERS,
        }

        models_url = f"{base_url}{MODELS_INFO}"
        models_response = request_session_http.get(
            url=models_url,
            headers=headers,
            timeout=60,
        )
        assert models_response.status_code == 200, (
            f"/v1/models failed: {models_response.status_code} {models_response.text[:200]}"
        )

        models = models_response.json().get("data", [])
        assert models, "no models returned from /v1/models"
        model_id = models[0].get("id", "")
        assert model_id, "first model from /v1/models has no 'id'"

        payload = {"model": model_id, "prompt": "Hello", "max_tokens": 16}

        LOGGER.info(f"MaaS RBAC: POST {model_url} with payload keys={list(payload.keys())}")

        chat_response = request_session_http.post(
            url=model_url,
            headers=headers,
            json=payload,
            timeout=60,
        )

        LOGGER.info(f"MaaS RBAC: POST {model_url} -> {chat_response.status_code}")

        assert chat_response.status_code == 200, (
            f"/v1/chat/completions failed: {chat_response.status_code} {chat_response.text[:200]} (url={model_url})"
        )
        chat_body = chat_response.json()
        choices = chat_body.get("choices", [])
        assert isinstance(choices, list) and choices, "'choices' missing or empty"
