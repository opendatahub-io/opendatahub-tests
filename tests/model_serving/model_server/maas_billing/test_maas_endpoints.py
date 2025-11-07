from utilities.plugins.constant import RestHeader, OpenAIEnpoints
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)
MODELS_INFO = OpenAIEnpoints.MODELS_INFO
CHAT_COMPLETIONS = OpenAIEnpoints.CHAT_COMPLETIONS


class TestMaasEndpoints:
    def test_model(self, request_session_http, base_url: str, minted_token: str) -> None:
        """Verify /v1/models endpoint is reachable and returns available models."""
        headers = {"Authorization": f"Bearer {minted_token}", **RestHeader.HEADERS}
        url = f"{base_url}{MODELS_INFO}"

        resp = request_session_http.get(url, headers=headers, timeout=60)
        assert resp.status_code == 200, f"/v1/models failed: {resp.status_code} {resp.text[:200]}"

        body = resp.json()
        assert isinstance(body.get("data"), list), "'data' missing or not a list"
        assert body["data"], "no models found"

    def test_chat_completions(
        self,
        request_session_http,
        base_url: str,
        minted_token: str,
        model_url: str,
    ) -> None:
        """
        Verify the chat completion endpoint /llm/<deployment>/v1/chat/completions
        responds correctly to a prompt request.

        """
        headers = {"Authorization": f"Bearer {minted_token}", **RestHeader.HEADERS}

        # 1) Pick a model id from /v1/models
        models_url = f"{base_url}{MODELS_INFO}"
        models_resp = request_session_http.get(models_url, headers=headers, timeout=60)
        assert models_resp.status_code == 200, f"/v1/models failed: {models_resp.status_code} {models_resp.text[:200]}"
        models = models_resp.json().get("data", [])
        assert models, "no models available"
        model_id = models[0].get("id", "")
        LOGGER.info("Using model_id=%s", model_id)

        # 2) Prepare the chat completion endpoint URL
        payload = {"model": model_id, "prompt": "Hello", "max_tokens": 50}
        LOGGER.info("POST %s with keys=%s", model_url, list(payload.keys()))
        resp = request_session_http.post(url=model_url, headers=headers, json=payload, timeout=60)
        LOGGER.info("POST %s -> %s", model_url, resp.status_code)
        assert resp.status_code == 200, (
            f"/v1/chat/completions failed: {resp.status_code} {resp.text[:200]} (url={model_url})"
        )

        body = resp.json()
        assert isinstance(body.get("choices"), list), "'choices' missing or not a list"
        if body["choices"]:
            msg = body["choices"][0].get("message", {}) or {}
            text = msg.get("content") or body["choices"][0].get("text", "")
            assert isinstance(text, str) and text.strip() != "", "first choice has no text content"
