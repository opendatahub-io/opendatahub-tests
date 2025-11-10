from utilities.plugins.constant import OpenAIEnpoints
from simple_logger.logger import get_logger
import requests

LOGGER = get_logger(name=__name__)
MODELS_INFO = OpenAIEnpoints.MODELS_INFO
CHAT_COMPLETIONS = OpenAIEnpoints.CHAT_COMPLETIONS


class TestMaasEndpoints:
    def test_model(
        self,
        maas_models: list,
    ) -> None:
        """Verify /v1/models endpoint returns at least one model."""
        assert isinstance(maas_models, list)
        assert maas_models, "no models returned from /v1/models"

        first = maas_models[0]
        assert "id" in first, "model entry missing 'id'"

    def test_chat_completions(
        self,
        request_session_http: requests.Session,
        model_url: str,
        maas_headers: dict,
        maas_models: list,
    ) -> None:
        """
        Verify /llm/<deployment>/v1/chat/completions responds to a simple prompt.
        """
        model_id = maas_models[0].get("id", "")
        LOGGER.info("Using model_id=%s", model_id)
        assert model_id, "first model from /v1/models has no 'id'"

        payload = {"model": model_id, "prompt": "Hello", "max_tokens": 50}
        LOGGER.info(f"POST {model_url} with keys={list(payload.keys())}")

        resp = request_session_http.post(
            url=model_url,
            headers=maas_headers,
            json=payload,
            timeout=60,
        )
        LOGGER.info(f"POST {model_url} -> {resp.status_code}")

        assert resp.status_code == 200, (
            f"/v1/chat/completions failed: {resp.status_code} {resp.text[:200]} (url={model_url})"
        )

        body = resp.json()
        choices = body.get("choices", [])
        assert isinstance(choices, list) and choices, "'choices' missing or empty"

        msg = choices[0].get("message", {}) or {}
        text = msg.get("content") or choices[0].get("text", "")
        assert isinstance(text, str) and text.strip(), "first choice has no text content"
