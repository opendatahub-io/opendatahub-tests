import pytest
import requests

from tests.ai_gateway.models_as_a_service.guardrails.constants import (
    CLEAN_PROMPT,
    GUARDRAIL_VIOLATION_ERROR_CODE,
    INPUT_RAIL_FLOW,
    OUTPUT_RAIL_FLOW,
    PII_EMAIL,
    PII_PROMPT,
    STREAMING_WITH_OUTPUT_GUARDRAIL_STATUS,
)
from tests.ai_gateway.models_as_a_service.guardrails.utils import chat_completion_payload, wait_for_chat_response
from tests.ai_gateway.models_as_a_service.utils import build_maas_headers
from utilities.resources.llm_inference_service import LLMInferenceService


@pytest.mark.usefixtures("guardrails_maas_auth_policy", "guardrail_checks_enforced")
@pytest.mark.ai_safety
class TestMaaSGuardrails:
    """Guardrails attached to MaaS subscriptions are enforced on inference through the MaaS gateway.

    The backend is llm-d-inference-sim in echo mode, so the completion repeats the prompt and
    output-phase verdicts are deterministic. Guardrail checks are NeMo e-mail detection rails
    exposed by one AIGuardrail and attached per subscription.
    """

    @pytest.mark.tier1
    def test_clean_prompt_passes_input_guardrail(
        self,
        request_session_http: requests.Session,
        guardrails_model_url: str,
        guardrails_echo_llmisvc: LLMInferenceService,
        input_guarded_api_key: str,
    ) -> None:
        """Given a subscription whose input-phase PII guardrail is confirmed active,
        when a prompt without PII is sent through MaaS,
        then the request reaches the model and the echoed completion is returned.
        """
        headers = build_maas_headers(token=input_guarded_api_key)
        # A 200 before the check selection header propagates would pass without screening anything.
        wait_for_chat_response(
            request_session_http=request_session_http,
            model_url=guardrails_model_url,
            headers=headers,
            payload=chat_completion_payload(model=guardrails_echo_llmisvc.name, prompt=PII_PROMPT),
            is_expected=lambda resp: resp.status_code == 403,
            description="input guardrail active for the subscription",
        )
        response = request_session_http.post(
            url=guardrails_model_url,
            headers=headers,
            json=chat_completion_payload(model=guardrails_echo_llmisvc.name, prompt=CLEAN_PROMPT),
            timeout=60,
        )
        assert response.status_code == 200, (
            f"Clean prompt rejected by the input guardrail: status={response.status_code}, body={response.text[:300]!r}"
        )
        content = response.json()["choices"][0]["message"]["content"]
        assert CLEAN_PROMPT in content, f"Expected echoed prompt in completion, got: {content!r}"

    @pytest.mark.tier1
    def test_pii_prompt_blocked_by_input_guardrail(
        self,
        request_session_http: requests.Session,
        guardrails_model_url: str,
        guardrails_echo_llmisvc: LLMInferenceService,
        input_guarded_api_key: str,
    ) -> None:
        """Given a subscription with an input-phase PII guardrail,
        when a prompt containing an e-mail address is sent through MaaS,
        then the gateway rejects it with 403 naming the blocking rail and the model is never answered.
        """
        response = wait_for_chat_response(
            request_session_http=request_session_http,
            model_url=guardrails_model_url,
            headers=build_maas_headers(token=input_guarded_api_key),
            payload=chat_completion_payload(model=guardrails_echo_llmisvc.name, prompt=PII_PROMPT),
            is_expected=lambda resp: resp.status_code == 403,
            description="PII prompt blocked by the input guardrail",
        )
        assert INPUT_RAIL_FLOW in response.text, f"Expected blocking rail in 403 body, got: {response.text!r}"
        assert PII_EMAIL not in response.text, "Blocked response must not echo the PII back"

    @pytest.mark.tier1
    def test_pii_response_blocked_by_output_guardrail(
        self,
        request_session_http: requests.Session,
        guardrails_model_url: str,
        guardrails_echo_llmisvc: LLMInferenceService,
        output_guarded_api_key: str,
    ) -> None:
        """Given a subscription with only an output-phase PII guardrail,
        when a prompt containing an e-mail address is echoed back by the model,
        then the completion is replaced by a content_blocked error naming the blocking rail.
        """
        response = wait_for_chat_response(
            request_session_http=request_session_http,
            model_url=guardrails_model_url,
            headers=build_maas_headers(token=output_guarded_api_key),
            payload=chat_completion_payload(model=guardrails_echo_llmisvc.name, prompt=PII_PROMPT),
            is_expected=lambda resp: resp.status_code == 200 and GUARDRAIL_VIOLATION_ERROR_CODE in resp.text,
            description="PII completion blocked by the output guardrail",
        )
        error = response.json()["error"]
        assert error["code"] == GUARDRAIL_VIOLATION_ERROR_CODE, f"Unexpected error document: {error}"
        assert error["type"] == "guardrail_violation", f"Unexpected error document: {error}"
        assert OUTPUT_RAIL_FLOW in error["message"], f"Expected blocking rail in message: {error}"
        assert PII_EMAIL not in response.text, "Blocked completion must not leak the PII"

    @pytest.mark.tier1
    def test_clean_response_passes_output_guardrail(
        self,
        request_session_http: requests.Session,
        guardrails_model_url: str,
        guardrails_echo_llmisvc: LLMInferenceService,
        output_guarded_api_key: str,
    ) -> None:
        """Given a subscription with only an output-phase PII guardrail,
        when the model's completion contains no PII,
        then the completion is returned unchanged.
        """
        response = wait_for_chat_response(
            request_session_http=request_session_http,
            model_url=guardrails_model_url,
            headers=build_maas_headers(token=output_guarded_api_key),
            payload=chat_completion_payload(model=guardrails_echo_llmisvc.name, prompt=CLEAN_PROMPT),
            is_expected=lambda resp: resp.status_code == 200,
            description="clean completion allowed by the output guardrail",
        )
        response_body = response.json()
        assert "error" not in response_body, f"Clean completion was replaced: {response_body}"
        content = response_body["choices"][0]["message"]["content"]
        assert CLEAN_PROMPT in content, f"Expected echoed prompt in completion, got: {content!r}"

    @pytest.mark.tier1
    def test_unguarded_subscription_not_screened(
        self,
        request_session_http: requests.Session,
        guardrails_model_url: str,
        guardrails_echo_llmisvc: LLMInferenceService,
        unguarded_api_key: str,
    ) -> None:
        """Given guarded subscriptions on the model and a second subscription without guardrails,
        when a PII prompt is sent with a key bound to the unguarded subscription,
        then no guardrail runs and the PII is echoed back, proving checks are selected per subscription.
        """
        response = wait_for_chat_response(
            request_session_http=request_session_http,
            model_url=guardrails_model_url,
            headers=build_maas_headers(token=unguarded_api_key),
            payload=chat_completion_payload(model=guardrails_echo_llmisvc.name, prompt=PII_PROMPT),
            is_expected=lambda resp: resp.status_code == 200,
            description="PII prompt allowed on the unguarded subscription",
        )
        content = response.json()["choices"][0]["message"]["content"]
        assert PII_EMAIL in content, f"Expected unscreened echo of the PII prompt, got: {content!r}"

    @pytest.mark.tier2
    def test_streaming_rejected_with_output_guardrail(
        self,
        request_session_http: requests.Session,
        guardrails_model_url: str,
        guardrails_echo_llmisvc: LLMInferenceService,
        output_guarded_api_key: str,
    ) -> None:
        """Given a subscription with an output-phase guardrail,
        when a streaming chat completion is requested,
        then the gateway rejects it with 422 because streamed output cannot be screened.
        """
        response = wait_for_chat_response(
            request_session_http=request_session_http,
            model_url=guardrails_model_url,
            headers=build_maas_headers(token=output_guarded_api_key),
            payload=chat_completion_payload(model=guardrails_echo_llmisvc.name, prompt=CLEAN_PROMPT, stream=True),
            is_expected=lambda resp: resp.status_code == STREAMING_WITH_OUTPUT_GUARDRAIL_STATUS,
            description="streaming request rejected by the output guardrail",
        )
        assert "streaming" in response.text, f"Expected streaming rejection reason, got: {response.text!r}"

    @pytest.mark.tier2
    def test_streaming_allowed_with_input_only_guardrail(
        self,
        request_session_http: requests.Session,
        guardrails_model_url: str,
        guardrails_echo_llmisvc: LLMInferenceService,
        input_guarded_api_key: str,
    ) -> None:
        """Given a subscription with only an input-phase guardrail,
        when a streaming chat completion with a clean prompt is requested,
        then the response streams as server-sent events.
        """
        response = wait_for_chat_response(
            request_session_http=request_session_http,
            model_url=guardrails_model_url,
            headers=build_maas_headers(token=input_guarded_api_key),
            payload=chat_completion_payload(model=guardrails_echo_llmisvc.name, prompt=CLEAN_PROMPT, stream=True),
            is_expected=lambda resp: resp.status_code == 200,
            description="streaming request allowed by the input-only guardrail",
        )
        assert response.headers.get("content-type", "").startswith("text/event-stream"), (
            f"Expected an SSE response, got content-type={response.headers.get('content-type')!r}"
        )
        assert "data: " in response.text, "Expected SSE data frames in the streamed response"

    @pytest.mark.tier2
    def test_client_supplied_guardrail_header_rejected(
        self,
        request_session_http: requests.Session,
        guardrails_model_url: str,
        guardrails_echo_llmisvc: LLMInferenceService,
        input_guarded_api_key: str,
        input_check_header: str,
    ) -> None:
        """Given a subscription with an input-phase guardrail,
        when a client sends its own guardrail selection header to switch the check off,
        then the MaaS auth policy denies the request before it reaches the guardrail or model.
        """
        response = wait_for_chat_response(
            request_session_http=request_session_http,
            model_url=guardrails_model_url,
            headers={**build_maas_headers(token=input_guarded_api_key), input_check_header: "false"},
            payload=chat_completion_payload(model=guardrails_echo_llmisvc.name, prompt=PII_PROMPT),
            is_expected=lambda resp: resp.status_code in (401, 403),
            description="client-supplied guardrail header rejected",
        )
        assert PII_EMAIL not in response.text, f"Denied request must not reach the model: {response.text!r}"
        assert INPUT_RAIL_FLOW not in response.text, (
            f"Request must be denied by the auth policy, not the guardrail: {response.text!r}"
        )
