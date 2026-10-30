import portforward
import pytest
import requests
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_gateway.models_as_a_service.observability.constants import (
    LOKI_GATEWAY_SERVICE_NAME,
    LOKI_GATEWAY_SERVICE_PORT,
    LOKI_QUERY_LOCAL_PORT,
    USAGE_LOGS_INFERENCE_CALLS,
    USAGE_LOGS_PROPAGATION_SLEEP,
    USAGE_LOGS_PROPAGATION_TIMEOUT,
)
from tests.ai_gateway.models_as_a_service.observability.utils import get_loki_usage_totals
from tests.ai_gateway.models_as_a_service.utils import get_total_tokens, verify_chat_completions
from utilities.resources.llm_inference_service import LLMInferenceService
from utilities.resources.maas_config import Config as MaaSConfig


@pytest.mark.usefixtures(
    "maas_inference_service_tinyllama_modelcar",
    "minimal_subscription_for_modelcar",
    "maas_auth_policy_tinyllama_modelcar",
    "usage_logs_monitoring_ready",
)
@pytest.mark.tier2
@pytest.mark.metrics
class TestMaaSUsageLogsInference:
    """Validate MaaS inference usage logging in the odh-observability LokiStack."""

    def test_non_streaming_inference_usage_logged_in_loki(
        self,
        request_session_http: requests.Session,
        usage_logs_model_url: str,
        usage_logs_maas_headers: dict,
        usage_logs_maas_models: list[dict],
        maas_inference_service_tinyllama_modelcar: LLMInferenceService,
        dsci_monitoring_namespace: str,
        current_client_token: str,
        maas_config_with_usage_logging_disabled: MaaSConfig,
    ) -> None:
        """Given usage logs are configured in DSCI and usageLogging is disabled on Config/default,
        when non-streaming inference requests are sent through the MaaS gateway, then the request
        count, successful request count, and total tokens logged in Loki for the model increase by
        exactly the expected deltas.

        Verifies the end-to-end usage-logs path: the MaaS gateway EnvoyFilter access logs flow
        through the data-science usage-logs collector into the S3-backed LokiStack application
        tenant, queryable per model.
        """
        # The MaaS gateway's access-log EnvoyFilter extracts `model` from the chat-completions
        # response body (the vLLM-served model name, e.g. "llm-oci-tinyllama"), not the
        # publisher-qualified id /v1/models advertises (e.g. "publishers/llm/models/llm-oci-tinyllama").
        # Loki's `model` label and the response's `model` field use that served name, so queries
        # and assertions must be keyed on it rather than on the /v1/models id.
        model_id = maas_inference_service_tinyllama_modelcar.name

        with portforward.forward(
            pod_or_service=LOKI_GATEWAY_SERVICE_NAME,
            namespace=dsci_monitoring_namespace,
            from_port=LOKI_QUERY_LOCAL_PORT,
            to_port=LOKI_GATEWAY_SERVICE_PORT,
            waiting=20,
        ):
            requests_before, successful_requests_before, tokens_before = get_loki_usage_totals(
                http_session=request_session_http,
                admin_token=current_client_token,
                model=model_id,
            )

            expected_total_tokens = 0
            for call_index in range(USAGE_LOGS_INFERENCE_CALLS):
                response = verify_chat_completions(
                    request_session_http=request_session_http,
                    model_url=usage_logs_model_url,
                    headers=usage_logs_maas_headers,
                    models_list=usage_logs_maas_models,
                    log_prefix=f"MaaS usage-logs [{call_index + 1}/{USAGE_LOGS_INFERENCE_CALLS}]",
                )
                # The Loki `model` label is taken from the response body, so it must match the
                # model id the queries are filtered by.
                assert response.json()["model"] == model_id, (
                    f"Response model '{response.json()['model']}' does not match the model id "
                    f"'{model_id}' the Loki queries are filtered by"
                )
                response_total_tokens = get_total_tokens(resp=response, fail_if_missing=True)
                assert response_total_tokens is not None
                expected_total_tokens += response_total_tokens

            expected_requests = requests_before + USAGE_LOGS_INFERENCE_CALLS
            expected_successful_requests = successful_requests_before + USAGE_LOGS_INFERENCE_CALLS
            expected_tokens = tokens_before + expected_total_tokens

            # Usage records propagate asynchronously (gateway EnvoyFilter -> OTel collector -> Loki).
            requests_after, successful_requests_after, tokens_after = None, None, None
            try:
                for requests_after, successful_requests_after, tokens_after in TimeoutSampler(
                    wait_timeout=USAGE_LOGS_PROPAGATION_TIMEOUT,
                    sleep=USAGE_LOGS_PROPAGATION_SLEEP,
                    func=get_loki_usage_totals,
                    sensitive_keys=frozenset({"admin_token"}),
                    http_session=request_session_http,
                    admin_token=current_client_token,
                    model=model_id,
                ):
                    if (
                        requests_after == expected_requests
                        and successful_requests_after == expected_successful_requests
                        and tokens_after == expected_tokens
                    ):
                        break
            except TimeoutExpiredError:
                pytest.fail(
                    f"Loki usage totals for model '{model_id}' did not reach the expected values within "
                    f"{USAGE_LOGS_PROPAGATION_TIMEOUT}s: "
                    f"expected (requests={expected_requests}, successful_requests={expected_successful_requests}, "
                    f"tokens={expected_tokens}), "
                    f"last observed (requests={requests_after}, successful_requests={successful_requests_after}, "
                    f"tokens={tokens_after})"
                )
