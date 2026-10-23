import portforward
import pytest
import requests
from ocp_resources.maas_subscription import MaaSSubscription
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_gateway.models_as_a_service.maas_subscription.utils import maas_body_routed_model_name
from tests.ai_gateway.models_as_a_service.observability.constants import (
    LOKI_GATEWAY_SERVICE_NAME,
    LOKI_GATEWAY_SERVICE_PORT,
    LOKI_QUERY_LOCAL_PORT,
    USAGE_LOGS_INFERENCE_CALLS,
    USAGE_LOGS_PROPAGATION_SLEEP,
    USAGE_LOGS_PROPAGATION_TIMEOUT,
    USAGE_LOGS_RATE_LIMIT_MAX_CALLS,
    USAGE_LOGS_RATE_LIMIT_TOKENS_PER_MINUTE,
)
from tests.ai_gateway.models_as_a_service.observability.utils import (
    get_loki_usage_totals,
    get_loki_usage_totals_with_rate_limits,
)
from tests.ai_gateway.models_as_a_service.utils import (
    get_total_tokens,
    verify_chat_completions,
    verify_streaming_chat_completions,
)
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
        # Requests must identify the model by its MaaS body-routed identity. The gateway also
        # lists models owned by other workloads, so the id is derived from the LLMInferenceService
        # rather than taken from the first /v1/models entry.
        request_model_id = maas_body_routed_model_name(llmisvc=maas_inference_service_tinyllama_modelcar)
        # The access-log EnvoyFilter records `model` from the chat-completions response body, which
        # carries the vLLM-served model name, so Loki queries and assertions are keyed on it.
        logged_model = maas_inference_service_tinyllama_modelcar.name

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
                model=logged_model,
            )

            expected_total_tokens = 0
            for call_index in range(USAGE_LOGS_INFERENCE_CALLS):
                response = verify_chat_completions(
                    request_session_http=request_session_http,
                    model_url=usage_logs_model_url,
                    headers=usage_logs_maas_headers,
                    model_id=request_model_id,
                    log_prefix=f"MaaS usage-logs [{call_index + 1}/{USAGE_LOGS_INFERENCE_CALLS}]",
                )
                # The Loki `model` label is taken from the response body, so it must match the
                # served model name the queries are filtered by.
                assert response.json()["model"] == logged_model, (
                    f"Response model '{response.json()['model']}' does not match the served model "
                    f"name '{logged_model}' the Loki queries are filtered by"
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
                    model=logged_model,
                ):
                    if (
                        requests_after == expected_requests
                        and successful_requests_after == expected_successful_requests
                        and tokens_after == expected_tokens
                    ):
                        break
            except TimeoutExpiredError:
                pytest.fail(
                    f"Loki usage totals for model '{logged_model}' did not reach the expected values within "
                    f"{USAGE_LOGS_PROPAGATION_TIMEOUT}s: "
                    f"expected (requests={expected_requests}, successful_requests={expected_successful_requests}, "
                    f"tokens={expected_tokens}), "
                    f"last observed (requests={requests_after}, successful_requests={successful_requests_after}, "
                    f"tokens={tokens_after})"
                )

    def test_streaming_inference_usage_logged_in_loki(
        self,
        request_session_http: requests.Session,
        usage_logs_model_url: str,
        usage_logs_maas_headers: dict,
        maas_inference_service_tinyllama_modelcar: LLMInferenceService,
        dsci_monitoring_namespace: str,
        current_client_token: str,
        maas_config_with_usage_logging_disabled: MaaSConfig,
    ) -> None:
        """Given usage logs are configured in DSCI and usageLogging is disabled on Config/default,
        when streaming (SSE) inference requests are sent through the MaaS gateway with
        stream_options.include_usage, then the request count, successful request count, and total
        tokens logged in Loki for the model increase by exactly the deltas reported by each
        stream's terminal usage event.

        Verifies the streaming branch of the usage-logs path: json_to_metadata cannot parse
        text/event-stream bodies, so the gateway EnvoyFilter's sse_tokens Lua filter reads the
        usage fields from the SSE chunks and records them through the same data-science
        usage-logs collector into the S3-backed LokiStack application tenant.
        """
        # Requests must identify the model by its MaaS body-routed identity; an explicit id is
        # used because /v1/models also lists models owned by other workloads.
        request_model_id = maas_body_routed_model_name(llmisvc=maas_inference_service_tinyllama_modelcar)
        # The Loki `model` label is taken from the response body, so it must match the served
        # model name the queries are filtered by — same as the non-streaming test.
        logged_model = maas_inference_service_tinyllama_modelcar.name

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
                model=logged_model,
            )

            expected_total_tokens = 0
            for call_index in range(USAGE_LOGS_INFERENCE_CALLS):
                served_model, response_total_tokens = verify_streaming_chat_completions(
                    request_session_http=request_session_http,
                    model_url=usage_logs_model_url,
                    headers=usage_logs_maas_headers,
                    model_id=request_model_id,
                    log_prefix=f"MaaS usage-logs streaming [{call_index + 1}/{USAGE_LOGS_INFERENCE_CALLS}]",
                )
                assert served_model == logged_model, (
                    f"Streamed model '{served_model}' does not match the served model name "
                    f"'{logged_model}' the Loki queries are filtered by"
                )
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
                    model=logged_model,
                ):
                    if (
                        requests_after == expected_requests
                        and successful_requests_after == expected_successful_requests
                        and tokens_after == expected_tokens
                    ):
                        break
            except TimeoutExpiredError:
                pytest.fail(
                    f"Loki usage totals for model '{logged_model}' did not reach the expected values within "
                    f"{USAGE_LOGS_PROPAGATION_TIMEOUT}s after streaming inference calls: "
                    f"expected (requests={expected_requests}, successful_requests={expected_successful_requests}, "
                    f"tokens={expected_tokens}), "
                    f"last observed (requests={requests_after}, successful_requests={successful_requests_after}, "
                    f"tokens={tokens_after})"
                )

    def test_rate_limited_inference_usage_logged_in_loki(
        self,
        request_session_http: requests.Session,
        usage_logs_model_url: str,
        usage_logs_rate_limited_headers: dict,
        usage_logs_rate_limit_subscription: MaaSSubscription,
        maas_inference_service_tinyllama_modelcar: LLMInferenceService,
        dsci_monitoring_namespace: str,
        current_client_token: str,
        maas_config_with_usage_logging_disabled: MaaSConfig,
    ) -> None:
        """Given usage logs are configured in DSCI and inference traffic is pinned to a
        subscription with a tiny token quota, when requests are sent through the MaaS gateway
        until the quota is exhausted and calls are rejected with HTTP 429, then every rejected
        call is logged in Loki with response_type="rate_limit", while only the successful calls
        increase the successful-request count and total tokens for the model.

        Verifies the rate-limited branch of the usage-logs path: the MaaS gateway EnvoyFilter
        access logs the token rate limiter's 429 responses (rate_limit response type, zero
        tokens) through the same data-science usage-logs collector into Loki as successful
        calls, so rate-limited usage is observable without being counted as consumption.
        """
        # Requests must identify the model by its MaaS body-routed identity; an explicit id is
        # used because /v1/models also lists models owned by other workloads.
        request_model_id = maas_body_routed_model_name(llmisvc=maas_inference_service_tinyllama_modelcar)
        # The Loki `model` label of successful calls comes from the response body (the served
        # model name), so the successful-requests and tokens queries are keyed on it — same as
        # the non-streaming test. Rate-limited calls are rejected before a model response
        # exists, so their count is queried by response_type and the dedicated subscription's
        # name, which the gateway access log records for rejected requests too (see
        # loki_usage_rate_limited_requests_query).
        logged_model = maas_inference_service_tinyllama_modelcar.name
        rate_limit_subscription = usage_logs_rate_limit_subscription.name

        with portforward.forward(
            pod_or_service=LOKI_GATEWAY_SERVICE_NAME,
            namespace=dsci_monitoring_namespace,
            from_port=LOKI_QUERY_LOCAL_PORT,
            to_port=LOKI_GATEWAY_SERVICE_PORT,
            waiting=20,
        ):
            successful_before, rate_limited_before, tokens_before = get_loki_usage_totals_with_rate_limits(
                http_session=request_session_http,
                admin_token=current_client_token,
                model=logged_model,
                subscription=rate_limit_subscription,
            )

            # The subscription's token quota is small enough that a few successful calls
            # exhaust it; the remaining calls are rejected with 429. Counts are derived from
            # the observed responses because the 1-minute quota window may reset mid-loop.
            successful_calls = 0
            rate_limited_calls = 0
            expected_total_tokens = 0
            for call_index in range(USAGE_LOGS_RATE_LIMIT_MAX_CALLS):
                response = verify_chat_completions(
                    request_session_http=request_session_http,
                    model_url=usage_logs_model_url,
                    headers=usage_logs_rate_limited_headers,
                    model_id=request_model_id,
                    log_prefix=f"MaaS usage-logs rate-limit [{call_index + 1}/{USAGE_LOGS_RATE_LIMIT_MAX_CALLS}]",
                    expected_status_codes=(200, 429),
                )
                if response.status_code == 429:
                    rate_limited_calls += 1
                    continue
                # The Loki `model` label is taken from the response body, so it must match the
                # served model name the successful-requests and tokens queries are filtered by.
                assert response.json()["model"] == logged_model, (
                    f"Response model '{response.json()['model']}' does not match the served model "
                    f"name '{logged_model}' the Loki queries are filtered by"
                )
                response_total_tokens = get_total_tokens(resp=response, fail_if_missing=True)
                assert response_total_tokens is not None
                expected_total_tokens += response_total_tokens
                successful_calls += 1

            # The subscription's Limitador counters start empty, so the first call always
            # succeeds; if no 429 is observed the quota was never exhausted.
            assert successful_calls > 0, (
                f"Expected at least one successful inference call before the token quota was "
                f"exhausted, got none in {USAGE_LOGS_RATE_LIMIT_MAX_CALLS} calls"
            )
            assert rate_limited_calls > 0, (
                f"Token quota of {USAGE_LOGS_RATE_LIMIT_TOKENS_PER_MINUTE} per minute was not "
                f"exhausted within {USAGE_LOGS_RATE_LIMIT_MAX_CALLS} inference calls; "
                f"no rate-limited (429) call was observed"
            )

            expected_successful_requests = successful_before + successful_calls
            expected_rate_limited_requests = rate_limited_before + rate_limited_calls
            expected_tokens = tokens_before + expected_total_tokens

            # Usage records propagate asynchronously (gateway EnvoyFilter -> OTel collector -> Loki).
            successful_after, rate_limited_after, tokens_after = None, None, None
            try:
                for successful_after, rate_limited_after, tokens_after in TimeoutSampler(
                    wait_timeout=USAGE_LOGS_PROPAGATION_TIMEOUT,
                    sleep=USAGE_LOGS_PROPAGATION_SLEEP,
                    func=get_loki_usage_totals_with_rate_limits,
                    sensitive_keys=frozenset({"admin_token"}),
                    http_session=request_session_http,
                    admin_token=current_client_token,
                    model=logged_model,
                    subscription=rate_limit_subscription,
                ):
                    if (
                        successful_after == expected_successful_requests
                        and rate_limited_after == expected_rate_limited_requests
                        and tokens_after == expected_tokens
                    ):
                        break
            except TimeoutExpiredError:
                pytest.fail(
                    f"Loki usage totals for model '{logged_model}' and subscription "
                    f"'{rate_limit_subscription}' did not reach the expected values within "
                    f"{USAGE_LOGS_PROPAGATION_TIMEOUT}s: "
                    f"expected (successful_requests={expected_successful_requests}, "
                    f"rate_limited_requests={expected_rate_limited_requests}, tokens={expected_tokens}), "
                    f"last observed (successful_requests={successful_after}, "
                    f"rate_limited_requests={rate_limited_after}, tokens={tokens_after})"
                )
