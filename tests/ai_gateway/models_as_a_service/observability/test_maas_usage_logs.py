import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.resource import ResourceEditor

from tests.ai_gateway.models_as_a_service.maas_subscription.utils import chat_payload_for_url, poll_expected_status
from tests.ai_gateway.models_as_a_service.observability.constants import (
    USAGE_LOGS_INFERENCE_CALLS,
    USAGE_LOGS_RATE_LIMIT_MAX_CALLS,
    USAGE_LOGS_RATE_LIMIT_TOKENS_PER_MINUTE,
)
from tests.ai_gateway.models_as_a_service.observability.utils import (
    get_loki_usage_record_count,
    get_loki_usage_totals_snapshot,
    loki_query_port_forward,
    send_non_streaming_inference_calls,
    wait_for_loki_usage_record,
    wait_for_loki_usage_totals,
    wait_for_usage_logs_envoy_filter_capture,
)
from tests.ai_gateway.models_as_a_service.utils import (
    build_maas_headers,
    create_api_key,
    revoke_api_key,
    verify_streaming_chat_completions,
)
from utilities.general import generate_random_name
from utilities.resources.maastenantconfig import MaasTenantConfig


@pytest.mark.usefixtures(
    "maas_inference_service_tinyllama_modelcar",
    "minimal_subscription_for_modelcar",
    "maas_auth_policy_tinyllama_modelcar",
    "usage_logs_monitoring_ready",
    "maas_config_with_usage_logging_disabled",
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
        request_model_id: str,
        logged_model: str,
        dsci_monitoring_namespace: str,
        current_client_token: str,
    ) -> None:
        """Given usage logs are configured in DSCI and usageLogging is disabled on Config/default,
        when non-streaming inference requests are sent through the MaaS gateway, then the request
        count, successful request count, and total tokens logged in Loki for the model increase by
        exactly the expected deltas.

        Verifies the end-to-end usage-logs path: the MaaS gateway EnvoyFilter access logs flow
        through the data-science usage-logs collector into the S3-backed LokiStack application
        tenant, queryable per model.
        """
        with loki_query_port_forward(namespace=dsci_monitoring_namespace):
            requests_before, successful_requests_before, tokens_before = get_loki_usage_totals_snapshot(
                http_session=request_session_http,
                admin_token=current_client_token,
                model=logged_model,
            )

            successful_calls, _, expected_total_tokens = send_non_streaming_inference_calls(
                request_session_http=request_session_http,
                model_url=usage_logs_model_url,
                headers=usage_logs_maas_headers,
                model_id=request_model_id,
                logged_model=logged_model,
                call_count=USAGE_LOGS_INFERENCE_CALLS,
                log_prefix="MaaS usage-logs",
            )

            expected_requests = requests_before + successful_calls
            expected_successful_requests = successful_requests_before + successful_calls
            expected_tokens = tokens_before + expected_total_tokens

            wait_for_loki_usage_totals(
                http_session=request_session_http,
                admin_token=current_client_token,
                model=logged_model,
                expected_totals=(expected_requests, expected_successful_requests, expected_tokens),
            )

    def test_streaming_inference_usage_logged_in_loki(
        self,
        request_session_http: requests.Session,
        usage_logs_model_url: str,
        usage_logs_maas_headers: dict,
        request_model_id: str,
        logged_model: str,
        dsci_monitoring_namespace: str,
        current_client_token: str,
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
        with loki_query_port_forward(namespace=dsci_monitoring_namespace):
            requests_before, successful_requests_before, tokens_before = get_loki_usage_totals_snapshot(
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

            wait_for_loki_usage_totals(
                http_session=request_session_http,
                admin_token=current_client_token,
                model=logged_model,
                expected_totals=(expected_requests, expected_successful_requests, expected_tokens),
                timeout_context="after streaming inference calls",
            )

    def test_rate_limited_inference_usage_logged_in_loki(
        self,
        request_session_http: requests.Session,
        usage_logs_model_url: str,
        usage_logs_rate_limited_headers: dict,
        usage_logs_rate_limit_subscription: MaaSSubscription,
        request_model_id: str,
        logged_model: str,
        dsci_monitoring_namespace: str,
        current_client_token: str,
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
        # Rate-limited calls are rejected before a model response exists, so their count is
        # queried by response_type and the dedicated subscription's name, which the gateway
        # access log records for rejected requests too (see loki_usage_rate_limited_requests_query).
        rate_limit_subscription = usage_logs_rate_limit_subscription.name

        with loki_query_port_forward(namespace=dsci_monitoring_namespace):
            successful_before, rate_limited_before, tokens_before = get_loki_usage_totals_snapshot(
                http_session=request_session_http,
                admin_token=current_client_token,
                model=logged_model,
                subscription=rate_limit_subscription,
            )

            # The subscription's token quota is small enough that a few successful calls
            # exhaust it; the remaining calls are rejected with 429. Counts are derived from
            # the observed responses because the 1-minute quota window may reset mid-loop.
            successful_calls, rate_limited_calls, expected_total_tokens = send_non_streaming_inference_calls(
                request_session_http=request_session_http,
                model_url=usage_logs_model_url,
                headers=usage_logs_rate_limited_headers,
                model_id=request_model_id,
                logged_model=logged_model,
                call_count=USAGE_LOGS_RATE_LIMIT_MAX_CALLS,
                log_prefix="MaaS usage-logs rate-limit",
                expected_status_codes=(200, 429),
            )

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

            wait_for_loki_usage_totals(
                http_session=request_session_http,
                admin_token=current_client_token,
                model=logged_model,
                expected_totals=(expected_successful_requests, expected_rate_limited_requests, expected_tokens),
                subscription=rate_limit_subscription,
            )

    @pytest.mark.slow
    def test_identity_capture_follows_tenant_settings(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        usage_logs_capture_subscription: MaaSSubscription,
        usage_logs_model_url: str,
        dsci_monitoring_namespace: str,
        current_client_token: str,
        base_url: str,
        maas_api_endpoints_ready: None,
        maas_gateway_auth_policy_ready: None,
    ) -> None:
        """Given capture toggles, when governed inference runs, then Loki logs contain only enabled identities."""
        gateway_probe = request_session_http.get(f"{base_url}/v1/models", timeout=30)
        assert gateway_probe.status_code in {200, 401, 403}, (
            f"MaaS API is not reachable through the gateway: HTTP {gateway_probe.status_code}"
        )
        subscription_name = usage_logs_capture_subscription.name
        organization_id = usage_logs_capture_subscription.instance.to_dict()["spec"]["tokenMetadata"]["organizationId"]
        tenant_config = MaasTenantConfig(
            client=admin_client,
            name="default-tenant",
            namespace=usage_logs_capture_subscription.namespace,
            ensure_exists=True,
        )

        with loki_query_port_forward(namespace=dsci_monitoring_namespace):
            for capture_user, capture_organization in ((False, False), (True, True), (True, False), (False, True)):
                editor = ResourceEditor(
                    patches={
                        tenant_config: {
                            "spec": {
                                "telemetry": {
                                    "enabled": True,
                                    "logs": {
                                        "captureUser": capture_user,
                                        "captureOrganization": capture_organization,
                                    },
                                }
                            }
                        }
                    }
                )
                try:
                    editor.update(backup_resources=True)
                    wait_for_usage_logs_envoy_filter_capture(
                        admin_client=admin_client,
                        capture_user=capture_user,
                        capture_organization=capture_organization,
                    )
                    key_name = generate_random_name(prefix="usage-capture-key")
                    _, key = create_api_key(
                        base_url=base_url,
                        ocp_user_token=current_client_token,
                        request_session_http=request_session_http,
                        api_key_name=key_name,
                        subscription=subscription_name,
                    )
                    try:
                        response = poll_expected_status(
                            request_session_http=request_session_http,
                            model_url=usage_logs_model_url,
                            headers=build_maas_headers(token=key["key"]),
                            payload=chat_payload_for_url(model_url=usage_logs_model_url),
                            expected_statuses={200},
                        )
                        assert response.status_code == 200, "Governed inference did not succeed"
                        record_count = wait_for_loki_usage_record(
                            http_session=request_session_http,
                            admin_token=current_client_token,
                            subscription=subscription_name,
                            key_name=key_name,
                        )
                        user_count = get_loki_usage_record_count(
                            http_session=request_session_http,
                            admin_token=current_client_token,
                            subscription=subscription_name,
                            key_name=key_name,
                            identity_field="user_id",
                        )
                        organization_count = get_loki_usage_record_count(
                            http_session=request_session_http,
                            admin_token=current_client_token,
                            subscription=subscription_name,
                            key_name=key_name,
                            identity_field="organization_id",
                            identity_value=organization_id if capture_organization else None,
                        )
                        assert user_count == (record_count if capture_user else 0), (
                            f"captureUser={capture_user}: {user_count} of {record_count} logs contain user_id"
                        )
                        assert organization_count == (record_count if capture_organization else 0), (
                            f"captureOrganization={capture_organization}: {organization_count} of {record_count} "
                            "logs contain the subscription organization_id"
                        )
                    finally:
                        revoke_response, _ = revoke_api_key(
                            request_session_http=request_session_http,
                            base_url=base_url,
                            key_id=key["id"],
                            ocp_user_token=current_client_token,
                        )
                        assert revoke_response.status_code in (200, 404), (
                            f"Unexpected revocation status for key id={key['id']}: {revoke_response.status_code}"
                        )
                finally:
                    editor.restore()
