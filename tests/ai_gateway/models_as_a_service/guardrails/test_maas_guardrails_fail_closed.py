import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.namespace import Namespace

from tests.ai_gateway.models_as_a_service.guardrails.constants import (
    AI_GUARDRAIL_PROVIDER_NOT_FOUND_REASON,
    AI_GUARDRAIL_TIMEOUT,
    CLEAN_PROMPT,
    EXT_AUTH_REASON_HEADER,
    GUARDRAIL_UNAVAILABLE_ERROR,
    INPUT_CHECK_NAME,
    INPUT_RAIL_FLOW,
    NEMO_PII_CONFIG_ID,
    PII_PROMPT,
)
from tests.ai_gateway.models_as_a_service.guardrails.utils import (
    chat_completion_payload,
    wait_for_ai_guardrail_binding_withdrawn,
    wait_for_chat_response,
    wait_for_guardrail_checks_compiled,
    wait_for_guardrail_unavailable_selection,
)
from tests.ai_gateway.models_as_a_service.utils import build_maas_headers
from utilities.general import generate_random_name
from utilities.resources.ai_guardrail import AIGuardrail
from utilities.resources.llm_inference_service import LLMInferenceService
from utilities.resources.maa_s_subscription import MaaSSubscription
from utilities.resources.nemo_guardrails import NemoGuardrails


@pytest.mark.usefixtures("guardrails_maas_auth_policy")
@pytest.mark.ai_safety
class TestMaaSGuardrailFailClosed:
    """MaaS selection rejects subscriptions whose guardrail attachments cannot be resolved."""

    @pytest.mark.tier1
    def test_unknown_guardrail_check_fails_closed(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        maas_api_infra_namespace: str,
        maas_subscription_namespace: Namespace,
        guardrails_model_ref: MaaSModelRef,
        pii_ai_guardrail: AIGuardrail,
    ) -> None:
        """Given a subscription referring to an unknown guardrail check,
        when MaaS selects that subscription for its model,
        then selection fails closed without returning any selected checks.
        """
        with MaaSSubscription(
            client=admin_client,
            name=generate_random_name(prefix="e2e-guardrails-invalid-check"),
            namespace=maas_subscription_namespace.name,
            owner={"groups": [{"name": "system:authenticated"}]},
            model_refs=[
                {
                    "name": guardrails_model_ref.name,
                    "namespace": guardrails_model_ref.namespace,
                    "tokenRateLimits": [{"limit": 100000, "window": "1m"}],
                }
            ],
            guardrails=[{"ref": {"name": pii_ai_guardrail.name}, "checks": ["does-not-exist"]}],
            teardown=True,
            wait_for_resource=True,
        ) as invalid_check_subscription:
            selection = wait_for_guardrail_unavailable_selection(
                request_session_http=request_session_http,
                maas_api_infra_namespace=maas_api_infra_namespace,
                subscription=invalid_check_subscription,
                model_ref=guardrails_model_ref,
            )

        assert selection["error"] == "guardrail_unavailable"
        assert "selectedGuardrailChecks" not in selection, f"Selection returned partial checks: {selection}"

    @pytest.mark.tier1
    def test_unavailable_nemo_provider_fails_closed(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        maas_api_infra_namespace: str,
        maas_subscription_namespace: Namespace,
        guardrails_model_ref: MaaSModelRef,
    ) -> None:
        """Given a policy whose NeMo provider does not exist,
        when MaaS selects a subscription using that policy,
        then selection fails closed without returning any selected checks.
        """
        with (
            AIGuardrail(
                client=admin_client,
                name=generate_random_name(prefix="e2e-guardrails-missing-nemo"),
                namespace=maas_subscription_namespace.name,
                provider={
                    "nemo": {"ref": {"name": "does-not-exist"}},
                    "timeout": AI_GUARDRAIL_TIMEOUT,
                },
                checks=[{"name": INPUT_CHECK_NAME, "configId": NEMO_PII_CONFIG_ID, "phases": ["Input"]}],
                teardown=True,
                wait_for_resource=True,
            ) as unavailable_provider_guardrail,
            MaaSSubscription(
                client=admin_client,
                name=generate_random_name(prefix="e2e-guardrails-unavailable-provider"),
                namespace=maas_subscription_namespace.name,
                owner={"groups": [{"name": "system:authenticated"}]},
                model_refs=[
                    {
                        "name": guardrails_model_ref.name,
                        "namespace": guardrails_model_ref.namespace,
                        "tokenRateLimits": [{"limit": 100000, "window": "1m"}],
                    }
                ],
                guardrails=[{"ref": {"name": unavailable_provider_guardrail.name}, "checks": [INPUT_CHECK_NAME]}],
                teardown=True,
                wait_for_resource=True,
            ) as unavailable_provider_subscription,
        ):
            selection = wait_for_guardrail_unavailable_selection(
                request_session_http=request_session_http,
                maas_api_infra_namespace=maas_api_infra_namespace,
                subscription=unavailable_provider_subscription,
                model_ref=guardrails_model_ref,
            )

        assert selection["error"] == "guardrail_unavailable"
        assert "selectedGuardrailChecks" not in selection, f"Selection returned partial checks: {selection}"

    @pytest.mark.tier1
    def test_missing_guardrail_policy_fails_closed(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        maas_api_infra_namespace: str,
        maas_subscription_namespace: Namespace,
        guardrails_model_ref: MaaSModelRef,
    ) -> None:
        """Given a subscription referring to a missing guardrail policy,
        when MaaS selects that subscription for its model,
        then selection fails closed without returning any selected checks.
        """
        with MaaSSubscription(
            client=admin_client,
            name=generate_random_name(prefix="e2e-guardrails-missing-policy"),
            namespace=maas_subscription_namespace.name,
            owner={"groups": [{"name": "system:authenticated"}]},
            model_refs=[
                {
                    "name": guardrails_model_ref.name,
                    "namespace": guardrails_model_ref.namespace,
                    "tokenRateLimits": [{"limit": 100000, "window": "1m"}],
                }
            ],
            guardrails=[{"ref": {"name": "does-not-exist"}, "checks": [INPUT_CHECK_NAME]}],
            teardown=True,
            wait_for_resource=True,
        ) as missing_policy_subscription:
            selection = wait_for_guardrail_unavailable_selection(
                request_session_http=request_session_http,
                maas_api_infra_namespace=maas_api_infra_namespace,
                subscription=missing_policy_subscription,
                model_ref=guardrails_model_ref,
            )

        assert selection["error"] == "guardrail_unavailable"
        assert "selectedGuardrailChecks" not in selection, f"Selection returned partial checks: {selection}"


@pytest.mark.usefixtures("guardrails_maas_auth_policy")
@pytest.mark.ai_safety
class TestMaaSGuardrailProviderRemoval:
    """Gateway requests fail closed after the NeMo provider of an enforced guardrail is removed."""

    @pytest.mark.tier1
    def test_removed_nemo_provider_fails_closed_at_gateway(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        maas_api_infra_namespace: str,
        guardrails_model_ref: MaaSModelRef,
        guardrails_model_url: str,
        guardrails_echo_llmisvc: LLMInferenceService,
        guardrails_nemo_server: NemoGuardrails,
        pii_ai_guardrail: AIGuardrail,
        input_guarded_subscription: MaaSSubscription,
        input_guarded_api_key: str,
        input_check_header: str,
    ) -> None:
        """Given a subscription whose input guardrail is enforced through an accepted NeMo provider,
        when the NeMo provider is deleted,
        then the policy binding is withdrawn and gateway requests are rejected with 403 instead of reaching the model.
        """
        headers = build_maas_headers(token=input_guarded_api_key)
        pii_payload = chat_completion_payload(model=guardrails_echo_llmisvc.name, prompt=PII_PROMPT)
        wait_for_guardrail_checks_compiled(admin_client=admin_client, check_headers=[input_check_header])
        # A 403 naming the rail proves NeMo evaluates requests before the provider is removed.
        wait_for_chat_response(
            request_session_http=request_session_http,
            model_url=guardrails_model_url,
            headers=headers,
            payload=pii_payload,
            is_expected=lambda resp: resp.status_code == 403 and INPUT_RAIL_FLOW in resp.text,
            description="PII prompt blocked by the NeMo input rail before provider removal",
        )
        allowed_response = request_session_http.post(
            url=guardrails_model_url,
            headers=headers,
            json=chat_completion_payload(model=guardrails_echo_llmisvc.name, prompt=CLEAN_PROMPT),
            timeout=60,
        )
        assert allowed_response.status_code == 200, (
            f"Clean prompt rejected before provider removal: status={allowed_response.status_code}"
        )

        guardrails_nemo_server.clean_up()

        wait_for_ai_guardrail_binding_withdrawn(
            ai_guardrail=pii_ai_guardrail,
            reason=AI_GUARDRAIL_PROVIDER_NOT_FOUND_REASON,
        )
        selection = wait_for_guardrail_unavailable_selection(
            request_session_http=request_session_http,
            maas_api_infra_namespace=maas_api_infra_namespace,
            subscription=input_guarded_subscription,
            model_ref=guardrails_model_ref,
        )
        assert selection["error"] == GUARDRAIL_UNAVAILABLE_ERROR
        assert "selectedGuardrailChecks" not in selection, f"Selection returned partial checks: {selection}"

        def fail_closed_rejection(response: requests.Response) -> bool:
            # NeMo would block this PII prompt and fail-closed rejects it, so a 2xx is a bypass at any point.
            if 200 <= response.status_code < 300:
                pytest.fail(f"PII prompt bypassed the unavailable guardrail: status={response.status_code}")
            # Authorino sends the reason header twice, so the value is matched as a substring.
            return response.status_code == 403 and (
                GUARDRAIL_UNAVAILABLE_ERROR in response.headers.get(EXT_AUTH_REASON_HEADER, "")
                or GUARDRAIL_UNAVAILABLE_ERROR in response.text
            )

        # The reason tells this 403 apart from the NeMo rail 403 seen before removal. It comes from
        # Authorino on the selection error, or from the closed Praxis check while a selection is cached.
        wait_for_chat_response(
            request_session_http=request_session_http,
            model_url=guardrails_model_url,
            headers=headers,
            payload=pii_payload,
            is_expected=fail_closed_rejection,
            description="PII prompt rejected fail-closed after provider removal",
        )
