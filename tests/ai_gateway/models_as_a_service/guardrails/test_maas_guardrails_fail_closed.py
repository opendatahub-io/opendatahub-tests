import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.namespace import Namespace

from tests.ai_gateway.models_as_a_service.guardrails.constants import (
    AI_GUARDRAIL_TIMEOUT,
    INPUT_CHECK_NAME,
    NEMO_PII_CONFIG_ID,
)
from tests.ai_gateway.models_as_a_service.guardrails.utils import wait_for_guardrail_unavailable_selection
from utilities.general import generate_random_name
from utilities.resources.ai_guardrail import AIGuardrail
from utilities.resources.maa_s_subscription import MaaSSubscription


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
