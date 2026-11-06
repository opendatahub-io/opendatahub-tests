"""Tests for AIGuardrail Output-phase checks on a tenant with API translation enabled.

With spec.apiTranslation.enabled on the tenant's MaasTenantConfig, the shared post-auth ExtProc
streams responses and the shipped Praxis image skips response checks. ai-gateway-controller
therefore refuses an AIGuardrail that offers an Output-phase check in that tenant, rather than
accepting it while its response checks silently never run. Input-only policies are unaffected.
"""

from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from timeout_sampler import TimeoutSampler

from tests.ai_gateway.models_as_a_service.guardrails.constants import (
    AI_GUARDRAIL_TIMEOUT,
    INPUT_CHECK_NAME,
    NEMO_PII_CONFIG_ID,
    OUTPUT_CHECK_NAME,
    OUTPUT_CHECKS_UNSUPPORTED_MESSAGE,
    OUTPUT_CHECKS_UNSUPPORTED_REASON,
)
from tests.ai_gateway.models_as_a_service.guardrails.utils import (
    ai_guardrail_warning_events,
    wait_for_ai_guardrail_bound,
    wait_for_ai_guardrail_refused,
)
from utilities.general import generate_random_name
from utilities.resources.ai_guardrail import AIGuardrail
from utilities.resources.maastenantconfig import MaasTenantConfig
from utilities.resources.nemo_guardrails import NemoGuardrails

_EVENT_TIMEOUT = 120
_EVENT_SLEEP = 5


@pytest.mark.ai_safety
class TestMaaSGuardrailOutputChecksWithApiTranslation:
    """AIGuardrails with Output-phase checks are refused in a tenant whose ExtProc streams responses."""

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "checks",
        [
            pytest.param(
                [{"name": OUTPUT_CHECK_NAME, "configId": NEMO_PII_CONFIG_ID, "phases": ["Output"]}],
                id="test_output_only_check",
            ),
            pytest.param(
                [
                    {"name": INPUT_CHECK_NAME, "configId": NEMO_PII_CONFIG_ID, "phases": ["Input"]},
                    {"name": OUTPUT_CHECK_NAME, "configId": NEMO_PII_CONFIG_ID, "phases": ["Output"]},
                ],
                id="test_separate_input_and_output_checks",
            ),
            pytest.param(
                [{"name": INPUT_CHECK_NAME, "configId": NEMO_PII_CONFIG_ID, "phases": ["Input", "Output"]}],
                id="test_check_with_both_phases",
            ),
        ],
    )
    def test_output_check_refused_on_api_translation_tenant(
        self,
        admin_client: DynamicClient,
        maas_subscription_namespace: Namespace,
        guardrails_nemo_server: NemoGuardrails,
        api_translation_tenant_config: MaasTenantConfig,
        checks: list[dict[str, Any]],
    ) -> None:
        """Given a tenant with API translation enabled,
        when an AIGuardrail offering any Output-phase check is created there,
        then Accepted is False with reason OutputChecksUnsupported, no endpoint or binding revision
        is published, and a matching Warning Event is recorded.
        """
        with AIGuardrail(
            client=admin_client,
            name=generate_random_name(prefix="maas-output-refused"),
            namespace=maas_subscription_namespace.name,
            provider={"nemo": {"ref": {"name": guardrails_nemo_server.name}}, "timeout": AI_GUARDRAIL_TIMEOUT},
            checks=checks,
            teardown=True,
        ) as ai_guardrail:
            accepted = wait_for_ai_guardrail_refused(ai_guardrail=ai_guardrail, reason=OUTPUT_CHECKS_UNSUPPORTED_REASON)
            assert OUTPUT_CHECKS_UNSUPPORTED_MESSAGE in accepted.get("message", ""), (
                f"Expected Accepted message to explain the API translation refusal, got: {accepted}"
            )
            status = ai_guardrail.instance.status
            assert not status.providerEndpoint, f"Expected no providerEndpoint, got: {status.providerEndpoint!r}"
            assert not status.bindingRevision, f"Expected no bindingRevision, got: {status.bindingRevision!r}"

            for events in TimeoutSampler(
                wait_timeout=_EVENT_TIMEOUT,
                sleep=_EVENT_SLEEP,
                func=ai_guardrail_warning_events,
                admin_client=admin_client,
                ai_guardrail=ai_guardrail,
                reason=OUTPUT_CHECKS_UNSUPPORTED_REASON,
            ):
                if events:
                    break
            assert any(OUTPUT_CHECKS_UNSUPPORTED_MESSAGE in (event.message or "") for event in events), (
                f"Expected an {OUTPUT_CHECKS_UNSUPPORTED_REASON} Warning Event explaining the refusal, "
                f"got: {[event.message for event in events]}"
            )

    @pytest.mark.tier1
    def test_input_only_check_accepted_on_api_translation_tenant(
        self,
        admin_client: DynamicClient,
        maas_subscription_namespace: Namespace,
        guardrails_nemo_server: NemoGuardrails,
        api_translation_tenant_config: MaasTenantConfig,
    ) -> None:
        """Given a tenant with API translation enabled,
        when an AIGuardrail offering only Input-phase checks is created there,
        then it is accepted and publishes its provider endpoint and binding revision.
        """
        with AIGuardrail(
            client=admin_client,
            name=generate_random_name(prefix="maas-input-accepted"),
            namespace=maas_subscription_namespace.name,
            provider={"nemo": {"ref": {"name": guardrails_nemo_server.name}}, "timeout": AI_GUARDRAIL_TIMEOUT},
            checks=[{"name": INPUT_CHECK_NAME, "configId": NEMO_PII_CONFIG_ID, "phases": ["Input"]}],
            teardown=True,
        ) as ai_guardrail:
            wait_for_ai_guardrail_bound(ai_guardrail=ai_guardrail)
            provider_endpoint = ai_guardrail.instance.status.providerEndpoint
            assert provider_endpoint, "Expected an accepted Input-only AIGuardrail to publish providerEndpoint"

    @pytest.mark.tier2
    def test_disabling_api_translation_accepts_refused_output_check(
        self,
        admin_client: DynamicClient,
        maas_subscription_namespace: Namespace,
        guardrails_nemo_server: NemoGuardrails,
        api_translation_tenant_config: MaasTenantConfig,
    ) -> None:
        """Given an AIGuardrail with an Output-phase check refused on a tenant with API translation enabled,
        when API translation is disabled on the tenant's MaasTenantConfig,
        then the policy is re-evaluated without being edited and becomes accepted.
        """
        with AIGuardrail(
            client=admin_client,
            name=generate_random_name(prefix="maas-output-recovered"),
            namespace=maas_subscription_namespace.name,
            provider={"nemo": {"ref": {"name": guardrails_nemo_server.name}}, "timeout": AI_GUARDRAIL_TIMEOUT},
            checks=[{"name": OUTPUT_CHECK_NAME, "configId": NEMO_PII_CONFIG_ID, "phases": ["Output"]}],
            teardown=True,
        ) as ai_guardrail:
            wait_for_ai_guardrail_refused(ai_guardrail=ai_guardrail, reason=OUTPUT_CHECKS_UNSUPPORTED_REASON)

            with ResourceEditor(
                patches={api_translation_tenant_config: {"spec": {"apiTranslation": {"enabled": False}}}}
            ):
                wait_for_ai_guardrail_bound(ai_guardrail=ai_guardrail)
