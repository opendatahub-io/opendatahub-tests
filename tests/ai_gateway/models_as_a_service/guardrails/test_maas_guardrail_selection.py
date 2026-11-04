import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.resource import ResourceEditor

from tests.ai_gateway.models_as_a_service.guardrails.constants import (
    INPUT_CHECK_NAME,
    NEMO_PHASE_CONFIG_ID,
    OUTPUT_CHECK_NAME,
)
from tests.ai_gateway.models_as_a_service.guardrails.utils import (
    guarded_maas_subscription,
    guardrail_check_header,
    guardrail_check_id,
    wait_for_ai_guardrail_bound,
    wait_for_guardrail_header_catalog,
    wait_for_selected_guardrail_checks,
    wait_for_selected_guardrail_checks_for_groups,
)
from tests.ai_gateway.models_as_a_service.maas_api_key.utils import MAAS_GATEWAY_AUTH_POLICY_NAME
from tests.ai_gateway.models_as_a_service.maas_subscription.utils import wait_for_model_ref_resolved_alias
from tests.ai_gateway.models_as_a_service.utils import AITENANT_INFRA_NAMESPACE, DEFAULT_MAAS_TENANT_NAME
from utilities.general import generate_random_name
from utilities.resources.ai_guardrail import AIGuardrail
from utilities.resources.aitenant import AITenant
from utilities.resources.maa_s_subscription import MaaSSubscription
from utilities.resources.maastenantconfig import MaasTenantConfig


@pytest.mark.usefixtures("guardrails_maas_auth_policy")
@pytest.mark.ai_safety
class TestMaaSGuardrailSelection:
    """MaaS selection returns the guardrail checks attached to each subscription."""

    @pytest.mark.tier1
    def test_selection_returns_only_checks_attached_to_subscription(
        self,
        request_session_http: requests.Session,
        maas_api_infra_namespace: str,
        guardrails_model_ref: MaaSModelRef,
        phase_ai_guardrail: AIGuardrail,
        phase_input_guarded_subscription: MaaSSubscription,
        input_output_guarded_subscription: MaaSSubscription,
    ) -> None:
        """Given subscriptions selecting one check or both checks from the same policy,
        when MaaS selects each subscription for the same model,
        then each response contains exactly the checks attached to that subscription.
        """
        input_check_id = guardrail_check_id(
            namespace=phase_ai_guardrail.namespace,
            guardrail_name=phase_ai_guardrail.name,
            check_name=INPUT_CHECK_NAME,
        )
        output_check_id = guardrail_check_id(
            namespace=phase_ai_guardrail.namespace,
            guardrail_name=phase_ai_guardrail.name,
            check_name=OUTPUT_CHECK_NAME,
        )

        input_only_selection = wait_for_selected_guardrail_checks(
            request_session_http=request_session_http,
            maas_api_infra_namespace=maas_api_infra_namespace,
            subscription=phase_input_guarded_subscription,
            model_ref=guardrails_model_ref,
            expected_check_ids={input_check_id},
        )
        input_output_selection = wait_for_selected_guardrail_checks(
            request_session_http=request_session_http,
            maas_api_infra_namespace=maas_api_infra_namespace,
            subscription=input_output_guarded_subscription,
            model_ref=guardrails_model_ref,
            expected_check_ids={input_check_id, output_check_id},
        )

        assert input_only_selection["selectedGuardrailChecks"] == {input_check_id: True}
        assert input_output_selection["selectedGuardrailChecks"] == {
            input_check_id: True,
            output_check_id: True,
        }


@pytest.mark.usefixtures("guardrails_maas_auth_policy")
@pytest.mark.ai_safety
class TestMaaSGuardrailReconciliation:
    """MaaS selection and gateway headers follow valid guardrail policy changes."""

    @pytest.mark.tier1
    def test_guardrail_selection_and_headers_follow_check_changes(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        maas_api_infra_namespace: str,
        guardrails_model_ref: MaaSModelRef,
        phase_ai_guardrail: AIGuardrail,
        phase_input_guarded_subscription: MaaSSubscription,
    ) -> None:
        """Given a subscription selecting one valid check,
        when the policy and subscription change from one check to two and back to one,
        then selection and the gateway AuthPolicy header catalog match each effective check set.
        """
        input_check = {"name": INPUT_CHECK_NAME, "configId": NEMO_PHASE_CONFIG_ID, "phases": ["Input"]}
        output_check = {"name": OUTPUT_CHECK_NAME, "configId": NEMO_PHASE_CONFIG_ID, "phases": ["Output"]}
        input_check_id = guardrail_check_id(
            namespace=phase_ai_guardrail.namespace,
            guardrail_name=phase_ai_guardrail.name,
            check_name=INPUT_CHECK_NAME,
        )
        output_check_id = guardrail_check_id(
            namespace=phase_ai_guardrail.namespace,
            guardrail_name=phase_ai_guardrail.name,
            check_name=OUTPUT_CHECK_NAME,
        )
        input_header = guardrail_check_header(
            namespace=phase_ai_guardrail.namespace,
            guardrail_name=phase_ai_guardrail.name,
            check_name=INPUT_CHECK_NAME,
        )
        output_header = guardrail_check_header(
            namespace=phase_ai_guardrail.namespace,
            guardrail_name=phase_ai_guardrail.name,
            check_name=OUTPUT_CHECK_NAME,
        )

        phase_ai_guardrail.update(
            resource_dict={
                "metadata": {"name": phase_ai_guardrail.name},
                "spec": {"checks": [input_check]},
            }
        )
        wait_for_ai_guardrail_bound(ai_guardrail=phase_ai_guardrail)
        input_selection = wait_for_selected_guardrail_checks(
            request_session_http=request_session_http,
            maas_api_infra_namespace=maas_api_infra_namespace,
            subscription=phase_input_guarded_subscription,
            model_ref=guardrails_model_ref,
            expected_check_ids={input_check_id},
        )
        input_headers = wait_for_guardrail_header_catalog(
            admin_client=admin_client,
            policy_name=MAAS_GATEWAY_AUTH_POLICY_NAME,
            expected_headers={input_header},
        )
        assert input_selection["selectedGuardrailChecks"] == {input_check_id: True}
        assert input_headers == {input_header}

        phase_ai_guardrail.update(
            resource_dict={
                "metadata": {"name": phase_ai_guardrail.name},
                "spec": {"checks": [input_check, output_check]},
            }
        )
        wait_for_ai_guardrail_bound(ai_guardrail=phase_ai_guardrail)
        phase_input_guarded_subscription.update(
            resource_dict={
                "metadata": {"name": phase_input_guarded_subscription.name},
                "spec": {
                    "guardrails": [
                        {
                            "ref": {"name": phase_ai_guardrail.name},
                            "checks": [INPUT_CHECK_NAME, OUTPUT_CHECK_NAME],
                        }
                    ]
                }
            }
        )
        phase_input_guarded_subscription.wait_for_condition(condition="Ready", status="True", timeout=300)
        input_output_selection = wait_for_selected_guardrail_checks(
            request_session_http=request_session_http,
            maas_api_infra_namespace=maas_api_infra_namespace,
            subscription=phase_input_guarded_subscription,
            model_ref=guardrails_model_ref,
            expected_check_ids={input_check_id, output_check_id},
        )
        input_output_headers = wait_for_guardrail_header_catalog(
            admin_client=admin_client,
            policy_name=MAAS_GATEWAY_AUTH_POLICY_NAME,
            expected_headers={input_header, output_header},
        )
        assert input_output_selection["selectedGuardrailChecks"] == {
            input_check_id: True,
            output_check_id: True,
        }
        assert input_output_headers == {input_header, output_header}

        phase_input_guarded_subscription.update(
            resource_dict={
                "metadata": {"name": phase_input_guarded_subscription.name},
                "spec": {
                    "guardrails": [
                        {
                            "ref": {"name": phase_ai_guardrail.name},
                            "checks": [INPUT_CHECK_NAME],
                        }
                    ]
                }
            }
        )
        phase_input_guarded_subscription.wait_for_condition(condition="Ready", status="True", timeout=300)
        final_selection = wait_for_selected_guardrail_checks(
            request_session_http=request_session_http,
            maas_api_infra_namespace=maas_api_infra_namespace,
            subscription=phase_input_guarded_subscription,
            model_ref=guardrails_model_ref,
            expected_check_ids={input_check_id},
        )
        final_headers = wait_for_guardrail_header_catalog(
            admin_client=admin_client,
            policy_name=MAAS_GATEWAY_AUTH_POLICY_NAME,
            expected_headers={input_header},
        )
        assert final_selection["selectedGuardrailChecks"] == {input_check_id: True}
        assert final_headers == {input_header}


@pytest.mark.usefixtures("guardrails_maas_auth_policy")
@pytest.mark.ai_safety
class TestMaaSGuardrailScopeComposition:
    """MaaS supports matching guardrail attachments from multiple scopes."""

    @pytest.mark.tier1
    def test_duplicate_guardrail_check_remains_selected_after_tenant_source_removal(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        maas_api_infra_namespace: str,
        guardrails_model_ref: MaaSModelRef,
        phase_ai_guardrail: AIGuardrail,
        phase_input_guarded_subscription: MaaSSubscription,
    ) -> None:
        """Given one check attached to both tenant and subscription scopes,
        when the tenant attachment is removed,
        then selection and gateway headers contain one check from the remaining subscription source.
        """
        input_check_id = guardrail_check_id(
            namespace=phase_ai_guardrail.namespace,
            guardrail_name=phase_ai_guardrail.name,
            check_name=INPUT_CHECK_NAME,
        )
        input_header = guardrail_check_header(
            namespace=phase_ai_guardrail.namespace,
            guardrail_name=phase_ai_guardrail.name,
            check_name=INPUT_CHECK_NAME,
        )
        tenant_config = MaasTenantConfig(
            client=admin_client,
            name=DEFAULT_MAAS_TENANT_NAME,
            namespace=phase_input_guarded_subscription.namespace,
            ensure_exists=True,
        )
        tenant_attachment = [
            {
                "ref": {"name": phase_ai_guardrail.name},
                "checks": [INPUT_CHECK_NAME],
            }
        ]

        with ResourceEditor(patches={tenant_config: {"spec": {"guardrails": tenant_attachment}}}):
            duplicate_selection = wait_for_selected_guardrail_checks(
                request_session_http=request_session_http,
                maas_api_infra_namespace=maas_api_infra_namespace,
                subscription=phase_input_guarded_subscription,
                model_ref=guardrails_model_ref,
                expected_check_ids={input_check_id},
            )
            duplicate_headers = wait_for_guardrail_header_catalog(
                admin_client=admin_client,
                policy_name=MAAS_GATEWAY_AUTH_POLICY_NAME,
                expected_headers={input_header},
            )
            assert duplicate_selection["selectedGuardrailChecks"] == {input_check_id: True}
            assert duplicate_headers == {input_header}

        remaining_selection = wait_for_selected_guardrail_checks(
            request_session_http=request_session_http,
            maas_api_infra_namespace=maas_api_infra_namespace,
            subscription=phase_input_guarded_subscription,
            model_ref=guardrails_model_ref,
            expected_check_ids={input_check_id},
        )
        remaining_headers = wait_for_guardrail_header_catalog(
            admin_client=admin_client,
            policy_name=MAAS_GATEWAY_AUTH_POLICY_NAME,
            expected_headers={input_header},
        )
        assert remaining_selection["selectedGuardrailChecks"] == {input_check_id: True}
        assert remaining_headers == {input_header}


@pytest.mark.usefixtures("guardrails_maas_auth_policy")
@pytest.mark.ai_safety
class TestMaaSGuardrailAttachmentScopes:
    """Each supported guardrail attachment scope contributes to MaaS selection."""

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "attachment_scope",
        [
            pytest.param("aitenant", id="test_aitenant_guardrails"),
            pytest.param("tenant_config", id="test_maas_tenant_config_guardrails"),
            pytest.param("subscription", id="test_subscription_guardrails"),
            pytest.param("subscription_model", id="test_subscription_model_guardrails"),
            pytest.param("model_ref", id="test_model_ref_guardrails"),
        ],
    )
    def test_each_guardrail_attachment_scope_selects_one_check(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        maas_api_infra_namespace: str,
        guardrails_model_ref: MaaSModelRef,
        unguarded_subscription: MaaSSubscription,
        phase_ai_guardrail: AIGuardrail,
        phase_model_ai_guardrail: AIGuardrail,
        attachment_scope: str,
    ) -> None:
        """Given no guardrail attachment is active,
        when one supported attachment scope selects an input check,
        then MaaS returns one check and removes it after that scope is restored.
        """
        is_model_scope = attachment_scope == "model_ref"
        selected_policy = phase_model_ai_guardrail if is_model_scope else phase_ai_guardrail
        selected_check_id = guardrail_check_id(
            namespace=selected_policy.namespace,
            guardrail_name=selected_policy.name,
            check_name=INPUT_CHECK_NAME,
        )
        selected_header = guardrail_check_header(
            namespace=selected_policy.namespace,
            guardrail_name=selected_policy.name,
            check_name=INPUT_CHECK_NAME,
        )
        empty_selection = wait_for_selected_guardrail_checks(
            request_session_http=request_session_http,
            maas_api_infra_namespace=maas_api_infra_namespace,
            subscription=unguarded_subscription,
            model_ref=guardrails_model_ref,
            expected_check_ids=set(),
        )
        empty_headers = wait_for_guardrail_header_catalog(
            admin_client=admin_client,
            policy_name=MAAS_GATEWAY_AUTH_POLICY_NAME,
            expected_headers=set(),
        )
        assert empty_selection["selectedGuardrailChecks"] == {}
        assert empty_headers == set()

        attachment = [{"ref": {"name": selected_policy.name}, "checks": [INPUT_CHECK_NAME]}]
        aitenant = AITenant(
            client=admin_client,
            name=unguarded_subscription.namespace,
            namespace=AITENANT_INFRA_NAMESPACE,
            ensure_exists=True,
        )
        tenant_config = MaasTenantConfig(
            client=admin_client,
            name=DEFAULT_MAAS_TENANT_NAME,
            namespace=unguarded_subscription.namespace,
            ensure_exists=True,
        )

        if attachment_scope == "aitenant":
            patch_resource = aitenant
            patch = {"spec": {"guardrails": attachment}}
        elif attachment_scope == "tenant_config":
            patch_resource = tenant_config
            patch = {"spec": {"guardrails": attachment}}
        elif attachment_scope == "subscription":
            patch_resource = unguarded_subscription
            patch = {"spec": {"guardrails": attachment}}
        elif attachment_scope == "subscription_model":
            patch_resource = unguarded_subscription
            patch = {
                "spec": {
                    "modelRefs": [
                        {
                            "name": guardrails_model_ref.name,
                            "namespace": guardrails_model_ref.namespace,
                            "tokenRateLimits": [{"limit": 100000, "window": "1m"}],
                            "guardrails": attachment,
                        }
                    ]
                }
            }
        else:
            patch_resource = guardrails_model_ref
            patch = {"spec": {"guardrails": attachment}}

        with ResourceEditor(patches={patch_resource: patch}):
            selected = wait_for_selected_guardrail_checks(
                request_session_http=request_session_http,
                maas_api_infra_namespace=maas_api_infra_namespace,
                subscription=unguarded_subscription,
                model_ref=guardrails_model_ref,
                expected_check_ids={selected_check_id},
            )
            selected_headers = wait_for_guardrail_header_catalog(
                admin_client=admin_client,
                policy_name=MAAS_GATEWAY_AUTH_POLICY_NAME,
                expected_headers={selected_header},
            )
            assert selected["selectedGuardrailChecks"] == {selected_check_id: True}
            assert selected_headers == {selected_header}

        restored_selection = wait_for_selected_guardrail_checks(
            request_session_http=request_session_http,
            maas_api_infra_namespace=maas_api_infra_namespace,
            subscription=unguarded_subscription,
            model_ref=guardrails_model_ref,
            expected_check_ids=set(),
        )
        restored_headers = wait_for_guardrail_header_catalog(
            admin_client=admin_client,
            policy_name=MAAS_GATEWAY_AUTH_POLICY_NAME,
            expected_headers=set(),
        )
        assert restored_selection["selectedGuardrailChecks"] == {}
        assert restored_headers == set()


@pytest.mark.usefixtures("guardrails_maas_auth_policy")
@pytest.mark.ai_safety
class TestMaaSGuardrailSubscriptionIsolation:
    """MaaS selects guardrails from the subscription chosen for the caller identity."""

    @pytest.mark.tier1
    def test_same_model_selects_guardrails_by_subscription_identity(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        maas_api_infra_namespace: str,
        guardrails_model_ref: MaaSModelRef,
        phase_ai_guardrail: AIGuardrail,
        phase_input_guarded_subscription: MaaSSubscription,
        guardrails_maas_auth_policy: MaaSAuthPolicy,
    ) -> None:
        """Given two subscriptions for one model with different owner groups and checks,
        when MaaS selects by each caller identity without an explicit subscription,
        then each identity receives only its subscription's guardrail check.
        """
        safety_group = "guardrails-safety-users"
        input_check_id = guardrail_check_id(
            namespace=phase_ai_guardrail.namespace,
            guardrail_name=phase_ai_guardrail.name,
            check_name=INPUT_CHECK_NAME,
        )
        output_check_id = guardrail_check_id(
            namespace=phase_ai_guardrail.namespace,
            guardrail_name=phase_ai_guardrail.name,
            check_name=OUTPUT_CHECK_NAME,
        )
        safety_subscription_name = generate_random_name(prefix="e2e-guardrails-safety")
        with (
            ResourceEditor(
                patches={
                    guardrails_maas_auth_policy: {
                        "spec": {
                            "subjects": {
                                "groups": [
                                    {"name": "system:authenticated"},
                                    {"name": safety_group},
                                ]
                            }
                        }
                    }
                }
            ),
            guarded_maas_subscription(
                admin_client=admin_client,
                namespace=phase_input_guarded_subscription.namespace,
                name=safety_subscription_name,
                model_ref=guardrails_model_ref,
                guardrails=[
                    {
                        "ref": {"name": phase_ai_guardrail.name},
                        "checks": [OUTPUT_CHECK_NAME],
                    }
                ],
                owner_groups=[safety_group],
            ) as safety_subscription,
        ):
            default_selection = wait_for_selected_guardrail_checks_for_groups(
                request_session_http=request_session_http,
                maas_api_infra_namespace=maas_api_infra_namespace,
                subscription=phase_input_guarded_subscription,
                model_ref=guardrails_model_ref,
                groups=["system:authenticated"],
                username="guardrails-default-user",
                expected_check_ids={input_check_id},
            )
            safety_selection = wait_for_selected_guardrail_checks_for_groups(
                request_session_http=request_session_http,
                maas_api_infra_namespace=maas_api_infra_namespace,
                subscription=safety_subscription,
                model_ref=guardrails_model_ref,
                groups=[safety_group],
                username="guardrails-safety-user",
                expected_check_ids={output_check_id},
            )

        assert default_selection["selectedGuardrailChecks"] == {input_check_id: True}
        assert safety_selection["selectedGuardrailChecks"] == {output_check_id: True}


@pytest.mark.usefixtures("guardrails_maas_auth_policy")
@pytest.mark.ai_safety
class TestMaaSGuardrailModelIdentifiers:
    """MaaS resolves equivalent model identifiers to the same guardrail selection."""

    @pytest.mark.tier1
    def test_model_identifier_forms_select_the_same_guardrails(
        self,
        request_session_http: requests.Session,
        maas_api_infra_namespace: str,
        guardrails_model_ref: MaaSModelRef,
        phase_ai_guardrail: AIGuardrail,
        phase_input_guarded_subscription: MaaSSubscription,
    ) -> None:
        """Given one guarded model subscription,
        when the model namespace/name, bare name, and resolved alias are requested,
        then every request returns the same authorized model and guardrail check.
        """
        input_check_id = guardrail_check_id(
            namespace=phase_ai_guardrail.namespace,
            guardrail_name=phase_ai_guardrail.name,
            check_name=INPUT_CHECK_NAME,
        )
        canonical_model = f"{guardrails_model_ref.namespace}/{guardrails_model_ref.name}"
        model_alias = wait_for_model_ref_resolved_alias(maas_model_ref=guardrails_model_ref)
        requested_models = [canonical_model, guardrails_model_ref.name, model_alias]

        selections = [
            wait_for_selected_guardrail_checks_for_groups(
                request_session_http=request_session_http,
                maas_api_infra_namespace=maas_api_infra_namespace,
                subscription=phase_input_guarded_subscription,
                model_ref=guardrails_model_ref,
                groups=["system:authenticated"],
                username="guardrails-model-identifier-user",
                expected_check_ids={input_check_id},
                requested_model=requested_model,
            )
            for requested_model in requested_models
        ]

        for selection in selections:
            assert selection["accessAllowed"] is True
            assert selection["resolvedModel"] == canonical_model
            assert selection["selectedGuardrailChecks"] == {input_check_id: True}
