"""Tests for AIGuardrail NeMo provider endpoint discovery and authenticated NeMo providers.

ai-gateway-controller never reads NemoGuardrails status.endpoint. It derives the provider's
in-cluster Service from status.workloadNamespace, which must be an AI Gateway infrastructure
namespace, and picks https when the provider carries the enable-auth annotation.
For an authenticated provider the tenant compiler sends the post-auth pod's ServiceAccount
token over TLS and grants that ServiceAccount get on the NeMo Service for kube-rbac-proxy.
"""

from urllib.parse import urlparse

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding

from tests.ai_gateway.models_as_a_service.guardrails.constants import (
    AI_GATEWAY_INFRA_NAMESPACES,
    AI_GUARDRAIL_TIMEOUT,
    CLEAN_PROMPT,
    INPUT_CHECK_NAME,
    INPUT_RAIL_FLOW,
    NEMO_AUTH_ANNOTATION,
    NEMO_AUTH_VOLUME_NAME,
    NEMO_PII_CONFIG_ID,
    PII_EMAIL,
    PII_PROMPT,
    PROVIDER_AVAILABLE_REASON,
    PROVIDER_READY_CONDITION,
)
from tests.ai_gateway.models_as_a_service.guardrails.utils import (
    ai_guardrail_condition,
    chat_completion_payload,
    nemo_service_url,
    verify_guardrail_callouts_authenticated,
    wait_for_ai_guardrail_bound,
    wait_for_ai_guardrail_provider_endpoint,
    wait_for_chat_response,
    wait_for_nemo_guardrails_ready,
)
from tests.ai_gateway.models_as_a_service.praxis.constants import LEGACY_IPP_POST_PROCESSING_NAME_BASE
from tests.ai_gateway.models_as_a_service.utils import build_maas_headers
from utilities.constants import MAAS_GATEWAY_NAMESPACE
from utilities.resources.ai_guardrail import AIGuardrail
from utilities.resources.llm_inference_service import LLMInferenceService
from utilities.resources.nemo_guardrails import NemoGuardrails


@pytest.mark.ai_safety
class TestMaaSGuardrailProviderEndpointDiscovery:
    """AIGuardrail derives its NeMo provider endpoint from the infra namespace workload and the auth setting."""

    @pytest.mark.tier1
    def test_unauthenticated_provider_endpoint_is_derived(
        self,
        maas_api_infra_namespace: str,
        guardrails_nemo_server: NemoGuardrails,
        pii_ai_guardrail: AIGuardrail,
    ) -> None:
        """Given a bound AIGuardrail whose NeMo provider has auth disabled,
        when ai-gateway-controller discovers the provider endpoint,
        then status.providerEndpoint is the http Service in the AI Gateway infra namespace and ProviderReady is True.
        """
        assert maas_api_infra_namespace in AI_GATEWAY_INFRA_NAMESPACES, (
            f"NeMo workload namespace {maas_api_infra_namespace!r} is not an AI Gateway infra namespace "
            f"{AI_GATEWAY_INFRA_NAMESPACES}"
        )

        expected_endpoint = nemo_service_url(
            nemo_guardrails=guardrails_nemo_server,
            workload_namespace=maas_api_infra_namespace,
            authenticated=False,
        )
        provider_endpoint = pii_ai_guardrail.instance.status.providerEndpoint
        assert provider_endpoint == expected_endpoint, (
            f"Expected providerEndpoint {expected_endpoint!r}, got: {provider_endpoint!r}"
        )
        provider_ready = ai_guardrail_condition(ai_guardrail=pii_ai_guardrail, condition_type=PROVIDER_READY_CONDITION)
        assert provider_ready.get("status") == "True", f"Expected ProviderReady True, got: {provider_ready}"
        assert provider_ready.get("reason") == PROVIDER_AVAILABLE_REASON, (
            f"Expected ProviderReady reason {PROVIDER_AVAILABLE_REASON!r}, got: {provider_ready}"
        )

    @pytest.mark.tier1
    def test_authenticated_provider_endpoint_is_https(
        self,
        maas_api_infra_namespace: str,
        auth_nemo_server: NemoGuardrails,
        auth_ai_guardrail: AIGuardrail,
    ) -> None:
        """Given a bound AIGuardrail whose NeMo provider has auth enabled,
        when ai-gateway-controller discovers the provider endpoint,
        then status.providerEndpoint is the https kube-rbac-proxy Service in the infra namespace,
        matches the endpoint TrustyAI publishes, and ProviderReady is True.
        """
        expected_endpoint = nemo_service_url(
            nemo_guardrails=auth_nemo_server,
            workload_namespace=maas_api_infra_namespace,
            authenticated=True,
        )
        provider_endpoint = auth_ai_guardrail.instance.status.providerEndpoint
        assert provider_endpoint == expected_endpoint, (
            f"Expected providerEndpoint {expected_endpoint!r}, got: {provider_endpoint!r}"
        )
        nemo_endpoint = auth_nemo_server.instance.status.endpoint
        assert nemo_endpoint == expected_endpoint, (
            f"Expected NemoGuardrails status.endpoint {expected_endpoint!r}, got: {nemo_endpoint!r}"
        )
        provider_ready = ai_guardrail_condition(ai_guardrail=auth_ai_guardrail, condition_type=PROVIDER_READY_CONDITION)
        assert provider_ready.get("status") == "True", f"Expected ProviderReady True, got: {provider_ready}"
        assert provider_ready.get("reason") == PROVIDER_AVAILABLE_REASON, (
            f"Expected ProviderReady reason {PROVIDER_AVAILABLE_REASON!r}, got: {provider_ready}"
        )

    @pytest.mark.tier2
    def test_enabling_auth_switches_provider_endpoint_to_https(
        self,
        admin_client: DynamicClient,
        guardrails_resource_suffix: str,
        maas_api_infra_namespace: str,
        maas_subscription_namespace: Namespace,
        guardrails_nemo_configmap: ConfigMap,
    ) -> None:
        """Given a bound AIGuardrail whose NeMo provider has auth disabled,
        when the enable-auth annotation is set to "true" on the provider,
        then status.providerEndpoint switches from http to https and the binding revision changes.
        """
        with (
            NemoGuardrails(
                client=admin_client,
                name=f"maas-nemo-toggle-{guardrails_resource_suffix}",
                namespace=maas_subscription_namespace.name,
                annotations={NEMO_AUTH_ANNOTATION: "false"},
                nemo_configs=[
                    {"name": NEMO_PII_CONFIG_ID, "configMaps": [guardrails_nemo_configmap.name], "default": True}
                ],
                replicas=1,
                expose_route=False,
                teardown=True,
            ) as nemo_server,
            AIGuardrail(
                client=admin_client,
                name=f"maas-toggle-guardrail-{guardrails_resource_suffix}",
                namespace=maas_subscription_namespace.name,
                provider={"nemo": {"ref": {"name": nemo_server.name}}, "timeout": AI_GUARDRAIL_TIMEOUT},
                checks=[{"name": INPUT_CHECK_NAME, "configId": NEMO_PII_CONFIG_ID, "phases": ["Input"]}],
                teardown=True,
            ) as ai_guardrail,
        ):
            wait_for_nemo_guardrails_ready(nemo_guardrails=nemo_server)
            wait_for_ai_guardrail_bound(ai_guardrail=ai_guardrail)
            wait_for_ai_guardrail_provider_endpoint(
                ai_guardrail=ai_guardrail,
                expected_endpoint=nemo_service_url(
                    nemo_guardrails=nemo_server, workload_namespace=maas_api_infra_namespace, authenticated=False
                ),
            )
            http_binding_revision = ai_guardrail.instance.status.bindingRevision

            nemo_server.update(
                resource_dict={"metadata": {"name": nemo_server.name, "annotations": {NEMO_AUTH_ANNOTATION: "true"}}}
            )

            wait_for_ai_guardrail_provider_endpoint(
                ai_guardrail=ai_guardrail,
                expected_endpoint=nemo_service_url(
                    nemo_guardrails=nemo_server, workload_namespace=maas_api_infra_namespace, authenticated=True
                ),
            )
            wait_for_ai_guardrail_bound(ai_guardrail=ai_guardrail)
            https_binding_revision = ai_guardrail.instance.status.bindingRevision
            assert https_binding_revision != http_binding_revision, (
                f"Expected bindingRevision to change after enabling auth, still {https_binding_revision!r}"
            )


@pytest.mark.usefixtures("guardrails_maas_auth_policy", "auth_guardrail_checks_enforced")
@pytest.mark.ai_safety
class TestMaaSGuardrailsAuthenticatedProvider:
    """Guardrail checks served by a NeMo provider behind kube-rbac-proxy are called with a token and enforced."""

    @pytest.mark.tier1
    def test_authenticated_callout_carries_caller_token_and_service_ca(
        self,
        admin_client: DynamicClient,
        auth_ai_guardrail: AIGuardrail,
        auth_input_check_header: str,
    ) -> None:
        """Given a subscription attaching a check from an authenticated NeMo provider,
        when the check is compiled into the post-auth Praxis ExtProc,
        then the callout targets the https providerEndpoint with the caller token file and the service CA.
        """
        verify_guardrail_callouts_authenticated(
            admin_client=admin_client,
            check_headers=[auth_input_check_header],
            expected_endpoint=auth_ai_guardrail.instance.status.providerEndpoint,
        )

    @pytest.mark.tier1
    def test_post_auth_caller_granted_access_to_nemo_service(
        self,
        admin_client: DynamicClient,
        maas_api_infra_namespace: str,
        auth_ai_guardrail: AIGuardrail,
    ) -> None:
        """Given a compiled check from an authenticated NeMo provider,
        when kube-rbac-proxy authorizes the post-auth pod's token,
        then the pod mounts the caller token volume and its ServiceAccount is granted only get on that NeMo Service.
        """
        post_auth_deployment = Deployment(
            client=admin_client,
            name=LEGACY_IPP_POST_PROCESSING_NAME_BASE,
            namespace=MAAS_GATEWAY_NAMESPACE,
            ensure_exists=True,
        )
        pod_spec = post_auth_deployment.instance.spec.template.spec
        volume_names = {volume.name for volume in pod_spec.volumes or []}
        assert NEMO_AUTH_VOLUME_NAME in volume_names, (
            f"Expected volume {NEMO_AUTH_VOLUME_NAME!r} on {post_auth_deployment.name}, got: {sorted(volume_names)}"
        )

        service_account = pod_spec.serviceAccountName
        service_name = urlparse(auth_ai_guardrail.instance.status.providerEndpoint).hostname.split(".")[0]
        caller_name = f"nemo-caller.{MAAS_GATEWAY_NAMESPACE}.{service_account}.{service_name}"

        role = Role(client=admin_client, name=caller_name, namespace=maas_api_infra_namespace)
        assert role.exists, f"Expected Role {caller_name!r} in {maas_api_infra_namespace!r}"
        rules = [
            {
                "apiGroups": list(rule.apiGroups or []),
                "resources": list(rule.resources or []),
                "resourceNames": list(rule.resourceNames or []),
                "verbs": list(rule.verbs or []),
            }
            for rule in role.instance.rules
        ]
        expected_rules = [
            {"apiGroups": [""], "resources": ["services"], "resourceNames": [service_name], "verbs": ["get"]}
        ]
        assert rules == expected_rules, f"Expected Role rules {expected_rules}, got: {rules}"

        role_binding = RoleBinding(client=admin_client, name=caller_name, namespace=maas_api_infra_namespace)
        assert role_binding.exists, f"Expected RoleBinding {caller_name!r} in {maas_api_infra_namespace!r}"
        subjects = [(subject.kind, subject.name, subject.namespace) for subject in role_binding.instance.subjects]
        assert subjects == [("ServiceAccount", service_account, MAAS_GATEWAY_NAMESPACE)], (
            f"Expected RoleBinding subject {service_account!r} in {MAAS_GATEWAY_NAMESPACE!r}, got: {subjects}"
        )
        assert role_binding.instance.roleRef.name == caller_name, (
            f"Expected RoleBinding to reference Role {caller_name!r}, got: {role_binding.instance.roleRef}"
        )

    @pytest.mark.tier1
    def test_pii_prompt_blocked_by_authenticated_input_guardrail(
        self,
        request_session_http: requests.Session,
        guardrails_model_url: str,
        guardrails_echo_llmisvc: LLMInferenceService,
        auth_input_guarded_api_key: str,
    ) -> None:
        """Given a subscription with an input-phase PII check served by an authenticated NeMo provider,
        when a prompt containing an e-mail address is sent through MaaS,
        then NeMo evaluates the authenticated call and the gateway rejects the request with 403 naming the rail.
        """
        response = wait_for_chat_response(
            request_session_http=request_session_http,
            model_url=guardrails_model_url,
            headers=build_maas_headers(token=auth_input_guarded_api_key),
            payload=chat_completion_payload(model=guardrails_echo_llmisvc.name, prompt=PII_PROMPT),
            is_expected=lambda resp: resp.status_code == 403,
            description="PII prompt blocked by the authenticated input guardrail",
        )
        if INPUT_RAIL_FLOW not in response.text:
            pytest.fail("Expected input blocking rail in 403 response; NeMo may have rejected the caller token")
        if PII_EMAIL in response.text:
            pytest.fail("Blocked response must not echo PII")

    @pytest.mark.tier1
    def test_clean_prompt_passes_authenticated_input_guardrail(
        self,
        request_session_http: requests.Session,
        guardrails_model_url: str,
        guardrails_echo_llmisvc: LLMInferenceService,
        auth_input_guarded_api_key: str,
    ) -> None:
        """Given a subscription whose authenticated input-phase PII check is confirmed active,
        when a prompt without PII is sent through MaaS,
        then kube-rbac-proxy accepts the caller token, NeMo allows the prompt, and the model answers.
        """
        headers = build_maas_headers(token=auth_input_guarded_api_key)
        # A 200 before the check selection header propagates would pass without calling NeMo at all.
        wait_for_chat_response(
            request_session_http=request_session_http,
            model_url=guardrails_model_url,
            headers=headers,
            payload=chat_completion_payload(model=guardrails_echo_llmisvc.name, prompt=PII_PROMPT),
            is_expected=lambda resp: resp.status_code == 403 and INPUT_RAIL_FLOW in resp.text,
            description="authenticated input guardrail active for the subscription",
        )
        response = request_session_http.post(
            url=guardrails_model_url,
            headers=headers,
            json=chat_completion_payload(model=guardrails_echo_llmisvc.name, prompt=CLEAN_PROMPT),
            timeout=60,
        )
        assert response.status_code == 200, (
            f"Clean prompt rejected by the authenticated input guardrail: status={response.status_code}"
        )
        content = response.json()["choices"][0]["message"]["content"]
        if CLEAN_PROMPT not in content:
            pytest.fail("Expected echoed prompt in completion")
