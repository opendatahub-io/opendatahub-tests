"""E2E: external models outside tenant namespaces, served by the default tenant.

Ported from models-as-a-service
``test/e2e/tests/test_tenant_external_model_default_tenant.py``
(RHOAIENG-101754, 3.5 → 3.6 compatibility). In 3.5 the default tenant served
ExternalModels in every namespace. With ai-gateway-controller (AGC) on the
Praxis dataplane, an ExternalModel without gatewayRefs in a namespace that has
no MaasTenantConfig and that no AITenant claims is served by the default
tenant (AITenant models-as-a-service) as an implicit attachment to its
Gateway: the same hashed HTTPRoute, DestinationRule and credential copy as an
explicit gatewayRefs attachment, with spec unchanged.

The models live in the llm namespace: the namespace 3.5 demos used, and one
the default Gateway's allowedRoutes already admits (a fresh user namespace
would need the default Gateway patched). The stack is 3.5-shaped: the provider
uses ``auth.type: simple``, which the AGC serves as ``apikey``. The provider
endpoint is a header-echo service (default httpbingo.org): under
``/anything/<tag>`` it echoes the request, so a response proves which binding
served it and which credential was injected.

Scenarios:
- The model and its provider reach Ready and serve inference through the
  default Gateway, with the credential delivered as a copy into the default
  tenant namespace.
- gatewayRefs replace the implicit attachment and remove its objects, without
  falling back to the default Gateway; removing them restores it.
- Deleting the model removes the implicit attachment's objects.

Skips unless the default tenant runs praxis-extproc (installed by the AGC): a
default tenant on IPP serves these models itself and the AGC leaves them
alone.
"""

from __future__ import annotations

import re
import time
from contextlib import ExitStack
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    READY_TIMEOUT,
    default_tenant_model_stack,
    delete_external_model_and_ref,
    gateway_ref,
    gateway_status_entry,
    get_json_or_none,
    infer_chat_completion,
    namespace_name,
    patch_external_model_gateway_refs,
    redact_sensitive,
    serve_in_default_tenant,
    wait_for_gateway_attachment_reason,
    wait_for_provider_ready,
    wait_implicitly_attached,
    wait_not_implicitly_attached,
)
from utilities.constants import MAAS_GATEWAY_NAME, MAAS_GATEWAY_NAMESPACE
from utilities.general import generate_random_name

LOGGER = structlog.get_logger(name=__name__)


UNMANAGED_GATEWAY_NAME = "e2e-dft-unmanaged-gateway"
# Replacing gatewayRefs tears down the implicit attachment's objects and, with
# them, the shared default-tenant ExtProc Deployment; the reinstall can lag the
# teardown by minutes (observed ~5 minutes, tracked against AGC), so restores
# after a refs transition get a wider window than a first-time attachment.
REFS_TRANSITION_RESTORE_TIMEOUT: int = 600
# While one model's gatewayRefs transition is in flight, poll the other model
# this long to catch a degradation window instead of sampling a single moment.
REFS_TRANSITION_PROBE_INTERVAL: int = 15
REFS_TRANSITION_PROBE_DURATION: int = 360


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
    "_require_praxis_default_tenant",
)
class TestDefaultTenantServesModelWithoutGatewayRefs:
    """A 3.5-shaped model in a namespace of no tenant is served through the
    default tenant's Gateway."""

    @pytest.mark.tier2
    def test_attached_to_default_gateway(
        self,
        admin_client: DynamicClient,
        implicit_model: dict[str, Any],
    ) -> None:
        """Given a model without gatewayRefs in a tenant-less namespace, when the AGC reconciles it,
        then the single result is for the default Gateway, names an attachment route on it,
        and the spec still names no Gateway.
        """
        model = implicit_model["model"]
        route = implicit_model["route"]
        route_name = route["metadata"]["name"]
        assert re.fullmatch(rf"external-model-{re.escape(model['name'])}-[0-9a-f]{{12}}", route_name), (
            f"{route_name} is not an attachment route name (external-model-<model>-<hash>)"
        )
        parents = [(parent["namespace"], parent["name"]) for parent in route["spec"]["parentRefs"]]
        assert parents == [(MAAS_GATEWAY_NAMESPACE, MAAS_GATEWAY_NAME)], f"unexpected route parentRefs: {parents}"
        external_model = get_json_or_none(
            client=admin_client, kind="externalmodel", name=model["name"], namespace=model["namespace"]
        )
        assert external_model is not None, f"ExternalModel {model['namespace']}/{model['name']} not found"
        spec = external_model["spec"]
        assert not spec.get("gatewayRefs"), f"the implicit attachment must not mutate spec: {spec}"
        status = external_model["status"]
        results = [(entry["namespace"], entry["name"]) for entry in status["gateways"]]
        assert results == [(MAAS_GATEWAY_NAMESPACE, MAAS_GATEWAY_NAME)], f"unexpected status.gateways: {results}"
        assert status["httpRouteName"] == route_name, (
            f"status.httpRouteName {status['httpRouteName']!r} must name the single attachment "
            f"route {route_name} (maas-controller resolves the model's route from it)"
        )

    @pytest.mark.tier2
    def test_provider_reports_ready(
        self,
        admin_client: DynamicClient,
        implicit_model: dict[str, Any],
    ) -> None:
        """Given a provider in a namespace with no MaasTenantConfig, when its status is read,
        then it reports its own validated Ready configuration instead of Pending.
        """
        model = implicit_model["model"]

        wait_for_provider_ready(
            admin_client=admin_client,
            provider_name=model["provider"],
            provider_namespace=model["namespace"],
            timeout=READY_TIMEOUT,
        )

    @pytest.mark.tier2
    def test_served_through_default_gateway(
        self,
        request_session_http: requests.Session,
        default_gateway_url: str,
        implicit_model: dict[str, Any],
    ) -> None:
        """Given the implicitly attached model, when inference runs through the default Gateway,
        then the provider serves it with the model's credential injected (``simple`` auth served as
        ``apikey``) and without the client's MaaS API key.
        """
        model = implicit_model["model"]
        response = infer_chat_completion(
            request_session_http=request_session_http,
            gateway_url=default_gateway_url,
            model=model,
            api_key=implicit_model["api_key"],
        )
        assert response.status_code == 200, (
            f"expected 200 through {MAAS_GATEWAY_NAME}, got {response.status_code}: {redact_sensitive(response.text)}"
        )
        assert model["path_tag"] in response.text, (
            f"response did not come from the model's provider: {redact_sensitive(response.text)}"
        )
        assert model["credential"] in response.text, "the model's credential did not reach the provider"
        # Compare before asserting so pytest's failure introspection cannot
        # render the client's MaaS API key as part of the operands.
        client_key_forwarded = implicit_model["api_key"] in response.text
        assert not client_key_forwarded, "the client's MaaS API key was forwarded to the provider"

    @pytest.mark.tier3
    def test_unknown_key_rejected(
        self,
        request_session_http: requests.Session,
        default_gateway_url: str,
        implicit_model: dict[str, Any],
    ) -> None:
        """Given an unknown API key, when inference runs through the default Gateway,
        then the request is rejected with 401/403.
        """
        response = infer_chat_completion(
            request_session_http=request_session_http,
            gateway_url=default_gateway_url,
            model=implicit_model["model"],
            api_key=f"sk-extmod-invalid-{generate_random_name()}",
            retries=1,
        )
        assert response.status_code in (401, 403), (
            f"unknown API key must be rejected, got {response.status_code}: {redact_sensitive(response.text)}"
        )


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
    "_require_praxis_default_tenant",
)
class TestImplicitAttachmentLifecycle:
    """gatewayRefs replace the implicit attachment and removing them restores
    it; deleting the model removes its objects."""

    @pytest.mark.tier3
    def test_gateway_refs_replace_implicit_attachment(
        self,
        admin_client: DynamicClient,
        default_tenant_model_namespace: Namespace,
        teardown_resources: bool,
    ) -> None:
        """Given an implicitly attached model, when gatewayRefs bind it to an unmanaged Gateway,
        then the binding is refused with GatewayNotManaged and the implicit attachment is removed
        without falling back to the default Gateway; removing gatewayRefs restores it.
        """
        with default_tenant_model_stack(
            admin_client,
            prefix="e2e-dft-refs",
            namespace=namespace_name(default_tenant_model_namespace),
            teardown=teardown_resources,
        ) as model:
            route_name = wait_implicitly_attached(admin_client=admin_client, model=model)["route"]["metadata"]["name"]

            patch_external_model_gateway_refs(
                admin_client=admin_client,
                name=model["name"],
                namespace=model["namespace"],
                refs=[gateway_ref(UNMANAGED_GATEWAY_NAME)],
            )
            wait_for_gateway_attachment_reason(
                admin_client=admin_client,
                model_name=model["name"],
                model_namespace=model["namespace"],
                gateway_name=UNMANAGED_GATEWAY_NAME,
                reason="GatewayNotManaged",
            )
            wait_not_implicitly_attached(admin_client=admin_client, model=model, route_name=route_name)

            patch_external_model_gateway_refs(
                admin_client=admin_client, name=model["name"], namespace=model["namespace"], refs=None
            )
            wait_implicitly_attached(admin_client=admin_client, model=model, timeout=REFS_TRANSITION_RESTORE_TIMEOUT)
            assert (
                gateway_status_entry(admin_client, model["name"], model["namespace"], UNMANAGED_GATEWAY_NAME) is None
            ), f"removing gatewayRefs must drop the {UNMANAGED_GATEWAY_NAME} result"

    @pytest.mark.tier3
    def test_gateway_refs_replace_does_not_degrade_other_models(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        base_url: str,
        default_gateway_url: str,
        default_tenant_model_namespace: Namespace,
        teardown_resources: bool,
    ) -> None:
        """Given two implicitly attached models sharing the default tenant's ExtProc
        Deployment, when one model's gatewayRefs are replaced and then removed, then the
        other model serves without interruption throughout and the edited model serves
        again once its implicit attachment is restored.
        """
        with ExitStack() as stack:
            namespace = namespace_name(namespace=default_tenant_model_namespace)
            model_a = stack.enter_context(
                cm=default_tenant_model_stack(
                    admin_client, prefix="e2e-dft-refs2", namespace=namespace, teardown=teardown_resources
                )
            )
            model_b = stack.enter_context(
                cm=default_tenant_model_stack(
                    admin_client, prefix="e2e-dft-coex", namespace=namespace, teardown=teardown_resources
                )
            )
            route_name_a = wait_implicitly_attached(admin_client=admin_client, model=model_a)["route"]["metadata"][
                "name"
            ]
            wait_implicitly_attached(admin_client=admin_client, model=model_b)
            api_key_a = stack.enter_context(
                cm=serve_in_default_tenant(
                    admin_client,
                    request_session_http,
                    ocp_token_for_actor,
                    base_url,
                    model_a,
                    teardown=teardown_resources,
                )
            )
            api_key_b = stack.enter_context(
                cm=serve_in_default_tenant(
                    admin_client,
                    request_session_http,
                    ocp_token_for_actor,
                    base_url,
                    model_b,
                    teardown=teardown_resources,
                )
            )

            def infer_and_expect_provider_served(model: dict[str, Any], api_key: str) -> None:
                response = infer_chat_completion(
                    request_session_http=request_session_http,
                    gateway_url=default_gateway_url,
                    model=model,
                    api_key=api_key,
                )
                assert response.status_code == 200, (
                    f"expected 200 for {model['name']}, got {response.status_code}: {redact_sensitive(response.text)}"
                )
                assert model["path_tag"] in response.text, (
                    f"response for {model['name']} did not come from its provider: {redact_sensitive(response.text)}"
                )

            infer_and_expect_provider_served(model=model_b, api_key=api_key_b)

            patch_external_model_gateway_refs(
                admin_client=admin_client,
                name=model_a["name"],
                namespace=model_a["namespace"],
                refs=[gateway_ref(UNMANAGED_GATEWAY_NAME)],
            )
            wait_for_gateway_attachment_reason(
                admin_client=admin_client,
                model_name=model_a["name"],
                model_namespace=model_a["namespace"],
                gateway_name=UNMANAGED_GATEWAY_NAME,
                reason="GatewayNotManaged",
            )
            wait_not_implicitly_attached(admin_client=admin_client, model=model_a, route_name=route_name_a)
            patch_external_model_gateway_refs(
                admin_client=admin_client, name=model_a["name"], namespace=model_a["namespace"], refs=None
            )
            # The two models share one default-tenant ExtProc Deployment, so the
            # teardown of model_a's implicit attachment can in principle take
            # model_b's serving path down with it. Poll across the whole
            # transition window instead of sampling one moment; keep the full
            # status history so a failure shows whether the degradation was a
            # blip or a sustained outage.
            probe_history: list[str] = []
            # Single-shot requests: the loop owns the cadence, so a failed
            # probe must surface here rather than be retried inside the
            # request helper.
            probe_started = time.monotonic()
            while (probe_elapsed := time.monotonic() - probe_started) < REFS_TRANSITION_PROBE_DURATION:
                response = infer_chat_completion(
                    request_session_http=request_session_http,
                    gateway_url=default_gateway_url,
                    model=model_b,
                    api_key=api_key_b,
                    retries=0,
                )
                # Monotonic elapsed labels include request and scheduling time.
                probe_history.append(f"{probe_elapsed:.0f}s:{response.status_code}")
                time.sleep(REFS_TRANSITION_PROBE_INTERVAL)
                assert response.status_code == 200, (
                    f"{model_b['name']} degraded while {model_a['name']} transitioned "
                    f"gatewayRefs; probe history [{', '.join(probe_history)}]: "
                    f"{redact_sensitive(response.text)}"
                )

            wait_implicitly_attached(admin_client=admin_client, model=model_a, timeout=REFS_TRANSITION_RESTORE_TIMEOUT)
            infer_and_expect_provider_served(model=model_a, api_key=api_key_a)
            infer_and_expect_provider_served(model=model_b, api_key=api_key_b)

    @pytest.mark.tier3
    def test_model_delete_removes_implicit_attachment(
        self,
        admin_client: DynamicClient,
        default_tenant_model_namespace: Namespace,
        teardown_resources: bool,
    ) -> None:
        """Given an implicitly attached model, when the model and its MaaSModelRef are deleted,
        then the implicit attachment and every object it published are removed.
        """
        with default_tenant_model_stack(
            admin_client,
            prefix="e2e-dft-del",
            namespace=namespace_name(default_tenant_model_namespace),
            teardown=teardown_resources,
        ) as model:
            route_name = wait_implicitly_attached(admin_client=admin_client, model=model)["route"]["metadata"]["name"]

            delete_external_model_and_ref(
                admin_client=admin_client, name=model["name"], namespace=model["namespace"], timeout=READY_TIMEOUT
            )
            wait_not_implicitly_attached(admin_client=admin_client, model=model, route_name=route_name)
