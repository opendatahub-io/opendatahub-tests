"""E2E: one external model shared across multiple tenant gateways.

P0 scenario 2:

- One shared model attached to two supported tenant gateways has correct
  routes (one HTTPRoute per attachment, in the model namespace) and
  per-gateway status.
- Requests verify that authorization and quota consumption remain isolated:
  a user authorized in one tenant cannot call the model through the other
  tenant's gateway (both tenants authorize the model, so the cross-tenant
  rejections prove key isolation rather than an absent grant), and
  exhausting tenant A's token budget does not consume tenant B's.
- A MaaSModelRef that does not name its tenant on a two-gateway model
  reports the ambiguity instead of guessing a gateway (see
  TestSharedModelAmbiguousModelRef).

Shares the tenant_external_model session tenants (see the package conftest's
tenant_external_model_tenants fixture); the module's user namespace, models
and tenant-scoped policies come from conftest fixtures and are cleaned up by
them.

The model is served by one MaaSModelRef per tenant (``spec.tenantRef``
pinned): per-gateway status makes the controller's tenant selection require
it for models attached to more than one gateway, and each tenant's
subscription references its own modelRef.

The route/status tests always run. The serving tests (auth and quota
isolation, real inference) additionally need two capabilities, each with its
own skip gate:

- the AGC's per-gateway ``status.gateways`` results: aggregate-status AGC
  builds write ``status.httpRouteName`` for single-attachment models only,
  so the MaaS stack cannot resolve the multi-gateway model's route and those
  tests skip there (see ``require_maas_route``);
- the client-facing ``spec.modelName`` carried as the model identity: the
  request path and the modelRefs' names disagree here by construction
  (per-tenant refs have distinct names), so path-routed requests only
  authorize once ``status.resolvedModelAlias`` carries the client-facing
  name and maas-api matches pairs against it (see
  ``require_client_name_alias``). Unknown-key rejection needs no model
  identity resolution and runs with per-gateway status alone.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

import pytest
import requests
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
    assert_returns_usage,
    assert_served_by_provider,
    gateway_status_entry,
    get_json_or_none,
    infer_external_chat,
    redact_sensitive,
    require_client_name_alias,
    wait_for_attachment_httproute,
    wait_shared_attachments_ready,
    wait_until,
)

QUOTA_EXHAUSTION_TIMEOUT = 90

# A MaaSAuthPolicy reporting Ready does not guarantee the derived AuthConfig is
# enforced on every gateway envoy yet; a foreign key briefly serving through
# the other tenant's gateway is that propagation window, not a steady-state
# isolation break, so the cross-gateway rejection is re-checked a bounded
# number of times before failing.
CROSS_GATEWAY_REJECTION_ATTEMPTS: int = 6
CROSS_GATEWAY_REJECTION_RETRY_INTERVAL: int = 5


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestSharedModelTwoGateways:
    """One model, two attachments: routes, status and serving on both."""

    @pytest.mark.tier2
    def test_one_httproute_per_attachment(
        self,
        admin_client: DynamicClient,
        tenant_external_model_tenants: dict[str, dict[str, str]],
        shared_model: dict[str, Any],
    ) -> None:
        """Each attachment has its own HTTPRoute in the model namespace."""
        model = shared_model["model"]
        routes_by_gateway: dict[str, list[dict[str, Any]]] = {}
        for case in tenant_external_model_tenants.values():
            routes = wait_for_attachment_httproute(
                admin_client=admin_client,
                route_namespace=model["namespace"],
                gateway_name=case["gateway_name"],
                model_name=model["name"],
            )
            assert routes, f"no HTTPRoute attached to {case['gateway_name']}"
            routes_by_gateway[case["gateway_name"]] = routes
        route_uids = {route["metadata"]["uid"] for routes in routes_by_gateway.values() for route in routes}
        assert len(route_uids) == 2, f"expected two distinct HTTPRoutes (one per gateway), got uids {route_uids}"

    @pytest.mark.tier2
    def test_per_gateway_status(
        self,
        admin_client: DynamicClient,
        tenant_external_model_tenants: dict[str, dict[str, str]],
        shared_model: dict[str, Any],
    ) -> None:
        """status.gateways carries an independent ready result per gateway."""
        model = shared_model["model"]
        cases = tuple(tenant_external_model_tenants.values())
        if gateway_status_entry(admin_client, model["name"], model["namespace"], cases[0]["gateway_name"]) is None:
            pytest.skip(
                "AGC reports aggregate model status only (no status.gateways entries); "
                "per-gateway readiness is covered by test_both_gateways_serve"
            )
        wait_shared_attachments_ready(admin_client=admin_client, cases=cases, model=model)

    @pytest.mark.tier2
    def test_both_gateways_serve(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        tenant_external_model_tenants: dict[str, dict[str, str]],
        shared_model: dict[str, Any],
        shared_serving: dict[str, Any],
    ) -> None:
        """Real inference succeeds through both gateways with the same provider
        binding and credential."""
        model = shared_model["model"]
        require_client_name_alias(
            admin_client=admin_client,
            model=model,
            case=tenant_external_model_tenants["tenant_a"],
        )
        for case_key, access_key in (("tenant_a", "access_a"), ("tenant_b", "access_b")):
            case = tenant_external_model_tenants[case_key]
            access = shared_serving[access_key]
            response = infer_external_chat(
                request_session_http=request_session_http,
                gateway_url=case["gateway_url"],
                model_namespace=model["namespace"],
                model_client_name=model["client_name"],
                api_key=access["api_key"],
                path_tag=model["path_tag"],
            )
            assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestTenantAuthIsolation:
    """Each tenant's key serves only through its own gateway.

    Both tenants authorize the model, so a cross-tenant rejection proves
    key isolation rather than an absent grant on the rejecting tenant's
    gateway."""

    @pytest.mark.tier2
    def test_tenant_keys_authorized_only_on_their_own_gateway(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        tenant_external_model_tenants: dict[str, dict[str, str]],
        isolated_keys: dict[str, Any],
    ) -> None:
        """Each tenant's key serves through its own gateway and is rejected
        by the other's.

        The own-gateway 200 is the positive control: the other tenant's
        stack is live and authorizing the model, so the cross-tenant 401/403
        is a key-isolation verdict, not a missing grant."""
        model = isolated_keys["model"]
        require_client_name_alias(
            admin_client=admin_client, model=model, case=tenant_external_model_tenants["tenant_a"]
        )
        for (case, other), api_key in (
            (
                (
                    tenant_external_model_tenants["tenant_a"],
                    tenant_external_model_tenants["tenant_b"],
                ),
                isolated_keys["key_a"],
            ),
            (
                (
                    tenant_external_model_tenants["tenant_b"],
                    tenant_external_model_tenants["tenant_a"],
                ),
                isolated_keys["key_b"],
            ),
        ):
            served = infer_external_chat(
                request_session_http=request_session_http,
                gateway_url=case["gateway_url"],
                model_namespace=model["namespace"],
                model_client_name=model["client_name"],
                api_key=api_key,
                path_tag=model["path_tag"],
            )
            assert_served_by_provider(response=served, path_tag=model["path_tag"], credential=model["credential"])
            rejected: requests.Response | None = None
            for attempt in range(1, CROSS_GATEWAY_REJECTION_ATTEMPTS + 1):
                rejected = infer_external_chat(
                    request_session_http=request_session_http,
                    gateway_url=other["gateway_url"],
                    model_namespace=model["namespace"],
                    model_client_name=model["client_name"],
                    api_key=api_key,
                    path_tag=model["path_tag"],
                    retries=1,
                )
                if rejected.status_code in (401, 403):
                    break
                # Retry while the other gateway's auth enforcement catches up
                # with the policy created moments ago by the class fixture.
                time.sleep(CROSS_GATEWAY_REJECTION_RETRY_INTERVAL)
            assert rejected is not None and rejected.status_code in (401, 403), (
                f"{case['tenant_label_name']}'s API key must be rejected by "
                f"{other['tenant_label_name']}'s gateway after "
                f"{attempt} attempts (got "
                f"{rejected.status_code if rejected is not None else 'no response'}: "
                f"{redact_sensitive(rejected.text) if rejected is not None else ''})"
            )

    @pytest.mark.tier2
    def test_unknown_key_rejected_on_both_gateways(
        self,
        request_session_http: requests.Session,
        tenant_external_model_tenants: dict[str, dict[str, str]],
        isolated_keys: dict[str, Any],
    ) -> None:
        """An unknown API key is rejected by both tenants' gateways."""
        model = isolated_keys["model"]
        for case in tenant_external_model_tenants.values():
            response = infer_external_chat(
                request_session_http=request_session_http,
                gateway_url=case["gateway_url"],
                model_namespace=model["namespace"],
                model_client_name=model["client_name"],
                api_key=f"sk-extmod-invalid-{uuid.uuid4().hex[:8]}",
                path_tag=model["path_tag"],
                retries=1,
            )
            assert response.status_code in (401, 403), (
                f"unknown API key must be rejected by {case['gateway_name']}, got "
                f"{response.status_code}: {redact_sensitive(response.text)}"
            )


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestSharedModelAmbiguousModelRef:
    """A MaaSModelRef that does not name its tenant must not guess one.

    A model attached to two gateways has no unambiguous route for a
    tenantRef-less modelRef: it must report the ambiguity (Failed phase, a
    message directing the user to ``spec.tenantRef``) instead of silently
    selecting one gateway, while the per-tenant modelRefs that do pin
    ``tenantRef`` stay Ready.
    """

    @pytest.mark.tier2
    def test_tenantless_model_ref_reports_ambiguity(
        self,
        admin_client: DynamicClient,
        ambiguous_model_ref: dict[str, Any],
    ) -> None:
        """The tenantRef-less modelRef fails with a message naming
        spec.tenantRef, and reports no endpoint."""
        model = ambiguous_model_ref["model"]

        def _check() -> dict[str, Any]:
            ref = get_json_or_none(
                client=admin_client,
                kind="maasmodelref",
                name=ambiguous_model_ref["ambiguous_ref"],
                namespace=model["namespace"],
            )
            assert ref is not None, "ambiguous MaaSModelRef not found"
            status = ref.get("status") or {}
            assert status.get("phase") == "Failed", (
                f"tenantRef-less modelRef for a two-gateway model must be Failed, "
                f"got phase={status.get('phase')!r}: {redact_sensitive(status)}"
            )
            blob = json.dumps(status)
            assert re.search(r"tenantref", blob, re.IGNORECASE), (
                f"failure must direct the user to spec.tenantRef: {redact_sensitive(status)}"
            )
            return status

        status = wait_until(
            check=_check,
            timeout=TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
            what=(
                f"MaaSModelRef {model['namespace']}/{ambiguous_model_ref['ambiguous_ref']} "
                "report the two-gateway ambiguity"
            ),
        )
        assert not status.get("endpoint"), f"ambiguous modelRef must not report an endpoint: {redact_sensitive(status)}"

    @pytest.mark.tier2
    def test_pinned_model_refs_stay_ready(
        self,
        admin_client: DynamicClient,
        tenant_external_model_tenants: dict[str, dict[str, str]],
        ambiguous_model_ref: dict[str, Any],
    ) -> None:
        """The per-tenant modelRefs pinning spec.tenantRef are unaffected by
        their ambiguous sibling: each resolves its own tenant's attachment
        (RuntimeReady, resolvedTenantRef and serving gateway set to its own
        tenant) with no ambiguity error.

        Full phase-Ready additionally needs a governance pairing and unique
        model identities (RHOAIENG-101609 — two per-tenant refs for one
        ExternalModel share a resolved alias today), which are separate
        gaps; this test proves the tenant selection the scenario is about."""
        model = ambiguous_model_ref["model"]
        cases = {case["tenant_label_name"]: case for case in tenant_external_model_tenants.values()}
        for tenant_label, ref_name in model["model_refs"].items():
            case = cases[tenant_label]

            def _check(ref_name: str = ref_name, case: dict[str, str] = case) -> dict[str, Any]:
                ref = get_json_or_none(
                    client=admin_client, kind="maasmodelref", name=ref_name, namespace=model["namespace"]
                )
                assert ref is not None, f"MaaSModelRef {model['namespace']}/{ref_name} not found"
                status = ref.get("status") or {}
                assert status.get("phase") != "Failed", (
                    f"pinned modelRef {ref_name} must not fail: {redact_sensitive(status)}"
                )
                conditions = {condition.get("type"): condition for condition in status.get("conditions") or []}
                runtime = conditions.get("RuntimeReady")
                assert runtime is not None and runtime.get("status") == "True", (
                    f"pinned modelRef {ref_name} runtime not ready: {redact_sensitive(status)}"
                )
                assert status.get("resolvedTenantRef") == case["tenant_label_name"], (
                    f"pinned modelRef {ref_name} resolved tenant "
                    f"{status.get('resolvedTenantRef')!r}, expected "
                    f"{case['tenant_label_name']!r}: {redact_sensitive(status)}"
                )
                assert status.get("httpRouteGatewayName") == case["gateway_name"], (
                    f"pinned modelRef {ref_name} resolved gateway "
                    f"{status.get('httpRouteGatewayName')!r}, expected "
                    f"{case['gateway_name']!r}: {redact_sensitive(status)}"
                )
                return status

            wait_until(
                check=_check,
                timeout=TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
                what=f"pinned MaaSModelRef {model['namespace']}/{ref_name} resolving tenant "
                f"{case['tenant_label_name']} (RuntimeReady, own gateway)",
            )


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestTenantQuotaIsolation:
    """Quota consumption is tracked per tenant: exhausting tenant A's token
    budget leaves tenant B's untouched.

    Token budgets are accounted from ``usage.total_tokens`` in inference
    responses, so this class's model is served by a dedicated in-cluster
    simulator (see ``usage_provider``) rather than the header-echo
    endpoint, whose responses carry no usage and could never exhaust a
    budget.
    """

    @pytest.mark.tier2
    def test_quota_exhaustion_isolated_per_tenant(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        tenant_external_model_tenants: dict[str, dict[str, str]],
        quota_model: dict[str, Any],
    ) -> None:
        """Exhausting tenant A's token budget leaves tenant B serving."""
        model = quota_model["model"]
        require_client_name_alias(
            admin_client=admin_client, model=model, case=tenant_external_model_tenants["tenant_a"]
        )
        provider_path = quota_model["usage"]["provider_path"]

        successes = 0
        limited = None
        deadline = time.time() + QUOTA_EXHAUSTION_TIMEOUT
        while time.time() < deadline:
            try:
                limited = infer_external_chat(
                    request_session_http=request_session_http,
                    gateway_url=tenant_external_model_tenants["tenant_a"]["gateway_url"],
                    model_namespace=model["namespace"],
                    model_client_name=model["client_name"],
                    api_key=quota_model["access_a"]["api_key"],
                    path_tag=model["path_tag"],
                    provider_path=provider_path,
                    retries=0,
                )
            except requests.RequestException:
                time.sleep(1.0)
                continue
            if limited.status_code == 200:
                successes += 1
            elif limited.status_code == 429:
                if successes:
                    break
            elif limited.status_code >= 500:
                time.sleep(1.0)
                continue
            else:
                raise AssertionError(
                    "unexpected inference response on tenant A: "
                    f"{limited.status_code}: {redact_sensitive(limited.text)}"
                )
            time.sleep(1.0)
        assert limited is not None and limited.status_code == 429, (
            f"tenant A did not hit its token budget after {successes} successes: "
            f"{redact_sensitive(limited.text if limited is not None else 'no response')}"
        )
        assert successes > 0, "tenant A hit 429 before any successful inference (limits not live?)"

        response = infer_external_chat(
            request_session_http=request_session_http,
            gateway_url=tenant_external_model_tenants["tenant_b"]["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=quota_model["access_b"]["api_key"],
            path_tag=model["path_tag"],
            provider_path=provider_path,
        )
        assert_returns_usage(response=response)

        again = infer_external_chat(
            request_session_http=request_session_http,
            gateway_url=tenant_external_model_tenants["tenant_b"]["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=quota_model["access_b"]["api_key"],
            path_tag=model["path_tag"],
            provider_path=provider_path,
        )
        assert_returns_usage(response=again)
