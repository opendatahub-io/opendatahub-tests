"""E2E: compatibility for legacy-shaped external models and legacy IPP.

Ported from models-as-a-service ``test/e2e/tests/test_tenant_external_model_compat.py``
(RHOAIENG-99761 P0 scenarios; runtime halves — upgrade-context preservation is
tracked by the separate upgrade-tests task):

4. Objects without gatewayRefs follow the recorded compatibility behavior:
   tenant-local resolution keeps the legacy resource contract, and adding
   gatewayRefs that name only the model's own tenant recreates nothing.
5. Legacy IPP behavior remains compatible during legacy/Praxis coexistence:
   with the default tenant explicitly on ipp, user-namespace/shared AGC
   models are ignored by the legacy IPP controller (no route on the default
   gateway, no IPP status writes, shared provider ownership stays defined),
   and the MaasTenantConfig backend selection / cleanup-claim handoff works
   in both directions.

The coexistence test is serial (it flips the default tenant's payload backend)
and gated on TENANT_EXTERNAL_MODEL_LEGACY_IPP_E2E=true because it requires the
MaaS IPP image pin containing the legacy-controller exclusion guard.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    assert_served_by_provider,
    attachment_httproutes,
    attachment_status_snapshot,
    destination_rules_referencing,
    external_model_route_name,
    gateway_ref,
    get_json_or_none,
    infer_external_chat,
    infer_for_shared_model,
    patch_external_model_gateway_refs,
    provider_external_name_service,
    wait_for_gateway_attachment_ready,
    wait_for_model_generation_observed,
)
from utilities.constants import MAAS_GATEWAY_NAME, MAAS_GATEWAY_NAMESPACE

LOGGER = structlog.get_logger(name=__name__)


RECONCILE_WAIT = int(os.environ.get("E2E_RECONCILE_WAIT", "8"))
SETTLE_POLL_INTERVAL = 5


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestOmittedGatewayRefsUnchanged:
    """A legacy-shaped model (no gatewayRefs) in the tenant namespace keeps
    the legacy resource contract, and adding own-tenant gatewayRefs recreates
    nothing."""

    @pytest.mark.tier2
    def test_omitted_gateway_refs_serve_via_tenant_local_resolution(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        tenant_external_model_tenants: dict[str, dict[str, str]],
        legacy_shaped_model: dict[str, Any],
    ) -> None:
        """Given a legacy-shaped (no gatewayRefs) model in the tenant namespace, when it serves inference,
        then the AGC's tenant-local route (external-model-<name>) carries it and the provider serves the request.
        """
        tenant_a = tenant_external_model_tenants["tenant_a"]
        model = legacy_shaped_model["model"]
        route_names = [route["metadata"]["name"] for route in legacy_shaped_model["routes"]]
        expected_route = external_model_route_name(model_name=model["name"])
        assert expected_route in route_names, (
            f"legacy-shaped model must be served by the AGC route {expected_route} "
            f"(tenant-local resolution keeps the AGC naming, not the legacy "
            f"'{model['name']}'), got {route_names}"
        )
        response = infer_for_shared_model(
            request_session_http=request_session_http,
            case=tenant_a,
            model=model,
            api_key=legacy_shaped_model["access"]["api_key"],
        )
        assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])

    @pytest.mark.tier2
    def test_adding_own_tenant_gateway_refs_recreates_nothing(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        tenant_external_model_tenants: dict[str, dict[str, str]],
        legacy_shaped_model: dict[str, Any],
    ) -> None:
        """Given a serving legacy-shaped model, when gatewayRefs naming only its own tenant are added,
        then no HTTPRoute or DestinationRule is recreated (identities must hold, not converge).
        """
        tenant_a = tenant_external_model_tenants["tenant_a"]
        model = legacy_shaped_model["model"]
        gateway = tenant_a["gateway_name"]

        def _snapshot() -> dict[str, Any]:
            routes = attachment_httproutes(
                admin_client=admin_client, route_namespace=model["namespace"], gateway_name=gateway
            )
            return {
                "route_identities": {
                    route["metadata"]["name"]: (
                        route["metadata"]["uid"],
                        route["metadata"].get("labels") or {},
                    )
                    for route in routes
                },
                "destination_rule_uids": {
                    rule["metadata"]["uid"]
                    for rule in destination_rules_referencing(admin_client, model["name"], MAAS_GATEWAY_NAMESPACE)
                },
            }

        before = _snapshot()
        expected_route = external_model_route_name(model_name=model["name"])
        assert expected_route in before["route_identities"], (
            f"AGC route {expected_route} not found before the gatewayRefs patch: {sorted(before['route_identities'])}"
        )
        assert before["destination_rule_uids"], (
            f"provider DestinationRule for {model['name']} must exist in "
            f"{MAAS_GATEWAY_NAMESPACE} while the model is serving"
        )

        patch_external_model_gateway_refs(
            admin_client=admin_client, name=model["name"], namespace=model["namespace"], refs=[gateway_ref(gateway)]
        )
        wait_for_model_generation_observed(admin_client=admin_client, name=model["name"], namespace=model["namespace"])
        wait_for_gateway_attachment_ready(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=gateway,
        )

        def _serving_config_unchanged() -> None:
            after = _snapshot()
            assert after["route_identities"] == before["route_identities"], (
                "adding own-tenant gatewayRefs recreated or relabeled HTTPRoutes: "
                f"{before['route_identities']} -> {after['route_identities']}"
            )
            assert after["destination_rule_uids"] == before["destination_rule_uids"], (
                "adding own-tenant gatewayRefs recreated DestinationRules: "
                f"{before['destination_rule_uids']} -> {after['destination_rule_uids']}"
            )

        assert_stays_absent(
            check=_serving_config_unchanged,
            timeout=RECONCILE_WAIT,
            what=f"serving config of {model['name']} staying unchanged after adding own-tenant gatewayRefs",
            interval=SETTLE_POLL_INTERVAL,
        )
        response = infer_for_shared_model(
            request_session_http=request_session_http,
            case=tenant_a,
            model=model,
            api_key=legacy_shaped_model["access"]["api_key"],
        )
        assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])


@pytest.mark.tier3
@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestLegacyIppCoexistence:
    """With the default tenant on ipp, the legacy controller ignores
    new-shaped AGC models and the backend handoff works.

    Serial: flips the default tenant's payload-processing backend. Gated on
    TENANT_EXTERNAL_MODEL_LEGACY_IPP_E2E=true (RHOAIENG-99761; requires the IPP
    image pin with the exclusion guard; the AGC companion deploy removes
    legacy IPP by default, so this also verifies the MaasTenantConfig backend
    selection redeploys it).
    """

    @pytest.mark.tier2
    def test_no_route_on_default_gateway(
        self,
        admin_client: DynamicClient,
        coexistence: dict[str, Any],
    ) -> None:
        """Given the default tenant on legacy IPP, when the new-shaped model is reconciled,
        then the legacy IPP controller attaches no route for it to the default gateway.
        """
        model = coexistence["model"]
        assert attachment_httproutes(admin_client, model["namespace"], MAAS_GATEWAY_NAME) == [], (
            f"legacy IPP must not attach a route for new-shaped model "
            f"{model['namespace']}/{model['name']} to the default gateway"
        )

    @pytest.mark.tier2
    def test_not_served_through_default_gateway(
        self, request_session_http: requests.Session, coexistence: dict[str, Any]
    ) -> None:
        """Given the new-shaped AGC model, when called through the default gateway,
        then the request is rejected (the legacy controller does not serve it).
        """
        model = coexistence["model"]
        response = infer_external_chat(
            request_session_http=request_session_http,
            gateway_url=coexistence["default_gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=coexistence["access"]["api_key"],
            path_tag=model["path_tag"],
            retries=1,
        )
        assert response.status_code in (401, 403, 404), (
            f"new-shaped AGC model must not be served on the default gateway, got {response.status_code}"
        )

    @pytest.mark.tier2
    def test_no_ipp_status_writes(
        self,
        admin_client: DynamicClient,
        coexistence: dict[str, Any],
    ) -> None:
        """Given the default tenant on legacy IPP, when the AGC model's status is read,
        then the AGC per-gateway results are unchanged (no legacy-controller writes).
        """
        model = coexistence["model"]
        status_after = attachment_status_snapshot(admin_client=admin_client, model=model)
        assert status_after == coexistence["status_before"], (
            f"legacy IPP overwrote the AGC status.gateways results: {coexistence['status_before']} -> {status_after}"
        )

    @pytest.mark.tier2
    def test_shared_provider_ownership_stays_defined(
        self,
        admin_client: DynamicClient,
        coexistence: dict[str, Any],
    ) -> None:
        """Given legacy/Praxis coexistence, when the provider's reconciled Service is read,
        then it keeps its identity (the legacy controller neither claims nor deletes it).
        """
        model = coexistence["model"]
        service = get_json_or_none(
            client=admin_client,
            kind="service",
            name=provider_external_name_service(model["provider"]),
            namespace=model["namespace"],
        )
        expected_uid = coexistence["provider_service_uid"]
        assert service is not None, "provider Service disappeared during legacy/Praxis coexistence"
        if expected_uid:
            assert service["metadata"]["uid"] == expected_uid, (
                "provider Service was recreated (ownership handoff broke)"
            )

    @pytest.mark.tier2
    def test_agc_tenant_gateway_still_serves(
        self, request_session_http: requests.Session, coexistence: dict[str, Any]
    ) -> None:
        """Given the default tenant on legacy IPP, when the model is called through its tenant gateway,
        then the AGC/Praxis path still serves it with the provider credential injected.
        """
        model = coexistence["model"]
        case_b = coexistence["case_b"]
        response = infer_for_shared_model(
            request_session_http=request_session_http,
            case=case_b,
            model=model,
            api_key=coexistence["access"]["api_key"],
        )
        assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])
