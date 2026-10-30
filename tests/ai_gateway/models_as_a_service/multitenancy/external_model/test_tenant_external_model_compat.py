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
from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.maas_subscription.utils import (
    MAAS_SUBSCRIPTION_NAMESPACE,
)
from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    PRAXIS_IMAGE_MARKER,
    admitted_user_namespace,
    assert_served_by_provider,
    attachment_httproutes,
    deployment_image,
    destination_rules_referencing,
    external_model_route_name,
    external_model_stack,
    gateway_ref,
    get_json_or_none,
    infer_external_chat,
    namespace_name,
    patch_external_model_gateway_refs,
    require_legacy_ipp_coexistence_support,
    serve_external_model_in_tenant,
    wait_for_attachment_httproute,
    wait_for_gateway_attachment_ready,
    wait_for_model_generation_observed,
    wait_until,
)
from tests.ai_gateway.models_as_a_service.multitenancy.utils import (
    tenant_gateway_hostname,
)
from utilities.constants import MAAS_GATEWAY_NAME, MAAS_GATEWAY_NAMESPACE
from utilities.general import generate_random_name
from utilities.resources.maastenantconfig import MaasTenantConfig

LOGGER = structlog.get_logger(name=__name__)

# Serialized with the tenant_ipp-flavored MaaS suites until the legacy IPP
# controllers skip gatewayRefs ExternalModels (image pin pending): without
# that guard they adopt this suite's models and flap status.httpRouteName.
pytestmark = pytest.mark.xdist_group(name="tenant_ipp")

# MaasTenantConfig contract for the default tenant's payload-processing
# backend selection (constants preserved from the MaaS suite's
# multitenancy_helpers).
TENANT_CONFIG_NAME = "default-tenant"
ANNOTATION_PAYLOAD_PROCESSING_TYPE = "maas.opendatahub.io/payload-processing-type"
PAYLOAD_PROCESSING_TYPE_PRAXIS = "praxis"
PAYLOAD_PROCESSING_TYPE_IPP = "ipp"

# The default tenant lives in the MaaS subscription namespace.
DEFAULT_TENANT_NAMESPACE = MAAS_SUBSCRIPTION_NAMESPACE

RECONCILE_WAIT = int(os.environ.get("E2E_RECONCILE_WAIT", "8"))
DEFAULT_BACKEND_SWITCH_TIMEOUT = int(os.environ.get("E2E_TENANT_BACKEND_SWITCH_TIMEOUT", "420"))
SETTLE_POLL_INTERVAL = 5


# ─── Default tenant payload-processing backend ──────────────────────────────


def _default_payload_processing_is_praxis(admin_client: DynamicClient) -> bool:
    """Whether the default tenant's payload-processing deployment runs praxis-extproc."""
    deployment = get_json_or_none(
        client=admin_client, kind="deployment", name="payload-processing", namespace=MAAS_GATEWAY_NAMESPACE
    )
    return PRAXIS_IMAGE_MARKER in deployment_image(deployment)


def _wait_default_backend(
    admin_client: DynamicClient,
    *,
    praxis: bool,
    timeout: int = DEFAULT_BACKEND_SWITCH_TIMEOUT,
) -> None:
    """Wait for the default tenant's payload-processing backend selection."""

    def _check() -> bool:
        deployment = get_json_or_none(
            client=admin_client, kind="deployment", name="payload-processing", namespace=MAAS_GATEWAY_NAMESPACE
        )
        assert deployment is not None, "default payload-processing deployment not found"
        current = _default_payload_processing_is_praxis(admin_client=admin_client)
        assert current == praxis, f"default payload-processing praxis={current}, expected {praxis}"
        # A spec-only check can pass while the switch is still rolling out;
        # require a live replica so the selected backend has actually started.
        status = deployment.get("status") or {}
        assert status.get("availableReplicas", 0) >= 1, (
            "default payload-processing deployment has no available replicas"
        )
        return True

    wait_until(
        check=_check,
        timeout=timeout,
        what=f"default payload-processing backend to become {'praxis' if praxis else 'legacy IPP'}",
    )


def _payload_processing_annotation(
    admin_client: DynamicClient,
    tenant_namespace: str,
) -> str | None:
    """Read the tenant config's payload-processing annotation.

    Returns None when the annotation is absent (backend selected by cluster
    default rather than an explicit override).
    """
    tenant_config = MaasTenantConfig(
        client=admin_client,
        name=TENANT_CONFIG_NAME,
        namespace=tenant_namespace,
        ensure_exists=True,
    )
    if not tenant_config.exists:
        return None
    metadata = tenant_config.instance.to_dict().get("metadata") or {}
    annotations = metadata.get("annotations") or {}
    return annotations.get(ANNOTATION_PAYLOAD_PROCESSING_TYPE)


def _patch_payload_processing_annotation(
    admin_client: DynamicClient,
    tenant_namespace: str,
    value: str | None,
) -> None:
    """Merge-patch the payload-processing annotation (None removes the key)."""
    tenant_config = MaasTenantConfig(
        client=admin_client,
        name=TENANT_CONFIG_NAME,
        namespace=tenant_namespace,
        ensure_exists=True,
    )
    tenant_config.update(
        resource_dict={"metadata": {"annotations": {ANNOTATION_PAYLOAD_PROCESSING_TYPE: value}}},
    )


def _ensure_payload_processing_type_on_tenant_config(
    admin_client: DynamicClient,
    tenant_namespace: str,
    payload_type: str,
) -> None:
    """Patch MaasTenantConfig to select a payload backend (skip when already set)."""
    if _payload_processing_annotation(admin_client, tenant_namespace) == payload_type:
        return
    _patch_payload_processing_annotation(
        admin_client=admin_client, tenant_namespace=tenant_namespace, value=payload_type
    )


def _restore_payload_processing_annotation(
    admin_client: DynamicClient,
    tenant_namespace: str,
    original: str | None,
) -> None:
    """Restore the payload-processing annotation to a captured value.

    ``original=None`` removes the annotation — a merge patch with a null
    value deletes the key — restoring cluster-default backend selection.
    """
    _patch_payload_processing_annotation(admin_client=admin_client, tenant_namespace=tenant_namespace, value=original)


def provider_external_name_service(provider_name: str) -> str:
    """The AGC's ExternalName Service name for a provider (``provider-<name>``)."""
    return f"provider-{provider_name}"


def _attachment_status_snapshot(admin_client: DynamicClient, model: dict[str, Any]) -> dict[Any, Any]:
    """Stable comparable snapshot of a model's attachment status.

    Captures BOTH condition shapes on every call: the model's top-level
    conditions and each per-gateway ``status.gateways`` entry. AGC builds
    that report aggregate status only leave ``status.gateways`` empty, and
    a legacy-controller write may touch either shape — including both in
    one snapshot means a change to one cannot hide behind the other.
    """
    external_model = get_json_or_none(admin_client, "externalmodel", model["name"], model["namespace"]) or {}
    status = external_model.get("status") or {}
    snapshot: dict[Any, Any] = {
        ("aggregate", ""): sorted(
            (condition.get("type"), condition.get("status"), condition.get("reason"))
            for condition in status.get("conditions") or []
        )
    }
    for entry in status.get("gateways") or []:
        snapshot[(entry.get("name"), entry.get("namespace"))] = sorted(
            (condition.get("type"), condition.get("status"), condition.get("reason"))
            for condition in entry.get("conditions") or []
        )
    return snapshot


def _serve_in_tenant(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token: str,
    tenant: dict[str, str],
    model: dict[str, Any],
) -> Any:
    """Serve one model through a tenant and yield the access dict (see serve_external_model_in_tenant)."""
    return serve_external_model_in_tenant(
        admin_client=admin_client,
        request_session_http=request_session_http,
        ocp_token=ocp_token,
        tenant=tenant,
        model_name=model["name"],
        model_namespace=model["namespace"],
        policy_name=f"{model['name']}-access",
        subscription_name=f"{model['name']}-sub",
    )


# ─── P0 #4 (runtime half): omitted gatewayRefs ──────────────────────────────


class TestOmittedGatewayRefsUnchanged:
    """A legacy-shaped model (no gatewayRefs) in the tenant namespace keeps
    the legacy resource contract, and adding own-tenant gatewayRefs recreates
    nothing."""

    @pytest.fixture(scope="class")
    def legacy_shaped_model(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        tenant_external_model_tenants: dict[str, dict[str, str]],
    ) -> Generator[dict[str, Any], Any, Any]:
        """A gatewayRefs-less model in tenant A's namespace, served through tenant A."""
        tenant_a = tenant_external_model_tenants["tenant_a"]
        with ExitStack() as stack:
            model = stack.enter_context(
                cm=external_model_stack(
                    admin_client,
                    name=f"e2e-cmp-legacy-{generate_random_name()}",
                    namespace=tenant_a["tenant_ns"],
                    gateway_refs=None,
                )
            )
            access = stack.enter_context(
                cm=_serve_in_tenant(admin_client, request_session_http, ocp_token_for_actor, tenant_a, model)
            )
            # The route lives in the model (tenant) namespace, attached to the
            # tenant's gateway via tenant-local resolution. The AGC names it
            # external-model-<model> — its own naming, not the <model> name the
            # pre-AGC legacy flow used. Scoped by model name: the conflicts
            # module also places models in this tenant namespace.
            routes = wait_for_attachment_httproute(
                admin_client=admin_client,
                route_namespace=model["namespace"],
                gateway_name=tenant_a["gateway_name"],
                model_name=model["name"],
            )
            yield {"model": model, "access": access, "routes": routes}

    @pytest.mark.tier2
    def test_omitted_gateway_refs_serve_via_tenant_local_resolution(
        self,
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
        response = infer_external_chat(
            gateway_url=tenant_a["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=legacy_shaped_model["access"]["api_key"],
            path_tag=model["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])

    @pytest.mark.tier2
    def test_adding_own_tenant_gateway_refs_recreates_nothing(
        self,
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
                # AGC keeps the provider DestinationRules in the Gateway
                # namespace, so that is where they are observable.
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

        # Adding gatewayRefs that name only the model's own tenant must not
        # recreate anything: same routes, same uids, same labels. Wait for
        # the AGC to observe the new generation first — attachment readiness
        # can return on the first poll while the status still reflects the
        # previous (already Ready) generation, and a recreation a few
        # seconds later would slip through an immediate comparison.
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

        # Then require the comparison to HOLD across a settle window (like
        # test_nothing_published_into_tenant): the home attachment must not
        # recreate the route at all, so equality must hold, not converge.
        settle_checks = max(1, RECONCILE_WAIT // SETTLE_POLL_INTERVAL) + 1
        for _ in range(settle_checks):
            after = _snapshot()
            assert after["route_identities"] == before["route_identities"], (
                "adding own-tenant gatewayRefs recreated or relabeled HTTPRoutes: "
                f"{before['route_identities']} -> {after['route_identities']}"
            )
            # The AGC keys the provider DestinationRule by provider and
            # tenant only — the gateway appears in labels, applied in
            # place — so the home attachment must not change the rule's
            # identity. A uid change here is an AGC regression.
            assert after["destination_rule_uids"] == before["destination_rule_uids"], (
                "adding own-tenant gatewayRefs recreated DestinationRules: "
                f"{before['destination_rule_uids']} -> {after['destination_rule_uids']}"
            )
            time.sleep(SETTLE_POLL_INTERVAL)
        response = infer_external_chat(
            gateway_url=tenant_a["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=legacy_shaped_model["access"]["api_key"],
            path_tag=model["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])


# ─── P0 #5: legacy IPP coexistence ──────────────────────────────────────────


@pytest.fixture(scope="class")
def compat_default_gateway_url(maas_host: str) -> str:
    """External URL of the default (models-as-a-service) Gateway."""
    return f"https://{tenant_gateway_hostname(gateway_name=MAAS_GATEWAY_NAME, maas_host=maas_host)}"


@pytest.mark.tier3
class TestLegacyIppCoexistence:
    """With the default tenant on ipp, the legacy controller ignores
    new-shaped AGC models and the backend handoff works.

    Serial: flips the default tenant's payload-processing backend. Gated on
    TENANT_EXTERNAL_MODEL_LEGACY_IPP_E2E=true (RHOAIENG-99761; requires the IPP
    image pin with the exclusion guard; the AGC companion deploy removes
    legacy IPP by default, so this also verifies the MaasTenantConfig backend
    selection redeploys it).
    """

    @pytest.fixture(scope="class")
    def coexistence(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        tenant_external_model_tenants: dict[str, dict[str, str]],
        compat_default_gateway_url: str,
    ) -> Generator[dict[str, Any], Any, Any]:
        """An AGC model in a user namespace, with the default tenant flipped to legacy IPP."""
        require_legacy_ipp_coexistence_support()
        if not _default_payload_processing_is_praxis(admin_client):
            pytest.skip(
                "default tenant payload-processing is not praxis; the coexistence "
                "handoff test needs a praxis starting point"
            )
        # Capture the default tenant's original payload-processing annotation
        # (including its absence) so teardown restores the cluster state
        # exactly: praxis may have been selected by cluster default, and
        # leaving an explicit override behind would change that selection.
        original_payload_annotation = _payload_processing_annotation(
            admin_client=admin_client, tenant_namespace=DEFAULT_TENANT_NAMESPACE
        )

        # An AGC-served model in a user namespace on a praxis tenant gateway
        # (tenant B from this module's shared tenants — the fixture layer
        # guarantees a praxis tenant with a selector-admitting gateway).
        case_b = tenant_external_model_tenants["tenant_b"]
        with ExitStack() as stack:
            user_ns = stack.enter_context(
                cm=admitted_user_namespace(admin_client, "e2e-extmod-ipp-user", case_b["gateway_name"])
            )
            model = stack.enter_context(
                cm=external_model_stack(
                    admin_client,
                    name=f"e2e-ipp-coex-{generate_random_name()}",
                    namespace=namespace_name(user_ns),
                    gateway_refs=[gateway_ref(case_b["gateway_name"])],
                )
            )
            access = stack.enter_context(
                cm=_serve_in_tenant(admin_client, request_session_http, ocp_token_for_actor, case_b, model)
            )

            wait_for_gateway_attachment_ready(
                admin_client=admin_client,
                model_name=model["name"],
                model_namespace=model["namespace"],
                gateway_name=case_b["gateway_name"],
            )
            provider_service = get_json_or_none(
                client=admin_client,
                kind="service",
                name=provider_external_name_service(model["provider"]),
                namespace=model["namespace"],
            )
            status_before = _attachment_status_snapshot(admin_client=admin_client, model=model)

            # Flip the default tenant to legacy IPP (backend selection +
            # cleanup/claim handoff) and wait for the legacy stack.
            _ensure_payload_processing_type_on_tenant_config(
                admin_client=admin_client,
                tenant_namespace=DEFAULT_TENANT_NAMESPACE,
                payload_type=PAYLOAD_PROCESSING_TYPE_IPP,
            )
            _wait_default_backend(admin_client=admin_client, praxis=False)
            # A live legacy replica is necessary but not sufficient for the
            # controller to have acted on the default tenant; settle before
            # the negative assertions so absence checks are meaningful.
            time.sleep(RECONCILE_WAIT)

            try:
                yield {
                    "case_b": case_b,
                    "model": model,
                    "access": access,
                    "compat_default_gateway_url": compat_default_gateway_url,
                    "status_before": status_before,
                    "provider_service_uid": (provider_service or {}).get("metadata", {}).get("uid"),
                }
            finally:
                # Claim handoff back to the original backend selection (the
                # starting backend is praxis — the skip above enforces it —
                # so praxis must come back either way); a failure here
                # surfaces as a fixture teardown error instead of poisoning
                # later tests. The user namespace, model stack and tenant
                # access clean up with the ExitStack afterwards.
                _restore_payload_processing_annotation(
                    admin_client, DEFAULT_TENANT_NAMESPACE, original_payload_annotation
                )
                _wait_default_backend(admin_client, praxis=True)

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

    def test_not_served_through_default_gateway(self, coexistence: dict[str, Any]) -> None:
        """Given the new-shaped AGC model, when called through the default gateway,
        then the request is rejected (the legacy controller does not serve it).
        """
        model = coexistence["model"]
        response = infer_external_chat(
            gateway_url=coexistence["compat_default_gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=coexistence["access"]["api_key"],
            path_tag=model["path_tag"],
            retries=1,
        )
        assert response.status_code in (401, 403, 404), (
            f"new-shaped AGC model must not be served on the default gateway, got {response.status_code}"
        )

    def test_no_ipp_status_writes(
        self,
        admin_client: DynamicClient,
        coexistence: dict[str, Any],
    ) -> None:
        """Given the default tenant on legacy IPP, when the AGC model's status is read,
        then the AGC per-gateway results are unchanged (no legacy-controller writes).
        """
        model = coexistence["model"]
        status_after = _attachment_status_snapshot(admin_client=admin_client, model=model)
        assert status_after == coexistence["status_before"], (
            f"legacy IPP overwrote the AGC status.gateways results: {coexistence['status_before']} -> {status_after}"
        )

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

    def test_agc_tenant_gateway_still_serves(self, coexistence: dict[str, Any]) -> None:
        """Given the default tenant on legacy IPP, when the model is called through its tenant gateway,
        then the AGC/Praxis path still serves it with the provider credential injected.
        """
        model = coexistence["model"]
        case_b = coexistence["case_b"]
        response = infer_external_chat(
            gateway_url=case_b["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=coexistence["access"]["api_key"],
            path_tag=model["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])
