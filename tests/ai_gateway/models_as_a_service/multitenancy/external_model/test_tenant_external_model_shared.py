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
tenant_external_model_tenants fixture); this module creates its own user
namespace with unique names.

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
from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_model_ref import MaaSModelRef

from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    EXTERNAL_ENDPOINT,
    TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
    admitted_user_namespace,
    assert_returns_usage,
    assert_served_by_provider,
    external_model_stack,
    gateway_ref,
    gateway_status_entry,
    get_json_or_none,
    infer_external_chat,
    redact_sensitive,
    require_client_name_alias,
    require_maas_route,
    serve_shared_model_in_tenant,
    usage_provider,
    wait_for_attachment_httproute,
    wait_shared_attachments_ready,
    wait_until,
)

LOGGER = structlog.get_logger(name=__name__)

# Serialized with tenant_ipp until the legacy IPP controllers skip
# gatewayRefs ExternalModels (image pin pending): without that guard they
# adopt this suite's models and flap status.httpRouteName.
pytestmark = pytest.mark.xdist_group(name="tenant_ipp")

# Tenant A gets a tiny budget so quota isolation is observable quickly;
# tenant B gets a large one (mirrors the rate-limit isolation module).
LIMITED_TOKEN_LIMIT = 3
UNLIMITED_TOKEN_LIMIT = 100

# Token budgets are accounted from usage.total_tokens in inference
# responses, which the header-echo endpoint never returns — so the quota
# model's provider is a dedicated in-cluster simulator instead. The gateway
# envoy verifies provider certificates, so the simulator's Service gets an
# OpenShift service serving certificate (signed by the cluster service CA),
# and the provider points envoy at the CA bundle mounted in every pod via
# spec.config["tls.caCertificates"] — see usage_provider.

# llm-d-inference-sim used by the quota test's dedicated provider:
# --mode random responses carry usage.total_tokens; --mode echo responses do not.


@pytest.fixture(scope="module")
def shared_env(
    admin_client: DynamicClient,
    tenant_external_model_tenants: dict[str, dict[str, str]],
    teardown_resources: bool,
) -> Generator[dict[str, Any], Any, Any]:
    """Module tenants plus this module's own user namespace and cleanup stack."""
    tenant_a = tenant_external_model_tenants["tenant_a"]
    tenant_b = tenant_external_model_tenants["tenant_b"]
    with ExitStack() as stack:
        user_ns = stack.enter_context(
            cm=admitted_user_namespace(
                admin_client,
                "e2e-extmod-shr-user",
                tenant_a["gateway_name"],
                tenant_b["gateway_name"],
                teardown=teardown_resources,
            )
        )
        LOGGER.info(f"[shared-model] user namespace {user_ns.name} admitted by both tenant gateways")
        yield {
            "tenant_a": tenant_a,
            "tenant_b": tenant_b,
            "user_ns": user_ns.name,
            "stack": stack,
            "teardown": teardown_resources,
        }


class TestSharedModelTwoGateways:
    """One model, two attachments: routes, status and serving on both."""

    @pytest.fixture(scope="class")
    def shared_model(self, admin_client: DynamicClient, shared_env: dict[str, Any]) -> dict[str, Any]:
        """The shared model attached to both tenants' gateways."""
        env = shared_env
        name = f"e2e-shr-shared-{uuid.uuid4().hex[:6]}"
        model = env["stack"].enter_context(
            cm=external_model_stack(
                admin_client,
                name=name,
                namespace=env["user_ns"],
                gateway_refs=[
                    gateway_ref(env["tenant_a"]["gateway_name"]),
                    gateway_ref(env["tenant_b"]["gateway_name"]),
                ],
                # The client-facing name deliberately differs from the CRD name:
                # per-tenant modelRefs necessarily have distinct names, so shared
                # requests must resolve through the client-facing identity (the
                # alias gate keys on that difference).
                model_name=f"{name}-client",
                tenant_refs=[
                    env["tenant_a"]["tenant_label_name"],
                    env["tenant_b"]["tenant_label_name"],
                ],
                teardown=env["teardown"],
            )
        )
        wait_shared_attachments_ready(admin_client=admin_client, env=env, model=model)
        return {"model": model}

    @pytest.fixture(scope="class")
    def shared_serving(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        shared_env: dict[str, Any],
        shared_model: dict[str, Any],
    ) -> dict[str, Any]:
        """Per-tenant auth/subscription stacks and API keys for the shared model."""
        env = shared_env
        model = shared_model["model"]
        require_maas_route(admin_client=admin_client, model=model)
        access_a = serve_shared_model_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            case=env["tenant_a"],
            model=model,
        )
        access_b = serve_shared_model_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            case=env["tenant_b"],
            model=model,
        )
        return {"access_a": access_a, "access_b": access_b}

    @pytest.mark.tier2
    def test_one_httproute_per_attachment(
        self,
        admin_client: DynamicClient,
        shared_env: dict[str, Any],
        shared_model: dict[str, Any],
    ) -> None:
        """Each attachment has its own HTTPRoute in the model namespace."""
        env = shared_env
        model = shared_model["model"]
        routes_by_gateway: dict[str, list[dict[str, Any]]] = {}
        for case in (env["tenant_a"], env["tenant_b"]):
            # Scoped by model name: other classes' models share this
            # namespace, and the exact-two-UIDs assertion below must not
            # count their routes.
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
        shared_env: dict[str, Any],
        shared_model: dict[str, Any],
    ) -> None:
        """status.gateways carries an independent ready result per gateway."""
        env = shared_env
        model = shared_model["model"]
        if (
            gateway_status_entry(admin_client, model["name"], model["namespace"], env["tenant_a"]["gateway_name"])
            is None
        ):
            pytest.skip(
                "AGC reports aggregate model status only (no status.gateways entries); "
                "per-gateway readiness is covered by test_both_gateways_serve"
            )
        wait_shared_attachments_ready(admin_client=admin_client, env=env, model=model)

    @pytest.mark.tier2
    def test_both_gateways_serve(
        self,
        shared_env: dict[str, Any],
        shared_model: dict[str, Any],
        shared_serving: dict[str, Any],
        admin_client: DynamicClient,
    ) -> None:
        """Real inference succeeds through both gateways with the same provider
        binding and credential."""
        env = shared_env
        model = shared_model["model"]
        require_client_name_alias(admin_client=admin_client, model=model, case=env["tenant_a"])
        for case, access in (
            (env["tenant_a"], shared_serving["access_a"]),
            (env["tenant_b"], shared_serving["access_b"]),
        ):
            response = infer_external_chat(
                gateway_url=case["gateway_url"],
                model_namespace=model["namespace"],
                model_client_name=model["client_name"],
                api_key=access["api_key"],
                path_tag=model["path_tag"],
            )
            assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])


class TestTenantAuthIsolation:
    """Each tenant's key serves only through its own gateway.

    Both tenants authorize the model, so a cross-tenant rejection proves
    key isolation rather than an absent grant on the rejecting tenant's
    gateway."""

    @pytest.fixture(scope="class")
    def isolated_keys(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        shared_env: dict[str, Any],
    ) -> dict[str, Any]:
        """Two tenants' API keys for the same two-gateway model."""
        env = shared_env
        name = f"e2e-shr-auth-{uuid.uuid4().hex[:6]}"
        model = env["stack"].enter_context(
            cm=external_model_stack(
                admin_client,
                name=name,
                namespace=env["user_ns"],
                gateway_refs=[
                    gateway_ref(env["tenant_a"]["gateway_name"]),
                    gateway_ref(env["tenant_b"]["gateway_name"]),
                ],
                model_name=f"{name}-client",
                # Both tenants authorize the model, so a cross-tenant rejection
                # proves key isolation rather than a missing grant on the
                # rejecting tenant.
                tenant_refs=[
                    env["tenant_a"]["tenant_label_name"],
                    env["tenant_b"]["tenant_label_name"],
                ],
                teardown=env["teardown"],
            )
        )
        wait_shared_attachments_ready(admin_client=admin_client, env=env, model=model)
        require_maas_route(admin_client=admin_client, model=model)
        access_a = serve_shared_model_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            case=env["tenant_a"],
            model=model,
        )
        access_b = serve_shared_model_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            case=env["tenant_b"],
            model=model,
        )
        return {
            "model": model,
            "key_a": access_a["api_key"],
            "key_b": access_b["api_key"],
        }

    @pytest.mark.tier2
    def test_tenant_keys_authorized_only_on_their_own_gateway(
        self,
        admin_client: DynamicClient,
        shared_env: dict[str, Any],
        isolated_keys: dict[str, Any],
    ) -> None:
        """Each tenant's key serves through its own gateway and is rejected
        by the other's.

        The own-gateway 200 is the positive control: the other tenant's
        stack is live and authorizing the model, so the cross-tenant 401/403
        is a key-isolation verdict, not a missing grant."""
        env = shared_env
        model = isolated_keys["model"]
        require_client_name_alias(admin_client=admin_client, model=model, case=env["tenant_a"])
        for case, other, api_key in (
            (env["tenant_a"], env["tenant_b"], isolated_keys["key_a"]),
            (env["tenant_b"], env["tenant_a"], isolated_keys["key_b"]),
        ):
            served = infer_external_chat(
                gateway_url=case["gateway_url"],
                model_namespace=model["namespace"],
                model_client_name=model["client_name"],
                api_key=api_key,
                path_tag=model["path_tag"],
            )
            assert_served_by_provider(response=served, path_tag=model["path_tag"], credential=model["credential"])
            rejected = infer_external_chat(
                gateway_url=other["gateway_url"],
                model_namespace=model["namespace"],
                model_client_name=model["client_name"],
                api_key=api_key,
                path_tag=model["path_tag"],
                retries=1,
            )
            assert rejected.status_code in (401, 403), (
                f"{case['tenant_label_name']}'s API key must be rejected by "
                f"{other['tenant_label_name']}'s gateway, got "
                f"{rejected.status_code}: {redact_sensitive(rejected.text)}"
            )

    @pytest.mark.tier2
    def test_unknown_key_rejected_on_both_gateways(
        self,
        shared_env: dict[str, Any],
        isolated_keys: dict[str, Any],
    ) -> None:
        """An unknown API key is rejected by both tenants' gateways."""
        env = shared_env
        model = isolated_keys["model"]
        for case in (env["tenant_a"], env["tenant_b"]):
            response = infer_external_chat(
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


class TestSharedModelAmbiguousModelRef:
    """A MaaSModelRef that does not name its tenant must not guess one.

    A model attached to two gateways has no unambiguous route for a
    tenantRef-less modelRef: it must report the ambiguity (Failed phase, a
    message directing the user to ``spec.tenantRef``) instead of silently
    selecting one gateway, while the per-tenant modelRefs that do pin
    ``tenantRef`` stay Ready.
    """

    @pytest.fixture(scope="class")
    def ambiguous_stack(
        self,
        admin_client: DynamicClient,
        shared_env: dict[str, Any],
    ) -> dict[str, Any]:
        """A two-gateway model plus a tenantRef-less modelRef naming it."""
        env = shared_env
        name = f"e2e-shr-amb-{uuid.uuid4().hex[:6]}"
        model = env["stack"].enter_context(
            cm=external_model_stack(
                admin_client,
                name=name,
                namespace=env["user_ns"],
                gateway_refs=[
                    gateway_ref(env["tenant_a"]["gateway_name"]),
                    gateway_ref(env["tenant_b"]["gateway_name"]),
                ],
                model_name=f"{name}-client",
                tenant_refs=[
                    env["tenant_a"]["tenant_label_name"],
                    env["tenant_b"]["tenant_label_name"],
                ],
                teardown=env["teardown"],
            )
        )
        wait_shared_attachments_ready(admin_client=admin_client, env=env, model=model)
        # The ambiguity verdict is produced from the AGC's per-gateway
        # status.gateways results; aggregate-status builds report no route
        # for multi-gateway models and cannot exercise this path.
        require_maas_route(admin_client=admin_client, model=model)
        ambiguous_ref = f"{name}-ambiguous"
        env["stack"].enter_context(
            cm=MaaSModelRef(
                client=admin_client,
                name=ambiguous_ref,
                namespace=env["user_ns"],
                # Same contract as the suite's per-tenant modelRefs: the
                # annotations name the provider endpoint, and modelRef
                # resolves within the modelRef's own namespace.
                annotations={
                    "maas.opendatahub.io/endpoint": EXTERNAL_ENDPOINT,
                    "maas.opendatahub.io/provider": "openai",
                },
                model_ref={"kind": "ExternalModel", "name": name},
                teardown=env["teardown"],
                wait_for_resource=True,
            )
        )
        return {"model": model, "ambiguous_ref": ambiguous_ref}

    @pytest.mark.tier2
    def test_tenantless_model_ref_reports_ambiguity(
        self,
        admin_client: DynamicClient,
        shared_env: dict[str, Any],
        ambiguous_stack: dict[str, Any],
    ) -> None:
        """The tenantRef-less modelRef fails with a message naming
        spec.tenantRef, and reports no endpoint."""
        _ = shared_env
        model = ambiguous_stack["model"]

        def _check() -> dict[str, Any]:
            ref = get_json_or_none(
                client=admin_client,
                kind="maasmodelref",
                name=ambiguous_stack["ambiguous_ref"],
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
                f"MaaSModelRef {model['namespace']}/{ambiguous_stack['ambiguous_ref']} report the two-gateway ambiguity"
            ),
        )
        assert not status.get("endpoint"), f"ambiguous modelRef must not report an endpoint: {redact_sensitive(status)}"

    @pytest.mark.tier2
    def test_pinned_model_refs_stay_ready(
        self,
        admin_client: DynamicClient,
        shared_env: dict[str, Any],
        ambiguous_stack: dict[str, Any],
    ) -> None:
        """The per-tenant modelRefs pinning spec.tenantRef are unaffected by
        their ambiguous sibling: each resolves its own tenant's attachment
        (RuntimeReady, resolvedTenantRef and serving gateway set to its own
        tenant) with no ambiguity error.

        Full phase-Ready additionally needs a governance pairing and unique
        model identities (RHOAIENG-101609 — two per-tenant refs for one
        ExternalModel share a resolved alias today), which are separate
        gaps; this test proves the tenant selection the scenario is about."""
        env = shared_env
        model = ambiguous_stack["model"]
        cases = {
            env["tenant_a"]["tenant_label_name"]: env["tenant_a"],
            env["tenant_b"]["tenant_label_name"]: env["tenant_b"],
        }
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
                conditions = {c.get("type"): c for c in status.get("conditions") or []}
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


class TestTenantQuotaIsolation:
    """Quota consumption is tracked per tenant: exhausting tenant A's token
    budget leaves tenant B's untouched.

    Token budgets are accounted from ``usage.total_tokens`` in inference
    responses, so this class's model is served by a dedicated in-cluster
    simulator (see ``usage_provider``) rather than the header-echo
    endpoint, whose responses carry no usage and could never exhaust a
    budget.
    """

    @pytest.fixture(scope="class")
    def quota_model(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        shared_env: dict[str, Any],
    ) -> dict[str, Any]:
        """The simulator-backed shared model served in both tenants with
        asymmetric token budgets."""
        env = shared_env
        name = f"e2e-shr-quota-{uuid.uuid4().hex[:6]}"
        client_name = f"{name}-client"
        usage = usage_provider(
            admin_client=admin_client, env=env, namespace=env["user_ns"], client_model_name=client_name
        )
        model = env["stack"].enter_context(
            cm=external_model_stack(
                admin_client,
                name=name,
                namespace=env["user_ns"],
                gateway_refs=[
                    gateway_ref(env["tenant_a"]["gateway_name"]),
                    gateway_ref(env["tenant_b"]["gateway_name"]),
                ],
                model_name=client_name,
                tenant_refs=[
                    env["tenant_a"]["tenant_label_name"],
                    env["tenant_b"]["tenant_label_name"],
                ],
                endpoint=usage["endpoint"],
                provider_path=usage["provider_path"],
                credential=usage["credential"],
                provider_config=usage["provider_config"],
                teardown=env["teardown"],
            )
        )
        wait_shared_attachments_ready(admin_client=admin_client, env=env, model=model)
        require_maas_route(admin_client=admin_client, model=model)
        access_a = serve_shared_model_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            case=env["tenant_a"],
            model=model,
            token_limit=LIMITED_TOKEN_LIMIT,
            window="1m",
        )
        access_b = serve_shared_model_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            case=env["tenant_b"],
            model=model,
            token_limit=UNLIMITED_TOKEN_LIMIT,
            window="1m",
        )
        return {"model": model, "access_a": access_a, "access_b": access_b, "usage": usage}

    @pytest.mark.tier2
    def test_quota_exhaustion_isolated_per_tenant(
        self,
        admin_client: DynamicClient,
        shared_env: dict[str, Any],
        quota_model: dict[str, Any],
    ) -> None:
        """Exhausting tenant A's token budget leaves tenant B serving."""
        env = shared_env
        model = quota_model["model"]
        require_client_name_alias(admin_client=admin_client, model=model, case=env["tenant_a"])
        # The quota model binds to the usage simulator, not the header-echo
        # endpoint: requests must carry the simulator's own path (the default
        # echo /anything/<tag> path would 404 on it).
        provider_path = quota_model["usage"]["provider_path"]

        # Drive tenant A past its budget: at least one 200 must precede the
        # 429 (a fresh budget always admits the first request — see
        # the rate-limit isolation module's _exhaust_until_429).
        successes = 0
        limited = None
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                limited = infer_external_chat(
                    gateway_url=env["tenant_a"]["gateway_url"],
                    model_namespace=model["namespace"],
                    model_client_name=model["client_name"],
                    api_key=quota_model["access_a"]["api_key"],
                    path_tag=model["path_tag"],
                    provider_path=provider_path,
                    # No helper retries: a 429 near the end of the window would
                    # otherwise sleep past the reset and come back 200, hiding
                    # the rejection this loop must observe (the route is warm).
                    retries=0,
                )
            except requests.RequestException:
                # retries=0 also disables the helper's connection-error
                # retries; a transient network failure must not end the
                # exhaustion loop before the budget is reached.
                time.sleep(1.0)
                continue
            if limited.status_code == 200:
                successes += 1
            elif limited.status_code == 429:
                if successes:
                    break
            elif limited.status_code >= 500:
                # Same for transient 5xx: retry inside the loop until the
                # deadline rather than failing the exhaustion drive.
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

        # Tenant B's budget is independent and still serves — with usage,
        # which is what keeps its own budget accountable.
        response = infer_external_chat(
            gateway_url=env["tenant_b"]["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=quota_model["access_b"]["api_key"],
            path_tag=model["path_tag"],
            provider_path=provider_path,
        )
        assert_returns_usage(response=response)

        # And tenant A stays limited while tenant B keeps serving.
        still_limited = infer_external_chat(
            gateway_url=env["tenant_a"]["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=quota_model["access_a"]["api_key"],
            path_tag=model["path_tag"],
            provider_path=provider_path,
            retries=1,
        )
        assert still_limited.status_code == 429, (
            f"tenant A should remain rate limited, got {still_limited.status_code}: "
            f"{redact_sensitive(still_limited.text)}"
        )
        again_b = infer_external_chat(
            gateway_url=env["tenant_b"]["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=quota_model["access_b"]["api_key"],
            path_tag=model["path_tag"],
            provider_path=provider_path,
        )
        assert_returns_usage(response=again_b)
