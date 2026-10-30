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
  tests skip there (see ``_require_maas_route``);
- the client-facing ``spec.modelName`` carried as the model identity: the
  request path and the modelRefs' names disagree here by construction
  (per-tenant refs have distinct names), so path-routed requests only
  authorize once ``status.resolvedModelAlias`` carries the client-facing
  name and maas-api matches pairs against it (see
  ``_require_client_name_alias``). Unknown-key rejection needs no model
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
from ocp_resources.deployment import Deployment
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.service import Service

from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    EXTERNAL_ENDPOINT,
    TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
    admitted_user_namespace,
    assert_served_by_provider,
    external_model_reports_maas_route,
    external_model_stack,
    gateway_ref,
    gateway_status_entry,
    get_json_or_none,
    infer_external_chat,
    maas_model_ref_alias,
    redact_sensitive,
    serve_external_model_in_tenant,
    wait_for_attachment_httproute,
    wait_for_gateway_attachment_ready,
    wait_until,
)
from utilities.image_constants import SharedImages

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
# spec.config["tls.caCertificates"] — see _usage_provider.
SA_SERVICE_CA = "/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"
SERVING_CERT_ANNOTATION = "service.beta.openshift.io/serving-cert-secret-name"

# llm-d-inference-sim used by the quota test's dedicated provider:
# --mode random responses carry usage.total_tokens; --mode echo responses do not.
LLM_D_INFERENCE_SIM_IMAGE: str = SharedImages.LLMD_INFERENCE_SIM_UPSTREAM


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


def _serve_in_tenant(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token: str,
    env: dict[str, Any],
    case: dict[str, str],
    model: dict[str, Any],
    **kwargs: Any,
) -> dict[str, str]:
    """Serve one shared model in one tenant and mint its API key."""
    ref_name = model["model_refs"][case["tenant_label_name"]]
    return env["stack"].enter_context(
        cm=serve_external_model_in_tenant(
            admin_client,
            request_session_http,
            ocp_token,
            tenant=case,
            model_name=ref_name,
            model_namespace=model["namespace"],
            policy_name=f"{model['name']}-access",
            subscription_name=f"{model['name']}-sub",
            teardown=env["teardown"],
            **kwargs,
        )
    )


def _wait_attachments_ready(admin_client: DynamicClient, env: dict[str, Any], model: dict[str, Any]) -> None:
    """Wait until both tenants' gateway attachments report ready."""
    for case in (env["tenant_a"], env["tenant_b"]):
        wait_for_gateway_attachment_ready(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=case["gateway_name"],
        )


def _require_maas_route(admin_client: DynamicClient, model: dict[str, Any]) -> None:
    """Skip when the MaaS stack cannot resolve this multi-gateway model.

    Aggregate-status AGC builds report no per-gateway ``status.gateways``
    results and no ``status.httpRouteName`` for multi-attachment models, so
    maas-controller cannot select a route: the MaaSModelRef never reports an
    endpoint and the subscription TRLP backend stays not-ready, which makes
    per-tenant auth keys and quotas impossible to wire for the shared model.
    """
    if not external_model_reports_maas_route(admin_client, model["name"], model["namespace"]):
        pytest.skip(
            "AGC reports aggregate model status only (no per-gateway status.gateways "
            "results); the MaaS stack cannot resolve a multi-gateway model's route, so "
            "per-tenant auth/quota serving for a shared model cannot be exercised"
        )


def _require_client_name_alias(
    admin_client: DynamicClient,
    model: dict[str, Any],
    case: dict[str, str],
) -> None:
    """Skip until the published model identity is the client-facing name.

    The per-tenant stacks go green with per-gateway status alone, but a
    path-routed request carries ``{namespace}/{client-facing name}`` while
    the modelRefs are named after their tenants: requests only authorize
    once the controller publishes ``spec.modelName`` as
    ``status.resolvedModelAlias`` and maas-api matches pairs against it.
    Both halves ship together, so one alias check gates both.
    """
    ref_name = model["model_refs"][case["tenant_label_name"]]
    alias = maas_model_ref_alias(admin_client=admin_client, namespace=model["namespace"], name=ref_name)
    if alias != model["client_name"]:
        pytest.skip(
            "client-facing model names are not resolvable through the gateway "
            f"yet: MaaSModelRef {model['namespace']}/{ref_name} "
            f"status.resolvedModelAlias={alias!r} != {model['client_name']!r}"
        )


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
        _wait_attachments_ready(admin_client=admin_client, env=env, model=model)
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
        _require_maas_route(admin_client=admin_client, model=model)
        access_a = _serve_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            case=env["tenant_a"],
            model=model,
        )
        access_b = _serve_in_tenant(
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
        _wait_attachments_ready(admin_client=admin_client, env=env, model=model)

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
        _require_client_name_alias(admin_client=admin_client, model=model, case=env["tenant_a"])
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
        _wait_attachments_ready(admin_client=admin_client, env=env, model=model)
        _require_maas_route(admin_client=admin_client, model=model)
        access_a = _serve_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            case=env["tenant_a"],
            model=model,
        )
        access_b = _serve_in_tenant(
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
        _require_client_name_alias(admin_client=admin_client, model=model, case=env["tenant_a"])
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
        _wait_attachments_ready(admin_client=admin_client, env=env, model=model)
        # The ambiguity verdict is produced from the AGC's per-gateway
        # status.gateways results; aggregate-status builds report no route
        # for multi-gateway models and cannot exercise this path.
        _require_maas_route(admin_client=admin_client, model=model)
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


def _usage_provider(
    admin_client: DynamicClient,
    env: dict[str, Any],
    namespace: str,
    client_model_name: str,
) -> dict[str, Any]:
    """Deploy the usage-returning provider the quota model binds to.

    A dedicated in-cluster ``llm-d-inference-sim`` (``--mode random``): its
    responses carry ``usage.total_tokens``, which token budgets are
    accounted from. The gateway envoy verifies provider certificates, so
    the simulator cannot serve a self-signed cert: its Service is annotated
    for an OpenShift service serving certificate (signed by the cluster
    service CA) and the provider points envoy at the CA bundle mounted in
    every pod through ``spec.config["tls.caCertificates"]``. The simulator
    validates the request body's ``model`` field, so it is configured with
    the client-facing model name the requests will carry.

    The simulator is deployed in the model's namespace and is dedicated to
    that model, so no rate limits apply on the provider hop: a 429 seen
    through a tenant gateway can only be that tenant's own budget.

    An echo endpoint cannot stand in even with a synthetic usage field: the
    telemetry pipeline drops reports whose body does not name the model and
    cannot read gzip-compressed responses (which httpbingo serves) — the
    simulator's plain OpenAI-shaped bodies satisfy all of it.
    """
    stack: ExitStack = env["stack"]
    suffix = uuid.uuid4().hex[:6]
    sim = f"e2e-usage-sim-{suffix}"
    cert_secret = f"{sim}-cert"
    dns_name = f"{sim}.{namespace}.svc.cluster.local"
    stack.enter_context(
        cm=Service(
            client=admin_client,
            name=sim,
            namespace=namespace,
            annotations={SERVING_CERT_ANNOTATION: cert_secret},
            selector={"app": sim},
            # The ExternalProvider CRD rejects endpoints carrying a port
            # and the AGC dials 443, so the Service listens on 443 and
            # maps to the simulator's container port.
            ports=[{"port": 443, "targetPort": 8000, "name": "https"}],
            teardown=env["teardown"],
            wait_for_resource=True,
        )
    )

    def _cert_injected() -> bool:
        secret = get_json_or_none(admin_client, "secret", cert_secret, namespace) or {}
        return bool((secret.get("data") or {}).get("tls.crt"))

    wait_until(check=_cert_injected, timeout=120, what=f"service-ca cert secret {namespace}/{cert_secret}")

    stack.enter_context(
        cm=Deployment(
            client=admin_client,
            name=sim,
            namespace=namespace,
            replicas=1,
            selector={"matchLabels": {"app": sim}},
            template={
                "metadata": {"labels": {"app": sim}},
                "spec": {
                    "containers": [
                        {
                            "name": "sim",
                            "image": LLM_D_INFERENCE_SIM_IMAGE,
                            "command": ["/app/llm-d-inference-sim"],
                            "args": [
                                "--port",
                                "8000",
                                "--model",
                                client_model_name,
                                "--mode",
                                "random",
                                "--no-mm-encoder-only",
                                "--ssl-certfile",
                                "/etc/sim/tls/tls.crt",
                                "--ssl-keyfile",
                                "/etc/sim/tls/tls.key",
                            ],
                            "ports": [{"containerPort": 8000, "name": "https", "protocol": "TCP"}],
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": "https",
                                    "scheme": "HTTPS",
                                }
                            },
                            "volumeMounts": [{"name": "tls", "mountPath": "/etc/sim/tls", "readOnly": True}],
                        }
                    ],
                    "volumes": [{"name": "tls", "secret": {"secretName": cert_secret}}],
                },
            },
            teardown=env["teardown"],
        )
    )

    def _ready() -> bool:
        deployment = get_json_or_none(admin_client, "deployment", sim, namespace) or {}
        status = deployment.get("status") or {}
        return (status.get("availableReplicas") or 0) >= 1

    wait_until(check=_ready, timeout=240, what=f"usage simulator deployment {namespace}/{sim}")
    return {
        "endpoint": dns_name,
        "provider_path": "/v1/chat/completions",
        "credential": f"sk-usage-{suffix}",
        "provider_config": {"tls.caCertificates": SA_SERVICE_CA},
    }


def _assert_returns_usage(response: requests.Response) -> None:
    """Assert an inference response carries OpenAI-style usage tokens.

    Token quotas are accounted from ``usage.total_tokens`` in the response,
    so a provider that returns no usage (e.g. the header-echo endpoint)
    can never exhaust a budget — quota assertions are only meaningful
    against responses that pass this check.
    """
    assert response.status_code == 200, (
        f"expected 200 from external model inference, got {response.status_code}: {redact_sensitive(response.text)}"
    )
    try:
        usage = (response.json() or {}).get("usage") or {}
    except ValueError:
        usage = {}
    total_tokens = usage.get("total_tokens")
    assert isinstance(total_tokens, int) and total_tokens > 0, (
        f"response carries no usage.total_tokens for token accounting: {redact_sensitive(response.text)}"
    )


class TestTenantQuotaIsolation:
    """Quota consumption is tracked per tenant: exhausting tenant A's token
    budget leaves tenant B's untouched.

    Token budgets are accounted from ``usage.total_tokens`` in inference
    responses, so this class's model is served by a dedicated in-cluster
    simulator (see ``_usage_provider``) rather than the header-echo
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
        usage = _usage_provider(
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
        _wait_attachments_ready(admin_client=admin_client, env=env, model=model)
        _require_maas_route(admin_client=admin_client, model=model)
        access_a = _serve_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            case=env["tenant_a"],
            model=model,
            token_limit=LIMITED_TOKEN_LIMIT,
            window="1m",
        )
        access_b = _serve_in_tenant(
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
        _require_client_name_alias(admin_client=admin_client, model=model, case=env["tenant_a"])

        # Drive tenant A past its budget: at least one 200 must precede the
        # 429 (a fresh budget always admits the first request — see
        # the rate-limit isolation module's _exhaust_until_429).
        successes = 0
        limited = None
        deadline = time.time() + 90
        while time.time() < deadline:
            limited = infer_external_chat(
                gateway_url=env["tenant_a"]["gateway_url"],
                model_namespace=model["namespace"],
                model_client_name=model["client_name"],
                api_key=quota_model["access_a"]["api_key"],
                path_tag=model["path_tag"],
            )
            if limited.status_code == 200:
                successes += 1
            elif limited.status_code == 429:
                if successes:
                    break
            else:
                raise AssertionError(f"unexpected inference response on tenant A: {redact_sensitive(limited.text)}")
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
        )
        _assert_returns_usage(response=response)

        # And tenant A stays limited while tenant B keeps serving.
        still_limited = infer_external_chat(
            gateway_url=env["tenant_a"]["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=quota_model["access_a"]["api_key"],
            path_tag=model["path_tag"],
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
        )
        _assert_returns_usage(response=again_b)
