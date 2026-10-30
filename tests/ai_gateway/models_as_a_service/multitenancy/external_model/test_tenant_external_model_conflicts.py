"""E2E: model-name conflicts and same-named objects across namespaces.

Ported from models-as-a-service ``test/e2e/tests/test_tenant_external_model_conflicts.py``
(RHAI-12994 P0 scenarios), adapted to this repository's wrapper/fixture style.

P0 scenarios:

7. Two ExternalModels with the same client-facing modelName attached to one
   gateway from different namespaces: the tenant's own model always wins,
   otherwise the oldest; requests reach only the winner's provider and the
   loser reports ModelNameConflict.
8. Same-named ExternalModels and ExternalProviders in two user namespaces
   attached to one gateway both serve correctly, with no overlay stable_id or
   DestinationRule collision (regression coverage for the shared-provider
   overlay collision bug).
9. A broken foreign model in another namespace (provider credential Secret
   missing) never breaks the tenant's own models on the same gateway.

Conflict pairs share the CR name across namespaces (the client-facing name
defaults to it), which keeps the client-facing name equal to the
MaaSModelRef name: path-based inference stays authorizable that way,
because the gateway auth matches ``{namespace}/{name}`` path identities
against MaaSModelRef names. The same-named scenario also needs distinct
client-facing names (a shared one is the conflict above), and models whose
client-facing name differs from their CRD name cannot be called on this
stack — the auth resolves the CRD name (status.resolvedModelAlias) while
the routes match the client-facing name — so its serving test proves the
default-named model with a real request and skips the overridden model's
request until that gap is fixed (RHOAIENG-101609).

Conflict CR names are unique per test: a previous test's models live until
module teardown, and a tenant-own model pre-holds its client-facing name
for the whole gateway — a shared name would make every later pair lose.

Shares the tenant_external_model session tenants (see conftest.tenant_external_model_tenants);
this module creates its own user namespaces with unique names.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    EXTERNAL_ENDPOINT,
    admitted_user_namespace,
    assert_served_by_provider,
    attachment_httproutes,
    destination_rules_referencing,
    external_model_binding,
    external_model_stack,
    gateway_ref,
    infer_external_chat,
    maas_model_ref_alias,
    redact_sensitive,
    route_is_for_model,
    serve_external_model_in_tenant,
    wait_for_attachment_httproute,
    wait_for_gateway_attachment_not_ready,
    wait_for_gateway_attachment_reason,
    wait_for_no_attachment_httproute,
    wait_for_not_found,
)
from utilities.constants import MAAS_GATEWAY_NAMESPACE
from utilities.resources.external_model import ExternalModel
from utilities.resources.external_provider import ExternalProvider

LOGGER = structlog.get_logger(name=__name__)

# The source module serializes with the tenant_ipp xdist group until the
# legacy IPP controllers skip gatewayRefs ExternalModels (image pin
# pending): without that guard they adopt this suite's models and flap
# status.httpRouteName. This repository has no xdist_group marker; the same
# serialization must be arranged through CI scheduling instead.


def _conflict_cr_name() -> str:
    """A CR name shared by one test's conflict pair, unique across tests."""
    return f"e2e-cfl-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def conflicts_env(
    admin_client: DynamicClient,
    tenant_external_model_tenants: dict[str, dict[str, str]],
) -> Generator[dict[str, Any], Any]:
    """Tenant A's case, two admitted user namespaces, and a module-wide resource stack."""
    tenant_a = tenant_external_model_tenants["tenant_a"]
    # Every model/policy created for this module is entered into one stack so
    # it lives until module teardown (the source tracked the same set in
    # TenantExternalModelResources): a tenant-own model pre-holds its
    # client-facing name on the gateway for its whole lifetime.
    stack = ExitStack()
    with stack:
        user_ns_1 = stack.enter_context(
            cm=admitted_user_namespace(admin_client, "e2e-extmod-cfl-1", tenant_a["gateway_name"])
        )
        user_ns_2 = stack.enter_context(
            cm=admitted_user_namespace(admin_client, "e2e-extmod-cfl-2", tenant_a["gateway_name"])
        )
        yield {
            "tenant_a": tenant_a,
            "user_ns_1": user_ns_1.name,
            "user_ns_2": user_ns_2.name,
            "stack": stack,
        }


def _provision(env: dict[str, Any], admin_client: DynamicClient, **kwargs: Any) -> dict[str, Any]:
    """Create one external model serving stack that lives until module teardown."""
    return env["stack"].enter_context(cm=external_model_stack(admin_client, **kwargs))


def _serve_in_tenant(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token: str,
    env: dict[str, Any],
    case: dict[str, str],
    model: dict[str, Any],
    **kwargs: Any,
) -> dict[str, str]:
    """Serve one model under a tenant and keep the policy/subscription until module teardown."""
    # Subscription and policy names must be unique within the tenant
    # namespace: same-named models in different namespaces would otherwise
    # share f"{name}-sub" and the second apply would clobber the first
    # (orphaning the first key).
    ns_suffix = model["namespace"].rsplit("-", 1)[-1]
    return env["stack"].enter_context(
        cm=serve_external_model_in_tenant(
            admin_client,
            request_session_http,
            ocp_token,
            tenant=case,
            model_name=model["name"],
            model_namespace=model["namespace"],
            policy_name=f"{model['name']}-{ns_suffix}-access",
            subscription_name=f"{model['name']}-{ns_suffix}-sub",
            **kwargs,
        )
    )


class TestModelNameConflict:
    """Same client-facing modelName on one gateway from two namespaces."""

    @pytest.mark.tier2
    def test_tenant_own_model_wins_over_older_user_namespace_model(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        conflicts_env: dict[str, Any],
    ) -> None:
        """Given the same client-facing modelName in a user namespace and the tenant's
        own namespace, when both attach to the tenant gateway, then the tenant's own
        model wins even though the user-namespace model is older; the loser reports
        ModelNameConflict and never receives traffic."""
        env = conflicts_env
        tenant_a = env["tenant_a"]
        gateway = gateway_ref(gateway_name=tenant_a["gateway_name"])
        # One shared CR name — and therefore client-facing name — across a
        # user namespace and the tenant namespace.
        name = _conflict_cr_name()

        # The older model lives in a user namespace...
        older = _provision(
            env=env,
            admin_client=admin_client,
            name=name,
            namespace=env["user_ns_1"],
            gateway_refs=[gateway],
        )
        # Guarantee the challenger is strictly older: creationTimestamp has
        # one-second resolution, and on a tie the AGC falls back to
        # namespace/name ordering — the test would risk passing through the
        # tiebreaker instead of the tenant-own rule.
        time.sleep(2)
        # ...the younger model lives in the tenant's own namespace.
        tenant_own = _provision(
            env=env,
            admin_client=admin_client,
            name=name,
            namespace=tenant_a["tenant_ns"],
            gateway_refs=[gateway],
        )
        access = _serve_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            case=tenant_a,
            model=tenant_own,
        )

        wait_for_gateway_attachment_reason(
            admin_client=admin_client,
            model_name=older["name"],
            model_namespace=older["namespace"],
            gateway_name=tenant_a["gateway_name"],
            reason="ModelNameConflict",
        )
        # The conflict contract's serving effect: the loser is skipped by
        # admission, so it must hold no route on the gateway (matched by
        # route name — the namespace hosts other tests' models).
        wait_for_no_attachment_httproute(
            admin_client=admin_client,
            route_namespace=older["namespace"],
            gateway_name=tenant_a["gateway_name"],
            model_name=older["name"],
        )
        # The winner serves with its own credential.
        response = infer_external_chat(
            gateway_url=tenant_a["gateway_url"],
            model_namespace=tenant_own["namespace"],
            model_client_name=tenant_own["client_name"],
            api_key=access["api_key"],
            path_tag=tenant_own["path_tag"],
        )
        assert_served_by_provider(
            response=response, path_tag=tenant_own["path_tag"], credential=tenant_own["credential"]
        )
        # Defense in depth only: this key is bound to the winner's
        # subscription, so auth denies the loser's path identity whether or
        # not a route lingers — the route absence above is the actual
        # conflict-suppression proof.
        denied = infer_external_chat(
            gateway_url=tenant_a["gateway_url"],
            model_namespace=older["namespace"],
            model_client_name=older["client_name"],
            api_key=access["api_key"],
            path_tag=older["path_tag"],
            retries=1,
        )
        assert denied.status_code != 200, (
            f"the conflicted loser must not serve, got {denied.status_code}: {redact_sensitive(denied.text)}"
        )

    @pytest.mark.tier2
    def test_oldest_wins_when_no_tenant_own_model(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        conflicts_env: dict[str, Any],
    ) -> None:
        """Given the same client-facing modelName in two user namespaces, when both
        attach to the tenant gateway and the tenant has no own model of that name,
        then the oldest model wins and the youngest reports ModelNameConflict."""
        env = conflicts_env
        tenant_a = env["tenant_a"]
        gateway = gateway_ref(gateway_name=tenant_a["gateway_name"])
        name = _conflict_cr_name()

        oldest = _provision(
            env=env,
            admin_client=admin_client,
            name=name,
            namespace=env["user_ns_1"],
            gateway_refs=[gateway],
        )
        # Guarantee distinct creation timestamps: the AGC decides the winner
        # by creation time (then namespace/name), and "oldest wins" is the
        # behavior under test.
        time.sleep(2)
        youngest = _provision(
            env=env,
            admin_client=admin_client,
            name=name,
            namespace=env["user_ns_2"],
            gateway_refs=[gateway],
        )
        access = _serve_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            case=tenant_a,
            model=oldest,
        )

        wait_for_gateway_attachment_reason(
            admin_client=admin_client,
            model_name=youngest["name"],
            model_namespace=youngest["namespace"],
            gateway_name=tenant_a["gateway_name"],
            reason="ModelNameConflict",
        )
        # The conflict contract's serving effect: the loser is skipped by
        # admission, so it must hold no route on the gateway (matched by
        # route name — the namespace hosts other tests' models).
        wait_for_no_attachment_httproute(
            admin_client=admin_client,
            route_namespace=youngest["namespace"],
            gateway_name=tenant_a["gateway_name"],
            model_name=youngest["name"],
        )
        response = infer_external_chat(
            gateway_url=tenant_a["gateway_url"],
            model_namespace=oldest["namespace"],
            model_client_name=oldest["client_name"],
            api_key=access["api_key"],
            path_tag=oldest["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=oldest["path_tag"], credential=oldest["credential"])
        # Defense in depth only: this key is bound to the winner's
        # subscription, so auth denies the loser's path identity whether or
        # not a route lingers — the route absence above is the actual
        # conflict-suppression proof.
        denied = infer_external_chat(
            gateway_url=tenant_a["gateway_url"],
            model_namespace=youngest["namespace"],
            model_client_name=youngest["client_name"],
            api_key=access["api_key"],
            path_tag=youngest["path_tag"],
            retries=1,
        )
        assert denied.status_code != 200, f"the conflicted loser must not serve, got {denied.status_code}"


class TestSameNamedObjectsTwoNamespaces:
    """Same-named ExternalModels/Providers/Secrets in two user namespaces on
    one gateway set up independent serving stacks without colliding (overlay
    collision regression)."""

    @pytest.fixture(scope="class")
    def same_named_pair(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        conflicts_env: dict[str, Any],
    ) -> dict[str, Any]:
        """Two same-named model stacks in two user namespaces, both admitted and served."""
        env = conflicts_env
        tenant_a = env["tenant_a"]
        gateway = gateway_ref(gateway_name=tenant_a["gateway_name"])
        shared_cr_name = f"e2e-cfl-same-{uuid.uuid4().hex[:6]}"
        shared_secret_name = f"{shared_cr_name}-provider-cred"

        # Identical CR, provider, and Secret names in both namespaces.
        # Distinct client-facing modelNames (a shared client name would be
        # the ModelNameConflict scenario instead): model 1 keeps the default
        # name (client == CRD name), model 2 overrides — see the serving
        # test for what that costs today.
        # Model 2 is created and admitted first: on a build whose attachment
        # stable_ids collide (the overlay bug this test regresses) the first
        # attachment keeps the shared id, so the callable model 1 would lose
        # its route and fail the serving proof — with model 1 first, the
        # collision would pass silently.
        model_2 = _provision(
            env=env,
            admin_client=admin_client,
            name=shared_cr_name,
            namespace=env["user_ns_2"],
            gateway_refs=[gateway],
            model_name=f"{shared_cr_name}-two",
            provider_name=shared_cr_name,
            credential_secret_name=shared_secret_name,
        )
        wait_for_attachment_httproute(
            admin_client=admin_client,
            route_namespace=model_2["namespace"],
            gateway_name=tenant_a["gateway_name"],
            model_name=model_2["name"],
        )
        model_1 = _provision(
            env=env,
            admin_client=admin_client,
            name=shared_cr_name,
            namespace=env["user_ns_1"],
            gateway_refs=[gateway],
            provider_name=shared_cr_name,
            credential_secret_name=shared_secret_name,
        )
        assert model_1["credential"] != model_2["credential"]
        assert model_1["path_tag"] != model_2["path_tag"]

        # Both models attach with their own admitted route and complete
        # their own MaaS stack (the subscriptions get unique names per
        # namespace — see _serve_in_tenant).
        for model in (model_1, model_2):
            wait_for_attachment_httproute(
                admin_client=admin_client,
                route_namespace=model["namespace"],
                gateway_name=tenant_a["gateway_name"],
                model_name=model["name"],
            )
        access_1 = _serve_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            case=tenant_a,
            model=model_1,
        )
        access_2 = _serve_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            case=tenant_a,
            model=model_2,
        )
        return {
            "shared_cr_name": shared_cr_name,
            "models": (model_1, model_2),
            "accesses": (access_1, access_2),
        }

    @pytest.mark.tier2
    def test_same_named_objects_both_attach(
        self,
        admin_client: DynamicClient,
        conflicts_env: dict[str, Any],
        same_named_pair: dict[str, Any],
    ) -> None:
        """Given same-named models in two user namespaces on one gateway, when both
        attach, then each holds its own admitted route and the tenant-published
        DestinationRules stay distinct objects (no shared-name collision)."""
        tenant_a = conflicts_env["tenant_a"]
        pair = same_named_pair

        for model in pair["models"]:
            routes = [
                route
                for route in attachment_httproutes(admin_client, model["namespace"], tenant_a["gateway_name"])
                if route_is_for_model(route, model["name"])
            ]
            assert len(routes) == 1, (
                f"expected one route for {model['namespace']}/{model['name']}, "
                f"got {[route['metadata']['name'] for route in routes]}"
            )

        # No DestinationRule collision: AGC keeps each model's provider
        # DestinationRule in the Gateway namespace under a name hashed over
        # the gateway and the provider's namespace, so the same-named
        # providers in two model namespaces must stay distinct objects.
        rules = destination_rules_referencing(  # noqa: FCN001 - namespace feeds *args
            admin_client, pair["shared_cr_name"], MAAS_GATEWAY_NAMESPACE
        )
        rule_uids = {rule["metadata"]["uid"] for rule in rules}
        assert len(rule_uids) >= 2, (
            f"expected distinct DestinationRules per same-named model in {MAAS_GATEWAY_NAMESPACE}, got uids {rule_uids}"
        )

    @pytest.mark.tier2
    def test_same_named_objects_both_serve(
        self,
        admin_client: DynamicClient,
        conflicts_env: dict[str, Any],
        same_named_pair: dict[str, Any],
    ) -> None:
        """Given same-named models in two user namespaces, when calling each through
        the tenant gateway, then both serve with their own credentials.

        Model 1's client-facing name equals its CRD name, so it is callable
        through the normal path contract. Model 2's client-facing name
        differs from its CRD name, which the gateway auth cannot resolve on
        this stack: the controller publishes status.resolvedModelAlias from
        the ExternalModel CRD name while the routes match the client-facing
        name, so overridden modelNames are unreachable end to end. Its
        request proof is skipped until the alias carries the client-facing
        name (RHOAIENG-101609).
        """
        tenant_a = conflicts_env["tenant_a"]
        (model_1, model_2) = same_named_pair["models"]
        (access_1, access_2) = same_named_pair["accesses"]

        response = infer_external_chat(
            gateway_url=tenant_a["gateway_url"],
            model_namespace=model_1["namespace"],
            model_client_name=model_1["client_name"],
            api_key=access_1["api_key"],
            path_tag=model_1["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=model_1["path_tag"], credential=model_1["credential"])

        alias = maas_model_ref_alias(admin_client=admin_client, namespace=model_2["namespace"], name=model_2["name"])
        if alias != model_2["client_name"]:
            pytest.skip(
                "client-facing model names are not resolvable through the gateway "
                f"yet: MaaSModelRef {model_2['namespace']}/{model_2['name']} "
                f"status.resolvedModelAlias={alias!r} != {model_2['client_name']!r}"
            )
        response = infer_external_chat(
            gateway_url=tenant_a["gateway_url"],
            model_namespace=model_2["namespace"],
            model_client_name=model_2["client_name"],
            api_key=access_2["api_key"],
            path_tag=model_2["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=model_2["path_tag"], credential=model_2["credential"])


class TestForeignBrokenModelIsolation:
    """A broken model from another namespace must not take down the tenant's
    own models on the same gateway.

    The broken model binds to a provider whose credential Secret does not
    exist, so its attachment can never become serving. The tenant's own
    healthy model on the same gateway must keep serving both while the
    broken model exists and after it is deleted."""

    @pytest.fixture(scope="class")
    def broken_pair(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        conflicts_env: dict[str, Any],
    ) -> dict[str, Any]:
        """A healthy served model plus a broken foreign model on the same gateway."""
        env = conflicts_env
        tenant_a = env["tenant_a"]
        gateway = gateway_ref(gateway_name=tenant_a["gateway_name"])

        healthy = _provision(
            env=env,
            admin_client=admin_client,
            name=f"e2e-cfl-ok-{uuid.uuid4().hex[:6]}",
            namespace=env["user_ns_1"],
            gateway_refs=[gateway],
        )
        access = _serve_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            case=tenant_a,
            model=healthy,
        )

        # The broken foreign model: its provider references a Secret that is
        # never created, so the attachment cannot deliver a credential or
        # publish a working route. Same gateway, different namespace.
        broken_name = f"e2e-cfl-broken-{uuid.uuid4().hex[:6]}"
        broken_provider = f"{broken_name}-provider"
        env["stack"].enter_context(
            cm=ExternalProvider(
                client=admin_client,
                name=broken_provider,
                namespace=env["user_ns_2"],
                provider="openai",
                endpoint=EXTERNAL_ENDPOINT,
                auth={"type": "apikey", "secretRef": {"name": f"{broken_name}-missing-cred"}},
                teardown=True,
                wait_for_resource=True,
            )
        )
        broken_model = env["stack"].enter_context(
            cm=ExternalModel(
                client=admin_client,
                name=broken_name,
                namespace=env["user_ns_2"],
                external_provider_refs=[external_model_binding(provider_name=broken_provider)],
                gateway_refs=[gateway],
                teardown=True,
                wait_for_resource=True,
            )
        )
        return {
            "healthy": healthy,
            "access": access,
            "broken_name": broken_name,
            "broken_model": broken_model,
        }

    @pytest.mark.tier2
    def test_broken_foreign_model_does_not_break_tenant_models(
        self,
        admin_client: DynamicClient,
        conflicts_env: dict[str, Any],
        broken_pair: dict[str, Any],
    ) -> None:
        """Given a broken foreign attachment on the tenant's gateway, when it fails,
        then the tenant's own healthy model on the same gateway keeps serving."""
        env = conflicts_env
        tenant_a = env["tenant_a"]
        broken_ns = env["user_ns_2"]
        wait_for_gateway_attachment_not_ready(
            admin_client=admin_client,
            model_name=broken_pair["broken_name"],
            model_namespace=broken_ns,
            gateway_name=tenant_a["gateway_name"],
        )
        healthy = broken_pair["healthy"]
        response = infer_external_chat(
            gateway_url=tenant_a["gateway_url"],
            model_namespace=healthy["namespace"],
            model_client_name=healthy["client_name"],
            api_key=broken_pair["access"]["api_key"],
            path_tag=healthy["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=healthy["path_tag"], credential=healthy["credential"])

    @pytest.mark.tier2
    def test_tenant_models_serve_after_broken_model_deleted(
        self,
        admin_client: DynamicClient,
        conflicts_env: dict[str, Any],
        broken_pair: dict[str, Any],
    ) -> None:
        """Given the broken foreign model, when it is deleted, then its failed
        attachment cleans up without disturbing the tenant's own serving model."""
        env = conflicts_env
        tenant_a = env["tenant_a"]
        broken_ns = env["user_ns_2"]
        broken_pair["broken_model"].delete()
        # delete() only fails loudly on an API error: confirm the object is
        # gone before the absence checks, so a failed deletion cannot pass
        # vacuously (the broken model may have held no route to begin with).
        wait_for_not_found(
            client=admin_client, kind="externalmodel", name=broken_pair["broken_name"], namespace=broken_ns
        )
        wait_for_no_attachment_httproute(
            admin_client=admin_client,
            route_namespace=broken_ns,
            gateway_name=tenant_a["gateway_name"],
            model_name=broken_pair["broken_name"],
        )
        healthy = broken_pair["healthy"]
        response = infer_external_chat(
            gateway_url=tenant_a["gateway_url"],
            model_namespace=healthy["namespace"],
            model_client_name=healthy["client_name"],
            api_key=broken_pair["access"]["api_key"],
            path_tag=healthy["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=healthy["path_tag"], credential=healthy["credential"])
