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
from contextlib import ExitStack
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    assert_served_by_provider,
    attachment_httproutes,
    conflict_cr_name,
    destination_rules_referencing,
    gateway_ref,
    infer_for_shared_model,
    maas_model_ref_alias,
    provision_model_in_stack,
    redact_sensitive,
    route_is_for_model,
    serve_in_tenant_with_ns_suffix,
    wait_for_gateway_attachment_not_ready,
    wait_for_gateway_attachment_reason,
    wait_for_no_attachment_httproute,
    wait_for_not_found,
)
from utilities.constants import MAAS_GATEWAY_NAMESPACE

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
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
        with ExitStack() as stack:
            env = conflicts_env
            tenant_a = env["tenant_a"]
            gateway = gateway_ref(gateway_name=tenant_a["gateway_name"])
            name = conflict_cr_name()

            older = provision_model_in_stack(
                stack=stack,
                admin_client=admin_client,
                name=name,
                namespace=env["user_ns_1"],
                gateway_refs=[gateway],
            )
            time.sleep(2)
            tenant_own = provision_model_in_stack(
                stack=stack,
                admin_client=admin_client,
                name=name,
                namespace=tenant_a["tenant_ns"],
                gateway_refs=[gateway],
            )
            access = serve_in_tenant_with_ns_suffix(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                stack=stack,
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
            wait_for_no_attachment_httproute(
                admin_client=admin_client,
                route_namespace=older["namespace"],
                gateway_name=tenant_a["gateway_name"],
                model_name=older["name"],
            )
            response = infer_for_shared_model(
                request_session_http=request_session_http,
                case=tenant_a,
                model=tenant_own,
                api_key=access["api_key"],
            )
            assert_served_by_provider(
                response=response, path_tag=tenant_own["path_tag"], credential=tenant_own["credential"]
            )
            denied = infer_for_shared_model(
                request_session_http=request_session_http,
                case=tenant_a,
                model=older,
                api_key=access["api_key"],
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
        with ExitStack() as stack:
            env = conflicts_env
            tenant_a = env["tenant_a"]
            gateway = gateway_ref(gateway_name=tenant_a["gateway_name"])
            name = conflict_cr_name()

            oldest = provision_model_in_stack(
                stack=stack,
                admin_client=admin_client,
                name=name,
                namespace=env["user_ns_1"],
                gateway_refs=[gateway],
            )
            time.sleep(2)
            youngest = provision_model_in_stack(
                stack=stack,
                admin_client=admin_client,
                name=name,
                namespace=env["user_ns_2"],
                gateway_refs=[gateway],
            )
            access = serve_in_tenant_with_ns_suffix(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                stack=stack,
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
            wait_for_no_attachment_httproute(
                admin_client=admin_client,
                route_namespace=youngest["namespace"],
                gateway_name=tenant_a["gateway_name"],
                model_name=youngest["name"],
            )
            response = infer_for_shared_model(
                request_session_http=request_session_http,
                case=tenant_a,
                model=oldest,
                api_key=access["api_key"],
            )
            assert_served_by_provider(response=response, path_tag=oldest["path_tag"], credential=oldest["credential"])
            denied = infer_for_shared_model(
                request_session_http=request_session_http,
                case=tenant_a,
                model=youngest,
                api_key=access["api_key"],
                retries=1,
            )
            assert denied.status_code != 200, f"the conflicted loser must not serve, got {denied.status_code}"


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestSameNamedObjectsTwoNamespaces:
    """Same-named ExternalModels/Providers/Secrets in two user namespaces on
    one gateway set up independent serving stacks without colliding (overlay
    collision regression)."""

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
        request_session_http: requests.Session,
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

        response = infer_for_shared_model(
            request_session_http=request_session_http,
            case=tenant_a,
            model=model_1,
            api_key=access_1["api_key"],
        )
        assert_served_by_provider(response=response, path_tag=model_1["path_tag"], credential=model_1["credential"])

        alias = maas_model_ref_alias(admin_client=admin_client, namespace=model_2["namespace"], name=model_2["name"])
        if alias != model_2["client_name"]:
            pytest.skip(
                "client-facing model names are not resolvable through the gateway "
                f"yet: MaaSModelRef {model_2['namespace']}/{model_2['name']} "
                f"status.resolvedModelAlias={alias!r} != {model_2['client_name']!r}"
            )
        response = infer_for_shared_model(
            request_session_http=request_session_http,
            case=tenant_a,
            model=model_2,
            api_key=access_2["api_key"],
        )
        assert_served_by_provider(response=response, path_tag=model_2["path_tag"], credential=model_2["credential"])


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestForeignBrokenModelIsolation:
    """A broken model from another namespace must not take down the tenant's
    own models on the same gateway.

    The broken model binds to a provider whose credential Secret does not
    exist, so its attachment can never become serving. The tenant's own
    healthy model on the same gateway must keep serving both while the
    broken model exists and after it is deleted."""

    @pytest.mark.tier2
    def test_broken_foreign_model_does_not_break_tenant_models(
        self,
        request_session_http: requests.Session,
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
        response = infer_for_shared_model(
            request_session_http=request_session_http,
            case=tenant_a,
            model=healthy,
            api_key=broken_pair["access"]["api_key"],
        )
        assert_served_by_provider(response=response, path_tag=healthy["path_tag"], credential=healthy["credential"])

    @pytest.mark.tier2
    def test_tenant_models_serve_after_broken_model_deleted(
        self,
        request_session_http: requests.Session,
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
        response = infer_for_shared_model(
            request_session_http=request_session_http,
            case=tenant_a,
            model=healthy,
            api_key=broken_pair["access"]["api_key"],
        )
        assert_served_by_provider(response=response, path_tag=healthy["path_tag"], credential=healthy["credential"])
