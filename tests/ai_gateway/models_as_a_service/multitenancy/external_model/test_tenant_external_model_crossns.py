"""E2E: ReferenceGrant-gated cross-namespace ExternalProvider sharing (P1).

Ported from models-as-a-service ``test/e2e/tests/test_tenant_external_model_crossns.py``
(RHOAIENG-90804), adapted to this repository's wrapper/fixture style.

P1 scenarios:

1. A model referencing an ExternalProvider in another namespace serves
   requests when a matching ReferenceGrant permits it.
2. A missing/non-matching grant rejects the reference descriptively; adding
   permission restores serving.
3. Revoking/narrowing the grant stops access and updates status and runtime
   configuration.
4. A cross-namespace provider uses its provider-local Secret by default and
   the model-local Secret when auth is overridden; a missing model-local
   override must fail without provider-namespace fallback.
5. A model referencing same-named providers from two namespaces serves
   through both.

Runs wherever the AGC ReferenceGrant validation and its RBAC are packaged
(capability-gated on the ReferenceGrant CRD; RHOAIENG-90804 is done).

Shares the tenant_external_model session tenants (see
conftest.tenant_external_model_tenants); this module creates its own user
namespaces with unique names.
"""

from __future__ import annotations

import uuid
from contextlib import ExitStack
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    assert_served_by_provider,
    crossns_model,
    crossns_model_with_provider,
    crossns_namespace,
    crossns_provider,
    crossns_secret_with_credential,
    crossns_twin_model,
    get_json_or_none,
    infer_external_chat,
    infer_for_shared_model,
    redact_sensitive,
    reference_grant_for,
    revoke_reference_grant,
    serve_model_in_tenant_stack,
    wait_for_credential_copy,
    wait_for_gateway_attachment_not_ready,
    wait_for_gateway_attachment_ready,
    wait_for_no_credential_copy,
)

LOGGER = structlog.get_logger(name=__name__)

GRANT_FAILURE_PATTERN = r"grant|authoriz|not allowed|permission|forbidden|reference"
MISSING_SECRET_PATTERN = r"secret"


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestCrossNsProviderGrant:
    """A ReferenceGrant in the provider namespace permits serving."""

    @pytest.mark.tier2
    def test_grant_permits_cross_namespace_provider(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        crossns_env: dict[str, Any],
    ) -> None:
        """Given a model referencing a provider in another namespace, when a
        matching ReferenceGrant exists, then the attachment turns ready, the
        provider-local credential is copied into the tenant namespace, and the
        model serves with that credential."""
        with ExitStack() as stack:
            env = crossns_env
            model = crossns_model(
                admin_client=admin_client, env=env, stack=stack, name=f"e2e-crossns-grant-{uuid.uuid4().hex[:6]}"
            )
            reference_grant_for(admin_client=admin_client, env=env, stack=stack, model=model)
            access = serve_model_in_tenant_stack(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                env=env,
                stack=stack,
                model=model,
            )

            wait_for_gateway_attachment_ready(
                admin_client=admin_client,
                model_name=model["name"],
                model_namespace=model["namespace"],
                gateway_name=env["tenant"]["gateway_name"],
            )
            wait_for_credential_copy(
                admin_client=admin_client,
                tenant_namespace=env["tenant"]["tenant_ns"],
                source_secret_name=model["secret_name"],
                source_namespace=model["secret_namespace"],
            )
            response = infer_for_shared_model(
                request_session_http=request_session_http,
                case=env["tenant"],
                model=model,
                api_key=access["api_key"],
            )
            assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestMissingGrant:
    """A missing or non-matching grant rejects the reference descriptively;
    adding permission restores serving."""

    @pytest.mark.tier2
    def test_missing_grant_rejects_then_restores(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        crossns_env: dict[str, Any],
    ) -> None:
        """Given a cross-ns model with no grant, when it is created and then a
        grant is added, then the reference is first rejected descriptively and
        afterwards serves normally."""
        with ExitStack() as stack:
            env = crossns_env
            model = crossns_model(
                admin_client=admin_client, env=env, stack=stack, name=f"e2e-crossns-missing-{uuid.uuid4().hex[:6]}"
            )
            gateway = env["tenant"]["gateway_name"]

            wait_for_gateway_attachment_not_ready(
                admin_client=admin_client,
                model_name=model["name"],
                model_namespace=model["namespace"],
                gateway_name=gateway,
                message_pattern=GRANT_FAILURE_PATTERN,
            )

            reference_grant_for(admin_client=admin_client, env=env, stack=stack, model=model)
            access = serve_model_in_tenant_stack(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                env=env,
                stack=stack,
                model=model,
            )
            wait_for_gateway_attachment_ready(
                admin_client=admin_client,
                model_name=model["name"],
                model_namespace=model["namespace"],
                gateway_name=gateway,
            )
            response = infer_for_shared_model(
                request_session_http=request_session_http,
                case=env["tenant"],
                model=model,
                api_key=access["api_key"],
            )
            assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])

    @pytest.mark.tier2
    def test_non_matching_grant_rejects(
        self,
        admin_client: DynamicClient,
        crossns_env: dict[str, Any],
    ) -> None:
        """Given a cross-ns model and a grant for a different source namespace,
        when the model attaches, then the reference stays rejected with an
        authorization failure."""
        with ExitStack() as stack:
            env = crossns_env
            model = crossns_model(
                admin_client=admin_client, env=env, stack=stack, name=f"e2e-crossns-wronggr-{uuid.uuid4().hex[:6]}"
            )
            gateway = env["tenant"]["gateway_name"]
            reference_grant_for(
                admin_client=admin_client,
                env=env,
                stack=stack,
                model=model,
                name_suffix="-wrong",
                from_namespace=f"{model['namespace']}-other",
            )

            wait_for_gateway_attachment_not_ready(
                admin_client=admin_client,
                model_name=model["name"],
                model_namespace=model["namespace"],
                gateway_name=gateway,
                message_pattern=GRANT_FAILURE_PATTERN,
            )


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestGrantRevocation:
    """Revoking or narrowing the grant stops access and updates status and
    runtime configuration."""

    @pytest.mark.tier2
    def test_revoke_stops_access(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        crossns_env: dict[str, Any],
    ) -> None:
        """Given a granted and serving cross-ns model, when the grant is
        deleted, then the attachment fails with an authorization message, the
        credential copy is withdrawn, and inference is denied."""
        with ExitStack() as stack:
            env = crossns_env
            model = crossns_model(
                admin_client=admin_client, env=env, stack=stack, name=f"e2e-crossns-revoke-{uuid.uuid4().hex[:6]}"
            )
            gateway = env["tenant"]["gateway_name"]
            grant = reference_grant_for(admin_client=admin_client, env=env, stack=stack, model=model)
            access = serve_model_in_tenant_stack(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                env=env,
                stack=stack,
                model=model,
            )
            wait_for_gateway_attachment_ready(
                admin_client=admin_client,
                model_name=model["name"],
                model_namespace=model["namespace"],
                gateway_name=gateway,
            )

            response = infer_for_shared_model(
                request_session_http=request_session_http,
                case=env["tenant"],
                model=model,
                api_key=access["api_key"],
            )
            assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])

            revoke_reference_grant(grant=grant)
            wait_for_gateway_attachment_not_ready(
                admin_client=admin_client,
                model_name=model["name"],
                model_namespace=model["namespace"],
                gateway_name=gateway,
                message_pattern=GRANT_FAILURE_PATTERN,
            )
            wait_for_no_credential_copy(
                admin_client=admin_client,
                tenant_namespace=env["tenant"]["tenant_ns"],
                source_secret_name=model["secret_name"],
                source_namespace=model["secret_namespace"],
            )
            denied = infer_for_shared_model(
                request_session_http=request_session_http,
                case=env["tenant"],
                model=model,
                api_key=access["api_key"],
                retries=1,
            )
            assert denied.status_code != 200, (
                f"inference must stop after the grant is revoked, "
                f"got {denied.status_code}: {redact_sensitive(denied.text)}"
            )

    @pytest.mark.tier2
    def test_narrowed_grant_stops_access(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        crossns_env: dict[str, Any],
    ) -> None:
        """Given a granted and serving cross-ns model, when the grant is
        replaced by one covering a different source namespace, then the
        attachment fails with an authorization message and inference stops."""
        with ExitStack() as stack:
            env = crossns_env
            model = crossns_model(
                admin_client=admin_client, env=env, stack=stack, name=f"e2e-crossns-narrow-{uuid.uuid4().hex[:6]}"
            )
            gateway = env["tenant"]["gateway_name"]
            grant = reference_grant_for(admin_client=admin_client, env=env, stack=stack, model=model)
            access = serve_model_in_tenant_stack(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                env=env,
                stack=stack,
                model=model,
            )
            wait_for_gateway_attachment_ready(
                admin_client=admin_client,
                model_name=model["name"],
                model_namespace=model["namespace"],
                gateway_name=gateway,
            )
            # Positive control: prove serving works before the narrowing, so
            # the post-narrowing denial is a verdict on the grant change and
            # not on a serving path that never worked.
            assert_served_by_provider(
                response=infer_for_shared_model(
                    request_session_http=request_session_http,
                    case=env["tenant"],
                    model=model,
                    api_key=access["api_key"],
                ),
                path_tag=model["path_tag"],
                credential=model["credential"],
            )

            revoke_reference_grant(grant=grant)
            reference_grant_for(
                admin_client=admin_client,
                env=env,
                stack=stack,
                model=model,
                name_suffix="-narrowed",
                from_namespace=f"{model['namespace']}-other",
            )
            wait_for_gateway_attachment_not_ready(
                admin_client=admin_client,
                model_name=model["name"],
                model_namespace=model["namespace"],
                gateway_name=gateway,
                message_pattern=GRANT_FAILURE_PATTERN,
            )
            denied = infer_for_shared_model(
                request_session_http=request_session_http,
                case=env["tenant"],
                model=model,
                api_key=access["api_key"],
                retries=1,
            )
            assert denied.status_code != 200, (
                f"inference must stop after the grant is narrowed, got {denied.status_code}"
            )


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestCredentialSourceSelection:
    """Provider-local Secret by default, model-local Secret on override, and
    no provider-namespace fallback for a missing override."""

    @pytest.mark.tier2
    def test_provider_local_credential_used_by_default(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        crossns_env: dict[str, Any],
    ) -> None:
        """Given same-named Secrets with different values in the provider and
        model namespaces, when a cross-ns model serves without and then with an
        auth override, then the provider-local value is injected by default and
        the model-local value once overridden."""
        with ExitStack() as stack:
            env = crossns_env
            shared_credential = "shared-cred"
            provider_credential = crossns_secret_with_credential(
                admin_client=admin_client,
                stack=stack,
                env=env,
                name=shared_credential,
                namespace=env["provider_ns"],
            )
            model_credential = crossns_secret_with_credential(
                admin_client=admin_client,
                stack=stack,
                env=env,
                name=shared_credential,
                namespace=env["model_ns"],
            )
            provider_name = f"e2e-crossns-credsel-{uuid.uuid4().hex[:6]}-provider"
            crossns_provider(
                admin_client=admin_client,
                stack=stack,
                env=env,
                name=provider_name,
                namespace=env["provider_ns"],
                secret_name=shared_credential,
                credential=provider_credential,
            )

            default_model = crossns_model_with_provider(
                admin_client=admin_client,
                env=env,
                stack=stack,
                name=f"e2e-crossns-creddef-{uuid.uuid4().hex[:6]}",
                provider_name=provider_name,
            )
            reference_grant_for(admin_client=admin_client, env=env, stack=stack, model=default_model)
            access = serve_model_in_tenant_stack(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                env=env,
                stack=stack,
                model=default_model,
            )
            response = infer_for_shared_model(
                request_session_http=request_session_http,
                case=env["tenant"],
                model=default_model,
                api_key=access["api_key"],
            )
            assert_served_by_provider(
                response=response, path_tag=default_model["path_tag"], credential=provider_credential
            )
            assert model_credential not in response.text, "model-local Secret must not be used without an auth override"

            override_model = crossns_model_with_provider(
                admin_client=admin_client,
                env=env,
                stack=stack,
                name=f"e2e-crossns-credovr-{uuid.uuid4().hex[:6]}",
                provider_name=provider_name,
                override_secret=shared_credential,
            )
            reference_grant_for(admin_client=admin_client, env=env, stack=stack, model=override_model)
            access = serve_model_in_tenant_stack(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                env=env,
                stack=stack,
                model=override_model,
            )
            response = infer_for_shared_model(
                request_session_http=request_session_http,
                case=env["tenant"],
                model=override_model,
                api_key=access["api_key"],
            )
            assert_served_by_provider(
                response=response, path_tag=override_model["path_tag"], credential=model_credential
            )
            assert provider_credential not in response.text, (
                "provider-local Secret must not be used when auth is overridden"
            )

    @pytest.mark.tier2
    def test_missing_model_local_override_fails_without_fallback(
        self,
        admin_client: DynamicClient,
        crossns_env: dict[str, Any],
    ) -> None:
        """Given a cross-ns model whose auth override points at a Secret that
        does not exist in the model namespace, when it attaches, then serving
        fails with a missing-Secret message instead of falling back to the
        provider-namespace Secret."""
        with ExitStack() as stack:
            env = crossns_env
            provider_name = f"e2e-crossns-missing-{uuid.uuid4().hex[:6]}-provider"
            crossns_provider(
                admin_client=admin_client,
                stack=stack,
                env=env,
                name=provider_name,
                namespace=env["provider_ns"],
            )

            # The override must resolve in the MODEL namespace only: the name
            # deliberately matches the provider-local Secret, so a resolver
            # that wrongly falls back to the provider namespace would find it
            # and serve instead of failing.
            missing_override = f"{provider_name}-cred"
            model = crossns_model_with_provider(
                admin_client=admin_client,
                env=env,
                stack=stack,
                name=f"e2e-crossns-nofb-{uuid.uuid4().hex[:6]}",
                provider_name=provider_name,
                override_secret=missing_override,
            )
            reference_grant_for(admin_client=admin_client, env=env, stack=stack, model=model)
            gateway = env["tenant"]["gateway_name"]
            wait_for_gateway_attachment_not_ready(
                admin_client=admin_client,
                model_name=model["name"],
                model_namespace=model["namespace"],
                gateway_name=gateway,
                message_pattern=MISSING_SECRET_PATTERN,
            )
            external_model = get_json_or_none(
                client=admin_client, kind="externalmodel", name=model["name"], namespace=model["namespace"]
            )
            assert external_model is not None, (
                f"externalmodel {model['namespace']}/{model['name']} must exist after the missing-secret failure"
            )
            phase = external_model["status"]["phase"]
            assert phase != "Ready", (
                f"model must report a non-Ready phase after the missing-secret failure, got {phase!r}"
            )


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestSameNamedProvidersTwoNamespaces:
    """A model referencing same-named providers from two namespaces serves
    through both."""

    @pytest.mark.tier2
    def test_same_named_providers_both_serve(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        crossns_env: dict[str, Any],
    ) -> None:
        """Given a model whose two provider refs point at same-named providers
        in two granted namespaces, when clients call it repeatedly, then both
        providers serve (identified by their injected credentials)."""
        with ExitStack() as stack:
            env = crossns_env
            case = env["tenant"]
            shared_provider = f"e2e-crossns-twin-{uuid.uuid4().hex[:6]}"
            client_tag = f"{shared_provider}-client"
            provider_namespaces = [
                crossns_namespace(admin_client=admin_client, stack=stack, env=env, prefix="e2e-extmod-crossns-p1"),
                crossns_namespace(admin_client=admin_client, stack=stack, env=env, prefix="e2e-extmod-crossns-p2"),
            ]
            credentials = [
                crossns_provider(
                    admin_client=admin_client,
                    stack=stack,
                    env=env,
                    name=shared_provider,
                    namespace=provider_namespace,
                    grant_from_model_ns=True,
                )
                for provider_namespace in provider_namespaces
            ]
            model = crossns_twin_model(
                admin_client=admin_client,
                stack=stack,
                env=env,
                name=f"e2e-crossns-twinmodel-{uuid.uuid4().hex[:6]}",
                provider_name=shared_provider,
                provider_namespaces=provider_namespaces,
            )

            access = serve_model_in_tenant_stack(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                env=env,
                stack=stack,
                model=model,
            )
            wait_for_gateway_attachment_ready(
                admin_client=admin_client,
                model_name=model["name"],
                model_namespace=model["namespace"],
                gateway_name=case["gateway_name"],
            )

            seen: set[str] = set()
            for _request_index in range(12):
                response = infer_external_chat(
                    request_session_http=request_session_http,
                    gateway_url=case["gateway_url"],
                    model_namespace=model["namespace"],
                    model_client_name=model["client_name"],
                    api_key=access["api_key"],
                    path_tag=client_tag,
                )
                assert response.status_code == 200, (
                    f"inference failed on multi-provider model: {redact_sensitive(response.text)}"
                )
                for provider_index, (credential, tag) in enumerate(
                    zip(credentials, model["path_tags"], strict=True), start=1
                ):
                    if credential in response.text:
                        assert tag in response.text, (
                            f"credential of provider ns {provider_index} served through the wrong "
                            f"destination: expected path tag {tag!r} in the response"
                        )
                        seen.add(f"provider-ns-{provider_index}")
            assert seen == {f"provider-ns-{index}" for index in (1, 2)}, (
                f"expected both same-named providers to serve, saw {seen}"
            )
