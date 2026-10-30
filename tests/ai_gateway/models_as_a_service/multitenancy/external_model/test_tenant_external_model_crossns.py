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

Gated on TENANT_EXTERNAL_MODEL_CROSSNS_E2E=true (AGC cross-namespace
provider support, RHOAIENG-90804, is done; the flag additionally requires
the AGC ReferenceGrant validation and its RBAC to be packaged).

Shares the tenant_external_model session tenants (see
conftest.tenant_external_model_tenants); this module creates its own user
namespaces with unique names.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.secret import Secret

from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    CREDENTIAL_DATA_KEY,
    EXTERNAL_API_FORMAT,
    EXTERNAL_ENDPOINT,
    EXTERNAL_TARGET_MODEL,
    admitted_user_namespace,
    assert_served_by_provider,
    external_model_binding,
    gateway_ref,
    get_json_or_none,
    infer_external_chat,
    namespace_name,
    redact_sensitive,
    require_crossns_provider_support,
    serve_external_model_in_tenant,
    wait_for_credential_copy,
    wait_for_gateway_attachment_not_ready,
    wait_for_gateway_attachment_ready,
    wait_for_no_credential_copy,
)
from utilities.resources.external_model import ExternalModel
from utilities.resources.external_provider import ExternalProvider
from utilities.resources.reference_grant import ReferenceGrant

LOGGER = structlog.get_logger(name=__name__)

# Serialized with tenant_ipp until the legacy IPP controllers skip
# gatewayRefs ExternalModels (image pin pending): without that guard they
# adopt this suite's models and flap status.httpRouteName.
pytestmark = pytest.mark.xdist_group(name="tenant_ipp")

# Failure text expected when a cross-namespace reference is not authorized.
GRANT_FAILURE_PATTERN = r"grant|authoriz|not allowed|permission|forbidden|reference"
# Failure text expected when a referenced Secret does not exist.
MISSING_SECRET_PATTERN = r"secret"


@pytest.fixture(scope="module")
def crossns_env(
    admin_client: DynamicClient,
    tenant_external_model_tenants: dict[str, dict[str, str]],
    teardown_resources: bool,
) -> Generator[dict[str, Any], Any]:
    """One tenant's case, admitted model/provider namespaces, and a module-wide resource stack."""
    require_crossns_provider_support(admin_client=admin_client)
    case = tenant_external_model_tenants["tenant_a"]
    with ExitStack() as stack:
        model_ns = stack.enter_context(
            cm=admitted_user_namespace(
                admin_client,
                "e2e-extmod-crossns-model",
                case["gateway_name"],
                teardown=teardown_resources,
            )
        )
        provider_ns = stack.enter_context(
            cm=admitted_user_namespace(
                admin_client,
                "e2e-extmod-crossns-prov",
                case["gateway_name"],
                teardown=teardown_resources,
            )
        )
        LOGGER.info(
            f"[crossns] model namespace {model_ns.name}, provider namespace {provider_ns.name} "
            f"on gateway {case['gateway_name']}"
        )
        yield {
            "tenant": case,
            "model_ns": namespace_name(model_ns),
            "provider_ns": namespace_name(provider_ns),
            "stack": stack,
            "teardown": teardown_resources,
        }


def _serve(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token: str,
    env: dict[str, Any],
    model: dict[str, Any],
    **kwargs: Any,
) -> dict[str, str]:
    """Serve one model under the tenant and keep the policy/subscription until module teardown."""
    return env["stack"].enter_context(
        cm=serve_external_model_in_tenant(
            admin_client,
            request_session_http,
            ocp_token,
            tenant=env["tenant"],
            model_name=model["name"],
            model_namespace=model["namespace"],
            policy_name=f"{model['name']}-access",
            subscription_name=f"{model['name']}-sub",
            teardown=env["teardown"],
            **kwargs,
        )
    )


def _model_with_provider(
    admin_client: DynamicClient,
    env: dict[str, Any],
    *,
    name: str,
    provider_name: str,
    override_secret: str | None = None,
) -> dict[str, Any]:
    """Create a cross-ns model bound to a caller-managed provider."""
    suffix = uuid.uuid4().hex[:6]
    path_tag = f"{name}-{suffix}"
    env["stack"].enter_context(
        cm=ExternalModel(
            client=admin_client,
            name=name,
            namespace=env["model_ns"],
            external_provider_refs=[
                external_model_binding(
                    provider_name=provider_name,
                    provider_namespace=env["provider_ns"],
                    path_tag=path_tag,
                    credential_secret=override_secret,
                )
            ],
            gateway_refs=[gateway_ref(gateway_name=env["tenant"]["gateway_name"])],
            teardown=env["teardown"],
            wait_for_resource=True,
        )
    )
    env["stack"].enter_context(
        cm=MaaSModelRef(
            client=admin_client,
            name=name,
            namespace=env["model_ns"],
            model_ref={
                "name": name,
                "namespace": env["model_ns"],
                "kind": "ExternalModel",
                "apiGroup": "inference.opendatahub.io",
            },
            teardown=env["teardown"],
            wait_for_resource=True,
        )
    )
    return {
        "name": name,
        "namespace": env["model_ns"],
        "client_name": name,
        "provider": provider_name,
        "provider_namespace": env["provider_ns"],
        "path_tag": path_tag,
    }


def _crossns_model(
    admin_client: DynamicClient,
    env: dict[str, Any],
    *,
    name: str,
    credential: str | None = None,
    override: str | None = None,
) -> dict[str, Any]:
    """A model in the model namespace referencing a provider in the provider
    namespace (no ReferenceGrant — callers add grants per scenario)."""
    suffix = uuid.uuid4().hex[:6]
    credential = credential or f"sk-extmod-crossns-{suffix}"  # pragma: allowlist secret
    provider_name = f"{name}-provider"
    secret_name = f"{provider_name}-cred"

    env["stack"].enter_context(
        cm=Secret(
            client=admin_client,
            name=secret_name,
            namespace=env["provider_ns"],
            type="Opaque",
            string_data={CREDENTIAL_DATA_KEY: credential},
            teardown=env["teardown"],
            wait_for_resource=True,
        )
    )
    env["stack"].enter_context(
        cm=ExternalProvider(
            client=admin_client,
            name=provider_name,
            namespace=env["provider_ns"],
            provider="openai",
            endpoint=EXTERNAL_ENDPOINT,
            auth={"type": "apikey", "secretRef": {"name": secret_name}},
            teardown=env["teardown"],
            wait_for_resource=True,
        )
    )

    override_secret = None
    if override is not None:
        override_secret = f"{name}-override-cred"
        env["stack"].enter_context(
            cm=Secret(
                client=admin_client,
                name=override_secret,
                namespace=env["model_ns"],
                type="Opaque",
                string_data={CREDENTIAL_DATA_KEY: override},
                teardown=env["teardown"],
                wait_for_resource=True,
            )
        )

    model = _model_with_provider(
        admin_client=admin_client,
        env=env,
        name=name,
        provider_name=provider_name,
        override_secret=override_secret,
    )
    return {
        **model,
        "secret_name": secret_name,
        "secret_namespace": env["provider_ns"],
        "credential": credential,
        "override_credential": override,
    }


def _apply_reference_grant(
    admin_client: DynamicClient,
    env: dict[str, Any],
    name: str,
    provider_namespace: str,
    *,
    from_namespace: str,
    to_name: str | None = None,
) -> ReferenceGrant:
    """Authorize ExternalModels in *from_namespace* to reference providers in
    the grant's namespace (P1 cross-namespace contract).

    Grants authorize at namespace granularity only: ReferenceGrant v1beta1's
    ``from`` has no ``name`` field, and the API server prunes unknown fields,
    so every ExternalModel in *from_namespace* is covered.
    """
    grant_from: dict[str, str] = {
        "group": "inference.opendatahub.io",
        "kind": "ExternalModel",
        "namespace": from_namespace,
    }
    grant_to: dict[str, str] = {
        "group": "inference.opendatahub.io",
        "kind": "ExternalProvider",
    }
    if to_name:
        grant_to["name"] = to_name
    return env["stack"].enter_context(
        cm=ReferenceGrant(
            client=admin_client,
            name=name,
            namespace=provider_namespace,
            from_=[grant_from],
            to=[grant_to],
            teardown=env["teardown"],
            wait_for_resource=True,
        )
    )


def _grant_for(
    admin_client: DynamicClient,
    env: dict[str, Any],
    model: dict[str, Any],
    *,
    name_suffix: str = "",
    from_namespace: str | None = None,
) -> ReferenceGrant:
    """Grant the model's namespace access to the model's provider."""
    return _apply_reference_grant(
        admin_client=admin_client,
        env=env,
        name=f"{model['name']}-grant{name_suffix}",
        provider_namespace=model["provider_namespace"],
        from_namespace=from_namespace or model["namespace"],
        to_name=model["provider"],
    )


def _revoke_grant(grant: ReferenceGrant) -> None:
    """Delete a ReferenceGrant (the attachment waits confirm the effect)."""
    grant.delete()


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
        env = crossns_env
        model = _crossns_model(admin_client=admin_client, env=env, name=f"e2e-crossns-grant-{uuid.uuid4().hex[:6]}")
        _grant_for(admin_client=admin_client, env=env, model=model)
        access = _serve(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            model=model,
        )

        wait_for_gateway_attachment_ready(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=env["tenant"]["gateway_name"],
        )
        # The provider-local (provider-namespace) credential is delivered as a
        # controller-owned copy into the tenant namespace and used upstream.
        wait_for_credential_copy(
            admin_client=admin_client,
            tenant_namespace=env["tenant"]["tenant_ns"],
            source_secret_name=model["secret_name"],
            source_namespace=model["secret_namespace"],
        )
        response = infer_external_chat(
            gateway_url=env["tenant"]["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=access["api_key"],
            path_tag=model["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])


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
        env = crossns_env
        model = _crossns_model(admin_client=admin_client, env=env, name=f"e2e-crossns-missing-{uuid.uuid4().hex[:6]}")
        gateway = env["tenant"]["gateway_name"]

        # No grant: the reference is rejected descriptively (not silently
        # resolved or served with a fallback provider).
        wait_for_gateway_attachment_not_ready(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=gateway,
            message_pattern=GRANT_FAILURE_PATTERN,
        )

        # Adding permission restores serving.
        _grant_for(admin_client=admin_client, env=env, model=model)
        access = _serve(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            model=model,
        )
        wait_for_gateway_attachment_ready(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=gateway,
        )
        response = infer_external_chat(
            gateway_url=env["tenant"]["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=access["api_key"],
            path_tag=model["path_tag"],
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
        env = crossns_env
        model = _crossns_model(admin_client=admin_client, env=env, name=f"e2e-crossns-wronggr-{uuid.uuid4().hex[:6]}")
        gateway = env["tenant"]["gateway_name"]
        # A grant that authorizes a different source namespace must not open
        # the model's reference.
        _grant_for(
            admin_client=admin_client,
            env=env,
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
        env = crossns_env
        model = _crossns_model(admin_client=admin_client, env=env, name=f"e2e-crossns-revoke-{uuid.uuid4().hex[:6]}")
        gateway = env["tenant"]["gateway_name"]
        grant = _grant_for(admin_client=admin_client, env=env, model=model)
        access = _serve(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            model=model,
        )
        wait_for_gateway_attachment_ready(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=gateway,
        )

        response = infer_external_chat(
            gateway_url=env["tenant"]["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=access["api_key"],
            path_tag=model["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])

        # Revoke: status flips to a failure mentioning authorization and the
        # runtime configuration (credential copy) is withdrawn.
        _revoke_grant(grant=grant)
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
        denied = infer_external_chat(
            gateway_url=env["tenant"]["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=access["api_key"],
            path_tag=model["path_tag"],
            retries=1,
        )
        assert denied.status_code != 200, (
            f"inference must stop after the grant is revoked, got {denied.status_code}: {redact_sensitive(denied.text)}"
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
        env = crossns_env
        model = _crossns_model(admin_client=admin_client, env=env, name=f"e2e-crossns-narrow-{uuid.uuid4().hex[:6]}")
        gateway = env["tenant"]["gateway_name"]
        grant = _grant_for(admin_client=admin_client, env=env, model=model)
        access = _serve(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            model=model,
        )
        wait_for_gateway_attachment_ready(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=gateway,
        )

        # Narrow the grant to a different source namespace: the model's
        # reference is no longer covered. The grant object comes from
        # _grant_for itself — reconstructing its name here would silently
        # no-op the revocation if that naming ever changed.
        _revoke_grant(grant=grant)
        _grant_for(
            admin_client=admin_client,
            env=env,
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
        denied = infer_external_chat(
            gateway_url=env["tenant"]["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=access["api_key"],
            path_tag=model["path_tag"],
            retries=1,
        )
        assert denied.status_code != 200, f"inference must stop after the grant is narrowed, got {denied.status_code}"


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
        env = crossns_env
        # Same-named Secrets in both namespaces with different values; the
        # provider-local one must be selected without an override.
        shared_credential = "shared-cred"
        provider_credential = f"sk-extmod-prov-{uuid.uuid4().hex[:8]}"  # pragma: allowlist secret
        env["stack"].enter_context(
            cm=Secret(
                client=admin_client,
                name=shared_credential,
                namespace=env["provider_ns"],
                type="Opaque",
                string_data={CREDENTIAL_DATA_KEY: provider_credential},
                teardown=env["teardown"],
                wait_for_resource=True,
            )
        )
        model_credential = f"sk-extmod-model-{uuid.uuid4().hex[:8]}"  # pragma: allowlist secret
        env["stack"].enter_context(
            cm=Secret(
                client=admin_client,
                name=shared_credential,
                namespace=env["model_ns"],
                type="Opaque",
                string_data={CREDENTIAL_DATA_KEY: model_credential},
                teardown=env["teardown"],
                wait_for_resource=True,
            )
        )

        provider_name = f"e2e-crossns-credsel-{uuid.uuid4().hex[:6]}-provider"
        env["stack"].enter_context(
            cm=ExternalProvider(
                client=admin_client,
                name=provider_name,
                namespace=env["provider_ns"],
                provider="openai",
                endpoint=EXTERNAL_ENDPOINT,
                auth={"type": "apikey", "secretRef": {"name": shared_credential}},
                teardown=env["teardown"],
                wait_for_resource=True,
            )
        )

        # Default: provider-local Secret.
        default_model = _model_with_provider(
            admin_client=admin_client,
            env=env,
            name=f"e2e-crossns-creddef-{uuid.uuid4().hex[:6]}",
            provider_name=provider_name,
        )
        _grant_for(admin_client=admin_client, env=env, model=default_model)
        access = _serve(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            model=default_model,
        )
        response = infer_external_chat(
            gateway_url=env["tenant"]["gateway_url"],
            model_namespace=default_model["namespace"],
            model_client_name=default_model["client_name"],
            api_key=access["api_key"],
            path_tag=default_model["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=default_model["path_tag"], credential=provider_credential)
        assert model_credential not in response.text, "model-local Secret must not be used without an auth override"

        # Override: model-local Secret wins.
        override_model = _model_with_provider(
            admin_client=admin_client,
            env=env,
            name=f"e2e-crossns-credovr-{uuid.uuid4().hex[:6]}",
            provider_name=provider_name,
            override_secret=shared_credential,
        )
        _grant_for(admin_client=admin_client, env=env, model=override_model)
        access = _serve(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            env=env,
            model=override_model,
        )
        response = infer_external_chat(
            gateway_url=env["tenant"]["gateway_url"],
            model_namespace=override_model["namespace"],
            model_client_name=override_model["client_name"],
            api_key=access["api_key"],
            path_tag=override_model["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=override_model["path_tag"], credential=model_credential)
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
        env = crossns_env
        provider_name = f"e2e-crossns-missing-{uuid.uuid4().hex[:6]}-provider"
        provider_credential = f"sk-extmod-prov-{uuid.uuid4().hex[:8]}"  # pragma: allowlist secret
        secret_name = f"{provider_name}-cred"
        env["stack"].enter_context(
            cm=Secret(
                client=admin_client,
                name=secret_name,
                namespace=env["provider_ns"],
                type="Opaque",
                string_data={CREDENTIAL_DATA_KEY: provider_credential},
                teardown=env["teardown"],
                wait_for_resource=True,
            )
        )
        env["stack"].enter_context(
            cm=ExternalProvider(
                client=admin_client,
                name=provider_name,
                namespace=env["provider_ns"],
                provider="openai",
                endpoint=EXTERNAL_ENDPOINT,
                auth={"type": "apikey", "secretRef": {"name": secret_name}},
                teardown=env["teardown"],
                wait_for_resource=True,
            )
        )

        # Override points at a Secret that does not exist in the model
        # namespace; serving must fail rather than fall back to the
        # provider-namespace Secret.
        missing_override = "does-not-exist-cred"
        model = _model_with_provider(
            admin_client=admin_client,
            env=env,
            name=f"e2e-crossns-nofb-{uuid.uuid4().hex[:6]}",
            provider_name=provider_name,
            override_secret=missing_override,
        )
        _grant_for(admin_client=admin_client, env=env, model=model)
        gateway = env["tenant"]["gateway_name"]
        wait_for_gateway_attachment_not_ready(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=gateway,
            message_pattern=MISSING_SECRET_PATTERN,
        )
        external_model = (
            get_json_or_none(
                client=admin_client, kind="externalmodel", name=model["name"], namespace=model["namespace"]
            )
            or {}
        )
        phase = (external_model.get("status") or {}).get("phase")
        assert phase and phase != "Ready", (
            f"model must report a present, non-Ready phase after the missing-secret failure, got {phase!r}"
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
        env = crossns_env
        case = env["tenant"]
        shared_provider = f"e2e-crossns-twin-{uuid.uuid4().hex[:6]}"
        credential_1 = f"sk-extmod-t1-{uuid.uuid4().hex[:8]}"  # pragma: allowlist secret
        credential_2 = f"sk-extmod-t2-{uuid.uuid4().hex[:8]}"  # pragma: allowlist secret
        shared_secret = f"{shared_provider}-cred"
        tag_1 = f"{shared_provider}-ns1"
        tag_2 = f"{shared_provider}-ns2"
        # The request path is the client's choice and is passed through to
        # whichever provider the gateway selects, so a single client tag is
        # used and the injected credentials distinguish the two providers.
        client_tag = f"{shared_provider}-client"

        # The two provider namespaces live until the end of the test (the
        # source deleted them in a finally block): the local stack unwinds in
        # reverse order even if the second creation fails.
        with ExitStack() as test_stack:
            provider_ns_1 = namespace_name(
                namespace=test_stack.enter_context(
                    cm=admitted_user_namespace(
                        admin_client,
                        "e2e-extmod-crossns-p1",
                        case["gateway_name"],
                        teardown=env["teardown"],
                    )
                )
            )
            provider_ns_2 = namespace_name(
                namespace=test_stack.enter_context(
                    cm=admitted_user_namespace(
                        admin_client,
                        "e2e-extmod-crossns-p2",
                        case["gateway_name"],
                        teardown=env["teardown"],
                    )
                )
            )
            for provider_ns, credential in ((provider_ns_1, credential_1), (provider_ns_2, credential_2)):
                env["stack"].enter_context(
                    cm=Secret(
                        client=admin_client,
                        name=shared_secret,
                        namespace=provider_ns,
                        type="Opaque",
                        string_data={CREDENTIAL_DATA_KEY: credential},
                        teardown=env["teardown"],
                        wait_for_resource=True,
                    )
                )
                env["stack"].enter_context(
                    cm=ExternalProvider(
                        client=admin_client,
                        name=shared_provider,
                        namespace=provider_ns,
                        provider="openai",
                        endpoint=EXTERNAL_ENDPOINT,
                        auth={"type": "apikey", "secretRef": {"name": shared_secret}},
                        teardown=env["teardown"],
                        wait_for_resource=True,
                    )
                )
                _apply_reference_grant(
                    admin_client=admin_client,
                    env=env,
                    name=f"{shared_provider}-grant",
                    provider_namespace=provider_ns,
                    from_namespace=env["model_ns"],
                    to_name=shared_provider,
                )

            name = f"e2e-crossns-twinmodel-{uuid.uuid4().hex[:6]}"
            env["stack"].enter_context(
                cm=ExternalModel(
                    client=admin_client,
                    name=name,
                    namespace=env["model_ns"],
                    external_provider_refs=[
                        external_model_binding(
                            provider_name=shared_provider,
                            provider_namespace=provider_ns_1,
                            path_tag=tag_1,
                        ),
                        {
                            "ref": {"name": shared_provider, "namespace": provider_ns_2},
                            "targetModel": EXTERNAL_TARGET_MODEL,
                            "apiFormat": EXTERNAL_API_FORMAT,
                            "path": f"/anything/{tag_2}",
                        },
                    ],
                    gateway_refs=[gateway_ref(gateway_name=case["gateway_name"])],
                    teardown=env["teardown"],
                    wait_for_resource=True,
                )
            )
            env["stack"].enter_context(
                cm=MaaSModelRef(
                    client=admin_client,
                    name=name,
                    namespace=env["model_ns"],
                    model_ref={
                        "name": name,
                        "namespace": env["model_ns"],
                        "kind": "ExternalModel",
                        "apiGroup": "inference.opendatahub.io",
                    },
                    teardown=env["teardown"],
                    wait_for_resource=True,
                )
            )
            model = {
                "name": name,
                "namespace": env["model_ns"],
                "client_name": name,
            }

            access = _serve(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                env=env,
                model=model,
            )
            wait_for_gateway_attachment_ready(
                admin_client=admin_client,
                model_name=name,
                model_namespace=env["model_ns"],
                gateway_name=case["gateway_name"],
            )

            # Weighted selection across the two bindings: both providers must
            # receive traffic across a dozen requests. The injected
            # credential — not the echoed path — identifies which same-named
            # provider served each request.
            seen: set[str] = set()
            for _request_index in range(12):
                response = infer_external_chat(
                    gateway_url=case["gateway_url"],
                    model_namespace=model["namespace"],
                    model_client_name=model["client_name"],
                    api_key=access["api_key"],
                    path_tag=client_tag,
                )
                assert response.status_code == 200, (
                    f"inference failed on multi-provider model: {redact_sensitive(response.text)}"
                )
                if credential_1 in response.text:
                    seen.add("provider-ns-1")
                if credential_2 in response.text:
                    seen.add("provider-ns-2")
            assert seen == {"provider-ns-1", "provider-ns-2"}, (
                f"expected both same-named providers to serve, saw {seen}"
            )
