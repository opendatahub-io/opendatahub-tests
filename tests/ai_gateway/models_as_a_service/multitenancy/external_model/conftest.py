"""Fixtures for multi-tenant external model tests (RHAI-12994 P0 port).

Provides two AGC/Praxis tenants with selector-admitting tenant gateways,
enriched with gateway/maas-api URLs, plus the capability gates that keep
the suite green (skipped) on clusters where the AGC companion or the
gatewayRefs additive schema is not distributed yet.
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
from ocp_resources.namespace import Namespace
from ocp_resources.project_project_openshift_io import Project

from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import (
    build_aitenant_test_context,
)
from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    EXTERNAL_ENDPOINT,
    admitted_user_namespace,
    deploy_aitenant_best_effort,
    extmod_tenant_test_params,
    namespace_name,
    per_tenant_processing_uses_praxis,
    provision_and_serve_in_tenant,
    record_tenant_external_model_evidence,
    require_maas_route,
    require_tenant_external_model_support,
    serve_in_tenant_and_cleanup,
    serve_shared_model_in_tenant,
    tenant_case,
    two_gateway_model,
    unadmitted_user_namespace,
    usage_provider,
)
from tests.ai_gateway.models_as_a_service.multitenancy.utils import (
    admit_namespace_to_gateway,
    isolation_bootstrap_gateway_context,
    tenant_gateway_external_route,
)
from tests.ai_gateway.models_as_a_service.utils import (
    bootstrap_gateway_ref,
    deploy_and_verify_aitenant_ready,
    host_from_ingress_domain,
    verify_maas_gateway_programmed,
)
from utilities.certificates_utils import router_ca_bundle_env

LOGGER = structlog.get_logger(name=__name__)

LIMITED_TOKEN_LIMIT = 3
UNLIMITED_TOKEN_LIMIT = 100


@pytest.fixture(scope="session", autouse=True)
def gateway_route_ca_bundle(admin_client: DynamicClient) -> Generator[str]:
    """Trust the cluster's router CA for gateway Route TLS (see router_ca_bundle_env)."""
    with router_ca_bundle_env(client=admin_client) as bundle:
        yield bundle


@pytest.fixture(scope="module")
def tenant_external_model_capabilities(admin_client: DynamicClient) -> dict[str, bool]:
    """Gate the suite on the installed AGC + ExternalModel CRD capabilities."""
    return require_tenant_external_model_support(admin_client=admin_client)


@pytest.fixture(scope="module")
def tenant_external_model_evidence(
    admin_client: DynamicClient,
    tenant_external_model_capabilities: dict[str, bool],
) -> dict[str, Any]:
    """Record installed AGC/Praxis versions and configuration as run evidence."""
    return record_tenant_external_model_evidence(admin_client=admin_client)


@pytest.fixture(scope="module")
def tenant_external_model_primary_test_params(admin_client: DynamicClient) -> dict[str, Any]:
    """Primary AITenant name and spec for the external model suite."""
    return extmod_tenant_test_params(tenant_label="alpha", admin_client=admin_client)


@pytest.fixture(scope="module")
def tenant_external_model_secondary_test_params(admin_client: DynamicClient) -> dict[str, Any]:
    """Secondary AITenant name and spec for the external model suite."""
    return extmod_tenant_test_params(tenant_label="beta", admin_client=admin_client)


@pytest.fixture(scope="module")
def tenant_external_model_tenants(
    admin_client: DynamicClient,
    tenant_external_model_capabilities: dict[str, bool],
    tenant_external_model_evidence: dict[str, Any],
    tenant_external_model_primary_test_params: dict[str, Any],
    tenant_external_model_secondary_test_params: dict[str, Any],
    aitenant_infra_namespace: str,
    maas_api_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[dict[str, dict[str, str]], Any, Any]:
    """Two AGC/Praxis tenants with selector-admitting tenant gateways.

    Each tenant gets a bootstrap Gateway exposed on an external route and an
    AITenant reconciled to Ready; the case dicts carry the gateway URL (for
    inference) and the per-tenant maas-api URL (for API key minting). Skips
    when either tenant's per-tenant payload-processing stack is not praxis
    (AGC contract inactive). Modules share these tenants; user namespaces,
    models and tenant-scoped policies are created per test with unique names.
    """
    maas_host = host_from_ingress_domain(client=admin_client)
    cases: dict[str, dict[str, str]] = {}
    with ExitStack() as stack:
        for key, params in (
            ("tenant_a", tenant_external_model_primary_test_params),
            ("tenant_b", tenant_external_model_secondary_test_params),
        ):
            gateway_name, gateway_namespace = bootstrap_gateway_ref(
                aitenant_name=params["aitenant_name"],
                aitenant_spec=params["aitenant_spec"],
            )
            gateway = stack.enter_context(
                cm=isolation_bootstrap_gateway_context(
                    admin_client=admin_client,
                    gateway_name=gateway_name,
                    api_namespace=maas_api_infra_namespace,
                    gateway_namespace=gateway_namespace,
                    teardown=teardown_resources,
                )
            )
            assert gateway.name is not None, "bootstrap Gateway has no name"
            verify_maas_gateway_programmed(gateway=gateway)
            stack.enter_context(
                cm=tenant_gateway_external_route(
                    admin_client=admin_client,
                    gateway_name=gateway.name,
                    maas_host=maas_host,
                    teardown=teardown_resources,
                )
            )
            aitenant = stack.enter_context(
                cm=deploy_aitenant_best_effort(
                    admin_client=admin_client,
                    aitenant_name=params["aitenant_name"],
                    cr_namespace=aitenant_infra_namespace,
                    aitenant_spec=params["aitenant_spec"],
                    teardown=teardown_resources,
                )
            )
            deploy_and_verify_aitenant_ready(aitenant=aitenant)
            if not per_tenant_processing_uses_praxis(admin_client, params["aitenant_name"]):
                pytest.fail(
                    f"per-tenant payload-processing for {params['aitenant_name']} is not "
                    "praxis-extproc; the AGC/Praxis multi-tenant contract is not active"
                )
            context = build_aitenant_test_context(aitenant=aitenant)
            admit_namespace_to_gateway(
                admin_client=admin_client,
                namespace=context["tenant_namespace_name"],
                gateway_name=gateway.name,
            )
            cases[key] = tenant_case(
                context=context,
                gateway_name=gateway.name,
                maas_host=maas_host,
            )
        yield cases


@pytest.fixture(scope="module")
def failure_env(
    admin_client: DynamicClient,
    tenant_external_model_tenants: dict[str, dict[str, str]],
    teardown_resources: bool,
) -> Generator[dict[str, Any], Any, Any]:
    """Test data for the failure-isolation scenario.

    One user namespace admitted to both tenants' gateways; cleanup is owned
    by this fixture and never exposed to the tests.
    """
    with ExitStack() as stack:
        user_ns = stack.enter_context(
            cm=admitted_user_namespace(
                admin_client,
                "e2e-extmod-fail-user",
                tenant_external_model_tenants["tenant_a"]["gateway_name"],
                tenant_external_model_tenants["tenant_b"]["gateway_name"],
                teardown=teardown_resources,
            )
        )
        yield {
            "tenant_a": tenant_external_model_tenants["tenant_a"],
            "tenant_b": tenant_external_model_tenants["tenant_b"],
            "user_ns": user_ns,
        }


@pytest.fixture(scope="class")
def failure_served(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token_for_actor: str,
    failure_env: dict[str, Any],
    teardown_resources: bool,
) -> Generator[dict[str, dict[str, Any]], Any, Any]:
    """Each tenant gets its own independent external model, served on its gateway."""
    with ExitStack() as stack:
        yield {
            "tenant_a": provision_and_serve_in_tenant(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                stack=stack,
                user_namespace=failure_env["user_ns"],
                case=failure_env["tenant_a"],
                teardown=teardown_resources,
            ),
            "tenant_b": provision_and_serve_in_tenant(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                stack=stack,
                user_namespace=failure_env["user_ns"],
                case=failure_env["tenant_b"],
                teardown=teardown_resources,
            ),
        }


@pytest.fixture(scope="module")
def shared_user_namespace(
    admin_client: DynamicClient,
    tenant_external_model_tenants: dict[str, dict[str, str]],
    teardown_resources: bool,
) -> Generator[Namespace | Project, Any, Any]:
    """User namespace admitted to both tenants' gateways for the shared module."""
    with ExitStack() as stack:
        yield stack.enter_context(
            cm=admitted_user_namespace(
                admin_client,
                "e2e-extmod-shr-user",
                tenant_external_model_tenants["tenant_a"]["gateway_name"],
                tenant_external_model_tenants["tenant_b"]["gateway_name"],
                teardown=teardown_resources,
            )
        )


@pytest.fixture(scope="class")
def shared_model(
    admin_client: DynamicClient,
    tenant_external_model_tenants: dict[str, dict[str, str]],
    shared_user_namespace: Namespace | Project,
    teardown_resources: bool,
) -> Generator[dict[str, Any], Any, Any]:
    """The shared model attached to both tenants' gateways."""
    cases = tenant_external_model_tenants
    name = f"e2e-shr-shared-{uuid.uuid4().hex[:6]}"
    with ExitStack() as stack:
        model = stack.enter_context(
            cm=two_gateway_model(
                admin_client,
                name=name,
                namespace=namespace_name(shared_user_namespace),
                cases=cases,
                teardown=teardown_resources,
            )
        )
        yield {"model": model}


@pytest.fixture(scope="class")
def shared_serving(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token_for_actor: str,
    tenant_external_model_tenants: dict[str, dict[str, str]],
    shared_model: dict[str, Any],
    teardown_resources: bool,
) -> Generator[dict[str, Any], Any, Any]:
    """Per-tenant auth/subscription stacks and API keys for the shared model."""
    model = shared_model["model"]
    require_maas_route(admin_client=admin_client, model=model)
    with ExitStack() as stack:
        yield {
            "tenant_a": serve_shared_model_in_tenant(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                stack=stack,
                case=tenant_external_model_tenants["tenant_a"],
                model=model,
                teardown=teardown_resources,
            ),
            "tenant_b": serve_shared_model_in_tenant(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                stack=stack,
                case=tenant_external_model_tenants["tenant_b"],
                model=model,
                teardown=teardown_resources,
            ),
        }


@pytest.fixture(scope="class")
def isolated_keys(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token_for_actor: str,
    tenant_external_model_tenants: dict[str, dict[str, str]],
    shared_user_namespace: Namespace | Project,
    teardown_resources: bool,
) -> Generator[dict[str, Any], Any, Any]:
    """Two tenants' API keys for the same two-gateway model."""
    cases = tenant_external_model_tenants
    name = f"e2e-shr-auth-{uuid.uuid4().hex[:6]}"
    with ExitStack() as stack:
        model = stack.enter_context(
            cm=two_gateway_model(
                admin_client,
                name=name,
                namespace=namespace_name(shared_user_namespace),
                cases=cases,
                teardown=teardown_resources,
            )
        )
        require_maas_route(admin_client=admin_client, model=model)
        access_a = serve_shared_model_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            stack=stack,
            case=cases["tenant_a"],
            model=model,
            teardown=teardown_resources,
        )
        access_b = serve_shared_model_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token_for_actor,
            stack=stack,
            case=cases["tenant_b"],
            model=model,
            teardown=teardown_resources,
        )
        yield {"model": model, "tenant_a": access_a["api_key"], "tenant_b": access_b["api_key"]}


@pytest.fixture(scope="class")
def ambiguous_model_ref(
    admin_client: DynamicClient,
    tenant_external_model_tenants: dict[str, dict[str, str]],
    shared_user_namespace: Namespace | Project,
    teardown_resources: bool,
) -> Generator[dict[str, Any], Any, Any]:
    """A two-gateway model plus a tenantRef-less modelRef naming it."""
    cases = tenant_external_model_tenants
    name = f"e2e-shr-amb-{uuid.uuid4().hex[:6]}"
    with ExitStack() as stack:
        model = stack.enter_context(
            cm=two_gateway_model(
                admin_client,
                name=name,
                namespace=namespace_name(shared_user_namespace),
                cases=cases,
                teardown=teardown_resources,
            )
        )
        require_maas_route(admin_client=admin_client, model=model)
        ambiguous_ref = f"{name}-ambiguous"
        stack.enter_context(
            cm=MaaSModelRef(
                client=admin_client,
                name=ambiguous_ref,
                namespace=namespace_name(shared_user_namespace),
                annotations={
                    "maas.opendatahub.io/endpoint": EXTERNAL_ENDPOINT,
                    "maas.opendatahub.io/provider": "openai",
                },
                model_ref={"kind": "ExternalModel", "name": name},
                teardown=teardown_resources,
                wait_for_resource=True,
            )
        )
        yield {"model": model, "ambiguous_ref": ambiguous_ref}


@pytest.fixture(scope="class")
def quota_model(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token_for_actor: str,
    tenant_external_model_tenants: dict[str, dict[str, str]],
    shared_user_namespace: Namespace | Project,
    teardown_resources: bool,
) -> Generator[dict[str, Any], Any, Any]:
    """The simulator-backed shared model served in both tenants with asymmetric token budgets."""
    cases = tenant_external_model_tenants
    name = f"e2e-shr-quota-{uuid.uuid4().hex[:6]}"
    client_name = f"{name}-client"
    with ExitStack() as stack:
        usage = usage_provider(
            admin_client=admin_client,
            stack=stack,
            namespace=namespace_name(shared_user_namespace),
            client_model_name=client_name,
            teardown=teardown_resources,
        )
        model = stack.enter_context(
            cm=two_gateway_model(
                admin_client,
                name=name,
                namespace=namespace_name(shared_user_namespace),
                cases=cases,
                teardown=teardown_resources,
                endpoint=usage["endpoint"],
                provider_path=usage["provider_path"],
                credential=usage["credential"],
                provider_config=usage["provider_config"],
            )
        )
        require_maas_route(admin_client=admin_client, model=model)
        yield {
            "model": model,
            "tenant_a": serve_shared_model_in_tenant(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                stack=stack,
                case=cases["tenant_a"],
                model=model,
                teardown=teardown_resources,
                token_limit=LIMITED_TOKEN_LIMIT,
                window="1m",
            ),
            "tenant_b": serve_shared_model_in_tenant(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                stack=stack,
                case=cases["tenant_b"],
                model=model,
                teardown=teardown_resources,
                token_limit=UNLIMITED_TOKEN_LIMIT,
                window="1m",
            ),
            "usage": usage,
        }


@pytest.fixture(scope="module")
def attachment_env(
    admin_client: DynamicClient,
    tenant_external_model_tenants: dict[str, dict[str, str]],
    teardown_resources: bool,
) -> Generator[dict[str, Any], Any, Any]:
    """User namespaces (admitted + deliberately unadmitted) on the shared tenants."""
    tenant_a = tenant_external_model_tenants["tenant_a"]
    tenant_b = tenant_external_model_tenants["tenant_b"]
    with ExitStack() as stack:
        user_ns = stack.enter_context(
            cm=admitted_user_namespace(
                admin_client,
                "e2e-extmod-att-user",
                tenant_a["gateway_name"],
                tenant_b["gateway_name"],
                teardown=teardown_resources,
            )
        )
        unadmitted_ns = stack.enter_context(
            cm=unadmitted_user_namespace(
                admin_client,
                "e2e-extmod-att-noaccess",
                teardown=teardown_resources,
            )
        )
        yield {
            "tenant_a": tenant_a,
            "tenant_b": tenant_b,
            "user_ns": user_ns.name,
            "unadmitted_ns": unadmitted_ns.name,
        }


@pytest.fixture(scope="class")
def attached_model(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token_for_actor: str,
    attachment_env: dict[str, Any],
    teardown_resources: bool,
) -> Generator[dict[str, Any], Any, Any]:
    """One model attached to (and served through) tenant B's gateway."""
    env = attachment_env
    tenant_b = env["tenant_b"]
    with ExitStack() as stack:
        model = stack.enter_context(
            cm=external_model_stack(
                admin_client,
                name=f"e2e-att-serve-{uuid.uuid4().hex[:6]}",
                namespace=env["user_ns"],
                gateway_refs=[gateway_ref(tenant_b["gateway_name"])],
                teardown=teardown_resources,
            )
        )
        access = stack.enter_context(
            cm=serve_in_tenant_and_cleanup(
                admin_client, request_session_http, ocp_token_for_actor, tenant_b, model, teardown=teardown_resources
            )
        )
        yield {"model": model, "access": access}


@pytest.fixture(scope="class")
def unrelated_model(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token_for_actor: str,
    attachment_env: dict[str, Any],
    teardown_resources: bool,
) -> Generator[dict[str, Any], Any, Any]:
    """A bystander model on tenant B's gateway that must survive cleanup."""
    env = attachment_env
    tenant_b = env["tenant_b"]
    with ExitStack() as stack:
        model = stack.enter_context(
            cm=external_model_stack(
                admin_client,
                name=f"e2e-att-unrel-{uuid.uuid4().hex[:6]}",
                namespace=env["user_ns"],
                gateway_refs=[gateway_ref(tenant_b["gateway_name"])],
                teardown=teardown_resources,
            )
        )
        access = stack.enter_context(
            cm=serve_in_tenant_and_cleanup(
                admin_client, request_session_http, ocp_token_for_actor, tenant_b, model, teardown=teardown_resources
            )
        )
        yield {"model": model, "access": access}


@pytest.fixture(scope="class")
def refused_model(
    admin_client: DynamicClient,
    attachment_env: dict[str, Any],
    teardown_resources: bool,
) -> Generator[dict[str, Any], Any, Any]:
    """One model attached to tenant B's gateway from an unadmitted namespace."""
    env = attachment_env
    tenant_b = env["tenant_b"]
    with external_model_stack(
        admin_client,
        name=f"e2e-att-noaccess-{uuid.uuid4().hex[:6]}",
        namespace=env["unadmitted_ns"],
        gateway_refs=[gateway_ref(tenant_b["gateway_name"])],
        teardown=teardown_resources,
    ) as model:
        yield model


@pytest.fixture(scope="class")
def recovery_env(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token_for_actor: str,
    attachment_env: dict[str, Any],
    teardown_resources: bool,
) -> Generator[dict[str, Any], Any, Any]:
    """A refused model and a healthy control model, each in its own namespace.

    The namespace context managers start (and own teardown of) the
    namespaces right after creation: pytest runs no teardown after a
    failed setup, so provisioning or serving failures must still delete
    the namespaces.
    """
    env = attachment_env
    tenant_b = env["tenant_b"]
    with ExitStack() as stack:
        refused_ns = stack.enter_context(
            cm=unadmitted_user_namespace(
                admin_client,
                "e2e-extmod-rec-noaccess",
                teardown=teardown_resources,
            )
        )
        control_ns = stack.enter_context(
            cm=admitted_user_namespace(
                admin_client,
                "e2e-extmod-rec-user",
                tenant_b["gateway_name"],
                teardown=teardown_resources,
            )
        )
        recovering_model = stack.enter_context(
            cm=external_model_stack(
                admin_client,
                name=f"e2e-rec-recover-{uuid.uuid4().hex[:6]}",
                namespace=namespace_name(refused_ns),
                gateway_refs=[gateway_ref(tenant_b["gateway_name"])],
                teardown=teardown_resources,
            )
        )
        control_model = stack.enter_context(
            cm=external_model_stack(
                admin_client,
                name=f"e2e-rec-ctrl-{uuid.uuid4().hex[:6]}",
                namespace=namespace_name(control_ns),
                gateway_refs=[gateway_ref(tenant_b["gateway_name"])],
                teardown=teardown_resources,
            )
        )
        control_access = stack.enter_context(
            cm=serve_in_tenant_and_cleanup(
                admin_client,
                request_session_http,
                ocp_token_for_actor,
                tenant_b,
                control_model,
                teardown=teardown_resources,
            )
        )
        yield {
            "tenant_b": tenant_b,
            "refused_ns": refused_ns.name,
            "control_ns": control_ns.name,
            "recovering_model": recovering_model,
            "control_model": control_model,
            "control_access": control_access,
        }


@pytest.fixture(scope="class")
def rotation_model(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token_for_actor: str,
    attachment_env: dict[str, Any],
    teardown_resources: bool,
) -> Generator[dict[str, Any], Any, Any]:
    """One serving model whose source credential Secret will be rotated.

    Its own namespace: the rotation rewrites this Secret in place, and
    the module's shared user namespace hosts other classes' models. The
    namespace context manager owns its teardown right after creation:
    pytest runs no teardown after a failed setup, so provisioning or
    serving failures must still delete the namespace.
    """
    env = attachment_env
    tenant_b = env["tenant_b"]
    with ExitStack() as stack:
        user_ns = stack.enter_context(
            cm=admitted_user_namespace(
                admin_client,
                "e2e-extmod-rot-user",
                tenant_b["gateway_name"],
                teardown=teardown_resources,
            )
        )
        model = stack.enter_context(
            cm=external_model_stack(
                admin_client,
                name=f"e2e-rot-{uuid.uuid4().hex[:6]}",
                namespace=namespace_name(user_ns),
                gateway_refs=[gateway_ref(tenant_b["gateway_name"])],
                teardown=teardown_resources,
            )
        )
        access = stack.enter_context(
            cm=serve_in_tenant_and_cleanup(
                admin_client, request_session_http, ocp_token_for_actor, tenant_b, model, teardown=teardown_resources
            )
        )
        yield {
            "tenant_b": tenant_b,
            "user_ns": user_ns.name,
            "model": model,
            "access": access,
        }
