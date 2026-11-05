"""Fixtures for multi-tenant external model tests (RHAI-12994 P0 port).

Provides two AGC/Praxis tenants with selector-admitting tenant gateways,
enriched with gateway/maas-api URLs, plus the capability gates that keep
the suite green (skipped) on clusters where the AGC companion or the
gatewayRefs additive schema is not distributed yet.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import (
    build_aitenant_test_context,
)
from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    admitted_user_namespace,
    deploy_aitenant_best_effort,
    extmod_tenant_test_params,
    per_tenant_processing_uses_praxis,
    provision_and_serve_in_tenant,
    record_tenant_external_model_evidence,
    require_tenant_external_model_support,
    tenant_case,
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
                pytest.skip(
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
