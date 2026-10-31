"""Fixtures for multi-tenant external model tests (RHAI-12994 P0 port).

Provides two AGC/Praxis tenants with selector-admitting tenant gateways,
enriched with gateway/maas-api URLs, plus the capability gates that keep
the suite green (skipped) on clusters where the AGC companion or the
gatewayRefs additive schema is not distributed yet.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import (
    build_aitenant_test_context,
)
from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    extmod_tenant_test_params,
    per_tenant_processing_uses_praxis,
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
    aitenant_from_spec,
    bootstrap_gateway_ref,
    deploy_and_verify_aitenant_ready,
    host_from_ingress_domain,
    verify_maas_gateway_programmed,
)
from utilities.certificates_utils import get_ca_bundle

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="session", autouse=True)
def gateway_route_ca_bundle(admin_client: DynamicClient) -> Generator[str]:
    """Trust the cluster's router CA for gateway Route TLS.

    Tenant-gateway Route certificates are signed by the cluster's router CA,
    which is not in the runner's trust store; without this, requests-based
    inference (``infer_external_chat``) fails TLS verification. Resolves the
    bundle the way ``utilities.autogluon`` does and exports it through
    ``REQUESTS_CA_BUNDLE`` for the whole suite. Yields the bundle path, or
    ``""`` when none is needed or available (managed clusters).
    """
    previous = os.environ.get("REQUESTS_CA_BUNDLE")
    bundle = os.environ.get("TENANT_EXTERNAL_MODEL_CA_BUNDLE", "")
    if not bundle:
        try:
            bundle = get_ca_bundle(client=admin_client)
        except Exception as ex:  # noqa: BLE001 - trust resolution is best-effort
            LOGGER.warning(
                "Failed to resolve the OpenShift router CA bundle; gateway "
                "inference falls back to the system trust store",
                error=str(ex),
            )
    if bundle:
        os.environ["REQUESTS_CA_BUNDLE"] = bundle
    try:
        yield bundle
    finally:
        if previous is None:
            os.environ.pop("REQUESTS_CA_BUNDLE", None)
        else:
            os.environ["REQUESTS_CA_BUNDLE"] = previous


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
def tenant_external_model_primary_test_params() -> dict[str, Any]:
    """Primary AITenant name and spec for the external model suite."""
    return extmod_tenant_test_params(tenant_label="a")


@pytest.fixture(scope="module")
def tenant_external_model_secondary_test_params() -> dict[str, Any]:
    """Secondary AITenant name and spec for the external model suite."""
    return extmod_tenant_test_params(tenant_label="b")


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
    # Computed directly (not via the class-scoped maas_host fixture, which a
    # module-scoped fixture cannot depend on).
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
                cm=aitenant_from_spec(
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
            # Source-suite bootstrap_aitenant_tenant step: admit the tenant
            # namespace to its own gateway so tenant-own models' routes are
            # accepted (listeners select the per-gateway access label).
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
