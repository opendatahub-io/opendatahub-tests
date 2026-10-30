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
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import (
    AITenantTestContext,
    build_aitenant_test_context,
)
from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    MAX_AITENANT_NAME_LENGTH,
    PER_TENANT_PROCESSING_DEPLOYMENT_PREFIX,
    PRAXIS_IMAGE_MARKER,
    deployment_image,
    get_json_or_none,
    record_tenant_external_model_evidence,
    require_tenant_external_model_support,
)
from tests.ai_gateway.models_as_a_service.multitenancy.utils import (
    isolation_bootstrap_gateway_context,
    tenant_gateway_external_route,
)
from tests.ai_gateway.models_as_a_service.utils import (
    aitenant_from_spec,
    bootstrap_gateway_ref,
    build_aitenant_spec,
    deploy_and_verify_aitenant_ready,
    host_from_ingress_domain,
    verify_maas_gateway_programmed,
)
from utilities.constants import MAAS_GATEWAY_NAMESPACE
from utilities.general import generate_random_name

TENANT_EXTERNAL_MODEL_NAME_PREFIX = "e2e-extmod"


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


def _per_tenant_processing_uses_praxis(
    admin_client: DynamicClient,
    tenant_name: str,
) -> bool:
    """Whether the tenant's payload-processing deployment runs praxis-extproc."""
    deployment = get_json_or_none(
        client=admin_client,
        kind="deployment",
        name=f"{PER_TENANT_PROCESSING_DEPLOYMENT_PREFIX}{tenant_name}",
        namespace=MAAS_GATEWAY_NAMESPACE,
    )
    return PRAXIS_IMAGE_MARKER in deployment_image(deployment)


def _extmod_tenant_test_params(tenant_label: str) -> dict[str, Any]:
    """AITenant name and spec for one external model suite tenant."""
    aitenant_name = f"{TENANT_EXTERNAL_MODEL_NAME_PREFIX}-{tenant_label}-{generate_random_name()}"
    assert len(aitenant_name) <= MAX_AITENANT_NAME_LENGTH, (
        f"tenant name {aitenant_name!r} is {len(aitenant_name)} characters; AGC per-tenant "
        f"praxis resource naming needs it to stay at or under {MAX_AITENANT_NAME_LENGTH} "
        "(63-char limit minus the longest resource base name)"
    )
    return {
        "aitenant_name": aitenant_name,
        "aitenant_spec": build_aitenant_spec(aitenant_name=aitenant_name),
    }


@pytest.fixture(scope="module")
def tenant_external_model_primary_test_params() -> dict[str, Any]:
    """Primary AITenant name and spec for the external model suite."""
    return _extmod_tenant_test_params(tenant_label="a")


@pytest.fixture(scope="module")
def tenant_external_model_secondary_test_params() -> dict[str, Any]:
    """Secondary AITenant name and spec for the external model suite."""
    return _extmod_tenant_test_params(tenant_label="b")


def _tenant_case(
    context: AITenantTestContext,
    gateway_name: str,
    maas_host: str,
) -> dict[str, str]:
    """Enrich an AITenant context with the external model suite's case keys."""
    gateway_host = f"{gateway_name}.{maas_host.removeprefix('maas.')}"
    return {
        "tenant_label_name": context["aitenant_name"],
        "tenant_ns": context["tenant_namespace_name"],
        "gateway_name": gateway_name,
        "gateway_host": gateway_host,
        "gateway_url": f"https://{gateway_host}",
        "maas_api_url": f"https://{gateway_host}/maas-api",
    }


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
            if not _per_tenant_processing_uses_praxis(admin_client, params["aitenant_name"]):
                pytest.skip(
                    f"per-tenant payload-processing for {params['aitenant_name']} is not "
                    "praxis-extproc; the AGC/Praxis multi-tenant contract is not active"
                )
            cases[key] = _tenant_case(
                context=build_aitenant_test_context(aitenant=aitenant),
                gateway_name=gateway.name,
                maas_host=maas_host,
            )
        yield cases
