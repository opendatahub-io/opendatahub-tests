"""Negative API tests for EvalHub single-tenancy mode.

Verifies that tenant-scoped endpoints reject requests when X-Tenant points to an
unauthorized namespace or when the X-Tenant header is omitted entirely.

In single-tenancy mode, kube-rbac-proxy performs a SubjectAccessReview against
the X-Tenant namespace. If the caller's SA has no Role in that namespace, the
SAR returns NoOpinion → 400 unable_to_authorize_request. If no X-Tenant header
is sent at all, the server returns 400 immediately.

Run in isolation:
    pytest tests/ai_safety/evalhub/single_tenancy/test_evalhub_single_tenancy_api_negative.py -m ai_safety
"""

from typing import Literal

import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route

from tests.ai_safety.evalhub.constants import (
    EVALHUB_COLLECTIONS_PATH,
    EVALHUB_JOBS_PATH,
    EVALHUB_PROVIDERS_PATH,
)
from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    validate_evalhub_post_denied,
    validate_evalhub_post_no_tenant,
    validate_evalhub_request_denied,
    validate_evalhub_request_no_tenant,
)

TenantScenario = Literal["cross_namespace", "missing_tenant"]
HttpMethod = Literal["GET", "POST"]


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-st-xns"})],
    indirect=True,
)
@pytest.mark.tier3
@pytest.mark.ai_safety
class TestEvalHubSingleTenancyTenantHeaderRejection:
    """Tenant-scoped API calls with invalid or missing X-Tenant are rejected."""

    @pytest.mark.parametrize(
        ("api_path", "http_method", "tenant_scenario", "endpoint_label"),
        [
            pytest.param(
                EVALHUB_PROVIDERS_PATH,
                "GET",
                "cross_namespace",
                "providers",
                id="test_get_providers_cross_namespace",
            ),
            pytest.param(
                EVALHUB_PROVIDERS_PATH,
                "GET",
                "missing_tenant",
                "providers",
                id="test_get_providers_missing_tenant",
            ),
            pytest.param(
                EVALHUB_COLLECTIONS_PATH,
                "GET",
                "cross_namespace",
                "collections",
                id="test_get_collections_cross_namespace",
            ),
            pytest.param(
                EVALHUB_COLLECTIONS_PATH,
                "GET",
                "missing_tenant",
                "collections",
                id="test_get_collections_missing_tenant",
            ),
            pytest.param(
                EVALHUB_JOBS_PATH,
                "POST",
                "cross_namespace",
                "jobs",
                id="test_post_jobs_cross_namespace",
            ),
            pytest.param(
                EVALHUB_JOBS_PATH,
                "POST",
                "missing_tenant",
                "jobs",
                id="test_post_jobs_missing_tenant",
            ),
        ],
    )
    def test_tenant_header_rejection(
        self,
        api_path: str,
        http_method: HttpMethod,
        tenant_scenario: TenantScenario,
        endpoint_label: str,
        evalhub_st_ready: None,
        evalhub_st_route: Route,
        evalhub_st_ca_bundle_file: str,
        evalhub_st_user_token: str,
        model_namespace: Namespace,
        second_namespace: Namespace,
    ) -> None:
        """Given: an authenticated user SA in single-tenancy mode.

        When: a tenant-scoped API call uses an unauthorized X-Tenant or omits the header.

        Then: the request is rejected with 400 or 403 (cross-namespace) or 400 (missing header).
        """
        host = evalhub_st_route.host
        token = evalhub_st_user_token
        ca_bundle_file = evalhub_st_ca_bundle_file

        if http_method == "GET":
            if tenant_scenario == "cross_namespace":
                validate_evalhub_request_denied(
                    host=host,
                    token=token,
                    path=api_path,
                    ca_bundle_file=ca_bundle_file,
                    tenant=second_namespace.name,
                )
                return

            validate_evalhub_request_no_tenant(
                host=host,
                token=token,
                path=api_path,
                ca_bundle_file=ca_bundle_file,
            )
            return

        payload = build_evalhub_job_payload(
            model_service_name="evalhub-st-emulator",
            tenant_namespace=(second_namespace.name if tenant_scenario == "cross_namespace" else model_namespace.name),
            job_name=f"evalhub-st-neg-{endpoint_label}",
        )

        if tenant_scenario == "cross_namespace":
            validate_evalhub_post_denied(
                host=host,
                token=token,
                path=api_path,
                ca_bundle_file=ca_bundle_file,
                tenant=second_namespace.name,
                payload=payload,
            )
            return

        validate_evalhub_post_no_tenant(
            host=host,
            token=token,
            path=api_path,
            ca_bundle_file=ca_bundle_file,
            payload=payload,
        )
