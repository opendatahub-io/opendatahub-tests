import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.model_explainability.evalhub.utils import (
    build_evalhub_job_payload,
    delete_evalhub_job,
    submit_evalhub_job,
    validate_evalhub_delete_denied,
    validate_evalhub_delete_no_tenant,
)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-job-delete-mt"},
        ),
    ],
    indirect=True,
)
@pytest.mark.model_explainability
class TestEvalHubJobDeleteMT:
    """Multi-tenancy tests for EvalHub job deletion (cancel).

    Three scenarios:
    - Authorized tenant: user with RBAC in tenant-a deletes a job → 200/204
    - Cross-tenant:      same user deletes for tenant-b → denied
    - Missing tenant:    DELETE without X-Tenant header → 400
    """

    def test_job_delete_authorized_tenant(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """User with evaluations-delete RBAC in tenant-a can delete a job."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="evalhub-delete-test-job",
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]

        response = delete_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        assert response.status_code in (200, 204), f"Expected 200 or 204 for job deletion, got {response.status_code}"

    def test_job_delete_cross_tenant_denied(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        tenant_b_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """User with RBAC in tenant-a is denied job deletion for tenant-b."""
        # Submit in tenant-a first to get a valid job ID
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="evalhub-delete-test-job",
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]

        # Try to delete via tenant-b
        validate_evalhub_delete_denied(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_b_namespace.name,
            job_id=job_id,
        )

    def test_job_delete_missing_tenant_rejected(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """Job deletion without X-Tenant header is rejected with 400."""
        # Submit in tenant-a first to get a valid job ID
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="evalhub-delete-test-job",
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]

        validate_evalhub_delete_no_tenant(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            job_id=job_id,
        )
