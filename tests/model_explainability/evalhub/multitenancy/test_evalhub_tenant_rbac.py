import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount

from tests.model_explainability.evalhub.constants import (
    EVALHUB_JOB_CONFIG_CLUSTERROLE,
    EVALHUB_JOBS_WRITER_CLUSTERROLE,
    EVALHUB_MT_CR_NAME,
)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-tenant-rbac"},
        ),
    ],
    indirect=True,
)
@pytest.mark.model_explainability
class TestEvalHubTenantRBAC:
    """Verify the operator provisions correct RBAC in tenant namespaces.

    When a namespace is labeled with evalhub.trustyai.opendatahub.io/tenant,
    the operator should create:
    - Job ServiceAccount
    - jobs-writer RoleBinding (API SA → batch/jobs create,delete)
    - job-config RoleBinding (API SA → configmaps create,get,list)
    - Service CA ConfigMap (with inject-cabundle annotation)
    """

    def test_job_service_account_created(
        self,
        admin_client: DynamicClient,
        tenant_a_namespace: Namespace,
        model_namespace: Namespace,
        tenant_a_rbac_ready: None,
    ) -> None:
        """Operator creates a job ServiceAccount in the tenant namespace."""
        # Name pattern: {instance.Name}-{instance.Namespace}-job
        expected_prefix = f"{EVALHUB_MT_CR_NAME}-{model_namespace.name}-job"
        sas = list(
            ServiceAccount.get(
                client=admin_client,
                namespace=tenant_a_namespace.name,
            )
        )
        sa_names = [sa.name for sa in sas]
        assert any(name.startswith(expected_prefix) for name in sa_names), (
            f"Expected job ServiceAccount starting with '{expected_prefix}' "
            f"in {tenant_a_namespace.name}, found: {sa_names}"
        )

    def test_jobs_writer_role_binding(
        self,
        admin_client: DynamicClient,
        tenant_a_namespace: Namespace,
        model_namespace: Namespace,
        tenant_a_rbac_ready: None,
    ) -> None:
        """Operator creates a jobs-writer RoleBinding for the API SA."""
        rbs = list(
            RoleBinding.get(
                client=admin_client,
                namespace=tenant_a_namespace.name,
            )
        )
        job_writer_rbs = [
            rb
            for rb in rbs
            if rb.name.startswith(EVALHUB_MT_CR_NAME) and rb.instance.roleRef.name == EVALHUB_JOBS_WRITER_CLUSTERROLE
        ]
        assert len(job_writer_rbs) == 1, (
            f"Expected 1 jobs-writer RoleBinding for '{EVALHUB_MT_CR_NAME}', "
            f"found {len(job_writer_rbs)}: {[rb.name for rb in rbs]}"
        )
        rb = job_writer_rbs[0]

        # Verify roleRef points to the correct ClusterRole
        assert rb.instance.roleRef.kind == "ClusterRole"
        assert rb.instance.roleRef.name == EVALHUB_JOBS_WRITER_CLUSTERROLE, (
            f"Expected roleRef '{EVALHUB_JOBS_WRITER_CLUSTERROLE}', got '{rb.instance.roleRef.name}'"
        )

        # Verify subject is the API SA from the instance namespace
        subjects = rb.instance.subjects
        assert len(subjects) == 1, f"Expected 1 subject, got {len(subjects)}"
        assert subjects[0].kind == "ServiceAccount"
        assert subjects[0].name == f"{EVALHUB_MT_CR_NAME}-service"
        assert subjects[0].namespace == model_namespace.name, (
            f"Expected SA namespace '{model_namespace.name}', got '{subjects[0].namespace}'"
        )

    def test_job_config_role_binding(
        self,
        admin_client: DynamicClient,
        tenant_a_namespace: Namespace,
        model_namespace: Namespace,
        tenant_a_rbac_ready: None,
    ) -> None:
        """Operator creates a job-config RoleBinding for the API SA."""
        rbs = list(
            RoleBinding.get(
                client=admin_client,
                namespace=tenant_a_namespace.name,
            )
        )
        job_config_rbs = [
            rb
            for rb in rbs
            if rb.name.startswith(EVALHUB_MT_CR_NAME) and rb.instance.roleRef.name == EVALHUB_JOB_CONFIG_CLUSTERROLE
        ]
        assert len(job_config_rbs) == 1, (
            f"Expected 1 job-config RoleBinding for '{EVALHUB_MT_CR_NAME}', "
            f"found {len(job_config_rbs)}: {[rb.name for rb in rbs]}"
        )
        rb = job_config_rbs[0]

        assert rb.instance.roleRef.kind == "ClusterRole"
        assert rb.instance.roleRef.name == EVALHUB_JOB_CONFIG_CLUSTERROLE, (
            f"Expected roleRef '{EVALHUB_JOB_CONFIG_CLUSTERROLE}', got '{rb.instance.roleRef.name}'"
        )

        subjects = rb.instance.subjects
        assert len(subjects) == 1
        assert subjects[0].kind == "ServiceAccount"
        assert subjects[0].name == f"{EVALHUB_MT_CR_NAME}-service"
        assert subjects[0].namespace == model_namespace.name

    def test_service_ca_configmap_created(
        self,
        admin_client: DynamicClient,
        tenant_a_namespace: Namespace,
        tenant_a_rbac_ready: None,
    ) -> None:
        """Operator creates a service CA ConfigMap with inject-cabundle annotation."""
        cm_name = f"{EVALHUB_MT_CR_NAME}-service-ca"
        cm = ConfigMap(
            client=admin_client,
            name=cm_name,
            namespace=tenant_a_namespace.name,
        )
        assert cm.exists, f"Expected ConfigMap '{cm_name}' in {tenant_a_namespace.name}"
        annotations = cm.instance.metadata.annotations or {}
        assert annotations.get("service.beta.openshift.io/inject-cabundle") == "true", (
            f"Expected inject-cabundle annotation, got: {annotations}"
        )
