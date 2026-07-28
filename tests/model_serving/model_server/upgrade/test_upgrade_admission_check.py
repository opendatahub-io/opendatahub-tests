import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.namespace import Namespace

from tests.model_serving.model_server.upgrade.admission_check_upgrade_config import (
    AC_ADMISSION_CHECK_NAME,
    AC_CLUSTER_QUEUE,
)
from utilities.constants import Timeout
from utilities.kueue_utils import (
    ClusterQueue,
    Workload,
    approve_admission_check_on_workload,
    check_admission_check_active,
    check_cluster_queue_has_admission_check,
    check_workload_admitted,
    check_workload_quota_reserved,
    wait_for_workload_condition,
)
from utilities.resources.admission_check import AdmissionCheck

pytestmark = [pytest.mark.kueue]

LOGGER = structlog.get_logger(name=__name__)


class TestAdmissionCheckPreUpgrade:
    """Pre-upgrade: submit Job gated by AdmissionCheck, verify it is blocked."""

    @pytest.mark.pre_upgrade
    @pytest.mark.dependency(name="ac_pre_job_exists")
    def test_job_exists(
        self,
        admission_check_job: Job,
    ) -> None:
        """Verify the batch Job exists on the cluster."""
        assert admission_check_job.exists, f"Job '{admission_check_job.name}' not found"
        LOGGER.info(
            "[PRE-UPGRADE] PASS: Job exists",
            job=admission_check_job.name,
        )

    @pytest.mark.pre_upgrade
    @pytest.mark.dependency(name="ac_pre_quota_reserved", depends=["ac_pre_job_exists"])
    def test_workload_quota_reserved(
        self,
        admin_client: DynamicClient,
        admission_check_namespace: Namespace,
        admission_check_job: Job,
        admission_check_workload: Workload,
    ) -> None:
        """Verify Kueue created a Workload and it has QuotaReserved=True."""
        assert admission_check_workload is not None, f"No Workload found for Job '{admission_check_job.name}'"

        wait_for_workload_condition(
            client=admin_client,
            workload_name=admission_check_workload.name,
            namespace=admission_check_namespace.name,
            condition_check=check_workload_quota_reserved,
            condition_name="QuotaReserved=True",
        )

        LOGGER.info(
            "[PRE-UPGRADE] PASS: Workload has QuotaReserved=True",
            workload=admission_check_workload.name,
        )

    @pytest.mark.pre_upgrade
    @pytest.mark.dependency(depends=["ac_pre_quota_reserved"])
    def test_workload_not_admitted(
        self,
        admin_client: DynamicClient,
        admission_check_namespace: Namespace,
        admission_check_workload: Workload,
    ) -> None:
        """Verify Workload is NOT Admitted (blocked by AdmissionCheck)."""
        refreshed = Workload(
            client=admin_client,
            name=admission_check_workload.name,
            namespace=admission_check_namespace.name,
        )
        assert not check_workload_admitted(workload=refreshed), (
            "Workload should NOT be Admitted — AdmissionCheck is pending"
        )
        LOGGER.info(
            "[PRE-UPGRADE] PASS: Workload is gated by AdmissionCheck",
            workload=admission_check_workload.name,
        )


class TestAdmissionCheckPostUpgrade:
    """Post-upgrade: verify AdmissionCheck still gates, then approve and validate admission."""

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="ac_exists")
    def test_admission_check_exists(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify AdmissionCheck resource still exists and is Active after upgrade."""
        ac = AdmissionCheck(client=admin_client, name=AC_ADMISSION_CHECK_NAME)
        assert ac.exists, f"AdmissionCheck '{AC_ADMISSION_CHECK_NAME}' not found after upgrade"
        assert check_admission_check_active(admission_check=ac), (
            f"AdmissionCheck '{AC_ADMISSION_CHECK_NAME}' is not Active after upgrade"
        )
        LOGGER.info(
            "[POST-UPGRADE] PASS: AdmissionCheck survived upgrade and is Active",
            admission_check=AC_ADMISSION_CHECK_NAME,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="ac_cq_references_check", depends=["ac_exists"])
    def test_cluster_queue_references_admission_check(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify ClusterQueue still references the AdmissionCheck in its strategy."""
        cq = ClusterQueue(client=admin_client, name=AC_CLUSTER_QUEUE)
        assert cq.exists, f"ClusterQueue '{AC_CLUSTER_QUEUE}' not found after upgrade"
        assert check_cluster_queue_has_admission_check(
            cluster_queue=cq, admission_check_name=AC_ADMISSION_CHECK_NAME
        ), f"ClusterQueue '{AC_CLUSTER_QUEUE}' no longer references AdmissionCheck '{AC_ADMISSION_CHECK_NAME}'"
        LOGGER.info(
            "[POST-UPGRADE] PASS: ClusterQueue strategy survived upgrade",
            cluster_queue=AC_CLUSTER_QUEUE,
            admission_check=AC_ADMISSION_CHECK_NAME,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="ac_workload_still_gated", depends=["ac_cq_references_check"])
    def test_workload_still_gated(
        self,
        admission_check_workload: Workload,
    ) -> None:
        """Verify the Workload still exists and is NOT Admitted after upgrade."""
        assert admission_check_workload.exists, f"Workload '{admission_check_workload.name}' not found after upgrade"
        assert not check_workload_admitted(workload=admission_check_workload), (
            "Workload should still be gated by AdmissionCheck after upgrade"
        )
        LOGGER.info(
            "[POST-UPGRADE] PASS: Workload still gated by AdmissionCheck after upgrade",
            workload=admission_check_workload.name,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="ac_approved", depends=["ac_workload_still_gated"])
    def test_approve_and_admit(
        self,
        admin_client: DynamicClient,
        admission_check_namespace: Namespace,
        admission_check_workload: Workload,
    ) -> None:
        """Approve AdmissionCheck and verify Workload becomes Admitted."""
        approve_admission_check_on_workload(
            workload=admission_check_workload,
            admission_check_name=AC_ADMISSION_CHECK_NAME,
        )
        LOGGER.info(
            "[POST-UPGRADE] Approved AdmissionCheck on Workload",
            admission_check=AC_ADMISSION_CHECK_NAME,
            workload=admission_check_workload.name,
        )

        wait_for_workload_condition(
            client=admin_client,
            workload_name=admission_check_workload.name,
            namespace=admission_check_namespace.name,
            condition_check=check_workload_admitted,
            condition_name=f"Admitted after approving AdmissionCheck '{AC_ADMISSION_CHECK_NAME}'",
        )
        LOGGER.info("[POST-UPGRADE] PASS: Workload admitted after AdmissionCheck approval")

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["ac_approved"])
    def test_job_completes_after_admission(
        self,
        admission_check_job: Job,
    ) -> None:
        """Verify Job completes after Workload is admitted."""
        admission_check_job.wait_for_condition(
            condition="Complete",
            status="True",
            timeout=Timeout.TIMEOUT_2MIN,
        )
        LOGGER.info(
            "[POST-UPGRADE] PASS: Job completed after admission",
            job=admission_check_job.name,
        )
