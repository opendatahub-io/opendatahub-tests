import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.namespace import Namespace
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from utilities.constants import Timeout
from tests.model_serving.model_server.upgrade.admission_check_upgrade_config import (
    AC_ADMISSION_CHECK_NAME,
    AC_CLUSTER_QUEUE,
)
from utilities.kueue_utils import (
    AdmissionCheck,
    ClusterQueue,
    Workload,
    approve_admission_check_on_workload,
    check_admission_check_active,
    check_cluster_queue_has_admission_check,
    check_workload_admitted,
    check_workload_quota_reserved,
)

pytestmark = [pytest.mark.kueue]

LOGGER = get_logger(name=__name__)


class TestAdmissionCheckPreUpgrade:
    """Pre-upgrade: submit Job gated by AdmissionCheck, verify it is blocked."""

    @pytest.mark.pre_upgrade
    def test_job_gated_by_admission_check(
        self,
        admin_client: DynamicClient,
        admission_check_namespace: Namespace,
        admission_check_job: Job,
        admission_check_workload: Workload,
    ) -> None:
        """Test steps:

        1. Verify the batch Job exists on the cluster.
        2. Verify Kueue created a Workload for the Job.
        3. Wait for Workload to have QuotaReserved=True.
        4. Verify Workload is NOT Admitted (blocked by AdmissionCheck).
        """
        assert admission_check_job.exists, f"Job '{admission_check_job.name}' not found"

        assert admission_check_workload is not None, f"No Workload found for Job '{admission_check_job.name}'"

        try:
            for workload in TimeoutSampler(
                wait_timeout=Timeout.TIMEOUT_2MIN,
                sleep=5,
                func=lambda: Workload(
                    client=admin_client,
                    name=admission_check_workload.name,
                    namespace=admission_check_namespace.name,
                ),
            ):
                if workload.exists and check_workload_quota_reserved(workload):
                    break
        except TimeoutExpiredError:
            pytest.fail(f"Workload '{admission_check_workload.name}' did not reach QuotaReserved=True")

        refreshed = Workload(
            client=admin_client,
            name=admission_check_workload.name,
            namespace=admission_check_namespace.name,
        )
        assert not check_workload_admitted(refreshed), "Workload should NOT be Admitted — AdmissionCheck is pending"

        LOGGER.info(
            f"[PRE-UPGRADE] PASS: Job is gated by AdmissionCheck, "
            f"job={admission_check_job.name}, workload={admission_check_workload.name}"
        )


class TestAdmissionCheckPostUpgrade:
    """Post-upgrade: verify AdmissionCheck still gates, then approve and validate admission."""

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="ac_exists")
    def test_admission_check_exists(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Test steps:

        1. Verify AdmissionCheck resource still exists after upgrade.
        2. Verify AdmissionCheck is still Active.
        3. Verify ClusterQueue still references the AdmissionCheck in its strategy.
        """
        ac = AdmissionCheck(client=admin_client, name=AC_ADMISSION_CHECK_NAME)
        assert ac.exists, f"AdmissionCheck '{AC_ADMISSION_CHECK_NAME}' not found after upgrade"
        assert check_admission_check_active(ac), (
            f"AdmissionCheck '{AC_ADMISSION_CHECK_NAME}' is not Active after upgrade"
        )

        cq = ClusterQueue(client=admin_client, name=AC_CLUSTER_QUEUE)
        assert cq.exists, f"ClusterQueue '{AC_CLUSTER_QUEUE}' not found after upgrade"
        assert check_cluster_queue_has_admission_check(cq, AC_ADMISSION_CHECK_NAME), (
            f"ClusterQueue '{AC_CLUSTER_QUEUE}' no longer references AdmissionCheck '{AC_ADMISSION_CHECK_NAME}'"
        )

        LOGGER.info(
            f"[POST-UPGRADE] PASS: AdmissionCheck and ClusterQueue strategy survived upgrade, "
            f"admission_check={AC_ADMISSION_CHECK_NAME}, cluster_queue={AC_CLUSTER_QUEUE}"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="ac_workload_still_gated", depends=["ac_exists"])
    def test_workload_still_gated(
        self,
        admin_client: DynamicClient,
        admission_check_namespace: Namespace,
        admission_check_workload: Workload,
    ) -> None:
        """Test steps:

        1. Verify the Workload still exists after upgrade.
        2. Verify it is still NOT Admitted (AdmissionCheck still pending).
        """
        assert admission_check_workload.exists, f"Workload '{admission_check_workload.name}' not found after upgrade"
        assert not check_workload_admitted(admission_check_workload), (
            "Workload should still be gated by AdmissionCheck after upgrade"
        )
        LOGGER.info(
            f"[POST-UPGRADE] PASS: Workload still gated by AdmissionCheck after upgrade, "
            f"workload={admission_check_workload.name}"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["ac_workload_still_gated"])
    def test_approve_and_admit(
        self,
        admin_client: DynamicClient,
        admission_check_namespace: Namespace,
        admission_check_job: Job,
        admission_check_workload: Workload,
    ) -> None:
        """Test steps:

        1. Approve AdmissionCheck by patching Workload status.
        2. Wait for Workload to become Admitted.
        3. Wait for Job to complete (unsuspended by Kueue).
        """
        approve_admission_check_on_workload(
            client=admin_client,
            workload=admission_check_workload,
            admission_check_name=AC_ADMISSION_CHECK_NAME,
        )
        LOGGER.info(
            f"[POST-UPGRADE] Approved AdmissionCheck on Workload, "
            f"admission_check={AC_ADMISSION_CHECK_NAME}, workload={admission_check_workload.name}"
        )

        try:
            for workload in TimeoutSampler(
                wait_timeout=Timeout.TIMEOUT_2MIN,
                sleep=5,
                func=lambda: Workload(
                    client=admin_client,
                    name=admission_check_workload.name,
                    namespace=admission_check_namespace.name,
                ),
            ):
                if workload.exists and check_workload_admitted(workload):
                    break
        except TimeoutExpiredError:
            pytest.fail(
                f"Workload '{admission_check_workload.name}' not Admitted after approving "
                f"AdmissionCheck '{AC_ADMISSION_CHECK_NAME}'"
            )
        LOGGER.info("[POST-UPGRADE] PASS: Workload admitted after AdmissionCheck approval")

        admission_check_job.wait_for_condition(
            condition="Complete",
            status="True",
            timeout=Timeout.TIMEOUT_2MIN,
        )
        LOGGER.info(f"[POST-UPGRADE] PASS: Job completed after admission, job={admission_check_job.name}")
