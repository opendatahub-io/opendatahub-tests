"""
Tests for InferenceService behavior when PVC cannot be provisioned.

A PVC referencing a non-existent StorageClass stays ``Pending`` indefinitely.
An InferenceService backed by that PVC should surface ``Ready=False`` with
volume-related reasons rather than silently hanging.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_server.kserve.negative.utils import (
    assert_kserve_control_plane_stable,
    snapshot_kserve_control_plane_restart_totals,
    wait_for_isvc_ready_false,
)
from utilities.constants import Timeout

pytestmark = [pytest.mark.rawdeployment]


@pytest.mark.tier2
class TestPvcFailures:
    """KServe surfaces failure when PVC storage cannot be provisioned."""

    def test_pvc_stays_pending_with_nonexistent_storage_class(
        self,
        bad_storage_class_pvc: PersistentVolumeClaim,
    ) -> None:
        """Given a PVC referencing a non-existent StorageClass, it must remain Pending.

        When:
            A PVC is created with ``storageClassName=nonexistent-sc``.

        Then:
            The PVC stays in ``Pending`` phase because no provisioner matches.
        """
        last_phase: str | None = None

        def _pvc_phase() -> str | None:
            bad_storage_class_pvc.update()
            status = getattr(bad_storage_class_pvc.instance, "status", None)
            return getattr(status, "phase", None) if status else None

        try:
            for last_phase in TimeoutSampler(
                wait_timeout=Timeout.TIMEOUT_2MIN,
                sleep=2,
                func=_pvc_phase,
            ):
                if last_phase == "Pending":
                    return
        except TimeoutExpiredError as exc:
            raise AssertionError(
                "Expected PVC phase Pending with non-existent StorageClass within timeout; "
                f"last observed phase: {last_phase!r}"
            ) from exc

    def test_isvc_reports_not_ready_with_bad_pvc(
        self,
        admin_client: DynamicClient,
        bad_pvc_ovms_isvc: InferenceService,
    ) -> None:
        """Given an ISVC backed by an unbound PVC, it must report Ready=False.

        When:
            An OVMS RawDeployment InferenceService is created with
            ``storage_uri=pvc://bad-sc-pvc/models/`` where the PVC is Pending.

        Then:
            ISVC conditions contain ``Ready=False`` with a volume-related reason.
            ``kserve-controller-manager`` and ``odh-model-controller`` remain Available,
            show no CrashLoopBackOff, and do not accumulate new container restarts.
        """
        applications_namespace: str = py_config["applications_namespace"]
        prior_restart_totals = snapshot_kserve_control_plane_restart_totals(
            admin_client=admin_client,
            applications_namespace=applications_namespace,
        )
        try:
            wait_for_isvc_ready_false(isvc=bad_pvc_ovms_isvc)
        finally:
            assert_kserve_control_plane_stable(
                admin_client=admin_client,
                applications_namespace=applications_namespace,
                prior_restart_totals=prior_restart_totals,
            )
