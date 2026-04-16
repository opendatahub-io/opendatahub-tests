"""
Tests for InferenceService behavior when S3 credentials are invalid.

This is distinct from ``platform/test_custom_resources.py`` (invalid *model path*
with a valid credential chain): here the *path* is valid but the secret carries
wrong AWS keys, matching a common customer misconfiguration (rotated keys, wrong
secret copy, or mixed dev/prod credentials).
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.kserve.negative.utils import (
    assert_kserve_control_plane_stable,
    snapshot_kserve_control_plane_restart_totals,
    wait_for_isvc_model_status_states,
)

pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.tier2
class TestInvalidS3Credentials:
    """KServe surfaces model load failure when the predictor cannot read S3."""

    def test_isvc_reports_failed_to_load_with_invalid_s3_credentials(
        self,
        admin_client: DynamicClient,
        invalid_s3_credentials_ovms_isvc: InferenceService,
    ) -> None:
        """Given a valid bucket path and invalid AWS keys, the ISVC must not silently succeed.

        When:
            An OVMS RawDeployment InferenceService is created with ``wait=False`` so the
            controller can reconcile while model download fails.

        Then:
            ``status.modelStatus`` reaches ``FailedToLoad`` with ``BlockedByFailedLoad``,
            so users and UIs see a clear failure instead of a stuck ``Pending`` state.
            ``kserve-controller-manager`` and ``odh-model-controller`` remain Available,
            show no CrashLoopBackOff, and do not accumulate new container restarts.

        Customer-style extensions (future work):
            - Wrong keys but correct secret *name* referenced from a different namespace.
            - Expired STS-style session tokens if the platform adds them.
            - TLS to custom S3 with ``serving.kserve.io/s3-verifyssl`` toggles.
            - Parallel healthy ISVC in the same namespace (neighbor isolation) under sibling Jiras.
        """
        applications_namespace: str = py_config["applications_namespace"]
        prior_restart_totals = snapshot_kserve_control_plane_restart_totals(
            admin_client=admin_client,
            applications_namespace=applications_namespace,
        )
        try:
            wait_for_isvc_model_status_states(
                isvc=invalid_s3_credentials_ovms_isvc,
                target_model_state="FailedToLoad",
                transition_status="BlockedByFailedLoad",
            )
        finally:
            assert_kserve_control_plane_stable(
                admin_client=admin_client,
                applications_namespace=applications_namespace,
                prior_restart_totals=prior_restart_totals,
            )
