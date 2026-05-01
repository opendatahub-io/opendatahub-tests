"""
Tests for InferenceService behavior when the model artifact is corrupted or truncated.

The storage-initializer may download the object successfully (the file exists in S3),
but the predictor container cannot parse or load it, producing a ``FailedToLoad``
status rather than silently hanging.
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


@pytest.mark.tier3
class TestCorruptedModelArtifacts:
    """KServe surfaces model load failure when the model artifact is invalid."""

    def test_isvc_reports_failed_to_load_with_corrupted_model(
        self,
        admin_client: DynamicClient,
        corrupted_model_ovms_isvc: InferenceService,
    ) -> None:
        """Given a valid S3 path containing a corrupted model, the ISVC must report failure.

        When:
            An OVMS RawDeployment InferenceService is created pointing to a
            zero-byte or non-existent model path in S3 so the storage-initializer
            or predictor fails to load the model.

        Then:
            ``status.modelStatus`` reaches ``FailedToLoad`` with ``BlockedByFailedLoad``.
            ``kserve-controller-manager`` and ``odh-model-controller`` remain Available,
            show no CrashLoopBackOff, and do not accumulate new container restarts.
        """
        applications_namespace: str = py_config["applications_namespace"]
        prior_restart_totals = snapshot_kserve_control_plane_restart_totals(
            admin_client=admin_client,
            applications_namespace=applications_namespace,
        )
        try:
            wait_for_isvc_model_status_states(
                isvc=corrupted_model_ovms_isvc,
                target_model_state="FailedToLoad",
                transition_status="BlockedByFailedLoad",
            )
        finally:
            assert_kserve_control_plane_stable(
                admin_client=admin_client,
                applications_namespace=applications_namespace,
                prior_restart_totals=prior_restart_totals,
            )
