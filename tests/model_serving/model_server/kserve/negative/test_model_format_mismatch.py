"""
Tests for InferenceService behavior when the declared model format does not match the runtime.

Deploying a model with ``modelFormat=pytorch`` against an OVMS runtime (which
expects ONNX/OpenVINO IR) forces a format mismatch that KServe should surface
as ``FailedToLoad`` rather than leaving the ISVC in a stuck or ambiguous state.
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
class TestModelFormatMismatch:
    """KServe surfaces model load failure when the format doesn't match the runtime."""

    def test_isvc_reports_failed_to_load_with_wrong_model_format(
        self,
        admin_client: DynamicClient,
        wrong_model_format_ovms_isvc: InferenceService,
    ) -> None:
        """Given a model format incompatible with the runtime, the ISVC must not silently succeed.

        When:
            An OVMS RawDeployment InferenceService is created declaring ``pytorch``
            format while the runtime only supports ONNX / OpenVINO IR.

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
                isvc=wrong_model_format_ovms_isvc,
                target_model_state="FailedToLoad",
                transition_status="BlockedByFailedLoad",
            )
        finally:
            assert_kserve_control_plane_stable(
                admin_client=admin_client,
                applications_namespace=applications_namespace,
                prior_restart_totals=prior_restart_totals,
            )
