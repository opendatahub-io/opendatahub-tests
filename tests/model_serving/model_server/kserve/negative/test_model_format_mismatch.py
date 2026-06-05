"""
Tests for InferenceService behavior when the declared model format is not
supported by the explicitly specified ServingRuntime.

KServe validates the model format against the runtime's ``supportedModelFormats``
before attempting to load. An unsupported format should be surfaced as
``FailedToLoad`` with ``InvalidSpec`` rather than silently succeeding or hanging.
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
class TestModelFormatMismatch:
    """KServe rejects an ISVC whose model format is not in the runtime's supportedModelFormats."""

    def test_isvc_reports_failed_to_load_with_wrong_model_format(
        self,
        admin_client: DynamicClient,
        wrong_model_format_ovms_isvc: InferenceService,
    ) -> None:
        """Given a model format not in the runtime's supportedModelFormats, KServe must reject the spec.

        When:
            An OVMS RawDeployment InferenceService is created declaring a format
            not listed in the runtime's ``supportedModelFormats``.

        Then:
            ``status.modelStatus`` reaches ``FailedToLoad`` with ``InvalidSpec``.
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
                transition_status="InvalidSpec",
            )
        finally:
            assert_kserve_control_plane_stable(
                admin_client=admin_client,
                applications_namespace=applications_namespace,
                prior_restart_totals=prior_restart_totals,
            )
